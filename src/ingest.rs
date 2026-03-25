//! Conversation JSONL parsing, chunking, and storage.
//!
//! Parses Claude Code conversation files (JSONL format), groups messages into
//! meaningful Q&A chunks, generates embeddings, and stores them in the database.

use std::collections::HashSet;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::db::{Database, DbError};
use crate::embed::{EmbedError, Embedder};
use crate::types::{Chunk, IngestOptions};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during conversation ingestion.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    /// Failed to read the input file.
    #[error("failed to read file: {0}")]
    Io(#[from] std::io::Error),

    /// A JSONL line could not be parsed.
    #[error("invalid JSONL at line {line}: {reason}")]
    InvalidJsonl { line: usize, reason: String },

    /// Embedding generation failed.
    #[error("embedding error: {0}")]
    Embed(#[from] EmbedError),

    /// Database operation failed.
    #[error("database error: {0}")]
    Db(#[from] DbError),
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Summary of an ingestion run for a single file.
#[derive(Debug)]
pub struct IngestResult {
    /// The session identifier for the ingested conversation.
    pub session_id: String,
    /// Number of chunks created and stored.
    pub chunks_created: usize,
    /// `true` if the session was already ingested and was skipped.
    pub skipped: bool,
}

// ---------------------------------------------------------------------------
// JSONL schema (deserialization)
// ---------------------------------------------------------------------------

/// A single line in a Claude Code conversation JSONL file.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawEntry {
    /// Entry type — only "user" and "assistant" carry conversation messages.
    #[serde(rename = "type")]
    entry_type: Option<String>,

    /// The session this entry belongs to.
    session_id: Option<String>,

    /// RFC 3339 timestamp.
    timestamp: Option<String>,

    /// The chat message payload.
    message: Option<RawMessage>,
}

/// The `message` field inside a JSONL entry.
#[derive(Debug, Deserialize)]
struct RawMessage {
    role: Option<String>,
    content: Option<RawContent>,
}

/// Message content — Claude Code uses either a plain string or an array of
/// content blocks (text, tool_use, tool_result, thinking, etc.).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A single content block within an array-style message.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        name: Option<String>,
        /// Captured for deserialization but not stored.
        #[serde(default)]
        #[allow(dead_code)]
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult { content: Option<ToolResultContent> },
    /// Thinking blocks, progress updates, and any other types we do not need
    /// to store verbatim. Captured so deserialization does not fail.
    #[serde(other)]
    Other,
}

/// Tool result content can itself be a string or an array of typed items.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

/// An item inside a tool_result content array.
#[derive(Debug, Deserialize)]
struct ToolResultBlock {
    /// Captured for deserialization but not stored.
    #[serde(rename = "type")]
    #[allow(dead_code)]
    block_type: Option<String>,
    text: Option<String>,
    content: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A parsed, usable conversation message.
struct ParsedMessage {
    role: String,
    text: String,
    timestamp: DateTime<Utc>,
}

/// Extract the displayable text from a `RawContent` value.
fn extract_text(content: &RawContent) -> String {
    match content {
        RawContent::Text(s) => s.clone(),
        RawContent::Blocks(blocks) => {
            let mut parts: Vec<String> = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => {
                        parts.push(text.clone());
                    }
                    ContentBlock::ToolUse { name, .. } => {
                        if let Some(name) = name {
                            parts.push(format!("[tool: {name}]"));
                        }
                    }
                    ContentBlock::ToolResult { content } => {
                        if let Some(c) = content {
                            match c {
                                ToolResultContent::Text(t) => parts.push(t.clone()),
                                ToolResultContent::Blocks(bs) => {
                                    for b in bs {
                                        if let Some(t) = &b.text {
                                            parts.push(t.clone());
                                        } else if let Some(t) = &b.content {
                                            parts.push(t.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ContentBlock::Other => {}
                }
            }
            parts.join("\n")
        }
    }
}

/// Parse a timestamp string (RFC 3339 or ISO 8601) into `DateTime<Utc>`.
fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Derive a deterministic session ID from a file path.
///
/// Uses the file stem (UUID portion) of the JSONL filename. Falls back to a
/// SHA-256 hash of the full path if the stem is not available.
fn derive_session_id(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(String::from)
        .unwrap_or_else(|| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        })
}

/// Parse a JSONL file into a sequence of conversation messages.
fn parse_jsonl(contents: &str) -> Result<(String, Vec<ParsedMessage>), IngestError> {
    let mut messages: Vec<ParsedMessage> = Vec::new();
    let mut session_id: Option<String> = None;

    for (idx, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let entry: RawEntry =
            serde_json::from_str(line).map_err(|e| IngestError::InvalidJsonl {
                line: idx + 1,
                reason: e.to_string(),
            })?;

        // Capture session_id from the first entry that has one.
        if session_id.is_none() {
            if let Some(ref sid) = entry.session_id {
                session_id = Some(sid.clone());
            }
        }

        // Only process actual conversation messages.
        let entry_type = match entry.entry_type.as_deref() {
            Some("user") | Some("assistant") => entry.entry_type.as_deref().unwrap_or_default(),
            _ => continue,
        };

        let message = match entry.message {
            Some(m) => m,
            None => continue,
        };

        // Determine the role — prefer message.role, fall back to entry type.
        let role = message.role.as_deref().unwrap_or(entry_type);

        let content = match message.content {
            Some(c) => c,
            None => continue,
        };

        let text = extract_text(&content);
        if text.trim().is_empty() {
            continue;
        }

        // Skip meta/command messages (local commands, slash commands, etc.).
        if text.starts_with("<local-command")
            || text.starts_with("<command-name>")
            || text.starts_with("<local-command-stdout>")
            || text.starts_with("<local-command-caveat>")
        {
            continue;
        }

        let timestamp = entry
            .timestamp
            .as_deref()
            .and_then(parse_timestamp)
            .unwrap_or_else(Utc::now);

        messages.push(ParsedMessage {
            role: role.to_string(),
            text,
            timestamp,
        });
    }

    let sid = session_id.unwrap_or_default();
    Ok((sid, messages))
}

/// Group parsed messages into Q&A pair chunks.
///
/// Each chunk pairs a user message with the concatenated assistant responses
/// that follow it, producing a coherent unit for retrieval. Consecutive
/// assistant messages without a preceding user message are grouped together
/// as a standalone assistant chunk.
fn chunk_messages(
    messages: Vec<ParsedMessage>,
    session_id: &str,
    max_chunk_tokens: usize,
) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: usize = 0;

    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];

        if msg.role == "user" {
            // Collect the user message.
            let user_text = &msg.text;
            let user_ts = msg.timestamp;

            // Gather following assistant responses.
            let mut assistant_parts: Vec<&str> = Vec::new();
            let mut j = i + 1;
            while j < messages.len() && messages[j].role != "user" {
                assistant_parts.push(&messages[j].text);
                j += 1;
            }

            let combined = if assistant_parts.is_empty() {
                format!("Q: {user_text}")
            } else {
                let assistant_text = assistant_parts.join("\n");
                format!("Q: {user_text}\n\nA: {assistant_text}")
            };

            // Split into sub-chunks if the text is too long.
            let sub_chunks = split_by_token_estimate(&combined, max_chunk_tokens);
            for sub in sub_chunks {
                let id = format!("{session_id}:{chunk_index:04}");
                chunks.push(Chunk {
                    id,
                    session_id: session_id.to_string(),
                    content: sub,
                    role: "qa".to_string(),
                    timestamp: user_ts,
                    embedding: None,
                });
                chunk_index += 1;
            }

            i = j;
        } else {
            // Standalone assistant message (no preceding user message).
            let sub_chunks = split_by_token_estimate(&msg.text, max_chunk_tokens);
            for sub in sub_chunks {
                let id = format!("{session_id}:{chunk_index:04}");
                chunks.push(Chunk {
                    id,
                    session_id: session_id.to_string(),
                    content: sub,
                    role: msg.role.clone(),
                    timestamp: msg.timestamp,
                    embedding: None,
                });
                chunk_index += 1;
            }
            i += 1;
        }
    }

    chunks
}

/// Rough token count estimate: ~4 characters per token for English/mixed text.
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Split text into pieces that fit within the token budget.
///
/// Tries to split on paragraph boundaries first, then on sentence boundaries.
fn split_by_token_estimate(text: &str, max_tokens: usize) -> Vec<String> {
    if estimate_tokens(text) <= max_tokens {
        return vec![text.to_string()];
    }

    let max_chars = max_tokens * 4;
    let mut result: Vec<String> = Vec::new();
    let mut current = String::new();

    // Split on double newlines (paragraphs) first.
    for paragraph in text.split("\n\n") {
        if current.len() + paragraph.len() + 2 > max_chars && !current.is_empty() {
            result.push(current.trim().to_string());
            current = String::new();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        // If a single paragraph exceeds the limit, split further on sentences.
        if paragraph.len() > max_chars {
            if !current.is_empty() {
                result.push(current.trim().to_string());
                current = String::new();
            }
            let mut buf = String::new();
            for sentence in paragraph.split_inclusive(". ") {
                if buf.len() + sentence.len() > max_chars && !buf.is_empty() {
                    result.push(buf.trim().to_string());
                    buf = String::new();
                }
                buf.push_str(sentence);
            }
            if !buf.trim().is_empty() {
                result.push(buf.trim().to_string());
            }
        } else {
            current.push_str(paragraph);
        }
    }

    if !current.trim().is_empty() {
        result.push(current.trim().to_string());
    }

    // Filter out any empty strings that may have slipped through.
    result.retain(|s| !s.is_empty());
    result
}

// ---------------------------------------------------------------------------
// Ingester
// ---------------------------------------------------------------------------

/// Orchestrates conversation ingestion: parsing, chunking, embedding, and storage.
pub struct Ingester {
    db: Database,
    embedder: Embedder,
}

impl Ingester {
    /// Create a new `Ingester` with the given database and embedder.
    pub fn new(db: Database, embedder: Embedder) -> Self {
        Self { db, embedder }
    }

    /// Ingest a single Claude Code conversation JSONL file.
    ///
    /// Returns an [`IngestResult`] summarising what was done. If the session
    /// has already been ingested (detected via deterministic chunk IDs and
    /// `INSERT OR REPLACE`), the operation is idempotent.
    pub async fn ingest_file(
        &mut self,
        path: &Path,
        options: &IngestOptions,
    ) -> Result<IngestResult, IngestError> {
        let contents = tokio::fs::read_to_string(path).await?;

        let (file_session_id, messages) = parse_jsonl(&contents)?;

        let session_id = options
            .session_id
            .clone()
            .or(if file_session_id.is_empty() {
                None
            } else {
                Some(file_session_id)
            })
            .unwrap_or_else(|| derive_session_id(path));

        if messages.is_empty() {
            return Ok(IngestResult {
                session_id,
                chunks_created: 0,
                skipped: false,
            });
        }

        let mut chunks = chunk_messages(messages, &session_id, options.max_chunk_tokens);

        // Generate embeddings unless told to skip.
        if !options.skip_embeddings {
            self.generate_embeddings(&mut chunks)?;
        }

        // Store chunks in the database.
        for chunk in &chunks {
            self.db.insert_chunk(chunk).await?;
        }

        let count = chunks.len();
        Ok(IngestResult {
            session_id,
            chunks_created: count,
            skipped: false,
        })
    }

    /// Ingest all `.jsonl` files in a directory (non-recursive by default,
    /// but also scans one level of subdirectories to match Claude Code's
    /// session/subagents layout).
    pub async fn ingest_directory(
        &mut self,
        dir: &Path,
        options: &IngestOptions,
    ) -> Result<Vec<IngestResult>, IngestError> {
        let mut results = Vec::new();
        let mut seen_sessions: HashSet<String> = HashSet::new();

        let jsonl_files = collect_jsonl_files(dir).await?;

        for path in jsonl_files {
            match self.ingest_file(&path, options).await {
                Ok(result) => {
                    if seen_sessions.contains(&result.session_id) {
                        // Already processed this session in this run.
                        results.push(IngestResult {
                            session_id: result.session_id,
                            chunks_created: 0,
                            skipped: true,
                        });
                    } else {
                        seen_sessions.insert(result.session_id.clone());
                        results.push(result);
                    }
                }
                Err(IngestError::InvalidJsonl { .. }) => {
                    // Skip malformed files in batch mode — log and continue.
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(results)
    }

    /// Generate embeddings for a batch of chunks, mutating them in place.
    fn generate_embeddings(&mut self, chunks: &mut [Chunk]) -> Result<(), IngestError> {
        const BATCH_SIZE: usize = 32;

        for batch_start in (0..chunks.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(chunks.len());
            let texts: Vec<&str> = chunks[batch_start..batch_end]
                .iter()
                .map(|c| c.content.as_str())
                .collect();

            let embeddings = self.embedder.embed_batch(&texts)?;

            for (chunk, embedding) in chunks[batch_start..batch_end]
                .iter_mut()
                .zip(embeddings.into_iter())
            {
                chunk.embedding = Some(embedding);
            }
        }

        Ok(())
    }
}

/// Collect all `.jsonl` files under `dir`, scanning up to two levels deep
/// (to cover Claude Code's `<session-uuid>.jsonl` and `subagents/*.jsonl`).
async fn collect_jsonl_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, IngestError> {
    let mut files = Vec::new();
    let mut read_dir = tokio::fs::read_dir(dir).await?;

    while let Some(entry) = read_dir.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            files.push(path);
        } else if path.is_dir() {
            // One level deeper (e.g., <uuid>/subagents/).
            if let Ok(mut sub_dir) = tokio::fs::read_dir(&path).await {
                while let Some(sub_entry) = sub_dir.next_entry().await? {
                    let sub_path = sub_entry.path();
                    if sub_path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                        files.push(sub_path);
                    } else if sub_path.is_dir() {
                        // Two levels deep (e.g., <uuid>/subagents/*.jsonl).
                        if let Ok(mut deep_dir) = tokio::fs::read_dir(&sub_path).await {
                            while let Some(deep_entry) = deep_dir.next_entry().await? {
                                let deep_path = deep_entry.path();
                                if deep_path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                                    files.push(deep_path);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort for deterministic processing order.
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_plain_string() {
        let content = RawContent::Text("hello world".into());
        assert_eq!(extract_text(&content), "hello world");
    }

    #[test]
    fn test_extract_text_blocks() {
        let content = RawContent::Blocks(vec![
            ContentBlock::Text {
                text: "first".into(),
            },
            ContentBlock::Text {
                text: "second".into(),
            },
        ]);
        assert_eq!(extract_text(&content), "first\nsecond");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn test_split_by_token_estimate_short() {
        let text = "short text";
        let result = split_by_token_estimate(text, 512);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "short text");
    }

    #[test]
    fn test_split_by_token_estimate_long() {
        // Create text that exceeds 10 tokens (~40 chars).
        let text = "a".repeat(200);
        let result = split_by_token_estimate(&text, 10);
        assert!(!result.is_empty());
        for chunk in &result {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_derive_session_id() {
        let path = Path::new("/some/path/abc-def-123.jsonl");
        assert_eq!(derive_session_id(path), "abc-def-123");
    }

    #[test]
    fn test_parse_jsonl_skips_non_message_types() {
        let jsonl = r#"{"type":"file-history-snapshot","sessionId":"s1","timestamp":"2025-01-01T00:00:00Z"}
{"type":"user","sessionId":"s1","timestamp":"2025-01-01T00:00:01Z","message":{"role":"user","content":"hello"}}
{"type":"assistant","sessionId":"s1","timestamp":"2025-01-01T00:00:02Z","message":{"role":"assistant","content":"hi there"}}
{"type":"progress","sessionId":"s1","timestamp":"2025-01-01T00:00:03Z"}"#;

        let (sid, msgs) = parse_jsonl(jsonl).unwrap();
        assert_eq!(sid, "s1");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
    }

    #[test]
    fn test_chunk_messages_qa_pairs() {
        let messages = vec![
            ParsedMessage {
                role: "user".into(),
                text: "What is Rust?".into(),
                timestamp: Utc::now(),
            },
            ParsedMessage {
                role: "assistant".into(),
                text: "Rust is a systems programming language.".into(),
                timestamp: Utc::now(),
            },
            ParsedMessage {
                role: "user".into(),
                text: "How do I install it?".into(),
                timestamp: Utc::now(),
            },
            ParsedMessage {
                role: "assistant".into(),
                text: "Use rustup.".into(),
                timestamp: Utc::now(),
            },
        ];

        let chunks = chunk_messages(messages, "test-session", 512);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.starts_with("Q: What is Rust?"));
        assert!(chunks[0].content.contains("A: Rust is a systems"));
        assert_eq!(chunks[0].id, "test-session:0000");
        assert_eq!(chunks[1].id, "test-session:0001");
        assert_eq!(chunks[0].role, "qa");
    }

    #[test]
    fn test_parse_jsonl_with_block_content() {
        let jsonl = r#"{"type":"assistant","sessionId":"s1","timestamp":"2025-01-01T00:00:00Z","message":{"role":"assistant","content":[{"type":"text","text":"hello from blocks"}]}}"#;

        let (_sid, msgs) = parse_jsonl(jsonl).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text, "hello from blocks");
    }

    #[test]
    fn test_parse_jsonl_skips_meta_messages() {
        let jsonl = r#"{"type":"user","sessionId":"s1","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"<local-command-caveat>some caveat</local-command-caveat>"}}"#;

        let (_sid, msgs) = parse_jsonl(jsonl).unwrap();
        assert!(msgs.is_empty());
    }
}
