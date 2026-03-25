use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A chunk of conversation content with its embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk.
    pub id: String,
    /// Session identifier grouping related chunks.
    pub session_id: String,
    /// The text content of the chunk.
    pub content: String,
    /// The role of the speaker (e.g. "human", "assistant").
    pub role: String,
    /// When this chunk was created.
    pub timestamp: DateTime<Utc>,
    /// The embedding vector for semantic search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// The source of a search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchSource {
    /// Full-text search via FTS5.
    Fts5,
    /// Vector similarity search.
    Vector,
    /// Combined hybrid search with RRF fusion.
    Hybrid,
}

/// A single search result with its relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matched chunk.
    pub chunk: Chunk,
    /// Relevance score (higher is more relevant).
    pub score: f64,
    /// Which search method produced this result.
    pub source: SearchSource,
}

/// Statistics about the memory database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of stored chunks.
    pub total_chunks: u64,
    /// Total number of distinct sessions.
    pub total_sessions: u64,
    /// Database file size in bytes.
    pub db_size_bytes: u64,
    /// Timestamp of the most recent chunk.
    pub latest_chunk_at: Option<DateTime<Utc>>,
    /// Timestamp of the oldest chunk.
    pub oldest_chunk_at: Option<DateTime<Utc>>,
}

/// Options controlling how conversations are ingested.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestOptions {
    /// Override the session ID instead of deriving it from the file.
    pub session_id: Option<String>,
    /// Maximum number of tokens per chunk before splitting.
    pub max_chunk_tokens: usize,
    /// Whether to skip generating embeddings during ingest.
    pub skip_embeddings: bool,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            session_id: None,
            max_chunk_tokens: 512,
            skip_embeddings: false,
        }
    }
}

/// Options controlling how searches are performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    /// The search query string.
    pub query: String,
    /// Maximum number of results to return.
    pub limit: usize,
    /// Half-life in days for time-decay scoring. Older chunks score lower.
    /// `None` disables time decay.
    pub time_decay_half_life_days: Option<f64>,
    /// Whether to use FTS5 keyword search.
    pub use_fts: bool,
    /// Whether to use vector similarity search.
    pub use_vector: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            limit: 10,
            time_decay_half_life_days: Some(30.0),
            use_fts: true,
            use_vector: true,
        }
    }
}
