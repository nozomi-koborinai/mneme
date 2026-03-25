use chrono::{DateTime, Utc};
use std::path::Path;

use crate::types::{Chunk, MemoryStats};

/// Errors specific to database operations.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("failed to open database: {0}")]
    Open(String),

    #[error("failed to initialize schema: {0}")]
    Schema(String),

    #[error("query failed: {0}")]
    Query(String),

    #[error("sync failed: {0}")]
    Sync(String),

    #[error("invalid data: {0}")]
    InvalidData(String),
}

/// Database handle wrapping a libsql connection.
pub struct Database {
    db: libsql::Database,
    conn: libsql::Connection,
}

impl Database {
    /// Connect to a local libsql database file.
    pub async fn new(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let db = libsql::Builder::new_local(path.as_ref())
            .build()
            .await
            .map_err(|e| DbError::Open(e.to_string()))?;

        let conn = db.connect().map_err(|e| DbError::Open(e.to_string()))?;

        Ok(Self { db, conn })
    }

    /// Connect to a libsql database with Turso remote sync (embedded replica).
    pub async fn new_with_sync(
        path: impl AsRef<Path>,
        url: impl Into<String>,
        token: impl Into<String>,
    ) -> Result<Self, DbError> {
        let db = libsql::Builder::new_remote_replica(path.as_ref(), url.into(), token.into())
            .build()
            .await
            .map_err(|e| DbError::Open(e.to_string()))?;

        let conn = db.connect().map_err(|e| DbError::Open(e.to_string()))?;

        Ok(Self { db, conn })
    }

    /// Create the database schema (tables, FTS5, triggers).
    ///
    /// Safe to call multiple times — uses `IF NOT EXISTS`.
    pub async fn initialize(&self) -> Result<(), DbError> {
        self.conn
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS chunks (
                    id         TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    timestamp  TEXT NOT NULL,
                    embedding  BLOB
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_session
                    ON chunks(session_id);

                CREATE INDEX IF NOT EXISTS idx_chunks_timestamp
                    ON chunks(timestamp);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(content, content_rowid='rowid', tokenize='trigram');

                -- Triggers to keep the FTS5 index in sync with the chunks table.
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                        VALUES ('delete', old.rowid, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                        VALUES ('delete', old.rowid, old.content);
                    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                ",
            )
            .await
            .map_err(|e| DbError::Schema(e.to_string()))?;

        Ok(())
    }

    /// Insert a chunk (with optional embedding) into the database.
    pub async fn insert_chunk(&self, chunk: &Chunk) -> Result<(), DbError> {
        let embedding_value: libsql::Value = match &chunk.embedding {
            Some(emb) => libsql::Value::Blob(embedding_to_bytes(emb)),
            None => libsql::Value::Null,
        };
        let timestamp_str = chunk.timestamp.to_rfc3339();

        self.conn
            .execute(
                "INSERT OR REPLACE INTO chunks (id, session_id, content, role, timestamp, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                libsql::params![
                    chunk.id.as_str(),
                    chunk.session_id.as_str(),
                    chunk.content.as_str(),
                    chunk.role.as_str(),
                    timestamp_str.as_str(),
                    embedding_value,
                ],
            )
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        Ok(())
    }

    /// Full-text search using the FTS5 index.
    pub async fn fts_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(Chunk, f64)>, DbError> {
        let mut rows = self
            .conn
            .query(
                "SELECT c.id, c.session_id, c.content, c.role, c.timestamp, c.embedding,
                        rank * -1.0 AS score
                 FROM chunks_fts f
                 JOIN chunks c ON c.rowid = f.rowid
                 WHERE chunks_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2",
                libsql::params![query, limit as i64],
            )
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DbError::Query(e.to_string()))?
        {
            let chunk = row_to_chunk(&row)?;
            let score: f64 = row
                .get::<f64>(6)
                .map_err(|e| DbError::InvalidData(e.to_string()))?;
            results.push((chunk, score));
        }
        Ok(results)
    }

    /// Vector similarity search using cosine similarity.
    ///
    /// Performs a brute-force scan — suitable for moderate dataset sizes.
    pub async fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(Chunk, f64)>, DbError> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, session_id, content, role, timestamp, embedding
                 FROM chunks
                 WHERE embedding IS NOT NULL",
                (),
            )
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut scored: Vec<(Chunk, f64)> = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DbError::Query(e.to_string()))?
        {
            let chunk = row_to_chunk(&row)?;
            if let Some(ref emb) = chunk.embedding {
                let sim = cosine_similarity(query_embedding, emb);
                scored.push((chunk, sim));
            }
        }

        // Sort descending by similarity and truncate.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    /// Return aggregate statistics about the database.
    pub async fn get_stats(&self) -> Result<MemoryStats, DbError> {
        let mut rows = self
            .conn
            .query(
                "SELECT
                    COUNT(*) AS total_chunks,
                    COUNT(DISTINCT session_id) AS total_sessions,
                    MIN(timestamp) AS oldest,
                    MAX(timestamp) AS latest
                 FROM chunks",
                (),
            )
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        let row = rows
            .next()
            .await
            .map_err(|e| DbError::Query(e.to_string()))?
            .ok_or_else(|| DbError::Query("empty result from stats query".into()))?;

        let total_chunks: i64 = row
            .get(0)
            .map_err(|e| DbError::InvalidData(e.to_string()))?;
        let total_sessions: i64 = row
            .get(1)
            .map_err(|e| DbError::InvalidData(e.to_string()))?;

        let oldest_str: Option<String> = row
            .get(2)
            .map_err(|e| DbError::InvalidData(e.to_string()))?;
        let latest_str: Option<String> = row
            .get(3)
            .map_err(|e| DbError::InvalidData(e.to_string()))?;

        let oldest_chunk_at = oldest_str.and_then(|s| parse_timestamp(&s));
        let latest_chunk_at = latest_str.and_then(|s| parse_timestamp(&s));

        Ok(MemoryStats {
            total_chunks: total_chunks as u64,
            total_sessions: total_sessions as u64,
            db_size_bytes: 0, // Populated by caller via filesystem stat.
            oldest_chunk_at,
            latest_chunk_at,
        })
    }

    /// Trigger a Turso sync for embedded replicas.
    ///
    /// No-op for local-only databases.
    pub async fn sync(&self) -> Result<(), DbError> {
        self.db
            .sync()
            .await
            .map_err(|e| DbError::Sync(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Serialize an f32 embedding vector to a little-endian byte blob.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

/// Deserialize a little-endian byte blob back to an f32 vector.
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Parse an RFC 3339 timestamp string into a `DateTime<Utc>`.
fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Extract a `Chunk` from a database row.
///
/// Expects columns: id(0), session_id(1), content(2), role(3), timestamp(4), embedding(5).
fn row_to_chunk(row: &libsql::Row) -> Result<Chunk, DbError> {
    let id: String = row
        .get(0)
        .map_err(|e| DbError::InvalidData(e.to_string()))?;
    let session_id: String = row
        .get(1)
        .map_err(|e| DbError::InvalidData(e.to_string()))?;
    let content: String = row
        .get(2)
        .map_err(|e| DbError::InvalidData(e.to_string()))?;
    let role: String = row
        .get(3)
        .map_err(|e| DbError::InvalidData(e.to_string()))?;
    let timestamp_str: String = row
        .get(4)
        .map_err(|e| DbError::InvalidData(e.to_string()))?;
    let embedding_blob: Option<Vec<u8>> = row
        .get(5)
        .map_err(|e| DbError::InvalidData(e.to_string()))?;

    let timestamp = parse_timestamp(&timestamp_str)
        .ok_or_else(|| DbError::InvalidData(format!("invalid timestamp: {timestamp_str}")))?;

    let embedding = embedding_blob.map(|b| bytes_to_embedding(&b));

    Ok(Chunk {
        id,
        session_id,
        content,
        role,
        timestamp,
        embedding,
    })
}
