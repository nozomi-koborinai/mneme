//! Hybrid search engine combining FTS5 keyword search, vector similarity search,
//! Reciprocal Rank Fusion (RRF), and exponential time decay scoring.

use std::collections::HashMap;

use chrono::Utc;

use crate::db::{Database, DbError};
use crate::embed::{EmbedError, Embedder};
use crate::types::{SearchOptions, SearchResult, SearchSource};

/// Errors that can occur during search operations.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("database error: {0}")]
    Database(#[from] DbError),

    #[error("embedding error: {0}")]
    Embedding(#[from] EmbedError),

    #[error("invalid query: {0}")]
    InvalidQuery(String),
}

/// Hybrid search engine that merges FTS5 keyword search and vector similarity
/// search using Reciprocal Rank Fusion (RRF) with exponential time decay.
pub struct Searcher {
    db: Database,
    embedder: Embedder,
}

impl Searcher {
    /// Create a new searcher backed by the given database and embedder.
    pub fn new(db: Database, embedder: Embedder) -> Self {
        Self { db, embedder }
    }

    /// Execute a hybrid search and return results ranked by fused score.
    ///
    /// The algorithm:
    /// 1. Run FTS5 keyword search and/or vector similarity search in parallel
    ///    (controlled by `options.use_fts` and `options.use_vector`).
    /// 2. Merge both ranked lists using Reciprocal Rank Fusion (RRF).
    /// 3. Apply exponential time decay to favour recent results.
    /// 4. Sort by final score descending and truncate to `options.limit`.
    pub async fn search(
        &mut self,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(SearchError::InvalidQuery(
                "search query must not be empty".into(),
            ));
        }

        if !options.use_fts && !options.use_vector {
            return Err(SearchError::InvalidQuery(
                "at least one search method (FTS or vector) must be enabled".into(),
            ));
        }

        // How many candidates to fetch from each backend. We over-fetch so that
        // RRF fusion has enough material to produce `limit` good results.
        let fetch_limit = options.limit * 3;

        // -- Run search backends -----------------------------------------------

        let fts_results = if options.use_fts {
            let escaped = escape_fts5_query(query);
            self.db.fts_search(&escaped, fetch_limit).await?
        } else {
            Vec::new()
        };

        let vector_results = if options.use_vector {
            let embedding = self.embedder.embed(query)?;
            self.db.vector_search(&embedding, fetch_limit).await?
        } else {
            Vec::new()
        };

        // -- If only one backend is active, skip RRF and go straight to decay --

        if !options.use_fts {
            return Ok(Self::apply_decay_and_collect(
                vector_results
                    .into_iter()
                    .map(|(chunk, _score)| (chunk, SearchSource::Vector)),
                options,
            ));
        }

        if !options.use_vector {
            return Ok(Self::apply_decay_and_collect(
                fts_results
                    .into_iter()
                    .map(|(chunk, _score)| (chunk, SearchSource::Fts5)),
                options,
            ));
        }

        // -- Reciprocal Rank Fusion --------------------------------------------

        let merged = rrf_merge(&fts_results, &vector_results);

        // -- Time decay and final assembly -------------------------------------

        Ok(Self::apply_decay_and_collect(merged.into_iter(), options))
    }

    /// Apply time decay to an iterator of (Chunk, SearchSource) pairs that
    /// already carry an implicit rank ordering, then sort and truncate.
    ///
    /// When called from single-backend paths the items are in score-descending
    /// order, so we assign 1-based ranks on the fly and compute RRF scores to
    /// keep scoring consistent across modes.
    fn apply_decay_and_collect(
        items: impl Iterator<Item = (crate::types::Chunk, SearchSource)>,
        options: &SearchOptions,
    ) -> Vec<SearchResult> {
        let now = Utc::now();
        let mut results: Vec<SearchResult> = items
            .enumerate()
            .map(|(rank, (chunk, source))| {
                // Use RRF-style scoring so that single-backend results are on a
                // comparable scale.
                let rrf_score = 1.0 / (RRF_K + rank as f64 + 1.0);
                let decay = time_decay(chunk.timestamp, now, options.time_decay_half_life_days);
                let final_score = rrf_score * decay;
                SearchResult {
                    chunk,
                    score: final_score,
                    source,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(options.limit);
        results
    }
}

// ---------------------------------------------------------------------------
// RRF fusion
// ---------------------------------------------------------------------------

/// The constant `k` in the RRF formula. A value of 60 is the standard default
/// from the original Cormack, Clarke & Buettcher (2009) paper. It controls how
/// much weight lower-ranked results receive: larger k → more uniform weighting.
const RRF_K: f64 = 60.0;

/// Intermediate entry used during RRF merging.
struct RrfEntry {
    chunk: crate::types::Chunk,
    rrf_score: f64,
    seen_fts: bool,
    seen_vector: bool,
}

/// Merge FTS5 and vector search results using Reciprocal Rank Fusion.
///
/// For each result list the RRF contribution of an item at 1-based rank `r` is:
///
/// ```text
/// score_i = 1 / (k + r)
/// ```
///
/// When the same chunk appears in both lists, its contributions are summed and
/// the source is marked as `Hybrid`.
fn rrf_merge(
    fts_results: &[(crate::types::Chunk, f64)],
    vector_results: &[(crate::types::Chunk, f64)],
) -> Vec<(crate::types::Chunk, SearchSource)> {
    let mut entries: HashMap<String, RrfEntry> = HashMap::new();

    // Score FTS5 results (1-based ranking).
    for (rank, (chunk, _score)) in fts_results.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f64 + 1.0);
        entries
            .entry(chunk.id.clone())
            .and_modify(|e| {
                e.rrf_score += rrf_score;
                e.seen_fts = true;
            })
            .or_insert(RrfEntry {
                chunk: chunk.clone(),
                rrf_score,
                seen_fts: true,
                seen_vector: false,
            });
    }

    // Score vector results (1-based ranking).
    for (rank, (chunk, _score)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f64 + 1.0);
        entries
            .entry(chunk.id.clone())
            .and_modify(|e| {
                e.rrf_score += rrf_score;
                e.seen_vector = true;
            })
            .or_insert(RrfEntry {
                chunk: chunk.clone(),
                rrf_score,
                seen_fts: false,
                seen_vector: true,
            });
    }

    // Sort by RRF score descending.
    let mut merged: Vec<_> = entries.into_values().collect();
    merged.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    merged
        .into_iter()
        .map(|e| {
            let source = match (e.seen_fts, e.seen_vector) {
                (true, true) => SearchSource::Hybrid,
                (true, false) => SearchSource::Fts5,
                (false, true) => SearchSource::Vector,
                // Unreachable: every entry was inserted from at least one list.
                (false, false) => SearchSource::Hybrid,
            };
            (e.chunk, source)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Time decay
// ---------------------------------------------------------------------------

/// Compute an exponential time-decay multiplier for a chunk.
///
/// The decay follows a half-life model:
///
/// ```text
/// decay = 0.5 ^ (age_days / half_life)
/// ```
///
/// - A chunk created *now* has decay = 1.0 (no penalty).
/// - A chunk `half_life` days old has decay = 0.5.
/// - A chunk `2 * half_life` days old has decay = 0.25, and so on.
///
/// If `half_life_days` is `None`, decay is disabled and the function returns 1.0.
fn time_decay(
    timestamp: chrono::DateTime<Utc>,
    now: chrono::DateTime<Utc>,
    half_life_days: Option<f64>,
) -> f64 {
    let half_life = match half_life_days {
        Some(hl) if hl > 0.0 => hl,
        _ => return 1.0,
    };

    let age_days = (now - timestamp).num_seconds().max(0) as f64 / 86_400.0;

    // 0.5^(age / half_life) = exp(age / half_life * ln(0.5))
    (age_days / half_life * std::f64::consts::LN_2.copysign(-1.0)).exp()
}

// ---------------------------------------------------------------------------
// FTS5 query escaping
// ---------------------------------------------------------------------------

/// Escape characters that have special meaning in FTS5 queries.
///
/// FTS5 with the trigram tokeniser treats most input literally, but certain
/// characters (double quotes, asterisks, carets, parentheses) can trigger
/// query syntax or prefix matching. We wrap each whitespace-separated token
/// in double quotes, escaping any embedded double quotes, so the query is
/// treated as a sequence of literal phrase matches joined by implicit AND.
fn escape_fts5_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|token| {
            let escaped = token.replace('"', "\"\"");
            format!("\"{escaped}\"")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_fts5_query_basic() {
        assert_eq!(escape_fts5_query("hello world"), "\"hello\" \"world\"");
    }

    #[test]
    fn test_escape_fts5_query_special_chars() {
        assert_eq!(escape_fts5_query("foo*bar"), "\"foo*bar\"");
        assert_eq!(escape_fts5_query("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn test_escape_fts5_query_empty() {
        assert_eq!(escape_fts5_query(""), "");
        assert_eq!(escape_fts5_query("   "), "");
    }

    #[test]
    fn test_time_decay_now() {
        let now = Utc::now();
        let decay = time_decay(now, now, Some(30.0));
        assert!((decay - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_decay_half_life() {
        let now = Utc::now();
        let thirty_days_ago = now - chrono::Duration::days(30);
        let decay = time_decay(thirty_days_ago, now, Some(30.0));
        assert!((decay - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_time_decay_disabled() {
        let now = Utc::now();
        let old = now - chrono::Duration::days(365);
        assert_eq!(time_decay(old, now, None), 1.0);
    }

    #[test]
    fn test_time_decay_negative_half_life() {
        let now = Utc::now();
        let old = now - chrono::Duration::days(30);
        // Non-positive half-life should disable decay.
        assert_eq!(time_decay(old, now, Some(0.0)), 1.0);
        assert_eq!(time_decay(old, now, Some(-5.0)), 1.0);
    }

    #[test]
    fn test_time_decay_future_timestamp() {
        let now = Utc::now();
        let future = now + chrono::Duration::days(10);
        // Future timestamps should not penalise — age is clamped to 0.
        let decay = time_decay(future, now, Some(30.0));
        assert!((decay - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_merge_deduplication() {
        use crate::types::Chunk;

        let chunk_a = Chunk {
            id: "a".into(),
            session_id: "s1".into(),
            content: "hello".into(),
            role: "human".into(),
            timestamp: Utc::now(),
            embedding: None,
        };
        let chunk_b = Chunk {
            id: "b".into(),
            session_id: "s1".into(),
            content: "world".into(),
            role: "assistant".into(),
            timestamp: Utc::now(),
            embedding: None,
        };

        // chunk_a appears in both lists → should be Hybrid
        let fts = vec![(chunk_a.clone(), 1.0), (chunk_b.clone(), 0.5)];
        let vec_results = vec![(chunk_a.clone(), 0.9)];

        let merged = rrf_merge(&fts, &vec_results);

        // chunk_a should have the highest RRF score (appears in both).
        assert_eq!(merged[0].0.id, "a");
        assert_eq!(merged[0].1, SearchSource::Hybrid);

        // chunk_b only appeared in FTS.
        let b_entry = merged.iter().find(|(c, _)| c.id == "b").unwrap();
        assert_eq!(b_entry.1, SearchSource::Fts5);

        // No duplicates.
        assert_eq!(merged.len(), 2);
    }
}
