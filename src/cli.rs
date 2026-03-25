//! CLI definition using clap derive API.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Long-term memory engine for Claude Code, powered by libsql.
#[derive(Parser)]
#[command(name = "mneme", version, about)]
pub struct Cli {
    /// Path to the database file.
    ///
    /// Defaults to `~/.local/share/mneme/mneme.db`.
    #[arg(long, env = "MNEME_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Turso database URL for cross-device sync.
    #[arg(long, env = "MNEME_TURSO_URL")]
    pub turso_url: Option<String>,

    /// Turso authentication token.
    #[arg(long, env = "MNEME_TURSO_TOKEN")]
    pub turso_token: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Ingest Claude Code conversation JSONL files.
    Ingest {
        /// Path to a JSONL file or directory containing JSONL files.
        path: PathBuf,

        /// Maximum tokens per chunk before splitting.
        #[arg(long, default_value = "512")]
        max_chunk_tokens: usize,

        /// Skip embedding generation (faster, but disables vector search for ingested chunks).
        #[arg(long)]
        skip_embeddings: bool,
    },

    /// Search past conversations.
    Search {
        /// The search query.
        query: String,

        /// Maximum number of results to return.
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Time decay half-life in days. Set to 0 to disable.
        #[arg(long, default_value = "30")]
        decay_days: f64,

        /// Disable FTS5 keyword search (vector only).
        #[arg(long)]
        no_fts: bool,

        /// Disable vector similarity search (FTS5 only).
        #[arg(long)]
        no_vector: bool,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Show memory database statistics.
    Stats {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
}

impl Cli {
    /// Resolve the database path, creating parent directories if needed.
    pub fn resolve_db_path(&self) -> anyhow::Result<PathBuf> {
        let path = match &self.db_path {
            Some(p) => p.clone(),
            None => default_db_path()?,
        };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(path)
    }
}

/// Return the default database path: `~/.local/share/mneme/mneme.db`.
fn default_db_path() -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| anyhow::anyhow!("HOME environment variable is not set"))?;

    Ok(PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("mneme")
        .join("mneme.db"))
}
