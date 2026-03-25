use std::path::Path;

use anyhow::Result;
use clap::Parser;

use mneme::cli::{Cli, Command};
use mneme::db::Database;
use mneme::embed::Embedder;
use mneme::ingest::Ingester;
use mneme::search::Searcher;
use mneme::types::{IngestOptions, SearchOptions};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = cli.resolve_db_path()?;

    let db = connect_db(&db_path, &cli.turso_url, &cli.turso_token).await?;
    db.initialize().await?;

    match cli.command {
        Command::Ingest {
            path,
            max_chunk_tokens,
            skip_embeddings,
        } => run_ingest(db, &path, max_chunk_tokens, skip_embeddings).await,

        Command::Search {
            query,
            limit,
            decay_days,
            no_fts,
            no_vector,
            json,
        } => run_search(db, &query, limit, decay_days, no_fts, no_vector, json).await,

        Command::Stats { json } => run_stats(db, &db_path, json).await,
    }
}

async fn connect_db(
    path: &Path,
    turso_url: &Option<String>,
    turso_token: &Option<String>,
) -> Result<Database> {
    match (turso_url, turso_token) {
        (Some(url), Some(token)) => {
            let db = Database::new_with_sync(path, url.clone(), token.clone()).await?;
            db.sync().await?;
            Ok(db)
        }
        _ => Ok(Database::new(path).await?),
    }
}

async fn run_ingest(
    db: Database,
    path: &Path,
    max_chunk_tokens: usize,
    skip_embeddings: bool,
) -> Result<()> {
    // Ingester requires an Embedder instance even when skip_embeddings is true.
    // Future improvement: support a no-op embedder mode.
    let embedder = Embedder::new()?;

    let mut ingester = Ingester::new(db, embedder);
    let options = IngestOptions {
        session_id: None,
        max_chunk_tokens,
        skip_embeddings,
    };

    if path.is_dir() {
        let results = ingester.ingest_directory(path, &options).await?;
        let total_chunks: usize = results.iter().map(|r| r.chunks_created).sum();
        let total_sessions = results.len();
        let skipped = results.iter().filter(|r| r.skipped).count();

        println!("Ingested {total_sessions} session(s), {total_chunks} chunk(s) created, {skipped} skipped.");
    } else {
        let result = ingester.ingest_file(path, &options).await?;
        if result.skipped {
            println!("Session {} already ingested, skipped.", result.session_id);
        } else {
            println!(
                "Ingested session {}: {} chunk(s) created.",
                result.session_id, result.chunks_created
            );
        }
    }

    Ok(())
}

async fn run_search(
    db: Database,
    query: &str,
    limit: usize,
    decay_days: f64,
    no_fts: bool,
    no_vector: bool,
    json: bool,
) -> Result<()> {
    let embedder = Embedder::new()?;
    let mut searcher = Searcher::new(db, embedder);

    let half_life = if decay_days > 0.0 {
        Some(decay_days)
    } else {
        None
    };

    let options = SearchOptions {
        query: query.to_string(),
        limit,
        time_decay_half_life_days: half_life,
        use_fts: !no_fts,
        use_vector: !no_vector,
    };

    let results = searcher.search(query, &options).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        println!(
            "--- [{}/{}] score: {:.4} | source: {:?} | {} ---",
            i + 1,
            results.len(),
            result.score,
            result.source,
            result.chunk.timestamp.format("%Y-%m-%d %H:%M"),
        );
        // Truncate long content for terminal display.
        let content = &result.chunk.content;
        if content.len() > 500 {
            println!("{}...\n", &content[..500]);
        } else {
            println!("{content}\n");
        }
    }

    Ok(())
}

async fn run_stats(db: Database, db_path: &Path, json: bool) -> Result<()> {
    let mut stats = db.get_stats().await?;

    // Fill in the database file size from the filesystem.
    if let Ok(metadata) = std::fs::metadata(db_path) {
        stats.db_size_bytes = metadata.len();
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
        return Ok(());
    }

    println!("mneme memory statistics");
    println!("=======================");
    println!("Chunks:    {}", stats.total_chunks);
    println!("Sessions:  {}", stats.total_sessions);
    println!("DB size:   {}", format_bytes(stats.db_size_bytes));
    if let Some(oldest) = stats.oldest_chunk_at {
        println!("Oldest:    {}", oldest.format("%Y-%m-%d %H:%M UTC"));
    }
    if let Some(latest) = stats.latest_chunk_at {
        println!("Latest:    {}", latest.format("%Y-%m-%d %H:%M UTC"));
    }

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
