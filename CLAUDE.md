# mneme

Long-term memory engine for Claude Code, powered by libsql.

## Architecture

Rust CLI binary with the following modules:

| Module | Responsibility |
|--------|---------------|
| `types.rs` | Shared types and data structures |
| `db.rs` | libsql connection, schema, migrations (Turso embedded replica) |
| `embed.rs` | Vector embedding generation via ONNX Runtime |
| `ingest.rs` | Conversation JSONL parsing, chunking, and storage |
| `search.rs` | FTS5 keyword search + vector similarity + RRF fusion + time decay |
| `cli.rs` | clap CLI definition (ingest, search, stats subcommands) |
| `main.rs` | Entry point |

## Tech Stack

| Purpose | Crate |
|---------|-------|
| CLI | `clap` (derive API) |
| Database | `libsql` (Turso embedded replica for cross-device sync) |
| Embeddings | `fastembed` (multilingual-e5-small, auto-downloaded) |
| Async runtime | `tokio` |
| Error handling | `anyhow` (application) + `thiserror` (library errors) |
| Serialization | `serde` + `serde_json` |

## Commands

```
mneme ingest <path>    # Ingest a Claude Code conversation JSONL file
mneme search <query>   # Search past conversations (FTS5 + vector + RRF)
mneme stats            # Show memory statistics
```

## Integration with Claude Code

mneme integrates via Claude Code hooks:

- `session_end` hook → `mneme ingest` (auto-index conversations)
- `session_start` hook → `mneme search` (auto-inject relevant context)

## Conventions

- All code, comments, documentation, and commit messages MUST be in English
- Follow Rust idioms and best practices (clippy-clean, rustfmt-formatted)
- Use `thiserror` for typed errors in library code, `anyhow` at the application boundary
- Prefer explicit error handling over `.unwrap()` / `.expect()`
- Keep modules focused: one responsibility per file
