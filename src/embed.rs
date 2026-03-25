use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

/// Errors specific to embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("failed to create ONNX Runtime session: {0}")]
    SessionInit(String),

    #[error("model not found at {0}")]
    ModelNotFound(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Generates text embeddings using the multilingual-e5-small ONNX model.
pub struct Embedder {
    session: Session,
}

impl Embedder {
    /// Load the multilingual-e5-small ONNX model.
    ///
    /// The model is expected at `~/.cache/mneme/models/multilingual-e5-small.onnx`.
    /// If not present, this returns an error indicating the model needs to be downloaded.
    pub fn new() -> Result<Self, EmbedError> {
        let model_path = Self::model_path()?;

        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound(format!(
                "model not found at {}. Download multilingual-e5-small.onnx and place it there.",
                model_path.display()
            )));
        }

        Self::from_path(&model_path)
    }

    /// Load an ONNX model from an explicit path.
    pub fn from_path(model_path: &Path) -> Result<Self, EmbedError> {
        let session = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(model_path))
            .map_err(|e| EmbedError::SessionInit(e.to_string()))?;

        Ok(Self { session })
    }

    /// Generate an embedding vector for a single text.
    ///
    /// Following the E5 convention, the input is prefixed with `"query: "`.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let results = self.embed_batch(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Inference("empty result from batch embedding".into()))
    }

    /// Generate embedding vectors for a batch of texts.
    ///
    /// Each input is prefixed with `"query: "` per the E5 convention.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let prefixed: Vec<String> = texts.iter().map(|t| format!("query: {t}")).collect();

        let array = ndarray::Array::from_shape_vec((prefixed.len(), 1), prefixed)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let string_tensor =
            Tensor::from_string_array(&array).map_err(|e| EmbedError::Inference(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![string_tensor])
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        // The output tensor has shape [batch_size, embedding_dim].
        let output_value = &outputs[0];
        let (shape, flat) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        if shape.len() < 2 {
            return Err(EmbedError::Inference(format!(
                "unexpected output shape: {shape:?}"
            )));
        }

        let embedding_dim = shape[shape.len() - 1] as usize;
        let embeddings: Vec<Vec<f32>> = flat.chunks_exact(embedding_dim).map(normalize).collect();

        Ok(embeddings)
    }

    /// Return the default model cache path.
    pub fn model_path() -> Result<PathBuf, EmbedError> {
        let home = std::env::var("HOME")
            .map_err(|_| EmbedError::Io(std::io::Error::other("HOME not set")))?;

        let path = PathBuf::from(home)
            .join(".cache")
            .join("mneme")
            .join("models")
            .join("multilingual-e5-small.onnx");

        Ok(path)
    }

    /// Ensure the model cache directory exists.
    pub fn ensure_cache_dir() -> Result<PathBuf, EmbedError> {
        let path = Self::model_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(path)
    }
}

/// L2-normalize a vector so cosine similarity becomes a dot product.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}
