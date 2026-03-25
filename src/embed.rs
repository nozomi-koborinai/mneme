use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

/// Errors specific to embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("failed to initialize embedding model: {0}")]
    ModelInit(String),

    #[error("inference failed: {0}")]
    Inference(String),
}

/// Generates text embeddings using the multilingual-e5-small model via fastembed.
///
/// The model is downloaded automatically on first use and cached locally.
pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Load the multilingual-e5-small embedding model.
    ///
    /// On first invocation the model files are downloaded from Hugging Face
    /// and cached in the fastembed cache directory. Subsequent calls reuse
    /// the cached files.
    pub fn new() -> Result<Self, EmbedError> {
        let options = TextInitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_show_download_progress(true);

        let model =
            TextEmbedding::try_new(options).map_err(|e| EmbedError::ModelInit(e.to_string()))?;

        Ok(Self { model })
    }

    /// Generate an embedding vector for a single text.
    ///
    /// Following the E5 convention, the input is prefixed with `"query: "` by fastembed.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let results = self
            .model
            .embed(vec![text], None)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Inference("empty result from embedding".into()))
    }

    /// Generate embedding vectors for a batch of texts.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.model
            .embed(texts, None)
            .map_err(|e| EmbedError::Inference(e.to_string()))
    }
}
