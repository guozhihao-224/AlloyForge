use anyhow::Result;
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I64,
    U32,
    U8,
}

pub trait Device: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_gpu(&self) -> bool;
    fn clone_box(&self) -> Box<dyn Device>;
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn Device> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub trait Tensor: Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn device(&self) -> &dyn Device;
    fn to_device(&self, dev: &dyn Device) -> Result<Box<dyn Tensor>>;
    fn to_dtype(&self, dt: DType) -> Result<Box<dyn Tensor>>;
    fn as_any(&self) -> &dyn Any;
}

pub trait MatmulOps: Send + Sync {
    fn matmul(&self, a: &dyn Tensor, b: &dyn Tensor) -> Result<Box<dyn Tensor>>;
}

pub trait AttentionOps: Send + Sync {
    fn attention(
        &self,
        q: &dyn Tensor,
        k: &dyn Tensor,
        v: &dyn Tensor,
        mask: Option<&dyn Tensor>,
        scale: f32,
        use_flash: bool,
    ) -> Result<Box<dyn Tensor>>;
}

pub trait RoPEOps: Send + Sync {
    fn apply(
        &self,
        q: &dyn Tensor,
        k: &dyn Tensor,
        cos: &dyn Tensor,
        sin: &dyn Tensor,
    ) -> Result<(Box<dyn Tensor>, Box<dyn Tensor>)>;
}

// ===== Model Abstraction =====

/// Core abstraction for language models
/// 
/// This trait defines the interface that all models must implement
/// to be used with the runtime system.
pub trait Model: Send + Sync {
    /// Reset the model's internal state (e.g., KV cache)
    fn reset_state(&mut self);
    
    /// Perform a forward pass for the given input token IDs
    /// Returns the logits for the next token prediction
    /// 
    /// # Arguments
    /// * `input_ids` - Input token IDs to process
    /// 
    /// # Returns
    /// A vector of logits with length equal to vocab_size
    fn forward_step(&mut self, input_ids: &[u32]) -> Result<Vec<f32>>;
    
    /// Get model configuration (optional, for introspection)
    fn config(&self) -> Option<ModelConfig> {
        None
    }
}

/// Model configuration metadata
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
}
