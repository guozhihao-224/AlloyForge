use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelConfig {
    pub vocab_size: usize,
}

pub fn load_config(_path: &str) -> Result<ModelConfig> {
    Ok(ModelConfig::default())
}

pub fn load_weights_mmap(_path: &str) -> Result<()> {
    Ok(())
}
