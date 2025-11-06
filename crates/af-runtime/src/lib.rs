use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};

// Re-export core types for convenience
pub use af_core::{Device, Model, ModelConfig};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Request {
    pub prompt: String,
    pub max_tokens: usize,
    pub images: Vec<String>,
    pub videos: Vec<String>,
}

pub struct SessionBuilder {
    device: Option<Arc<dyn Device>>,
    enable_flash_attn: bool,
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBuilder {
    pub fn new() -> Self {
        Self {
            device: None,
            enable_flash_attn: false,
        }
    }

    pub fn device(mut self, device: Arc<dyn Device>) -> Self {
        self.device = Some(device);
        self
    }

    pub fn enable_flash_attn(mut self, enable: bool) -> Self {
        self.enable_flash_attn = enable;
        self
    }

    pub fn build(self, model: Box<dyn Model>) -> Result<Session> {
        Ok(Session {
            device: self.device,
            enable_flash_attn: self.enable_flash_attn,
            model,
        })
    }
}

pub struct Session {
    device: Option<Arc<dyn Device>>,
    enable_flash_attn: bool,
    model: Box<dyn Model>,
}

impl Session {
    pub fn generate(&mut self, _req: Request) -> Result<String> {
        let _device = self.device.as_ref().map(|d| d.name()).unwrap_or("cpu");
        let _flash = self.enable_flash_attn;
        self.model.reset_state();
        
        // TODO: Implement full generation loop
        // 1. Tokenize prompt
        // 2. Prefill phase
        // 3. Decode loop with sampling
        // 4. Decode tokens back to text
        
        Ok(String::from("[placeholder response]"))
    }
}
