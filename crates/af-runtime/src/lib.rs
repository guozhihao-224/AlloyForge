use std::sync::Arc;

use af_core::Device;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub trait Model: Send + Sync {
    fn reset_state(&self) {}
    fn forward_step(&self, _input_ids: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("forward_step not implemented")
    }
}

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

    pub fn build(self, model: Arc<dyn Model>) -> Result<Session> {
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
    model: Arc<dyn Model>,
}

impl Session {
    pub fn generate(&mut self, _req: Request) -> Result<String> {
        let _device = self.device.as_ref().map(|d| d.name()).unwrap_or("cpu");
        let _flash = self.enable_flash_attn;
        self.model.reset_state();
        Ok(String::from("[placeholder response]"))
    }
}
