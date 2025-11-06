use std::sync::Arc;

use af_core::Model;
use af_modalities::{Projector, VisionEncoder};
use anyhow::Result;

pub mod qwen2;

pub struct PlaceholderVisionModel {
    #[allow(dead_code)]
    vision: Arc<dyn VisionEncoder>,
    #[allow(dead_code)]
    projector: Arc<dyn Projector>,
}

impl PlaceholderVisionModel {
    pub fn new(vision: Arc<dyn VisionEncoder>, projector: Arc<dyn Projector>) -> Self {
        Self { vision, projector }
    }
}

impl Model for PlaceholderVisionModel {
    fn reset_state(&mut self) {
        // No state to reset for placeholder
    }

    fn forward_step(&mut self, _input_ids: &[u32]) -> Result<Vec<f32>> {
        Ok(Vec::new())
    }
}
