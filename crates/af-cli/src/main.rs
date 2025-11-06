use std::any::Any;
use std::sync::Arc;

use af_core::Device;
use af_runtime::{Model, Request, SessionBuilder};
use anyhow::Result;

struct DummyDevice;

impl Device for DummyDevice {
    fn name(&self) -> &'static str {
        "dummy"
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn Device> {
        Box::new(DummyDevice)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct EchoModel;

impl Model for EchoModel {
    fn reset_state(&mut self) {
        // No state to reset for echo model
    }

    fn forward_step(&mut self, _input_ids: &[u32]) -> Result<Vec<f32>> {
        Ok(vec![0.0; 1000]) // Return dummy logits
    }
}

fn main() -> Result<()> {
    let device: Arc<dyn Device> = Arc::new(DummyDevice);
    let model = Box::new(EchoModel);
    let mut session = SessionBuilder::new().device(device).build(model)?;
    let response = session.generate(Request {
        prompt: "hello".into(),
        max_tokens: 16,
        images: vec![],
        videos: vec![],
    })?;
    println!("{}", response);
    Ok(())
}
