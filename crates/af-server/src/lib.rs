use af_runtime::{Model, SessionBuilder};
use anyhow::Result;
use std::sync::Arc;

pub fn build_app(model: Arc<dyn Model>) -> Result<()> {
    let _session = SessionBuilder::new().build(model)?;
    // Server implementation placeholder
    Ok(())
}
