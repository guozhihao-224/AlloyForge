use af_runtime::{Model, SessionBuilder};
use anyhow::Result;

pub fn build_app(model: Box<dyn Model>) -> Result<()> {
    let _session = SessionBuilder::new().build(model)?;
    // Server implementation placeholder
    Ok(())
}
