use anyhow::Result;

pub trait VisionEncoder: Send + Sync {
    fn encode(&self, _pixels_nchw: &[f32]) -> Result<Vec<f32>> {
        anyhow::bail!("vision encoder not implemented")
    }
}

pub trait AudioEncoder: Send + Sync {
    fn encode(&self, _pcm: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        anyhow::bail!("audio encoder not implemented")
    }
}

pub trait Projector: Send + Sync {
    fn project(&self, _embeddings: &[f32]) -> Result<Vec<f32>> {
        anyhow::bail!("projector not implemented")
    }
}
