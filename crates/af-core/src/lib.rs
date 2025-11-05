use anyhow::Result;

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
