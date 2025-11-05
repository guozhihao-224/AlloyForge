use af_core::{
    AttentionOps, DType, Device as DeviceTrait, MatmulOps, RoPEOps, Tensor as TensorTrait,
};
use anyhow::{Result, bail};

fn map_dtype(dtype: candle_core::DType) -> DType {
    use candle_core::DType as CDType;
    match dtype {
        CDType::F32 => DType::F32,
        CDType::F16 => DType::F16,
        CDType::BF16 => DType::BF16,
        CDType::I64 => DType::I64,
        CDType::U32 => DType::U32,
        CDType::U8 => DType::U8,
        _ => DType::F32,
    }
}

#[derive(Clone)]
pub struct CandleDevice(pub candle_core::Device);

impl CandleDevice {
    pub fn cpu() -> Self {
        Self(candle_core::Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(index: usize) -> Result<Self> {
        Ok(Self(candle_core::Device::new_cuda(index)?))
    }

    #[cfg(feature = "metal")]
    pub fn metal(index: usize) -> Result<Self> {
        Ok(Self(candle_core::Device::new_metal(index)?))
    }

    pub fn inner(&self) -> &candle_core::Device {
        &self.0
    }
}

impl DeviceTrait for CandleDevice {
    fn name(&self) -> &'static str {
        "candle"
    }

    fn is_gpu(&self) -> bool {
        !matches!(self.0, candle_core::Device::Cpu)
    }

    fn clone_box(&self) -> Box<dyn DeviceTrait> {
        Box::new(self.clone())
    }
}

pub struct CandleTensor {
    inner: candle_core::Tensor,
    shape: Vec<usize>,
    dtype: DType,
    device: CandleDevice,
}

impl CandleTensor {
    pub fn from_tensor(tensor: candle_core::Tensor, device: CandleDevice) -> Self {
        let shape = tensor.dims().to_vec();
        let dtype = map_dtype(tensor.dtype());
        Self {
            inner: tensor,
            shape,
            dtype,
            device,
        }
    }

    pub fn inner(&self) -> &candle_core::Tensor {
        &self.inner
    }
}

impl TensorTrait for CandleTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &dyn DeviceTrait {
        &self.device
    }

    fn to_device(&self, _device: &dyn DeviceTrait) -> Result<Box<dyn TensorTrait>> {
        bail!("tensor to_device not implemented yet")
    }

    fn to_dtype(&self, _dt: DType) -> Result<Box<dyn TensorTrait>> {
        bail!("tensor to_dtype not implemented yet")
    }
}

pub struct CandleOps;

impl MatmulOps for CandleOps {
    fn matmul(&self, _a: &dyn TensorTrait, _b: &dyn TensorTrait) -> Result<Box<dyn TensorTrait>> {
        bail!("matmul not implemented yet")
    }
}

impl AttentionOps for CandleOps {
    fn attention(
        &self,
        _q: &dyn TensorTrait,
        _k: &dyn TensorTrait,
        _v: &dyn TensorTrait,
        _mask: Option<&dyn TensorTrait>,
        _scale: f32,
        _use_flash: bool,
    ) -> Result<Box<dyn TensorTrait>> {
        bail!("attention not implemented yet")
    }
}

impl RoPEOps for CandleOps {
    fn apply(
        &self,
        _q: &dyn TensorTrait,
        _k: &dyn TensorTrait,
        _cos: &dyn TensorTrait,
        _sin: &dyn TensorTrait,
    ) -> Result<(Box<dyn TensorTrait>, Box<dyn TensorTrait>)> {
        bail!("rope not implemented yet")
    }
}
