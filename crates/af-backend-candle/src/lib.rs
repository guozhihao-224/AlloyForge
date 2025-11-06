use af_core::{
    AttentionOps, DType, Device as DeviceTrait, MatmulOps, RoPEOps, Tensor as TensorTrait,
};
use anyhow::{Context, Result};
use std::any::Any;

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

    fn as_any(&self) -> &dyn Any {
        self
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

    /// Helper method to downcast from trait object
    pub fn try_from_dyn(tensor: &dyn TensorTrait) -> Result<&Self> {
        tensor
            .as_any()
            .downcast_ref::<Self>()
            .context("Expected CandleTensor")
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

    fn to_device(&self, device: &dyn DeviceTrait) -> Result<Box<dyn TensorTrait>> {
        let target = device
            .as_any()
            .downcast_ref::<CandleDevice>()
            .context("Expected CandleDevice")?;

        let new_tensor = self.inner.to_device(target.inner())?;
        Ok(Box::new(Self::from_tensor(new_tensor, target.clone())))
    }

    fn to_dtype(&self, dt: DType) -> Result<Box<dyn TensorTrait>> {
        use candle_core::DType as CDType;
        let target_dtype = match dt {
            DType::F32 => CDType::F32,
            DType::F16 => CDType::F16,
            DType::BF16 => CDType::BF16,
            DType::I64 => CDType::I64,
            DType::U32 => CDType::U32,
            DType::U8 => CDType::U8,
        };

        let new_tensor = self.inner.to_dtype(target_dtype)?;
        Ok(Box::new(Self::from_tensor(new_tensor, self.device.clone())))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct CandleOps;

impl MatmulOps for CandleOps {
    fn matmul(&self, a: &dyn TensorTrait, b: &dyn TensorTrait) -> Result<Box<dyn TensorTrait>> {
        let a_candle = CandleTensor::try_from_dyn(a)?;
        let b_candle = CandleTensor::try_from_dyn(b)?;

        let result = a_candle.inner().matmul(b_candle.inner())?;
        Ok(Box::new(CandleTensor::from_tensor(
            result,
            a_candle.device.clone(),
        )))
    }
}

impl AttentionOps for CandleOps {
    fn attention(
        &self,
        q: &dyn TensorTrait,
        k: &dyn TensorTrait,
        v: &dyn TensorTrait,
        mask: Option<&dyn TensorTrait>,
        scale: f32,
        _use_flash: bool,
    ) -> Result<Box<dyn TensorTrait>> {
        let q_candle = CandleTensor::try_from_dyn(q)?;
        let k_candle = CandleTensor::try_from_dyn(k)?;
        let v_candle = CandleTensor::try_from_dyn(v)?;

        #[cfg(feature = "flash-attn")]
        if _use_flash {
            let output = candle_flash_attn::flash_attn(
                q_candle.inner(),
                k_candle.inner(),
                v_candle.inner(),
                scale,
                mask.is_some(),
            )?;
            return Ok(Box::new(CandleTensor::from_tensor(
                output,
                q_candle.device.clone(),
            )));
        }

        // Fallback: Standard attention implementation
        // Q @ K^T
        use candle_core::D;
        let attn_weights = q_candle
            .inner()
            .matmul(&k_candle.inner().transpose(D::Minus2, D::Minus1)?)?;

        // Scale
        let attn_weights = (attn_weights * scale as f64)?;

        // Add mask if provided
        let attn_weights = if let Some(mask) = mask {
            let mask_candle = CandleTensor::try_from_dyn(mask)?;
            attn_weights.broadcast_add(mask_candle.inner())?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // @ V
        let output = attn_weights.matmul(v_candle.inner())?;

        Ok(Box::new(CandleTensor::from_tensor(
            output,
            q_candle.device.clone(),
        )))
    }
}

impl RoPEOps for CandleOps {
    fn apply(
        &self,
        q: &dyn TensorTrait,
        k: &dyn TensorTrait,
        cos: &dyn TensorTrait,
        sin: &dyn TensorTrait,
    ) -> Result<(Box<dyn TensorTrait>, Box<dyn TensorTrait>)> {
        let q_candle = CandleTensor::try_from_dyn(q)?;
        let k_candle = CandleTensor::try_from_dyn(k)?;
        let cos_candle = CandleTensor::try_from_dyn(cos)?;
        let sin_candle = CandleTensor::try_from_dyn(sin)?;

        // RoPE formula: x_rotated = x * cos + rotate_half(x) * sin
        let (q_embed, k_embed) = apply_rope_inner(
            q_candle.inner(),
            k_candle.inner(),
            cos_candle.inner(),
            sin_candle.inner(),
        )?;

        Ok((
            Box::new(CandleTensor::from_tensor(q_embed, q_candle.device.clone())),
            Box::new(CandleTensor::from_tensor(k_embed, k_candle.device.clone())),
        ))
    }
}

/// Helper function to rotate half of the tensor
fn rotate_half(x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
    use candle_core::D;
    let last_dim = x.dim(D::Minus1)?;
    let half_dim = last_dim / 2;

    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

    // Negate x2 and concatenate with x1
    let x2_neg = x2.affine(-1.0, 0.0)?;
    Ok(candle_core::Tensor::cat(&[&x2_neg, &x1], D::Minus1)?)
}

/// Apply RoPE to query and key tensors
fn apply_rope_inner(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    cos: &candle_core::Tensor,
    sin: &candle_core::Tensor,
) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
    // q_embed = q * cos + rotate_half(q) * sin
    let q_embed = (q.broadcast_mul(cos)? + rotate_half(q)?.broadcast_mul(sin)?)?;

    // k_embed = k * cos + rotate_half(k) * sin
    let k_embed = (k.broadcast_mul(cos)? + rotate_half(k)?.broadcast_mul(sin)?)?;

    Ok((q_embed, k_embed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use af_core::MatmulOps;

    #[test]
    fn test_matmul_basic() -> Result<()> {
        let device = CandleDevice::cpu();
        let ops = CandleOps;

        // Create test tensors [2, 3] @ [3, 4] = [2, 4]
        let a = candle_core::Tensor::randn(0f32, 1f32, (2, 3), device.inner())?;
        let b = candle_core::Tensor::randn(0f32, 1f32, (3, 4), device.inner())?;

        let a_wrapped = CandleTensor::from_tensor(a.clone(), device.clone());
        let b_wrapped = CandleTensor::from_tensor(b.clone(), device.clone());

        let result = ops.matmul(&a_wrapped, &b_wrapped)?;

        assert_eq!(result.shape(), &[2, 4]);
        println!("✅ Basic matmul test passed: [2,3] @ [3,4] = [2,4]");
        Ok(())
    }

    #[test]
    fn test_tensor_dtype_conversion() -> Result<()> {
        let device = CandleDevice::cpu();
        let tensor = candle_core::Tensor::randn(0f32, 1f32, (2, 3), device.inner())?;
        let wrapped = CandleTensor::from_tensor(tensor, device.clone());

        // Test to_dtype
        let f16_tensor = wrapped.to_dtype(DType::F16)?;
        assert_eq!(f16_tensor.dtype(), DType::F16);
        assert_eq!(f16_tensor.shape(), &[2, 3]);

        println!("✅ Dtype conversion test passed");
        Ok(())
    }

    #[test]
    fn test_device_trait() -> Result<()> {
        let device = CandleDevice::cpu();
        assert_eq!(device.name(), "candle");
        assert!(!device.is_gpu());

        println!("✅ Device trait test passed");
        Ok(())
    }

    #[test]
    fn test_mlp_forward() -> Result<()> {
        use candle_nn::{Activation, VarBuilder, linear_no_bias};

        let device = CandleDevice::cpu();

        // Create dummy weights for MLP
        let hidden_size = 64;
        let intermediate_size = 256;
        let batch_size = 2;
        let seq_len = 4;

        // Create VarBuilder with random initialization
        let dtype = candle_core::DType::F32;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device.inner());

        // Build MLP layers (SwiGLU style: gate_proj, up_proj, down_proj)
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down"))?;
        let act_fn = Activation::Silu;

        // Create input tensor [batch, seq, hidden]
        let input = candle_core::Tensor::randn(
            0f32,
            1f32,
            (batch_size, seq_len, hidden_size),
            device.inner(),
        )?;

        // Forward pass: SwiGLU(x) = (gate(x) * silu(up(x))) @ down
        let gate_out = input.apply(&gate_proj)?.apply(&act_fn)?;
        let up_out = input.apply(&up_proj)?;
        let combined = (gate_out * up_out)?;
        let output = combined.apply(&down_proj)?;

        // Check output shape
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

        println!(
            "✅ MLP forward test passed: [{}, {}, {}]",
            batch_size, seq_len, hidden_size
        );
        Ok(())
    }

    #[test]
    fn test_rope_apply() -> Result<()> {
        use af_core::RoPEOps;

        let device = CandleDevice::cpu();
        let ops = CandleOps;

        // Create test tensors for RoPE
        // Shape: [batch, num_heads, seq_len, head_dim]
        let batch = 2;
        let num_heads = 8;
        let seq_len = 16;
        let head_dim = 64;

        let q = candle_core::Tensor::randn(
            0f32,
            1f32,
            (batch, num_heads, seq_len, head_dim),
            device.inner(),
        )?;
        let k = candle_core::Tensor::randn(
            0f32,
            1f32,
            (batch, num_heads, seq_len, head_dim),
            device.inner(),
        )?;

        // cos/sin for RoPE: [batch, 1, seq_len, head_dim]
        let cos =
            candle_core::Tensor::randn(0f32, 1f32, (batch, 1, seq_len, head_dim), device.inner())?;
        let sin =
            candle_core::Tensor::randn(0f32, 1f32, (batch, 1, seq_len, head_dim), device.inner())?;

        let q_wrapped = CandleTensor::from_tensor(q, device.clone());
        let k_wrapped = CandleTensor::from_tensor(k, device.clone());
        let cos_wrapped = CandleTensor::from_tensor(cos, device.clone());
        let sin_wrapped = CandleTensor::from_tensor(sin, device.clone());

        let (q_embed, k_embed) = ops.apply(&q_wrapped, &k_wrapped, &cos_wrapped, &sin_wrapped)?;

        // Check output shapes
        assert_eq!(q_embed.shape(), &[batch, num_heads, seq_len, head_dim]);
        assert_eq!(k_embed.shape(), &[batch, num_heads, seq_len, head_dim]);

        println!(
            "✅ RoPE apply test passed: [{}, {}, {}, {}]",
            batch, num_heads, seq_len, head_dim
        );
        Ok(())
    }

    #[test]
    fn test_attention() -> Result<()> {
        use af_core::AttentionOps;

        let device = CandleDevice::cpu();
        let ops = CandleOps;

        // Create test tensors for attention
        // Shape: [batch, num_heads, seq_len, head_dim]
        let batch = 2;
        let num_heads = 8;
        let seq_len = 16;
        let head_dim = 64;

        let q = candle_core::Tensor::randn(
            0f32,
            1f32,
            (batch, num_heads, seq_len, head_dim),
            device.inner(),
        )?;
        let k = candle_core::Tensor::randn(
            0f32,
            1f32,
            (batch, num_heads, seq_len, head_dim),
            device.inner(),
        )?;
        let v = candle_core::Tensor::randn(
            0f32,
            1f32,
            (batch, num_heads, seq_len, head_dim),
            device.inner(),
        )?;

        let q_wrapped = CandleTensor::from_tensor(q, device.clone());
        let k_wrapped = CandleTensor::from_tensor(k, device.clone());
        let v_wrapped = CandleTensor::from_tensor(v, device.clone());

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = ops.attention(&q_wrapped, &k_wrapped, &v_wrapped, None, scale, false)?;

        // Check output shape
        assert_eq!(output.shape(), &[batch, num_heads, seq_len, head_dim]);

        println!(
            "✅ Attention test passed: [{}, {}, {}, {}]",
            batch, num_heads, seq_len, head_dim
        );
        Ok(())
    }
}
