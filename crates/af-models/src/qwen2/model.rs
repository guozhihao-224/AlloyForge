use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};
use std::path::Path;

use super::config::Qwen2Config;

// ===== 辅助函数 =====

/// Repeat KV heads for Grouped Query Attention
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        let xs = xs
            .unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
        Ok(xs)
    }
}

/// RoPE 位置编码的旋转半张量操作
pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let half_dim = last_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    let x2_neg = x2.affine(-1.0, 0.0)?;
    Ok(Tensor::cat(&[&x2_neg, &x1], D::Minus1)?)
}

/// 应用 RoPE 位置编码
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    to_f32: bool,
) -> Result<(Tensor, Tensor)> {
    let orig_dtype = q.dtype();
    let q = if to_f32 { &q.to_dtype(DType::F32)? } else { q };
    let k = if to_f32 { &k.to_dtype(DType::F32)? } else { k };
    
    let q_embed = (q.broadcast_mul(cos)? + rotate_half(q)?.broadcast_mul(sin)?)?;
    let k_embed = (k.broadcast_mul(cos)? + rotate_half(k)?.broadcast_mul(sin)?)?;
    
    let q_embed = if to_f32 { q_embed.to_dtype(orig_dtype)? } else { q_embed };
    let k_embed = if to_f32 { k_embed.to_dtype(orig_dtype)? } else { k_embed };
    
    Ok((q_embed, k_embed))
}

// ===== RoPE =====

/// RoPE 旋转位置编码
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn forward(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let positions: Vec<_> = positions.iter().map(|&p| p as u32).collect();
        let positions = Tensor::new(&positions[..], self.cos.device())?;
        let cos = self.cos.index_select(&positions, 0)?;
        let sin = self.sin.index_select(&positions, 0)?;
        Ok((cos, sin))
    }
}

// ===== MLP =====

/// MLP 层 (SwiGLU 架构)
#[derive(Debug, Clone)]
pub struct Qwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2MLP {
    pub fn new(config: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Qwen2MLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // SwiGLU: gate(x) * silu(up(x))
        let lhs = xs.apply(&self.gate_proj)?.silu()?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ===== Attention =====

/// Qwen2 Attention 层（带 KV Cache 支持）
#[derive(Debug, Clone)]
pub struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen2Attention {
    pub fn new(config: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_size / num_heads;
        
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        
        // KV Cache
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        
        // Repeat KV for GQA
        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;
        let query_states = query_states.contiguous()?;
        
        // Attention
        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            
            #[cfg(not(feature = "flash-attn"))]
            {
                let attn_weights = query_states.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
                let attn_weights = (attn_weights * scale)?;
                let attn_weights = match attention_mask {
                    None => attn_weights,
                    Some(mask) => attn_weights.broadcast_add(mask)?,
                };
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&value_states)?
            }
            
            #[cfg(feature = "flash-attn")]
            {
                let query_states = query_states.transpose(1, 2)?;
                let key_states = key_states.transpose(1, 2)?;
                let value_states = value_states.transpose(1, 2)?;
                let attn_output = candle_flash_attn::flash_attn(
                    &query_states,
                    &key_states,
                    &value_states,
                    scale as f32,
                    attention_mask.is_some(),
                )?;
                attn_output.transpose(1, 2)?
            }
        };
        
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, q_len, self.hidden_size))?;
        Ok(attn_output.apply(&self.o_proj)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ===== DecoderLayer =====

/// Qwen2 Decoder Layer
#[derive(Debug, Clone)]
pub struct Qwen2DecoderLayer {
    self_attn: Qwen2Attention,
    mlp: Qwen2MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2DecoderLayer {
    pub fn new(config: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen2Attention::new(config, vb.pp("self_attn"))?;
        let mlp = Qwen2MLP::new(config, vb.pp("mlp"))?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = (xs + residual)?;
        
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        Ok((residual + xs)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ===== Qwen2Model =====

/// 完整的 Qwen2 模型
pub struct Qwen2Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen2DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: RotaryEmbedding,
    config: Qwen2Config,
    device: Device,
    dtype: DType,
}

impl Qwen2Model {
    /// 从预训练模型目录加载 Qwen2 模型
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // 1. 加载配置
        let config_path = model_dir.as_ref().join("config.json");
        let config = Qwen2Config::from_file(config_path)?;
        
        // 2. 查找并加载权重
        let safetensors_files = af_io::find_safetensors_files(&model_dir)?;
        if safetensors_files.is_empty() {
            anyhow::bail!("No safetensors files found in {:?}", model_dir.as_ref());
        }
        
        let vb = af_io::load_safetensors_mmap(&safetensors_files, dtype, device)?;
        
        // 3. 构建模型
        Self::new(&config, vb.pp("model"))
    }

    pub fn new(config: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = Qwen2DecoderLayer::new(config, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        
        let lm_head = if config.tie_word_embeddings {
            // 共享 embedding 和 lm_head 权重
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        
        let rotary_emb = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
        )?;
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config: config.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Forward pass
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        
        // 1. Embedding
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        
        // 2. RoPE
        let positions: Vec<_> = (seqlen_offset..seqlen_offset + seq_len).collect();
        let (cos, sin) = self.rotary_emb.forward(&positions)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        
        // 3. Attention mask (causal)
        let attention_mask = self.prepare_causal_attention_mask(seq_len, seqlen_offset)?;
        
        // 4. 通过所有 layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, Some(&attention_mask))?;
        }
        
        // 5. Final norm + LM head
        let hidden_states = self.norm.forward(&hidden_states)?;
        Ok(self.lm_head.forward(&hidden_states)?)
    }

    fn prepare_causal_attention_mask(
        &self,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i + seqlen_offset >= j {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Ok(Tensor::from_slice(&mask, (1, 1, tgt_len, tgt_len), &self.device)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }
}

