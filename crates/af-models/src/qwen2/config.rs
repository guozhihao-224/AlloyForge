use anyhow::Result;
use candle_nn::Activation;
use serde::Deserialize;
use std::path::Path;

/// Qwen2 模型配置（纯文本 LLM）
/// 支持 Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B 等
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen2Config {
    /// 隐藏层大小
    pub hidden_size: usize,
    
    /// FFN 中间层大小
    pub intermediate_size: usize,
    
    /// 注意力头数量
    pub num_attention_heads: usize,
    
    /// Transformer 层数
    pub num_hidden_layers: usize,
    
    /// KV 头数量（用于 GQA）
    pub num_key_value_heads: usize,
    
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    
    /// RoPE theta
    pub rope_theta: f32,
    
    /// 词表大小
    pub vocab_size: usize,
    
    /// 激活函数
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    
    /// 最大位置编码长度
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    
    /// 是否共享 embedding 和 lm_head 权重
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    
    /// BOS token ID
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    
    /// EOS token ID
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    
    /// 数据类型（用于推断）
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

fn default_max_position_embeddings() -> usize {
    32768
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl Qwen2Config {
    /// 从 config.json 文件加载配置
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config_str = std::fs::read(path.as_ref())?;
        let config: Self = serde_json::from_slice(&config_str)?;
        Ok(config)
    }
    
    /// 获取每个注意力头的维度
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    
    /// 获取 KV 头的分组数（用于 GQA）
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_0_5b_params() {
        // Qwen2-0.5B 的参数
        let config = Qwen2Config {
            hidden_size: 896,
            intermediate_size: 4864,
            num_attention_heads: 14,
            num_hidden_layers: 24,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            vocab_size: 151936,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            tie_word_embeddings: true,
            bos_token_id: Some(151643),
            eos_token_id: Some(151645),
            torch_dtype: Some("bfloat16".to_string()),
        };
        
        assert_eq!(config.head_dim(), 64);  // 896 / 14 = 64
        assert_eq!(config.num_kv_groups(), 7);  // 14 / 2 = 7 (GQA)
    }
}

