use anyhow::Result;
use candle_core::{DType, Device};

#[test]
#[ignore] // 需要实际的模型文件
fn test_qwen2_load_model() -> Result<()> {
    // 这个测试需要真实的 Qwen2-0.5B 模型文件
    // 下载模型：huggingface-cli download Qwen/Qwen2-0.5B --local-dir /path/to/Qwen2-0.5B

    let model_path =
        std::env::var("QWEN2_MODEL_PATH").unwrap_or_else(|_| "/path/to/Qwen2-0.5B".to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("⚠️  Model not found at {}", model_path);
        eprintln!("   Set QWEN2_MODEL_PATH environment variable to test with real model");
        return Ok(());
    }

    let device = Device::Cpu;
    let dtype = DType::BF16;

    println!("📦 Loading Qwen2 model from {}...", model_path);
    let mut model = af_models::qwen2::Qwen2Model::from_pretrained(&model_path, dtype, &device)?;

    println!("✅ Model loaded successfully!");
    println!("   Config: {:?}", model.config());

    // 测试 forward
    let input_ids = candle_core::Tensor::new(&[151643u32, 108386u32, 151645u32], &device)?;
    let input_ids = input_ids.unsqueeze(0)?; // [1, seq_len]

    println!("🔄 Running forward pass...");
    let logits = model.forward(&input_ids, 0)?;

    let (_batch, seq_len, vocab_size) = logits.dims3()?;
    println!("✅ Forward pass successful!");
    println!("   Output shape: [1, {}, {}]", seq_len, vocab_size);

    assert_eq!(vocab_size, model.config().vocab_size);

    Ok(())
}

#[test]
fn test_qwen2_config_loading() -> Result<()> {
    // 测试配置加载（不需要模型文件）
    use af_models::qwen2::Qwen2Config;

    // 创建一个临时配置文件
    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("test_qwen2_config.json");

    let config_json = r#"{
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": true,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "torch_dtype": "bfloat16"
    }"#;

    std::fs::write(&config_path, config_json)?;

    let config = Qwen2Config::from_file(&config_path)?;

    assert_eq!(config.hidden_size, 896);
    assert_eq!(config.num_attention_heads, 14);
    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.num_kv_groups(), 7);

    // 清理
    std::fs::remove_file(&config_path)?;

    println!("✅ Qwen2 config loading test passed");
    Ok(())
}
