/// 快速验证测试 - 只生成 10 个 tokens
use af_models::qwen2::Qwen2Model;
use af_ops::sampling;
use af_tokenizer::Tokenizer;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[test]
#[ignore]
fn test_quick_generation() -> Result<()> {
    let model_path = std::env::var("QWEN2_MODEL_PATH")
        .unwrap_or_else(|_| "./../../Qwen2-0.5B".to_string());

    println!("⚡ Quick generation test");
    
    let device = Device::Cpu;
    let mut model = Qwen2Model::from_pretrained(&model_path, DType::F32, &device)?;
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))?;

    // 简单的 prompt
    let prompt = "Hello, Introducing Xiaomi, a company";
    let input_ids = tokenizer.encode(prompt, false)?;
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;

    // Prefill
    let mut logits = model.forward(&input_tensor, 0)?;

    // 只生成 30 个 tokens
    let mut generated_ids = input_ids.clone();
    for i in 0..30 {
        let seq_len = logits.dim(1)?;
        let last_logits = logits
            .narrow(1, seq_len - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?;

        let logits_vec = last_logits.to_vec1::<f32>()?;
        let next_token = sampling::greedy(&logits_vec)
            .ok_or_else(|| anyhow::anyhow!("Failed to sample"))?;

        if Some(next_token) == tokenizer.eos_token_id() {
            break;
        }

        generated_ids.push(next_token);

        let next_tensor = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        logits = model.forward(&next_tensor, input_ids.len() + i)?;
    }

    let output = tokenizer.decode(&generated_ids, true)?;
    println!("✅ Generated: {}", output);
    
    assert!(generated_ids.len() > input_ids.len());
    println!("✅ Quick test passed!");
    Ok(())
}

