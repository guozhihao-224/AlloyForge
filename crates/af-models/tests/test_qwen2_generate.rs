use af_models::qwen2::Qwen2Model;
use af_ops::sampling;
use af_tokenizer::Tokenizer;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[test]
#[ignore] // 需要真实模型，使用 `-- --ignored` 运行
fn test_qwen2_generate() -> Result<()> {
    let model_path = std::env::var("QWEN2_MODEL_PATH")
        .unwrap_or_else(|_| "./../../Qwen2-0.5B".to_string());

    println!("🔧 Loading model from: {}", model_path);

    // 1. 加载模型和 tokenizer
    let device = Device::Cpu;
    let mut model = Qwen2Model::from_pretrained(&model_path, DType::F32, &device)?;
    println!("✅ Model loaded");

    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    println!("✅ Tokenizer loaded");
    println!("   Vocab size: {}", tokenizer.vocab_size());
    println!("   EOS token ID: {:?}", tokenizer.eos_token_id());

    // 2. Tokenize prompt
    let prompt = "你好";
    println!("\n🎯 Prompt: \"{}\"", prompt);
    
    let input_ids = tokenizer.encode(prompt, false)?;
    println!("📝 Encoded tokens: {:?}", input_ids);
    
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;
    println!("   Input shape: {:?}", input_tensor.shape());

    // 3. Prefill 阶段
    println!("\n🔄 Prefill stage...");
    let mut logits = model.forward(&input_tensor, 0)?;
    println!("✅ Prefill done, logits shape: {:?}", logits.shape());

    // 4. Decode 循环
    println!("\n🔄 Decode stage...");
    let mut generated_ids = input_ids.clone();
    let max_new_tokens = 1000;

    for i in 0..max_new_tokens {
        // 获取最后一个 token 的 logits
        let seq_len = logits.dim(1)?;
        let last_logits = logits
            .narrow(1, seq_len - 1, 1)?  // [batch, 1, vocab]
            .squeeze(0)?                  // [1, vocab]
            .squeeze(0)?;                 // [vocab]

        // 采样下一个 token
        let logits_vec = last_logits.to_vec1::<f32>()?;
        let next_token = sampling::greedy(&logits_vec)
            .ok_or_else(|| anyhow::anyhow!("Failed to sample token"))?;

        // 检查 EOS
        if Some(next_token) == tokenizer.eos_token_id() {
            println!("🛑 EOS token detected at step {}", i);
            break;
        }

        generated_ids.push(next_token);

        // 部分解码显示进度
        if i % 5 == 0 || i < 5 {
            let partial = tokenizer.decode(&generated_ids, false)?;
            println!("   Step {}: {}", i, partial);
        }

        // Forward 下一个 token
        let next_tensor = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let position = input_ids.len() + i;
        logits = model.forward(&next_tensor, position)?;
    }

    // 5. 最终解码
    println!("\n📤 Final output:");
    let output = tokenizer.decode(&generated_ids, true)?;
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{}", output);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n📊 Stats:");
    println!("   Input tokens: {}", input_ids.len());
    println!("   Generated tokens: {}", generated_ids.len() - input_ids.len());
    println!("   Total tokens: {}", generated_ids.len());

    // 验证生成了新内容
    assert!(
        generated_ids.len() > input_ids.len(),
        "Model should generate new tokens"
    );
    assert!(output.len() >= prompt.len(), "Output should not be shorter than input");

    println!("\n✅ Test passed!");
    Ok(())
}

#[test]
#[ignore]
fn test_qwen2_generate_english() -> Result<()> {
    let model_path = std::env::var("QWEN2_MODEL_PATH")
        .unwrap_or_else(|_| "./../../Qwen2-0.5B".to_string());

    println!("🔧 Loading model from: {}", model_path);

    let device = Device::Cpu;
    let mut model = Qwen2Model::from_pretrained(&model_path, DType::F32, &device)?;
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))?;

    println!("✅ Model and tokenizer loaded");

    // 英文 prompt
    let prompt = "Once upon a time";
    println!("\n🎯 Prompt: \"{}\"", prompt);

    let input_ids = tokenizer.encode(prompt, false)?;
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;

    // Prefill
    let mut logits = model.forward(&input_tensor, 0)?;

    // Decode
    let mut generated_ids = input_ids.clone();
    for i in 0..50 {
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
    println!("\n📤 Generated:\n{}", output);

    assert!(generated_ids.len() > input_ids.len());
    println!("\n✅ English generation test passed!");
    Ok(())
}

