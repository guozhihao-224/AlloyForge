# AlloyForge

基于 Candle 的 Rust 多模态大模型推理库，专注于高性能、易用的 LLM 推理能力。

## 特性

- 🚀 **高性能**：基于 Candle 框架，支持 CPU/CUDA/Metal 加速
- 🎯 **多模态**：支持视觉、语言和语音模态
- 🔧 **易于使用**：简洁的 API 设计，快速上手
- 🛡️ **内存安全**：得益于 Rust 的所有权系统
- 📦 **轻量级**：最小化依赖，编译产物小巧
- ⚡ **GPU 加速**：可选 CUDA/Metal 支持
- 🧠 **注意力优化**：可选 Flash Attention 支持

## 项目结构

```
alloyforge/
├── Cargo.toml              # Workspace 配置
├── crates/                 # 所有库 crate
│   ├── af-core/           # 核心抽象层
│   │                       # - Device/Tensor traits (设备和张量抽象)
│   │                       # - Model trait (模型接口)
│   │                       # - 基础算子 traits (MatmulOps/AttentionOps/RoPE)
│   ├── af-backend-candle/ # Candle 后端实现
│   │                       # - 实现 af-core 的所有 traits
│   ├── af-ops/            # 通用算子库
│   │                       # - 采样策略 (greedy/temperature/top-p/top-k)
│   │                       # - 优化器 (未来)
│   ├── af-io/             # I/O 工具
│   │                       # - SafeTensors 加载 (mmap)
│   │                       # - GGUF 支持 (未来)
│   ├── af-tokenizer/      # 分词器封装
│   │                       # - tokenizers crate 的简化接口
│   ├── af-modalities/     # 模态接口
│   │                       # - VisionEncoder/AudioEncoder/Projector traits
│   ├── af-models/         # 具体模型实现
│   │                       # - Qwen2/Llama/MiniCPM 等
│   │                       # - 实现 af-core::Model trait
│   ├── af-runtime/        # 运行时管理
│   │                       # - Session 会话管理
│   │                       # - 生成循环控制
│   │                       # - 采样策略集成
│   ├── af-cli/            # 命令行工具
│   └── af-server/         # OpenAI 兼容服务 (可选)
├── examples/              # 示例代码
└── docs/                  # 文档 (待添加)
```

## 快速开始

### 安装依赖

```toml
[dependencies]
af-runtime = { path = "crates/af-runtime" }
af-backend-candle = { path = "crates/af-backend-candle" }
```

### 基础使用

```rust
use af_models::qwen2::Qwen2ModelWrapper;
use af_runtime::{SessionBuilder, Request};
use candle_core::{DType, Device};
use std::sync::Arc;

// 1. 加载模型
let model = Qwen2ModelWrapper::from_pretrained(
    "/path/to/Qwen2-0.5B",
    DType::BF16,
    &Device::Cpu,
)?;

// 2. 创建会话
let mut session = SessionBuilder::new()
    .build(Box::new(model))?;

// 3. 生成文本
let response = session.generate(Request {
    prompt: "你好".into(),
    max_tokens: 128,
    ..Default::default()
})?;

println!("Response: {}", response);
```

## 编译

```bash
# 基础编译 (CPU)
cargo build --release

# 启用 CUDA
cargo build --release --features cuda

# 启用 Flash Attention
cargo build --release --features cuda,flash-attn

# 运行 CLI
cargo run --bin af-cli --release
```

## 开发

```bash
# 检查所有 workspace
cargo check --workspace

# 运行测试
cargo test --workspace

# 格式化代码
cargo fmt --all

# Lint
cargo clippy --workspace -- -D warnings
```

## 架构设计

### 核心原则

- **无通用计算图**：围绕自回归 LLM 的固定执行管线
- **后端可插拔**：通过 trait 抽象，支持多后端（默认 Candle）
- **模态可扩展**：统一接口，易于添加新模态编码器
- **性能优先**：KV cache、Flash Attention、量化等优化
- **依赖倒置**：抽象在底层，实现依赖抽象

### 架构分层

```
┌─────────────────────────────────────────┐
│  Layer 4: Application                   │  应用层
│  - af-cli (命令行)                       │
│  - af-server (API 服务)                  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Layer 3: Runtime                       │  运行时层
│  - af-runtime (会话管理、生成循环)         │
└──────┬────────────────────────┬─────────┘
       │                        │
┌──────▼──────┐        ┌───────▼─────────┐
│  Layer 2:   │        │   Layer 2:      │  实现层
│  Models     │        │   Operations    │
│  - af-models│        │   - af-ops      │
│  (Qwen2等)   │        │   - af-io       │
│             │        │   - af-tokenizer│
└──────┬──────┘        └───────┬─────────┘
       │                        │
       └────────┬───────────────┘
                │
┌───────────────▼─────────────────────────┐
│  Layer 1: Core Abstractions              │  抽象层
│  - af-core                               │
│    • Device trait (设备抽象)             │
│    • Tensor trait (张量抽象)             │
│    • Model trait (模型接口) ★            │
│    • MatmulOps/AttentionOps (算子)      │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│  Layer 0: Backend                         │  后端层
│  - af-backend-candle                      │
│    • 实现所有 core traits                  │
│    • Candle 框架适配                       │
└──────────────────────────────────────────┘
```

### 依赖关系图

```
             af-cli, af-server (应用)
                      ↓
                 af-runtime (运行时)
                   ↙    ↓    ↘
          af-models  af-ops  af-tokenizer
              ↓         ↓        ↓
          af-io    af-modalities
              ↓         ↓
           ╔══════════════════╗
           ║    af-core       ║  ← 所有抽象都在这里
           ║  (核心 traits)   ║
           ╚════════╤═════════╝
                    ↓
           af-backend-candle (后端实现)
```
## 添加自定义模型

添加新模型非常简单：

```rust
use af_core::{Model, ModelConfig};
use anyhow::Result;

// 1. 定义你的模型结构
pub struct MyCustomModel {
    // 模型参数
}

// 2. 实现 Model trait
impl Model for MyCustomModel {
    fn reset_state(&mut self) {
        // 重置 KV cache 或其他状态
    }

    fn forward_step(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        // 实现前向传播
        // 返回 vocab_size 长度的 logits
        todo!()
    }

    fn config(&self) -> Option<ModelConfig> {
        Some(ModelConfig {
            vocab_size: 32000,
            max_position_embeddings: 2048,
            hidden_size: 768,
            num_layers: 12,
        })
    }
}

// 3. 使用你的模型
let model = Box::new(MyCustomModel { /* ... */ });
let mut session = SessionBuilder::new().build(model)?;
```

**关键优势**：

- ✅ **只需实现 2 个方法**：`reset_state` 和 `forward_step`
- ✅ **无需修改框架代码**：完全解耦
- ✅ **自动获得所有功能**：采样、会话管理等
- ✅ **类型安全**：编译时检查接口正确性

## 路线图

### 已完成

- [x] **M0: 基础架构 + Candle 后端**
  - [x] 核心抽象层设计（af-core）
  - [x] Candle 后端适配（af-backend-candle）
  - [x] Model trait 定义
  - [x] 采样策略实现（greedy/temperature/top-p/top-k）
  - [x] Qwen2 模型实现
  - [x] 架构重构（依赖倒置）

### 进行中

- [🔄] **M1: 文本 LLM CPU 推理 + KV cache**
  - [x] Qwen2 模型 + KV Cache
  - [x] Model trait 实现
  - [ ] 完整生成循环（prefill + decode）
  - [ ] Tokenizer 集成
  - [ ] 端到端测试

### 计划中

- [ ] **M2: CUDA 支持 + Flash Attention**

  - [ ] CUDA 后端支持
  - [ ] Flash Attention 集成
  - [ ] 性能基准测试

- [ ] **M3: 图像模态 (CLIP/ViT) + 图文问答**

  - [ ] VisionEncoder 实现
  - [ ] 图文多模态模型
  - [ ] LLaVA 风格模型支持

- [ ] **M4: 视频模态 + 长上下文优化**

  - [ ] 视频编码器
  - [ ] 长上下文优化
  - [ ] 流式处理

- [ ] **M5: 量化支持 (int8/int4)**

  - [ ] GPTQ/AWQ 量化
  - [ ] 动态量化
  - [ ] 性能评估

- [ ] **M6: Python 绑定 + 服务化**
  - [ ] PyO3 Python 绑定
  - [ ] OpenAI 兼容 API
  - [ ] 部署文档

## 许可证

Apache-2.0

## 致谢

- Hugging Face Candle 团队
- 所有开源模型的贡献者
