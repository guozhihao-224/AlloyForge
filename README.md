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
│   ├── af-core/           # 核心抽象 (Device/Tensor/Ops traits)
│   ├── af-backend-candle/ # Candle 后端适配
│   ├── af-ops/            # 通用算子 (采样/优化器等)
│   ├── af-runtime/        # 会话管理、KV cache、批处理
│   ├── af-modalities/     # 模态接口 (Vision/Audio/Text)
│   ├── af-models/         # 模型适配器
│   ├── af-io/             # 权重加载 (safetensors/GGUF)
│   ├── af-tokenizer/      # 分词器封装
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
use af_runtime::{SessionBuilder, Request};
use af_backend_candle::CandleDevice;

// 创建会话
let device = CandleDevice::cpu();
let mut session = SessionBuilder::new()
    .device(device)
    .build(model)?;

// 生成文本
let response = session.generate(Request {
    prompt: "你好".into(),
    max_tokens: 128,
    ..Default::default()
})?;
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

### 依赖关系

```
af-core (核心 traits)
    ↓
af-backend-candle (Candle 实现)
    ↓
af-ops, af-modalities, af-io, af-tokenizer
    ↓
af-runtime (会话管理)
    ↓
af-models (具体模型)
    ↓
af-cli, af-server (应用层)
```

## 路线图

- [x] M0: 基础架构 + Candle 后端
- [ ] M1: 文本 LLM CPU 推理 + KV cache
- [ ] M2: CUDA 支持 + Flash Attention
- [ ] M3: 图像模态 (CLIP/ViT) + 图文问答
- [ ] M4: 视频模态 + 长上下文优化
- [ ] M5: 量化支持 (int8/int4)
- [ ] M6: Python 绑定 + 服务化

## 许可证

Apache-2.0

## 致谢

- Hugging Face Candle 团队
- 所有开源模型的贡献者

