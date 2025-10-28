# FireRedASR vLLM 集成 - 快速开始指南

本指南帮助您快速上手 FireRedASR 的 vLLM 集成。

## 📋 目录结构

```
vllm-abo/
├── vllm/model_executor/models/
│   ├── fireredasr_vllm.py              # 主实现文件
│   ├── FIREREDASR_VLLM_README.md       # 详细文档
│   └── MIGRATION_GUIDE.md              # 迁移指南
├── examples/
│   └── fireredasr_vllm_example.py      # 使用示例
├── tests/
│   └── test_fireredasr_vllm.py         # 单元测试
├── scripts/
│   └── verify_fireredasr_integration.py # 验证脚本
└── FIREREDASR_QUICKSTART.md            # 本文档
```

## 🚀 快速开始（5 分钟）

### 1. 验证环境

```bash
# 运行验证脚本
python scripts/verify_fireredasr_integration.py --model-dir /path/to/your/model

# 如果有任何错误，按照提示修复
```

### 2. 最小示例

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="fireredasr",
    trust_remote_code=True,
    override_neuron_config={
        "encoder_path": "/path/to/asr_encoder.pth.tar",
        "cmvn_path": "/path/to/cmvn.ark",
        "llm_dir": "/path/to/Qwen2-7B-Instruct",
    }
)

# 准备输入
prompts = [{
    "prompt": "<|SPEECH|>",
    "multi_modal_data": {"audio": "/path/to/audio.wav"}
}]

# 生成转录
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

# 获取结果
print(outputs[0].outputs[0].text)
```

### 3. 运行示例

```bash
# 查看完整示例
python examples/fireredasr_vllm_example.py
```

## 📦 安装要求

### 必需依赖

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install vllm>=0.3.0
pip install fireredasr
```

### 模型文件

确保有以下文件：

```
model_dir/
├── asr_encoder.pth.tar      # ASR 编码器
├── model.pth.tar            # 投影层权重
├── cmvn.ark                 # CMVN 统计
└── Qwen2-7B-Instruct/       # LLM 模型
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── *.safetensors          # 模型权重
```

## 🔧 核心配置

### 基础配置

```python
llm = LLM(
    model="fireredasr",
    trust_remote_code=True,
    override_neuron_config={
        # 必需配置
        "encoder_path": "/path/to/asr_encoder.pth.tar",
        "cmvn_path": "/path/to/cmvn.ark",
        "llm_dir": "/path/to/Qwen2-7B-Instruct",
        
        # 可选配置
        "freeze_encoder": True,      # 是否冻结编码器
        "freeze_llm": False,         # 是否冻结 LLM
        "encoder_downsample_rate": 4, # 下采样率
    }
)
```

### 性能配置

```python
# 单 GPU 配置
llm = LLM(
    model="fireredasr",
    tensor_parallel_size=1,
    max_num_seqs=16,              # 批大小
    gpu_memory_utilization=0.85,
)

# 多 GPU 配置
llm = LLM(
    model="fireredasr",
    tensor_parallel_size=4,       # 4-GPU 并行
    max_num_seqs=32,
    gpu_memory_utilization=0.90,
)
```

### 采样配置

```python
# ASR 推荐：贪婪解码
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
    repetition_penalty=1.0,
)

# Beam search（更高质量）
sampling_params = SamplingParams(
    best_of=5,
    use_beam_search=True,
    temperature=0.0,
    max_tokens=100,
)

# 带随机性的采样
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)
```

## 📝 常见用例

### 用例 1: 单个音频转录

```python
from vllm import LLM, SamplingParams

llm = LLM(model="fireredasr", override_neuron_config={...})

prompts = [{
    "prompt": "<|SPEECH|>",
    "multi_modal_data": {"audio": "audio.wav"}
}]

outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=100))
print(outputs[0].outputs[0].text)
```

### 用例 2: 批量转录

```python
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

prompts = [
    {"prompt": "<|SPEECH|>", "multi_modal_data": {"audio": f}}
    for f in audio_files
]

outputs = llm.generate(prompts, sampling_params)

for audio, output in zip(audio_files, outputs):
    print(f"{audio}: {output.outputs[0].text}")
```

### 用例 3: 处理原始音频张量

```python
import torch

# 假设已有音频张量（例如从实时流获取）
audio_tensor = torch.randn(1, 16000 * 5)  # 5 秒，16kHz

prompts = [{
    "prompt": "<|SPEECH|>",
    "multi_modal_data": {"audio": audio_tensor}
}]

outputs = llm.generate(prompts, sampling_params)
print(outputs[0].outputs[0].text)
```

### 用例 4: 异步处理（高并发）

```python
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio

async def transcribe_async(audio_files):
    engine = AsyncLLMEngine.from_engine_args(...)
    
    tasks = []
    for audio_file in audio_files:
        prompt = {
            "prompt": "<|SPEECH|>",
            "multi_modal_data": {"audio": audio_file}
        }
        task = engine.generate(prompt, sampling_params)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# 运行
audio_files = ["audio1.wav", "audio2.wav", ...]
results = asyncio.run(transcribe_async(audio_files))
```

## 🎯 性能调优

### 延迟优化（单请求）

```python
llm = LLM(
    model="fireredasr",
    max_num_seqs=1,               # 单个序列
    gpu_memory_utilization=0.8,
    enforce_eager=True,           # 禁用 CUDA graph（减少首次延迟）
)
```

### 吞吐量优化（批处理）

```python
llm = LLM(
    model="fireredasr",
    max_num_seqs=64,              # 大批量
    gpu_memory_utilization=0.95,
    max_model_len=2048,
)
```

### 内存优化

```python
llm = LLM(
    model="fireredasr",
    gpu_memory_utilization=0.7,   # 降低内存使用
    max_num_seqs=8,
    enable_prefix_caching=True,   # 启用前缀缓存
)
```

## 🐛 故障排除

### 问题 1: ImportError: FireRedASR not installed

```bash
# 解决方案
pip install fireredasr
```

### 问题 2: 找不到模型文件

```python
# 检查路径是否正确
import os
encoder_path = "/path/to/asr_encoder.pth.tar"
assert os.path.exists(encoder_path), f"Encoder not found: {encoder_path}"
```

### 问题 3: CUDA out of memory

```python
# 解决方案 1: 降低批大小
llm = LLM(model="fireredasr", max_num_seqs=8)

# 解决方案 2: 降低 GPU 内存利用率
llm = LLM(model="fireredasr", gpu_memory_utilization=0.7)

# 解决方案 3: 使用张量并行
llm = LLM(model="fireredasr", tensor_parallel_size=2)
```

### 问题 4: 输出为空或不正确

```python
# 检查 1: 验证 speech_token_id
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/Qwen2-7B-Instruct")
speech_token = "<|SPEECH|>"
speech_token_id = tokenizer.convert_tokens_to_ids(speech_token)
print(f"Speech token ID: {speech_token_id}")

# 检查 2: 增加 max_tokens
sampling_params = SamplingParams(max_tokens=200)  # 增大

# 检查 3: 检查音频质量
# 确保音频采样率为 16kHz，格式正确
```

### 问题 5: 速度慢

```python
# 诊断步骤
import time

# 1. 预热模型
dummy_prompt = [{"prompt": "<|SPEECH|>", "multi_modal_data": {"audio": dummy_audio}}]
_ = llm.generate(dummy_prompt, sampling_params)

# 2. 测量不同阶段的时间
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start
print(f"Total time: {elapsed:.2f}s")
print(f"Throughput: {len(outputs)/elapsed:.2f} requests/s")

# 3. 考虑批处理
# 将多个请求合并为一个批次
```

## 📚 更多资源

- **详细文档**: `vllm/model_executor/models/FIREREDASR_VLLM_README.md`
- **迁移指南**: `vllm/model_executor/models/MIGRATION_GUIDE.md`
- **示例代码**: `examples/fireredasr_vllm_example.py`
- **单元测试**: `tests/test_fireredasr_vllm.py`
- **验证脚本**: `scripts/verify_fireredasr_integration.py`

## 🔍 检查清单

在部署前，确保：

- [ ] Python >= 3.8
- [ ] 所有依赖已安装（torch, transformers, vllm, fireredasr）
- [ ] 模型文件完整（encoder, projector, cmvn, llm）
- [ ] 验证脚本通过
- [ ] 测试单个音频转录正常
- [ ] 测试批处理正常
- [ ] 性能满足需求
- [ ] 内存使用在合理范围内

## 💡 最佳实践

1. **重用 LLM 实例**: 初始化一次，多次使用
2. **批处理优先**: 合并多个请求提高吞吐量
3. **预热模型**: 首次调用前预热以减少延迟
4. **监控性能**: 定期检查吞吐量和延迟指标
5. **合理配置内存**: 留 10-20% GPU 内存给其他进程
6. **使用异步 API**: 高并发场景使用 AsyncLLMEngine
7. **启用缓存**: 相同音频会自动缓存编码器输出

## 🚦 下一步

1. **运行验证脚本**:
   ```bash
   python scripts/verify_fireredasr_integration.py --model-dir /your/model/dir
   ```

2. **测试基础功能**:
   ```bash
   python examples/fireredasr_vllm_example.py
   ```

3. **集成到您的应用**:
   - 参考示例代码
   - 根据需求调整配置
   - 测试性能和准确性

4. **性能调优**:
   - 调整批大小
   - 测试不同的采样参数
   - 根据硬件调整并行度

## 📧 获取帮助

如果遇到问题：

1. 查看详细文档（FIREREDASR_VLLM_README.md）
2. 运行验证脚本诊断问题
3. 查看单元测试了解正确用法
4. 参考迁移指南了解与原始实现的差异

---

**祝您使用愉快！** 🎉

