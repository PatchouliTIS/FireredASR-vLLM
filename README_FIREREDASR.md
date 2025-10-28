# FireRedASR Integration for vLLM

This document describes the integration of FireRedASR (LLM mode) into the vLLM framework.

## Overview

FireRedASR is an automatic speech recognition (ASR) model that combines:
- A speech encoder (from FireRedASR)
- An adapter/projector layer
- A language model (Qwen2) for text generation

The integration allows FireRedASR to run efficiently within vLLM's infrastructure, leveraging features like:
- Dynamic batching
- KV cache management
- Distributed inference
- Async request handling

## Model Directory Structure

### Expected Structure

A typical FireRedASR model directory has the following structure:

```
fireredasr_models/
├── FireRedASR-LLM-L/
│   ├── cmvn.ark                    # CMVN normalization statistics
│   ├── model.pth.tar               # Main model checkpoint
│   ├── asr_encoder.pth.tar         # ASR encoder weights
│   ├── config.yaml                 # Original FireRedASR config
│   ├── config.json                 # vLLM config (created by setup script)
│   └── Qwen2-7B-Instruct/          # LLM directory (often a symlink)
└── Qwen2-7B-Instruct/               # Actual Qwen2 model files
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

### Symlink Support

The integration fully supports symlinked LLM directories. For example, `FireRedASR-LLM-L/Qwen2-7B-Instruct` can be a symlink pointing to `../Qwen2-7B-Instruct`.

## Setup

### Prerequisites

1. Ensure FireRedASR is installed:
```bash
# Install FireRedASR package (if available)
pip install fireredasr
```

2. The integration files are already in place:
- Config: `vllm/transformers_utils/configs/fireredasr.py`
- Model: `vllm/model_executor/models/fireredasr_vllm.py`

### Model Setup

Since FireRedASR models don't typically include a `config.json` file, you need to create one for vLLM to recognize the model type:

#### Option 1: Use the Setup Script (Recommended)

```bash
# Run the setup helper script
python setup_fireredasr.py /path/to/FireRedASR-LLM-L

# The script will:
# 1. Validate the model structure
# 2. Check for required files
# 3. Resolve symlinks
# 4. Create config.json automatically
```

#### Option 2: Manual Setup

Create a `config.json` file in your FireRedASR model directory:

```json
{
  "model_type": "fireredasr",
  "architectures": ["FireRedAsrForConditionalGeneration"],
  "asr_type": "llm",
  "encoder_dim": 512,
  "encoder_downsample_rate": 4,
  "freeze_encoder": true,
  "sampling_rate": 16000,
  "default_speech_token": "<speech>",
  "speech_token_id": 151659
}
```

#### Option 3: Automatic Detection

The FireRedAsrConfig class can automatically detect FireRedASR directories even without config.json. It looks for:
- `cmvn.ark`
- `asr_encoder.pth.tar`
- `model.pth.tar`

If these files are present, it will create a config.json automatically on first use.

## Usage

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize model
# Example path: /path/to/FireRedASR-LLM-L
model_path = "/apdcephfs_qy2/share_303477892/patchychen/fireredasr_models/FireRedASR-LLM-L"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=2048,
)

# Prepare input
prompts = [{
    "prompt": "",  # FireRedASR uses empty prompt
    "multi_modal_data": {
        "audio": "/path/to/audio.wav"
    }
}]

# ASR-optimized sampling
sampling_params = SamplingParams(
    temperature=0.0,  # Greedy decoding
    max_tokens=256,   # ASR outputs are typically short
)

# Generate transcription
outputs = llm.generate(prompts, sampling_params)

# Get result
for output in outputs:
    transcription = output.outputs[0].text
    print(f"Transcription: {transcription}")
```

### Async Usage

```python
import asyncio
from vllm.v1.engine.async_llm import AsyncLLM
from vllm import AsyncEngineArgs, SamplingParams

async def transcribe():
    # Initialize async engine
    engine_args = AsyncEngineArgs(
        model="/path/to/fireredasr_model",
        trust_remote_code=True,
    )
    async_llm = AsyncLLM.from_engine_args(engine_args)

    # Process audio
    prompt = {
        "prompt": "",
        "multi_modal_data": {"audio": "audio.wav"}
    }

    async for output in async_llm.generate(
        prompt=prompt,
        sampling_params=SamplingParams(temperature=0.0),
        request_id="asr_001"
    ):
        if output.outputs:
            print(output.outputs[0].text)

    async_llm.shutdown()

# Run
asyncio.run(transcribe())
```

### Batch Processing

```python
# Process multiple audio files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

prompts = [
    {
        "prompt": "",
        "multi_modal_data": {"audio": audio_file}
    }
    for audio_file in audio_files
]

outputs = llm.generate(prompts, sampling_params)
```

## Configuration

### Automatic Path Resolution

The FireRedASR config automatically resolves all sub-model paths when you provide the root model directory:

```python
from vllm.transformers_utils.configs.fireredasr import FireRedAsrConfig

# Paths are auto-resolved
config = FireRedAsrConfig.from_pretrained("/path/to/fireredasr_model")

print(config.cmvn_path)      # /path/to/fireredasr_model/cmvn.ark
print(config.encoder_path)   # /path/to/fireredasr_model/asr_encoder.pth.tar
print(config.llm_dir)        # /path/to/fireredasr_model/Qwen2-7B-Instruct/
```

### Creating config.json (Optional)

If your model directory doesn't have a config.json, you can create one:

```python
python test_fireredasr.py create_config /path/to/fireredasr_model
```

Or manually create `config.json`:

```json
{
  "model_type": "fireredasr",
  "architectures": ["FireRedAsrForConditionalGeneration"],
  "asr_type": "llm",
  "encoder_dim": 512,
  "encoder_downsample_rate": 4,
  "freeze_encoder": true,
  "sampling_rate": 16000,
  "default_speech_token": "<speech>",
  "speech_token_id": 151659
}
```

## Key Features

### 1. Automatic Path Resolution
- Model paths are automatically resolved from the root directory
- No need to specify individual component paths

### 2. CMVN Normalization
- CMVN file is automatically located and loaded
- Can be overridden if needed via `mm_kwargs`

### 3. VllmConfig Integration
- Properly integrated with vLLM's configuration system
- Supports all vLLM features (quantization, distributed inference, etc.)

### 4. Multi-Modal Processing
- Registered in vLLM's multi-modal registry
- Supports batch processing of audio inputs

## Testing

Run the provided test suite:

```bash
# Test configuration loading
python test_fireredasr.py config

# Test simple inference
python test_fireredasr.py simple

# Test async inference
python test_fireredasr.py async

# Test batch processing
python test_fireredasr.py batch
```

## Troubleshooting

### Common Issues

#### 1. Model Type Not Recognized
**Problem**: vLLM doesn't recognize the model as FireRedASR
**Solution**:
- Run `python setup_fireredasr.py /path/to/model` to create config.json
- Or manually create config.json with `"model_type": "fireredasr"`

#### 2. Symlink Resolution Issues
**Problem**: LLM directory symlink not resolving correctly
**Solution**:
- Ensure the symlink target exists
- Check permissions on both the symlink and target
- The integration automatically uses `os.path.realpath()` to resolve symlinks

#### 3. Missing Files
**Error**: "Missing required file: cmvn.ark"
**Solution**:
1. Verify all required files are present:
   - cmvn.ark
   - asr_encoder.pth.tar
   - model.pth.tar
2. Check file permissions
3. Use absolute paths

#### 4. Import Errors
**Error**: "ImportError: No module named 'fireredasr'"
**Solution**:
1. Install FireRedASR: `pip install fireredasr`
2. Check Python environment
3. Verify installation: `python -c "import fireredasr"`

#### 5. Memory Issues
**Problem**: Out of memory errors
**Solution**:
- Reduce `max_model_len`
- Use tensor parallelism: `tensor_parallel_size=2`
- Enable CPU offloading if available

### Performance Optimization
1. **ASR-specific settings**:
   - Always use `temperature=0.0` for deterministic output
   - Set `max_tokens=256` (ASR outputs are typically short)

2. **Batch processing**:
   - Process multiple audio files in parallel
   - Use async mode for better throughput

3. **GPU optimization**:
   - Enable tensor parallelism for multi-GPU
   - Use appropriate batch sizes

## Advanced Usage

### Custom CMVN Path

```python
prompts = [{
    "prompt": "",
    "multi_modal_data": {"audio": "audio.wav"},
    "mm_kwargs": {
        "cmvn_path": "/custom/path/to/cmvn.ark"
    }
}]
```

### Distributed Inference

```python
llm = LLM(
    model="/path/to/fireredasr_model",
    tensor_parallel_size=4,  # Use 4 GPUs
    trust_remote_code=True,
)
```

## Implementation Details

The integration consists of three main components:

1. **FireRedAsrConfig** (`vllm/transformers_utils/configs/fireredasr.py`)
   - Handles configuration and automatic path resolution
   - Compatible with HuggingFace transformers

2. **FireRedAsrForConditionalGeneration** (`vllm/model_executor/models/fireredasr_vllm.py`)
   - Main model class that initializes encoder, projector, and LLM
   - Properly uses VllmConfig for initialization
   - Handles multi-modal embedding generation

3. **FireRedAsrMultiModalProcessor**
   - Processes audio inputs using ASRFeatExtractor
   - Automatically retrieves CMVN path from config
   - Generates speech features and lengths

## Contributing

When modifying the FireRedASR integration:
1. Ensure VllmConfig is properly used in model initialization
2. Test automatic path resolution with various directory structures
3. Verify batch processing works correctly
4. Check async mode functionality

## License

This integration follows vLLM's Apache 2.0 license.