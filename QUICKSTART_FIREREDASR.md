# FireRedASR Quick Start Guide

## 🚀 Quick Setup (3 Steps)

### Step 1: Prepare Your Model

Your FireRedASR model directory should look like:
```
FireRedASR-LLM-L/
├── cmvn.ark
├── asr_encoder.pth.tar
├── model.pth.tar
└── Qwen2-7B-Instruct/ (can be symlink)
```

### Step 2: Run Setup Script

```bash
# Setup the model (creates config.json)
python setup_fireredasr.py /path/to/FireRedASR-LLM-L
```

### Step 3: Use the Model

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="/path/to/FireRedASR-LLM-L",
    trust_remote_code=True,
)

# Transcribe audio
prompts = [{
    "prompt": "",
    "multi_modal_data": {"audio": "audio.wav"}
}]

outputs = llm.generate(
    prompts,
    SamplingParams(temperature=0.0, max_tokens=256)
)

print(outputs[0].outputs[0].text)
```

## 📋 Complete Setup Checklist

- [ ] FireRedASR package installed: `pip install fireredasr`
- [ ] Model files present (cmvn.ark, encoders, etc.)
- [ ] Qwen2 LLM directory available (symlink OK)
- [ ] Run `setup_fireredasr.py` to create config.json
- [ ] Test with sample audio file

## 🔧 Environment Variables (Optional)

```bash
# For testing
export FIREREDASR_MODEL_PATH=/path/to/FireRedASR-LLM-L
export TEST_AUDIO_FILE=/path/to/test.wav

# Run tests
python test_fireredasr.py simple
```

## ⚠️ Common Gotchas

1. **No config.json**: Run the setup script first!
2. **Symlink issues**: Make sure symlink targets exist
3. **Memory**: Use `tensor_parallel_size` for large models

## 📚 Full Documentation

See [README_FIREREDASR.md](README_FIREREDASR.md) for detailed documentation.