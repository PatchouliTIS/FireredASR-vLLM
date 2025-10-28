"""
Test and usage examples for FireRedASR in vLLM.

This script demonstrates how to use the FireRedASR model with vLLM for speech-to-text tasks.
"""

import asyncio
import os
from typing import Optional

import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, LLM, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


def test_fireredasr_simple():
    """
    Simple synchronous test for FireRedASR model.

    Expected directory structure:
    FireRedASR-LLM-L/
    ├── cmvn.ark
    ├── asr_encoder.pth.tar
    ├── model.pth.tar
    ├── config.json (created by setup_fireredasr.py)
    └── Qwen2-7B-Instruct -> ../Qwen2-7B-Instruct (symlink)
    """
    # Path to your FireRedASR model directory
    # Example: /apdcephfs_qy2/share_303477892/patchychen/fireredasr_models/FireRedASR-LLM-L
    model_path = os.environ.get("FIREREDASR_MODEL_PATH", "/path/to/FireRedASR-LLM-L")

    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        print("Set FIREREDASR_MODEL_PATH environment variable or edit the path in the script")
        return

    print(f"Loading FireRedASR model from: {model_path}")

    # Initialize the model
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
    )

    # Prepare input
    audio_file = os.environ.get("TEST_AUDIO_FILE", "/path/to/audio.wav")

    if not os.path.exists(audio_file):
        print(f"Warning: Audio file does not exist: {audio_file}")
        print("Using placeholder path. Set TEST_AUDIO_FILE environment variable")

    prompts = [{
        "prompt": "",  # FireRedASR uses empty prompt
        "multi_modal_data": {
            "audio": audio_file
        }
    }]

    # Set sampling parameters for ASR
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for ASR
        top_p=1.0,
        max_tokens=256,  # ASR outputs are typically short
    )

    # Generate transcription
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Audio: {audio_file}")
        print(f"Transcription: {generated_text}")
        print("-" * 50)


async def test_fireredasr_async():
    """
    Asynchronous test for FireRedASR model using AsyncLLM.
    """
    model_path = "/path/to/fireredasr_model"

    # Create engine arguments
    engine_args = AsyncEngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        tensor_parallel_size=1,
    )

    # Initialize AsyncLLM
    async_llm = AsyncLLM.from_engine_args(engine_args)

    # Prepare input
    audio_file = "/path/to/audio.wav"

    prompt = {
        "prompt": "",
        "multi_modal_data": {
            "audio": audio_file
        }
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    # Generate asynchronously
    request_id = "test_request_001"

    async for output in async_llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id
    ):
        if output.outputs:
            for completion in output.outputs:
                print(f"Transcription: {completion.text}")

    # Clean up
    async_llm.shutdown()


async def test_fireredasr_batch():
    """
    Test batch processing of multiple audio files.
    """
    model_path = "/path/to/fireredasr_model"

    # Initialize
    async_llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            max_model_len=2048,
        )
    )

    # Multiple audio files
    audio_files = [
        "/path/to/audio1.wav",
        "/path/to/audio2.wav",
        "/path/to/audio3.wav",
    ]

    # Create tasks for concurrent processing
    tasks = []
    for i, audio_file in enumerate(audio_files):
        prompt = {
            "prompt": "",
            "multi_modal_data": {"audio": audio_file}
        }

        task = async_llm.generate(
            prompt=prompt,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
            request_id=f"request_{i}"
        )
        tasks.append(task)

    # Process all requests
    for i, task in enumerate(tasks):
        print(f"Processing audio {i+1}/{len(tasks)}")
        async for output in task:
            if output.outputs:
                transcription = output.outputs[0].text
                print(f"  Audio {i+1}: {transcription}")

    # Clean up
    async_llm.shutdown()


def test_model_loading():
    """
    Test that the model configuration is correctly loaded.
    """
    from vllm.transformers_utils.configs.fireredasr import FireRedAsrConfig

    # Test path resolution
    model_dir = "/path/to/fireredasr_model"

    # Load config
    config = FireRedAsrConfig.from_pretrained(model_dir)

    print("Model Configuration:")
    print(f"  Model directory: {config.model_dir}")
    print(f"  CMVN path: {config.cmvn_path}")
    print(f"  Encoder path: {config.encoder_path}")
    print(f"  LLM directory: {config.llm_dir}")
    print(f"  ASR type: {config.asr_type}")

    # Validate paths
    if config.cmvn_path and os.path.exists(config.cmvn_path):
        print("  ✓ CMVN file found")
    else:
        print("  ✗ CMVN file not found")

    if config.encoder_path and os.path.exists(config.encoder_path):
        print("  ✓ Encoder file found")
    else:
        print("  ✗ Encoder file not found")

    if config.llm_dir and os.path.exists(config.llm_dir):
        print("  ✓ LLM directory found")
    else:
        print("  ✗ LLM directory not found")


def create_model_config_json(model_dir: str):
    """
    Helper function to create a config.json for FireRedASR model.

    Args:
        model_dir: Directory containing the FireRedASR model files
    """
    import json

    config = {
        "model_type": "fireredasr",
        "architectures": ["FireRedAsrForConditionalGeneration"],
        "asr_type": "llm",
        "encoder_dim": 512,
        "encoder_downsample_rate": 4,
        "freeze_encoder": True,
        "sampling_rate": 16000,
        "default_speech_token": "<speech>",
        "speech_token_id": 151659,
    }

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created config.json at {config_path}")


if __name__ == "__main__":
    # Choose which test to run
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]

        if test_name == "simple":
            test_fireredasr_simple()
        elif test_name == "async":
            asyncio.run(test_fireredasr_async())
        elif test_name == "batch":
            asyncio.run(test_fireredasr_batch())
        elif test_name == "config":
            test_model_loading()
        elif test_name == "create_config":
            if len(sys.argv) > 2:
                create_model_config_json(sys.argv[2])
            else:
                print("Usage: python test_fireredasr.py create_config <model_dir>")
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: simple, async, batch, config, create_config")
    else:
        print("Usage: python test_fireredasr.py <test_name>")
        print("Available tests: simple, async, batch, config, create_config")
        print("\nExample:")
        print("  python test_fireredasr.py simple")
        print("  python test_fireredasr.py create_config /path/to/model")