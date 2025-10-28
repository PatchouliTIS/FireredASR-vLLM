#!/usr/bin/env python3
"""
Setup helper for FireRedASR models in vLLM.

This script helps prepare FireRedASR models for use with vLLM by:
1. Verifying the model directory structure
2. Resolving symlinks
3. Creating the necessary config.json file
4. Validating the setup

Usage:
    python setup_fireredasr.py /path/to/FireRedASR-LLM-L
"""

import argparse
import json
import os
import sys
from pathlib import Path


def check_model_structure(model_dir: Path) -> dict:
    """Check if the model directory has the expected structure."""
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "info": {}
    }

    # Check required files
    required_files = [
        "cmvn.ark",
        "asr_encoder.pth.tar",
        "model.pth.tar"
    ]

    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            results["errors"].append(f"Missing required file: {file}")
            results["valid"] = False
        else:
            results["info"][file] = str(file_path)

    # Check for LLM directory (can be symlink)
    llm_dirs = [
        "Qwen2-7B-Instruct",
        "Qwen2-1.5B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-0.5B-Instruct"
    ]

    found_llm = None
    for llm_dir in llm_dirs:
        llm_path = model_dir / llm_dir
        if llm_path.exists():
            found_llm = llm_dir
            # Check if it's a symlink and resolve it
            if llm_path.is_symlink():
                real_path = llm_path.resolve()
                results["info"][f"llm_dir_symlink"] = str(llm_path)
                results["info"][f"llm_dir_real"] = str(real_path)

                # Verify the target exists and has required files
                if not real_path.exists():
                    results["errors"].append(f"Symlink {llm_dir} points to non-existent path: {real_path}")
                    results["valid"] = False
                elif not (real_path / "config.json").exists():
                    results["warnings"].append(f"LLM directory {real_path} missing config.json")
            else:
                results["info"]["llm_dir"] = str(llm_path)
            break

    if not found_llm:
        # Try to find any directory containing "qwen"
        for item in model_dir.iterdir():
            if item.is_dir() and "qwen" in item.name.lower():
                found_llm = item.name
                results["info"]["llm_dir"] = str(item)
                results["warnings"].append(f"Using non-standard LLM directory: {item.name}")
                break

    if not found_llm:
        results["errors"].append("No Qwen LLM directory found")
        results["valid"] = False

    # Check optional files
    optional_files = ["config.yaml", "config.json", "README.md"]
    for file in optional_files:
        file_path = model_dir / file
        if file_path.exists():
            results["info"][file] = str(file_path)

    return results


def create_config_json(model_dir: Path, force: bool = False) -> bool:
    """Create a config.json file for the FireRedASR model."""
    config_path = model_dir / "config.json"

    if config_path.exists() and not force:
        print(f"config.json already exists at {config_path}")
        print("Use --force to overwrite")
        return False

    config = {
        "model_type": "fireredasr",
        "architectures": ["FireRedAsrForConditionalGeneration"],
        "asr_type": "llm",
        "encoder_dim": 512,
        "encoder_downsample_rate": 4,
        "freeze_encoder": True,
        "sampling_rate": 16000,
        "default_speech_token": "<speech>",
        "speech_token_id": 151659
    }

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Created config.json at {config_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create config.json: {e}")
        return False


def print_test_code(model_dir: Path):
    """Print example code for testing the model."""
    print("\n" + "="*60)
    print("TEST CODE")
    print("="*60)
    print(f"""
# Test with vLLM
from vllm import LLM, SamplingParams

# Initialize model
model_path = "{model_dir}"
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=2048,
)

# Prepare test input
audio_file = "/path/to/test.wav"  # Replace with actual audio file
prompts = [{{
    "prompt": "",
    "multi_modal_data": {{
        "audio": audio_file
    }}
}}]

# Run inference
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
)

outputs = llm.generate(prompts, sampling_params)

# Print result
for output in outputs:
    print(f"Transcription: {{output.outputs[0].text}}")
""")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Setup FireRedASR model for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check and setup a model
    python setup_fireredasr.py /path/to/FireRedASR-LLM-L

    # Force recreate config.json
    python setup_fireredasr.py /path/to/FireRedASR-LLM-L --force

    # Only validate without making changes
    python setup_fireredasr.py /path/to/FireRedASR-LLM-L --check-only
        """
    )

    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to FireRedASR model directory (e.g., FireRedASR-LLM-L)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing config.json"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check model structure without creating files"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()

    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        sys.exit(1)

    print(f"Checking FireRedASR model at: {model_dir}")
    print("-" * 60)

    # Check model structure
    results = check_model_structure(model_dir)

    # Print results
    if results["errors"]:
        print("\n❌ ERRORS:")
        for error in results["errors"]:
            print(f"  - {error}")

    if results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"]:
            print(f"  - {warning}")

    if args.verbose or not results["valid"]:
        print("\n📁 MODEL STRUCTURE:")
        for key, value in results["info"].items():
            print(f"  {key}: {value}")

    if not results["valid"]:
        print("\n✗ Model validation failed. Please fix the errors above.")
        sys.exit(1)

    print("\n✓ Model structure is valid")

    # Create config.json if needed
    if not args.check_only:
        config_created = create_config_json(model_dir, args.force)

        if config_created or (model_dir / "config.json").exists():
            print("\n✅ Model is ready for use with vLLM!")
            print_test_code(model_dir)
        else:
            print("\n⚠️  config.json was not created")
            print("The model may still work but vLLM might not auto-detect it as FireRedASR")
    else:
        print("\n(Check-only mode, no files were created)")

        if not (model_dir / "config.json").exists():
            print("ℹ️  Run without --check-only to create config.json")


if __name__ == "__main__":
    main()