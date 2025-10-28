#!/usr/bin/env python3
"""
验证 FireRedASR vLLM 集成是否正确配置。

运行此脚本以检查所有依赖和配置是否就绪。

Usage:
    python scripts/verify_fireredasr_integration.py --model-dir /path/to/model
"""

import argparse
import os
import sys
from pathlib import Path


class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(message, status="info"):
    """Print colored status message."""
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.ENDC} {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")
    elif status == "info":
        print(f"{Colors.BLUE}ℹ{Colors.ENDC} {message}")


def check_python_version():
    """Check Python version."""
    print_status("Checking Python version...", "info")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - OK", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Need >= 3.8", "error")
        return False


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print_status(f"{package_name} - Installed", "success")
        return True
    except ImportError:
        print_status(f"{package_name} - Not installed", "error")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print_status("Checking dependencies...", "info")
    
    dependencies = {
        "torch": "torch",
        "transformers": "transformers",
        "vllm": "vllm",
        "fireredasr": "fireredasr",
    }
    
    all_installed = True
    for package, import_name in dependencies.items():
        if not check_package(package, import_name):
            all_installed = False
    
    return all_installed


def check_vllm_integration():
    """Check if FireRedASR vLLM integration is available."""
    print_status("Checking vLLM integration...", "info")
    
    try:
        from vllm.model_executor.models.fireredasr_vllm import (
            FireRedAsrConfig,
            FireRedAsrForConditionalGeneration,
        )
        print_status("FireRedASR vLLM integration - Available", "success")
        return True
    except ImportError as e:
        print_status(f"FireRedASR vLLM integration - Not found: {e}", "error")
        return False


def check_model_files(model_dir):
    """Check if all required model files exist."""
    print_status(f"Checking model files in {model_dir}...", "info")
    
    if not model_dir:
        print_status("No model directory specified, skipping file checks", "warning")
        return True
    
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print_status(f"Model directory does not exist: {model_dir}", "error")
        return False
    
    required_files = {
        "asr_encoder.pth.tar": "ASR Encoder checkpoint",
        "model.pth.tar": "Model checkpoint (projector)",
        "cmvn.ark": "CMVN statistics",
        "Qwen2-7B-Instruct": "LLM directory",
    }
    
    all_exist = True
    for filename, description in required_files.items():
        filepath = model_path / filename
        if filepath.exists():
            print_status(f"{description} - Found", "success")
        else:
            print_status(f"{description} - Missing: {filename}", "error")
            all_exist = False
    
    return all_exist


def check_llm_config(model_dir):
    """Check LLM configuration files."""
    if not model_dir:
        return True
    
    print_status("Checking LLM configuration...", "info")
    
    llm_dir = Path(model_dir) / "Qwen2-7B-Instruct"
    
    if not llm_dir.exists():
        print_status("LLM directory not found", "error")
        return False
    
    required_llm_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    
    all_exist = True
    for filename in required_llm_files:
        filepath = llm_dir / filename
        if filepath.exists():
            print_status(f"  {filename} - Found", "success")
        else:
            print_status(f"  {filename} - Missing", "error")
            all_exist = False
    
    # Check for model weights (at least one should exist)
    weight_patterns = ["*.safetensors", "*.bin", "*.pth"]
    weights_found = False
    for pattern in weight_patterns:
        if list(llm_dir.glob(pattern)):
            weights_found = True
            break
    
    if weights_found:
        print_status("  Model weights - Found", "success")
    else:
        print_status("  Model weights - Missing", "error")
        all_exist = False
    
    return all_exist


def test_basic_import():
    """Test basic imports."""
    print_status("Testing basic imports...", "info")
    
    try:
        from vllm.model_executor.models.fireredasr_vllm import (
            FireRedAsrConfig,
            FireRedAsrInputs,
            FireRedAsrEncoder,
            FireRedAsrProjector,
            FireRedAsrForConditionalGeneration,
            FireRedAsrMultiModalProcessor,
        )
        print_status("All classes imported successfully", "success")
        return True
    except Exception as e:
        print_status(f"Import failed: {e}", "error")
        return False


def test_config_creation():
    """Test config creation."""
    print_status("Testing config creation...", "info")
    
    try:
        from vllm.model_executor.models.fireredasr_vllm import FireRedAsrConfig
        
        config = FireRedAsrConfig(
            encoder_path="/path/to/encoder.pth.tar",
            encoder_dim=512,
            llm_dir="/path/to/llm",
            cmvn_path="/path/to/cmvn.ark",
        )
        
        assert config.encoder_dim == 512
        assert config.encoder_path == "/path/to/encoder.pth.tar"
        
        print_status("Config creation - OK", "success")
        return True
    except Exception as e:
        print_status(f"Config creation failed: {e}", "error")
        return False


def test_tensor_operations():
    """Test basic tensor operations."""
    print_status("Testing tensor operations...", "info")
    
    try:
        import torch
        from vllm.model_executor.models.fireredasr_vllm import FireRedAsrInputs
        
        # Create sample inputs
        inputs: FireRedAsrInputs = {
            "speech_features": torch.randn(2, 100, 80),
            "speech_lengths": torch.tensor([100, 90]),
        }
        
        assert inputs["speech_features"].shape == (2, 100, 80)
        assert inputs["speech_lengths"].shape == (2,)
        
        print_status("Tensor operations - OK", "success")
        return True
    except Exception as e:
        print_status(f"Tensor operations failed: {e}", "error")
        return False


def run_full_verification(model_dir=None):
    """Run full verification."""
    print(f"\n{Colors.BOLD}FireRedASR vLLM Integration Verification{Colors.ENDC}\n")
    print("=" * 60)
    
    results = []
    
    # System checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("vLLM Integration", check_vllm_integration()))
    
    # Model file checks
    if model_dir:
        results.append(("Model Files", check_model_files(model_dir)))
        results.append(("LLM Config", check_llm_config(model_dir)))
    
    # Functional tests
    results.append(("Basic Imports", test_basic_import()))
    results.append(("Config Creation", test_config_creation()))
    results.append(("Tensor Operations", test_tensor_operations()))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"\n{Colors.BOLD}Verification Summary{Colors.ENDC}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "success" if result else "error"
        print_status(f"{name}: {'PASS' if result else 'FAIL'}", status)
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed! ({passed}/{total}){Colors.ENDC}\n")
        print("Your FireRedASR vLLM integration is ready to use.\n")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some checks failed ({passed}/{total}){Colors.ENDC}\n")
        print("Please fix the errors above before using the integration.\n")
        
        # Provide help
        print(f"{Colors.BOLD}Next steps:{Colors.ENDC}")
        print("1. Install missing dependencies:")
        print("   pip install torch transformers vllm fireredasr")
        print("2. Ensure model files are in the correct location")
        print("3. Check the integration documentation in FIREREDASR_VLLM_README.md")
        print()
        
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify FireRedASR vLLM integration setup"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to FireRedASR model directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    success = run_full_verification(args.model_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

