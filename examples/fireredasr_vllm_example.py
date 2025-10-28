"""
Example usage of FireRedASR with vLLM integration.

This example demonstrates how to use the FireRedASR model with vLLM
for efficient speech-to-text transcription.
"""

from vllm import LLM, SamplingParams


def main():
    """Example of using FireRedASR with vLLM."""
    
    # Model configuration
    model_dir = "/path/to/fireredasr/model"  # Update with your model path
    
    # Initialize LLM with FireRedASR
    # The model needs:
    # 1. encoder_path: path to asr_encoder.pth.tar
    # 2. cmvn_path: path to cmvn.ark
    # 3. llm_dir: path to Qwen2-7B-Instruct
    
    llm = LLM(
        model="fireredasr",
        trust_remote_code=True,
        # Additional config can be passed via --override-config
        override_neuron_config={
            "encoder_path": f"{model_dir}/asr_encoder.pth.tar",
            "cmvn_path": f"{model_dir}/cmvn.ark",
            "llm_dir": f"{model_dir}/Qwen2-7B-Instruct",
            "freeze_encoder": True,
            "freeze_llm": False,
            "encoder_downsample_rate": 4,
        }
    )
    
    # Sampling parameters for ASR
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for ASR
        max_tokens=100,    # Adjust based on expected transcription length
        repetition_penalty=1.0,
    )
    
    # Prepare audio inputs
    # Option 1: Audio file paths
    audio_paths = [
        "/path/to/audio1.wav",
        "/path/to/audio2.wav",
    ]
    
    prompts = [
        {
            "prompt": "<|SPEECH|>",  # Speech token placeholder
            "multi_modal_data": {
                "audio": audio_path
            }
        }
        for audio_path in audio_paths
    ]
    
    # Generate transcriptions
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for idx, output in enumerate(outputs):
        print(f"Audio {idx + 1}: {audio_paths[idx]}")
        print(f"Transcription: {output.outputs[0].text}")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")
        print("-" * 50)


def example_with_raw_audio():
    """Example using raw audio tensors instead of file paths."""
    import torch
    from vllm import LLM, SamplingParams
    
    model_dir = "/path/to/fireredasr/model"
    
    llm = LLM(
        model="fireredasr",
        trust_remote_code=True,
        override_neuron_config={
            "encoder_path": f"{model_dir}/asr_encoder.pth.tar",
            "cmvn_path": f"{model_dir}/cmvn.ark",
            "llm_dir": f"{model_dir}/Qwen2-7B-Instruct",
        }
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
    )
    
    # Create dummy audio tensor (batch_size, time_steps)
    # In practice, load this from your audio processing pipeline
    dummy_audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
    
    prompts = [
        {
            "prompt": "<|SPEECH|>",
            "multi_modal_data": {
                "audio": dummy_audio
            }
        }
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Transcription: {output.outputs[0].text}")


def batch_transcription_example():
    """Example of batch transcription for efficiency."""
    from vllm import LLM, SamplingParams
    import glob
    
    model_dir = "/path/to/fireredasr/model"
    
    llm = LLM(
        model="fireredasr",
        trust_remote_code=True,
        tensor_parallel_size=1,  # Adjust based on available GPUs
        override_neuron_config={
            "encoder_path": f"{model_dir}/asr_encoder.pth.tar",
            "cmvn_path": f"{model_dir}/cmvn.ark",
            "llm_dir": f"{model_dir}/Qwen2-7B-Instruct",
        }
    )
    
    # ASR-optimized sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,           # Greedy for accuracy
        max_tokens=200,            # Max transcription length
        repetition_penalty=1.0,    # Can adjust if needed
        # For beam search:
        # best_of=5,
        # use_beam_search=True,
    )
    
    # Load all audio files from directory
    audio_dir = "/path/to/audio/directory"
    audio_files = glob.glob(f"{audio_dir}/*.wav")
    
    print(f"Processing {len(audio_files)} audio files...")
    
    # Create prompts for batch processing
    prompts = [
        {
            "prompt": "<|SPEECH|>",
            "multi_modal_data": {"audio": audio_file}
        }
        for audio_file in audio_files
    ]
    
    # Process in batch - vLLM will handle batching automatically
    outputs = llm.generate(prompts, sampling_params)
    
    # Save results
    results = []
    for audio_file, output in zip(audio_files, outputs):
        transcription = output.outputs[0].text
        results.append({
            "file": audio_file,
            "transcription": transcription,
            "num_tokens": len(output.outputs[0].token_ids),
        })
        print(f"{audio_file}: {transcription}")
    
    return results


if __name__ == "__main__":
    print("FireRedASR vLLM Integration Examples")
    print("=" * 60)
    
    print("\n1. Basic Usage")
    print("-" * 60)
    # Uncomment to run:
    # main()
    
    print("\n2. Raw Audio Tensor Usage")
    print("-" * 60)
    # Uncomment to run:
    # example_with_raw_audio()
    
    print("\n3. Batch Transcription")
    print("-" * 60)
    # Uncomment to run:
    # batch_transcription_example()
    
    print("\nNote: Update model paths and uncomment examples to run.")

