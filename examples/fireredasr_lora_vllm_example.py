"""
Example usage of FireRedASR with vLLM integration.

This example demonstrates how to use the FireRedASR model with vLLM
for efficient speech-to-text transcription.

IMPORTANT: Before running this example, ensure your model directory has a config.json file.
You can create one using:
    python fireredasr_setup_config.py /path/to/your/model/directory/

The model directory should contain:
- config.json (FireRedASR configuration)
- asr_encoder.pth.tar (ASR encoder checkpoint)
- cmvn.ark (CMVN statistics)
- Qwen2-7B-Instruct/ (LLM directory - base model without LoRA merged)
- model.pth.tar (optional, main model checkpoint)

For LoRA usage, you also need:
- fireredasr_lora_adapter/ (LoRA adapter directory containing adapter_config.json and adapter_model.safetensors)
"""

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import os
import glob
import time
import soundfile as sf


def get_audio_duration(audio_path):
    """获取单个音频文件时长（秒）"""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"Warning: Could not read duration for {audio_path}: {e}")
        return 0.0


def print_stats(name, elapsed_time, num_samples, total_audio_duration):
    """打印性能统计信息"""
    qps = num_samples / elapsed_time if elapsed_time > 0 else 0
    rtf = elapsed_time / total_audio_duration if total_audio_duration > 0 else 0

    print(f"\n{'='*60}")
    print(f"[{name}] 性能统计:")
    print(f"  - 样本数量: {num_samples}")
    print(f"  - 音频总时长: {total_audio_duration:.2f} 秒")
    print(f"  - 端到端时延: {elapsed_time:.3f} 秒")
    print(f"  - 平均时延: {elapsed_time/num_samples:.3f} 秒/样本" if num_samples > 0 else "  - 平均时延: N/A")
    print(f"  - QPS: {qps:.2f} 样本/秒")
    print(f"  - RTF: {rtf:.4f} (越小越好, <1 表示实时)")
    print(f"  - 速度: {1/rtf:.2f}x 实时" if rtf > 0 else "  - 速度: N/A")
    print(f"{'='*60}\n")


def main():
    """Example of using FireRedASR with vLLM and LoRA support."""

    # Model configuration
    model_dir = "/home/ray/pretrained_models/FireRedASR-LLM-L/"  # Update with your model path
    tokenizer_dir = "/home/ray/pretrained_models/tokenizer/"
    lora_adapter_path = "/home/ray/pretrained_models/FireRedASR-LLM-L/fireredasr_lora_adapter/"  # LoRA adapter directory

    # Initialize LLM with FireRedASR
    # The model directory should contain:
    # 1. config.json with FireRedASR configuration
    # 2. asr_encoder.pth.tar
    # 3. cmvn.ark
    # 4. Qwen2-7B-Instruct/ subdirectory (base model without LoRA merged)
    # 5. fireredasr_lora_adapter/ (LoRA adapter with adapter_config.json and adapter_model.safetensors)

    # Note: The FireRedAsrConfig should be saved as config.json in the model directory
    # with all necessary paths configured

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        disable_log_stats=True,  # Disable logging for cleaner output
        tokenizer=tokenizer_dir,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        dtype="float16",
        mm_processor_cache_gb=0,  # disable mm_processor cache
        max_num_seqs=2,  # set max_num_seqs to 1 to avoid batching
        # LoRA configurations
        enable_lora=True,         # Enable LoRA support
        max_lora_rank=64,         # Must be >= your LoRA rank (r=64)
        max_loras=1,              # Maximum number of LoRAs to load simultaneously
    )

    # Create LoRA request
    lora_request = LoRARequest(
        lora_name="fireredasr-asr",      # Human-readable name
        lora_int_id=1,                    # Unique integer ID
        lora_path=lora_adapter_path,      # Path to LoRA adapter directory
    )

    # Sampling parameters for ASR
    sampling_params = SamplingParams(
        temperature=1.0,  # Greedy decoding for ASR
        max_tokens=4096,    # Adjust based on expected transcription length
        repetition_penalty=1.0,
        top_p=1.0,
    )

    # Prepare audio inputs
    # Option 1: Audio file paths
    audio_paths = [
        "examples/wav/BAC009S0764W0121.wav",
    ]

    # 计算音频总时长
    total_audio_duration = sum(get_audio_duration(p) for p in audio_paths)
    print(f"音频文件总时长: {total_audio_duration:.2f} 秒")
    print(f"Using LoRA adapter: {lora_request.lora_name}")

    prompts = [
        {
            "prompt": "<|im_start|>user\n<speech>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",  # Speech token placeholder
            "multi_modal_data": {
                "audio": audio_path
            }
        }
        for audio_path in audio_paths
    ]

    # Generate transcriptions with LoRA
    start_time = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,  # Pass LoRA request
    )
    elapsed_time = time.perf_counter() - start_time

    # Print results
    for idx, output in enumerate(outputs):
        audio_duration = get_audio_duration(audio_paths[idx])
        print(f"Audio {idx + 1}: {audio_paths[idx]}")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Transcription: {output.outputs[0].text}")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")
        print("-" * 50)

    # 打印性能统计
    print_stats("vLLM with LoRA", elapsed_time, len(audio_paths), total_audio_duration)


def batch_transcription_with_chunking():
    """
    Memory-optimized batch transcription using chunking with LoRA support.

    This approach processes audio files in smaller batches to avoid OOM errors
    when dealing with large datasets (e.g., 200+ audio files).
    """
    import glob
    import torch

    model_dir = "/apdcephfs_qy2/share_303477892/patchychen/29_FireRedASR/pretrained_models/FireRedASR-LLM-L/"  # Update with your model path
    tokenizer_dir = "/apdcephfs_qy2/share_303477892/patchychen/29_FireRedASR/pretrained_models/tokenizer/"
    lora_adapter_path = "/apdcephfs_qy2/share_303477892/patchychen/29_FireRedASR/pretrained_models/FireRedASR-LLM-L/lora_adapter/"  # LoRA adapter directory

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        disable_log_stats=True,
        tokenizer=tokenizer_dir,
        gpu_memory_utilization=0.85,  # Reduced from 0.95 to leave more headroom
        max_model_len=8192,
        mm_processor_cache_gb=0,  # disable mm_processor cache
        max_num_seqs=2,  # set max_num_seqs to 1 to avoid batching
        dtype="auto",
        # LoRA configurations
        enable_lora=True,         # Enable LoRA support
        max_lora_rank=64,         # Must be >= your LoRA rank (r=64)
        max_loras=1,              # Maximum number of LoRAs to load simultaneously
    )

    # Create LoRA request
    lora_request = LoRARequest(
        lora_name="fireredasr-asr",
        lora_int_id=1,
        lora_path=lora_adapter_path,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4096,
        repetition_penalty=1.0,
        top_p=1.0,
    )

    # Load all audio files
    audio_dir = "/apdcephfs_qy2/share_303477892/patchychen/data/asr_data_yace_251223_final/ch/ch_sentence_denoise_wav_dir/"  # Change this to your audio directory

    # Get all .wav files from the directory
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    audio_files.sort()
    audio_files = audio_files[:32]
    audio_files = ["/apdcephfs_qy2/share_303477892/patchychen/data/asr_data_yace_251223_final/ch/ch_sentence_denoise_wav_dir/-6399VpPb1c_1280x720.f140_transform_denoise_16k_seg0_speaker_1_denoise.wav"]

    print(f"Total audio files: {len(audio_files)}")
    print(f"audio_file: {audio_files[0]}")
    print(f"Using LoRA adapter: {lora_request.lora_name}")

    # 计算音频总时长
    audio_durations = {f: get_audio_duration(f) for f in audio_files}
    total_audio_duration = sum(audio_durations.values())
    print(f"音频文件总时长: {total_audio_duration:.2f} 秒")

    # Process in chunks to avoid OOM
    BATCH_SIZE = 32  # Adjust based on your GPU memory and audio length
    all_results = []

    start_time = time.perf_counter()
    for chunk_idx in range(0, len(audio_files), BATCH_SIZE):
        chunk_files = audio_files[chunk_idx:chunk_idx + BATCH_SIZE]

        chunk_num = chunk_idx // BATCH_SIZE + 1
        total_chunks = (len(audio_files) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nProcessing chunk {chunk_num}/{total_chunks}")
        print(f"Files {chunk_idx + 1} to {min(chunk_idx + BATCH_SIZE, len(audio_files))} of {len(audio_files)}")

        # Create prompts for this chunk
        chunk_prompts = [
            {
                "prompt": "<|im_start|>user\n<speech>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"audio": audio_file}
            }
            for audio_file in chunk_files
        ]

        # Process chunk with LoRA
        chunk_outputs = llm.generate(
            chunk_prompts,
            sampling_params,
            lora_request=lora_request,  # Pass LoRA request
        )

        # Collect results
        for audio_file, output in zip(chunk_files, chunk_outputs):
            transcription = output.outputs[0].text
            duration = audio_durations[audio_file]
            all_results.append({
                "file": audio_file,
                "transcription": transcription,
                "num_tokens": len(output.outputs[0].token_ids),
                "audio_duration": duration,
            })
            print(f"[{len(all_results)}/{len(audio_files)}] ({duration:.2f}s) {audio_file.split('/')[-1]}: {transcription}")

        # Optional: Clear CUDA cache between chunks to reduce fragmentation
        if chunk_idx + BATCH_SIZE < len(audio_files):
            torch.cuda.empty_cache()
            print(f"Memory cleared for next chunk")

    print(f"\n✓ Successfully processed all {len(all_results)} audio files")
    elapsed_time = time.perf_counter() - start_time

    # 打印性能统计
    print_stats("vLLM Batch with LoRA", elapsed_time, len(all_results), total_audio_duration)

    return all_results


if __name__ == "__main__":
    print("FireRedASR vLLM Integration Examples")
    print("=" * 60)

    # print("\n1. Basic Usage")
    # print("-" * 60)
    # main()

    print("\n3. Batch Transcription with Chunking (Memory-optimized)")
    print("-" * 60)
    print("This is the RECOMMENDED approach for large datasets (200+ files)")
    batch_transcription_with_chunking()

    print("\nNote: Update model paths and uncomment examples to run.")