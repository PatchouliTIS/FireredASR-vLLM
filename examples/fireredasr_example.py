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
- Qwen2-7B-Instruct/ (LLM directory)
- model.pth.tar (optional, main model checkpoint)
"""

from vllm import LLM, SamplingParams

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

def estimate_max_tokens_from_audio(audio_path: str) -> int:
    """根据音频时长估算 max_tokens"""
    info = sf.info(audio_path)
    duration_seconds = info.duration

    # fbank 帧数 (100 fps)
    feat_frames = int(duration_seconds * 100)

    # Encoder Conv2dSubsampling
    padded = feat_frames + 6
    after_conv1 = (padded - 3) // 2 + 1
    encoder_frames = (after_conv1 - 3) // 2 + 1

    # Adapter 下采样
    speech_frames = encoder_frames // 2

    return max(1, speech_frames)

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

def batch_transcription_with_chunking():
    """
    Memory-optimized batch transcription using chunking.

    This approach processes audio files in smaller batches to avoid OOM errors
    when dealing with large datasets (e.g., 200+ audio files).
    """
    import glob
    import torch

    model_dir = "/home/ray/pretrained_models/FireRedASR-LLM-L/"  # Update with your model path
    tokenizer_dir = "/home/ray/pretrained_models/tokenizer/"

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        disable_log_stats=True,
        tokenizer=tokenizer_dir,
        gpu_memory_utilization=0.80,  # Reduced from 0.95 to leave more headroom
        max_model_len=4096,
        mm_processor_cache_gb=0, # disable mm_processor cache
        max_num_seqs=32, # set max_num_seqs to 1 to avoid batching
        dtype="bfloat16",
        enforce_eager=False
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
        # repetition_penalty=1.2,
        top_p=1.0,
    )

    # Load all audio files
    audio_dir = "/apdcephfs_qy2/share_303477892/patchychen/data/asr_data_yace_251223_final/ch/ch_sentence_denoise_wav_dir/"  # Change this to your audio directory

    # Get all .wav files from the directory
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    audio_files.sort()
    audio_files = audio_files[:32]
    # audio_files = ["/apdcephfs_qy2/share_303477892/patchychen/data/asr_data_yace_251223_final/ch/ch_sentence_denoise_wav_dir/-6399VpPb1c_1280x720.f140_transform_denoise_16k_seg0_speaker_1_denoise.wav"]

    print(f"Total audio files: {len(audio_files)}")
    print(f"audio_file: {audio_files[0]}")

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

        # 计算当前批次的 max_tokens（取最大值）
        batch_max_tokens = max(
            estimate_max_tokens_from_audio(f) for f in chunk_files
        )

        sampling_params.max_tokens = batch_max_tokens

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

        # Process chunk
        chunk_outputs = llm.generate(chunk_prompts, sampling_params)

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
    print_stats("vLLM Batch Chunking", elapsed_time, len(all_results), total_audio_duration)

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