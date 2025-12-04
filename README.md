<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## Summary

The current repo is a specialized adaptation tailored to the original FireredASR-LLM model architecture and input parameters, containing extensive hard-coded elements. Significant work remains to be done before it can be merged into the main vLLM branch:

- [ ] Modify the FireredASR-LLM model files to match the standard loading procedure in vLLM
- [ ] Modify the input format to support raw features data
- [ ] Remove the separate fireredasr directory in `vllm/model_executor/models`


## Getting Started

1. Run `merge_lora_weights.py` under the directory of `FireRedASR-LLM-L` to get the complete Qwen2-7B LLM model with LoRA weights.

2. Run `save_tokenizer.py` to get the specific tokenizer of Qwen2-7B model.

3. Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):
  
    Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

  - [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
  - [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
  - [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
