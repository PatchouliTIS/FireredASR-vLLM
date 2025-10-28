"""
FireRedASR LLM mode integration for vLLM.

This module implements the vLLM integration for FireRedASR's LLM-based ASR model,
which uses a speech encoder + projector + Qwen2 LLM architecture.
"""
import os
from copy import deepcopy
from typing import Any, Iterable, List, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
from vllm.inputs import InputContext
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, NestedTensors
from vllm.multimodal.parse import (
    AudioEmbeddingItems,
    AudioProcessorItems,
    BaseProcessingInfo,
    MultiModalDataItems,
    MultiModalFieldConfig,
    MultiModalProcessorBase,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    MultiModalDataDict,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors, SequenceData

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP

# Import FireRedASR components
try:
    from .fireredasr.data.asr_feat import ASRFeatExtractor
    from .fireredasr.models.fireredasr_aed import FireRedAsrAed
    from .fireredasr.models.module.adapter import Adapter
except ImportError:
    ASRFeatExtractor = None
    FireRedAsrAed = None
    Adapter = None

# Import FireRedASR config from transformers_utils
from vllm.transformers_utils.configs.fireredasr import FireRedAsrConfig


# ============= Data Structures =============

class FireRedAsrInputs(TypedDict):
    """Type definition for FireRedASR audio inputs."""
    speech_features: torch.Tensor  # Shape: (batch, time, feat_dim)
    speech_lengths: torch.Tensor   # Shape: (batch,)


# ============= Processing Components =============

class FireRedAsrProcessingInfo(BaseProcessingInfo):
    """Processing information for FireRedASR."""
    
    def get_hf_config(self) -> PretrainedConfig:
        return FireRedAsrConfig.from_pretrained(self.ctx.model_config.model)
    
    def get_supported_mm_limits(self) -> dict[str, Optional[int]]:
        return {"audio": None}  # No limit on number of audio inputs
    
    def get_mm_max_tokens_per_item(self, seq_len: int) -> dict[str, int]:
        """Estimate max tokens per audio item."""
        hf_config = self.get_hf_config()
        # Approximate: audio length / downsample_rate
        # This is a rough estimate, actual value depends on audio duration
        max_audio_tokens = seq_len // hf_config.encoder_downsample_rate
        return {"audio": max_audio_tokens}


class FireRedAsrMultiModalProcessor(BaseMultiModalProcessor[FireRedAsrProcessingInfo]):
    """Multimodal processor for FireRedASR."""
    
    def _get_mm_fields_config(
        self,
        hf_inputs: dict[str, Any],
        hf_processor_mm_kwargs: dict[str, Any],
    ) -> dict[str, MultiModalFieldConfig]:
        """Configure multimodal fields."""
        return {
            "speech_features": MultiModalFieldConfig.batched("audio"),
            "speech_lengths": MultiModalFieldConfig.batched("audio"),
            "projected_lengths": MultiModalFieldConfig.batched("audio"),
        }
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, Any],
        mm_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Process audio data using ASRFeatExtractor."""
        if ASRFeatExtractor is None:
            raise ImportError(
                "FireRedASR is not installed. Please install it to use this model."
            )

        # Get configuration
        hf_config = self.info.get_hf_config()

        # Get CMVN path from config (auto-resolved) or mm_kwargs (user override)
        cmvn_path = mm_kwargs.get("cmvn_path", hf_config.cmvn_path)
        if cmvn_path is None:
            raise ValueError(
                "cmvn_path could not be resolved. Please ensure the model directory "
                "contains 'cmvn.ark' or provide cmvn_path explicitly."
            )

        if not os.path.exists(cmvn_path):
            raise FileNotFoundError(f"CMVN file not found at {cmvn_path}")

        feat_extractor = ASRFeatExtractor(cmvn_path)

        # Process audio data
        audio_data = mm_data.get("audio", [])
        if not audio_data:
            # No audio, return empty features
            return {
                "speech_features": torch.empty(0, 0, hf_config.encoder_dim),
                "speech_lengths": torch.empty(0, dtype=torch.long),
            }

        # Extract features
        # audio_data should be list of file paths or audio arrays
        if isinstance(audio_data, list):
            feats, lengths, _ = feat_extractor(audio_data)
        else:
            feats, lengths, _ = feat_extractor([audio_data])

        # Compute projected lengths (after encoder downsampling)
        # This matches what the Adapter.forward will produce
        projected_lengths = lengths // hf_config.encoder_downsample_rate

        return {
            "speech_features": feats,
            "speech_lengths": lengths,
            "projected_lengths": projected_lengths,
        }
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: dict[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptUpdate]:
        """
        Get prompt updates for FireRedASR.
        
        This method computes the projected lengths (after encoder downsampling)
        and creates prompt updates that expand <speech> tokens to match the
        projector output length.
        """
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        
        # Get speech token and token ID
        speech_token = hf_config.default_speech_token
        vocab = tokenizer.get_vocab()
        speech_token_id = vocab[speech_token]
        
        # Get projected lengths (computed in _call_hf_processor)
        projected_lengths = out_mm_kwargs.get("projected_lengths")
        if projected_lengths is None:
            # Fallback: compute from speech_lengths
            speech_lengths = out_mm_kwargs.get("speech_lengths")
            if speech_lengths is not None:
                # Apply encoder downsampling rate
                projected_lengths = speech_lengths // hf_config.encoder_downsample_rate
            else:
                # Ultimate fallback
                projected_lengths = torch.tensor([100] * len(mm_items.get_items("audio", AudioProcessorItems)))
        
        def get_replacement_fireredasr(item_idx: int) -> list[int]:
            """Get replacement tokens for a specific audio item."""
            if isinstance(projected_lengths, torch.Tensor):
                num_tokens = int(projected_lengths[item_idx].item())
            else:
                num_tokens = projected_lengths[item_idx]
            return [speech_token_id] * num_tokens
        
        return [
            PromptReplacement(
                modality="audio",
                target=speech_token_id,
                replacement=get_replacement_fireredasr,
            )
        ]


# ============= Model Components =============

class FireRedAsrEncoder(nn.Module):
    """
    Wrapper for FireRedASR's speech encoder.
    Loads the encoder from FireRedAsrAed model.
    """
    
    def __init__(self, encoder_path: str):
        super().__init__()
        
        if FireRedAsrAed is None:
            raise ImportError(
                "FireRedASR is not installed. Please install it to use this model."
            )
        
        # Load encoder from checkpoint
        package = torch.load(encoder_path, map_location="cpu")
        model = FireRedAsrAed.from_args(package["args"])
        
        if "model_state_dict" in package:
            model.load_state_dict(package["model_state_dict"], strict=False)
        
        self.encoder = model.encoder
        self.encoder_dim = self.encoder.odim
    
    def forward(
        self,
        speech_features: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            speech_features: (batch, time, feat_dim)
            speech_lengths: (batch,)
        
        Returns:
            encoder_outputs: (batch, time', encoder_dim)
            output_lengths: (batch,)
            encoder_mask: (batch, 1, time')
        """
        return self.encoder(speech_features, speech_lengths)


class FireRedAsrProjector(nn.Module):
    """
    Adapter/Projector that maps encoder outputs to LLM embedding space.
    This is a wrapper around FireRedASR's Adapter module.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 4,
    ):
        super().__init__()
        
        if Adapter is None:
            raise ImportError(
                "FireRedASR is not installed. Please install it to use this model."
            )
        
        self.adapter = Adapter(encoder_dim, llm_dim, downsample_rate)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch, time, encoder_dim)
            output_lengths: (batch,)
        
        Returns:
            projected_features: (batch, time', llm_dim)
            projected_lengths: (batch,)
        """
        return self.adapter(encoder_outputs, output_lengths)


# ============= Main Model =============

@MULTIMODAL_REGISTRY.register_processor(
    FireRedAsrMultiModalProcessor,
    info=FireRedAsrProcessingInfo,
)
class FireRedAsrForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    FireRedASR model for conditional generation in vLLM.

    Architecture:
        Audio -> Encoder -> Projector -> LLM (Qwen2)
    """
    
    supports_multimodal: bool = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        # Extract configurations from vllm_config
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config

        # Verify config type
        if not hasattr(config, 'model_type') or config.model_type != 'fireredasr':
            from vllm.transformers_utils.configs.fireredasr import FireRedAsrConfig
            if not isinstance(config, FireRedAsrConfig):
                raise ValueError(f"Expected FireRedAsrConfig, got {type(config)}")

        # Initialize speech encoder
        if config.encoder_path is None:
            raise ValueError("encoder_path must be provided in config. "
                           "Please ensure the model directory contains 'asr_encoder.pth.tar'")

        if not os.path.exists(config.encoder_path):
            raise FileNotFoundError(f"Encoder not found at {config.encoder_path}")

        self.speech_encoder = FireRedAsrEncoder(config.encoder_path)

        # Freeze encoder if required
        if config.freeze_encoder:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False
            self.speech_encoder.eval()

        # Get actual encoder dimension from loaded encoder
        encoder_dim = self.speech_encoder.encoder_dim

        # Initialize LLM
        if config.llm_dir is None:
            raise ValueError("llm_dir must be provided in config. "
                           "Please ensure the model directory contains a Qwen2 subdirectory")

        if not os.path.exists(config.llm_dir):
            raise FileNotFoundError(f"LLM directory not found at {config.llm_dir}")

        # Create a modified vllm_config for the LLM
        llm_vllm_config = self._create_llm_vllm_config(vllm_config, config.llm_dir)

        self.language_model = init_vllm_registered_model(
            vllm_config=llm_vllm_config,
            hf_config=llm_vllm_config.model_config.hf_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        # Get LLM hidden size
        llm_config = llm_vllm_config.model_config.hf_config
        llm_dim = llm_config.hidden_size

        # Initialize projector
        self.projector = FireRedAsrProjector(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            downsample_rate=config.encoder_downsample_rate,
        )

        # Initialize sampler (if needed for standalone use)
        self.sampler = get_sampler()

    def _create_llm_vllm_config(self, base_vllm_config: VllmConfig, llm_dir: str) -> VllmConfig:
        """Create a VllmConfig for the internal LLM."""
        from vllm.config import ModelConfig
        from transformers import AutoConfig

        # Load LLM config from directory
        llm_hf_config = AutoConfig.from_pretrained(llm_dir)

        # Create new ModelConfig for the LLM
        llm_model_config = ModelConfig(
            model=llm_dir,
            tokenizer=llm_dir,
            tokenizer_mode=base_vllm_config.model_config.tokenizer_mode,
            trust_remote_code=base_vllm_config.model_config.trust_remote_code,
            dtype=base_vllm_config.model_config.dtype,
            seed=base_vllm_config.model_config.seed,
            revision=base_vllm_config.model_config.revision,
            code_revision=base_vllm_config.model_config.code_revision,
            rope_scaling=base_vllm_config.model_config.rope_scaling,
            rope_theta=base_vllm_config.model_config.rope_theta,
            tokenizer_revision=base_vllm_config.model_config.tokenizer_revision,
            max_model_len=base_vllm_config.model_config.max_model_len,
            spec_decoding_config=base_vllm_config.model_config.spec_decoding_config,
            quantization=base_vllm_config.model_config.quantization,
            quantization_param_path=base_vllm_config.model_config.quantization_param_path,
            enforce_eager=base_vllm_config.model_config.enforce_eager,
            max_seq_len_to_capture=base_vllm_config.model_config.max_seq_len_to_capture,
            max_logprobs=base_vllm_config.model_config.max_logprobs,
            disable_sliding_window=base_vllm_config.model_config.disable_sliding_window,
            skip_tokenizer_init=base_vllm_config.model_config.skip_tokenizer_init,
            served_model_name=base_vllm_config.model_config.served_model_name,
            # Set the HF config
            hf_config=llm_hf_config,
        )

        # Create a copy of base_vllm_config with the new model_config
        llm_vllm_config = deepcopy(base_vllm_config)
        llm_vllm_config.model_config = llm_model_config

        return llm_vllm_config
    
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        """
        Get placeholder text for FireRedASR audio inputs.
        """
        if modality == "audio":
            return "<speech>"
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def get_language_model(self) -> torch.nn.Module:
        """
        Returns the underlying language model used for text generation.
        """
        return self.language_model
    
    def _validate_and_reshape_mm_tensor(
        self,
        mm_input: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        name: str,
    ) -> torch.Tensor:
        """Validate and reshape multimodal tensor input."""
        if mm_input is None:
            return torch.empty(0, device=self.device)
        
        if isinstance(mm_input, list):
            if not mm_input:
                return torch.empty(0, device=self.device)
            mm_input = torch.stack(mm_input)
        
        return mm_input
    
    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[FireRedAsrInputs]:
        """Parse and validate audio inputs."""
        speech_features = kwargs.pop("speech_features", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        
        if speech_features is None:
            return None
        
        speech_features = self._validate_and_reshape_mm_tensor(
            speech_features, "speech_features"
        )
        speech_lengths = self._validate_and_reshape_mm_tensor(
            speech_lengths, "speech_lengths"
        )
        
        if speech_features.numel() == 0:
            return None
        
        return FireRedAsrInputs(
            speech_features=speech_features,
            speech_lengths=speech_lengths,
        )
    
    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> MultiModalEmbeddings:
        """
        Process audio inputs and generate embeddings.

        Returns:
            Tuple of tensors, one per audio item, each with shape (num_tokens, llm_dim)
        """
        audio_inputs = self._parse_and_validate_audio_input(**kwargs)
        if audio_inputs is None:
            return tuple()

        speech_features = audio_inputs["speech_features"]
        speech_lengths = audio_inputs["speech_lengths"]

        # Run encoder
        encoder_outputs, output_lengths, _ = self.speech_encoder(
            speech_features, speech_lengths
        )

        # Run projector
        projected_features, projected_lengths = self.projector(
            encoder_outputs, output_lengths
        )

        # Store projected lengths for later use in input embedding merging
        self._cached_projected_lengths = projected_lengths

        # Return as tuple of tensors, one per audio item
        batch_size = projected_features.size(0)
        audio_embeddings_list = []

        for i in range(batch_size):
            actual_len = int(projected_lengths[i].item())
            audio_embeddings_list.append(projected_features[i, :actual_len])

        return tuple(audio_embeddings_list)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        """
        Merge text and audio embeddings using vLLM's standard utilities.
        """
        # Get text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if multimodal_embeddings is None or not multimodal_embeddings:
            return inputs_embeds

        # Use vLLM's standard multimodal embedding merging
        # merge_multimodal_embeddings handles the merging by matching
        # placeholder token IDs in input_ids with the multimodal embeddings
        inputs_embeds = merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            multimodal_embeddings,
            self.config.speech_token_id
        )

        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass through the model."""

        if intermediate_tensors is not None:
            inputs_embeds = None

        # Process multimodal inputs if needed
        # In V1 the inputs_embeds should always be generated at model runner.
        if inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)

            inputs_embeds = self.get_input_embeddings(
                input_ids, multimodal_embeddings
            )
            input_ids = None  # Don't use input_ids when using embeddings

        # Forward through language model
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states, sampling_metadata)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample from logits."""
        return self.sampler(logits, sampling_metadata)
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights using AutoWeightsLoader for proper vLLM integration."""
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

