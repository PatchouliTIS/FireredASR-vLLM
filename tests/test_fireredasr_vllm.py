"""
Unit tests for FireRedASR vLLM integration.

To run these tests:
    pytest tests/test_fireredasr_vllm.py -v
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

# Test imports
try:
    from vllm.model_executor.models.fireredasr_vllm import (
        FireRedAsrConfig,
        FireRedAsrInputs,
        FireRedAsrEncoder,
        FireRedAsrProjector,
        FireRedAsrForConditionalGeneration,
        FireRedAsrMultiModalProcessor,
        FireRedAsrProcessingInfo,
    )
    FIREREDASR_VLLM_AVAILABLE = True
except ImportError as e:
    FIREREDASR_VLLM_AVAILABLE = False
    import_error = str(e)


pytestmark = pytest.mark.skipif(
    not FIREREDASR_VLLM_AVAILABLE,
    reason=f"FireRedASR vLLM module not available: {import_error if not FIREREDASR_VLLM_AVAILABLE else ''}"
)


class TestFireRedAsrConfig:
    """Test FireRedAsrConfig class."""
    
    def test_config_initialization(self):
        """Test basic config initialization."""
        config = FireRedAsrConfig(
            encoder_path="/path/to/encoder.pth.tar",
            encoder_dim=512,
            llm_dir="/path/to/llm",
            cmvn_path="/path/to/cmvn.ark",
        )
        
        assert config.encoder_path == "/path/to/encoder.pth.tar"
        assert config.encoder_dim == 512
        assert config.freeze_encoder is True  # Default
        assert config.encoder_downsample_rate == 4  # Default
        assert config.llm_dir == "/path/to/llm"
        assert config.cmvn_path == "/path/to/cmvn.ark"
    
    def test_config_with_custom_values(self):
        """Test config with custom values."""
        config = FireRedAsrConfig(
            encoder_path="/path/to/encoder.pth.tar",
            encoder_dim=768,
            freeze_encoder=False,
            encoder_downsample_rate=8,
            llm_dir="/path/to/llm",
            default_speech_token="<|AUDIO|>",
            speech_token_id=12345,
        )
        
        assert config.encoder_dim == 768
        assert config.freeze_encoder is False
        assert config.encoder_downsample_rate == 8
        assert config.default_speech_token == "<|AUDIO|>"
        assert config.speech_token_id == 12345


class TestFireRedAsrInputs:
    """Test FireRedAsrInputs TypedDict."""
    
    def test_inputs_structure(self):
        """Test inputs data structure."""
        batch_size = 2
        time_steps = 100
        feat_dim = 80
        
        inputs: FireRedAsrInputs = {
            "speech_features": torch.randn(batch_size, time_steps, feat_dim),
            "speech_lengths": torch.tensor([100, 80]),
        }
        
        assert inputs["speech_features"].shape == (batch_size, time_steps, feat_dim)
        assert inputs["speech_lengths"].shape == (batch_size,)
        assert inputs["speech_lengths"][0].item() == 100
        assert inputs["speech_lengths"][1].item() == 80


@pytest.mark.skipif(
    not FIREREDASR_VLLM_AVAILABLE,
    reason="FireRedASR dependencies not available"
)
class TestFireRedAsrEncoder:
    """Test FireRedAsrEncoder class."""
    
    @patch('vllm.model_executor.models.fireredasr_vllm.torch.load')
    @patch('vllm.model_executor.models.fireredasr_vllm.FireRedAsrAed')
    def test_encoder_initialization(self, mock_aed_class, mock_torch_load):
        """Test encoder initialization."""
        # Mock the loaded checkpoint
        mock_package = {
            "args": MagicMock(),
            "model_state_dict": {},
        }
        mock_torch_load.return_value = mock_package
        
        # Mock the model
        mock_encoder = MagicMock()
        mock_encoder.odim = 512
        mock_model = MagicMock()
        mock_model.encoder = mock_encoder
        mock_aed_class.from_args.return_value = mock_model
        
        # Initialize encoder
        encoder = FireRedAsrEncoder(encoder_path="/path/to/encoder.pth.tar")
        
        assert encoder.encoder_dim == 512
        mock_torch_load.assert_called_once()
        mock_aed_class.from_args.assert_called_once()
    
    @patch('vllm.model_executor.models.fireredasr_vllm.torch.load')
    @patch('vllm.model_executor.models.fireredasr_vllm.FireRedAsrAed')
    def test_encoder_forward(self, mock_aed_class, mock_torch_load):
        """Test encoder forward pass."""
        # Setup mocks
        mock_package = {"args": MagicMock(), "model_state_dict": {}}
        mock_torch_load.return_value = mock_package
        
        mock_encoder = MagicMock()
        mock_encoder.odim = 512
        mock_encoder.return_value = (
            torch.randn(2, 50, 512),  # encoder outputs
            torch.tensor([50, 45]),    # lengths
            torch.ones(2, 1, 50),      # mask
        )
        mock_model = MagicMock()
        mock_model.encoder = mock_encoder
        mock_aed_class.from_args.return_value = mock_model
        
        encoder = FireRedAsrEncoder(encoder_path="/path/to/encoder.pth.tar")
        
        # Test forward
        speech_features = torch.randn(2, 100, 80)
        speech_lengths = torch.tensor([100, 90])
        
        outputs, out_lengths, mask = encoder(speech_features, speech_lengths)
        
        assert outputs.shape == (2, 50, 512)
        assert out_lengths.shape == (2,)
        assert mask.shape == (2, 1, 50)


@pytest.mark.skipif(
    not FIREREDASR_VLLM_AVAILABLE,
    reason="FireRedASR dependencies not available"
)
class TestFireRedAsrProjector:
    """Test FireRedAsrProjector class."""
    
    @patch('vllm.model_executor.models.fireredasr_vllm.Adapter')
    def test_projector_initialization(self, mock_adapter_class):
        """Test projector initialization."""
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        
        projector = FireRedAsrProjector(
            encoder_dim=512,
            llm_dim=2048,
            downsample_rate=4,
        )
        
        mock_adapter_class.assert_called_once_with(512, 2048, 4)
    
    @patch('vllm.model_executor.models.fireredasr_vllm.Adapter')
    def test_projector_forward(self, mock_adapter_class):
        """Test projector forward pass."""
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter.return_value = (
            torch.randn(2, 12, 2048),  # projected features (downsampled)
            torch.tensor([12, 11]),     # projected lengths
        )
        mock_adapter_class.return_value = mock_adapter
        
        projector = FireRedAsrProjector(512, 2048, 4)
        
        # Test forward
        encoder_outputs = torch.randn(2, 50, 512)
        output_lengths = torch.tensor([50, 45])
        
        proj_features, proj_lengths = projector(encoder_outputs, output_lengths)
        
        assert proj_features.shape == (2, 12, 2048)
        assert proj_lengths.shape == (2,)
        mock_adapter.assert_called_once()


class TestFireRedAsrProcessingInfo:
    """Test FireRedAsrProcessingInfo class."""
    
    @patch.object(FireRedAsrProcessingInfo, 'get_hf_config')
    def test_supported_mm_limits(self, mock_get_config):
        """Test supported multimodal limits."""
        mock_ctx = MagicMock()
        info = FireRedAsrProcessingInfo(ctx=mock_ctx)
        
        limits = info.get_supported_mm_limits()
        
        assert "audio" in limits
        assert limits["audio"] is None  # No limit
    
    @patch.object(FireRedAsrProcessingInfo, 'get_hf_config')
    def test_mm_max_tokens_per_item(self, mock_get_config):
        """Test max tokens per audio item."""
        mock_config = FireRedAsrConfig(
            encoder_downsample_rate=4,
        )
        mock_get_config.return_value = mock_config
        
        mock_ctx = MagicMock()
        info = FireRedAsrProcessingInfo(ctx=mock_ctx)
        
        seq_len = 400
        max_tokens = info.get_mm_max_tokens_per_item(seq_len)
        
        assert "audio" in max_tokens
        assert max_tokens["audio"] == seq_len // 4  # 100


class TestFireRedAsrMultiModalProcessor:
    """Test FireRedAsrMultiModalProcessor class."""
    
    def test_get_mm_fields_config(self):
        """Test multimodal fields configuration."""
        mock_info = MagicMock(spec=FireRedAsrProcessingInfo)
        processor = FireRedAsrMultiModalProcessor(info=mock_info)
        
        fields_config = processor._get_mm_fields_config(
            hf_inputs={},
            hf_processor_mm_kwargs={},
        )
        
        assert "speech_features" in fields_config
        assert "speech_lengths" in fields_config
    
    @patch('vllm.model_executor.models.fireredasr_vllm.ASRFeatExtractor')
    def test_call_hf_processor_with_audio(self, mock_feat_extractor_class):
        """Test audio processing."""
        # Mock feature extractor
        mock_extractor = MagicMock()
        mock_extractor.return_value = (
            torch.randn(1, 100, 80),  # features
            torch.tensor([100]),       # lengths
            [5.0],                     # durations
        )
        mock_feat_extractor_class.return_value = mock_extractor
        
        # Mock info
        mock_config = FireRedAsrConfig(cmvn_path="/path/to/cmvn.ark")
        mock_info = MagicMock(spec=FireRedAsrProcessingInfo)
        mock_info.get_hf_config.return_value = mock_config
        
        processor = FireRedAsrMultiModalProcessor(info=mock_info)
        
        # Process audio
        result = processor._call_hf_processor(
            prompt="<speech>",
            mm_data={"audio": ["/path/to/audio.wav"]},
            mm_kwargs={"cmvn_path": "/path/to/cmvn.ark"},
        )
        
        assert "speech_features" in result
        assert "speech_lengths" in result
        assert result["speech_features"].shape[0] == 1  # batch size
        mock_feat_extractor_class.assert_called_once()
    
    @patch('vllm.model_executor.models.fireredasr_vllm.ASRFeatExtractor')
    def test_call_hf_processor_without_audio(self, mock_feat_extractor_class):
        """Test processing without audio."""
        mock_config = FireRedAsrConfig(
            cmvn_path="/path/to/cmvn.ark",
            encoder_dim=512,
        )
        mock_info = MagicMock(spec=FireRedAsrProcessingInfo)
        mock_info.get_hf_config.return_value = mock_config
        
        processor = FireRedAsrMultiModalProcessor(info=mock_info)
        
        # Process without audio
        result = processor._call_hf_processor(
            prompt="",
            mm_data={},
            mm_kwargs={"cmvn_path": "/path/to/cmvn.ark"},
        )
        
        assert "speech_features" in result
        assert "speech_lengths" in result
        assert result["speech_features"].numel() == 0  # Empty tensor
        mock_feat_extractor_class.assert_not_called()


@pytest.mark.skipif(
    not FIREREDASR_VLLM_AVAILABLE,
    reason="Full model tests require all dependencies"
)
class TestFireRedAsrForConditionalGeneration:
    """Test main model class."""
    
    def test_validate_and_reshape_mm_tensor_list(self):
        """Test tensor validation with list input."""
        mock_config = FireRedAsrConfig()
        mock_multimodal_config = MagicMock()
        
        # We need to mock all the initialization
        with patch.object(
            FireRedAsrForConditionalGeneration,
            '__init__',
            lambda self, *args, **kwargs: None
        ):
            model = FireRedAsrForConditionalGeneration(
                mock_config, mock_multimodal_config
            )
            
            # Manually set device property
            model._device = torch.device("cpu")
            
            # Test with list of tensors
            tensor_list = [torch.randn(10, 80), torch.randn(12, 80)]
            result = model._validate_and_reshape_mm_tensor(tensor_list, "test")
            
            assert result.shape[0] == 2
            assert result.shape[2] == 80
    
    def test_validate_and_reshape_mm_tensor_single(self):
        """Test tensor validation with single tensor."""
        mock_config = FireRedAsrConfig()
        mock_multimodal_config = MagicMock()
        
        with patch.object(
            FireRedAsrForConditionalGeneration,
            '__init__',
            lambda self, *args, **kwargs: None
        ):
            model = FireRedAsrForConditionalGeneration(
                mock_config, mock_multimodal_config
            )
            model._device = torch.device("cpu")
            
            # Test with single tensor
            tensor = torch.randn(2, 10, 80)
            result = model._validate_and_reshape_mm_tensor(tensor, "test")
            
            assert result.shape == tensor.shape
    
    def test_parse_and_validate_audio_input(self):
        """Test audio input parsing."""
        mock_config = FireRedAsrConfig()
        mock_multimodal_config = MagicMock()
        
        with patch.object(
            FireRedAsrForConditionalGeneration,
            '__init__',
            lambda self, *args, **kwargs: None
        ):
            model = FireRedAsrForConditionalGeneration(
                mock_config, mock_multimodal_config
            )
            model._device = torch.device("cpu")
            
            # Test with valid inputs
            kwargs = {
                "speech_features": torch.randn(2, 100, 80),
                "speech_lengths": torch.tensor([100, 90]),
            }
            
            result = model._parse_and_validate_audio_input(**kwargs)
            
            assert result is not None
            assert "speech_features" in result
            assert "speech_lengths" in result
            assert result["speech_features"].shape == (2, 100, 80)


def test_module_imports():
    """Test that all required components can be imported."""
    assert FIREREDASR_VLLM_AVAILABLE, "FireRedASR vLLM module should be importable"
    
    # Check key classes exist
    assert FireRedAsrConfig is not None
    assert FireRedAsrInputs is not None
    assert FireRedAsrEncoder is not None
    assert FireRedAsrProjector is not None
    assert FireRedAsrForConditionalGeneration is not None
    assert FireRedAsrMultiModalProcessor is not None
    assert FireRedAsrProcessingInfo is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

