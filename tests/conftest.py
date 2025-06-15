import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration dictionary for testing."""
    return {
        "model": {
            "name": "phantom-wan",
            "version": "2.1",
            "num_frames": 16,
            "frame_rate": 8,
            "resolution": [512, 512],
        },
        "inference": {
            "batch_size": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "seed": 42,
        },
        "paths": {
            "model_dir": "/tmp/models",
            "output_dir": "/tmp/outputs",
            "cache_dir": "/tmp/cache",
        },
    }


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image for testing."""
    img_path = temp_dir / "sample_image.png"
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_video_frames(temp_dir: Path) -> Path:
    """Create sample video frames for testing."""
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    for i in range(16):
        img = Image.new("RGB", (512, 512), color=(i * 16, i * 16, i * 16))
        img.save(frames_dir / f"frame_{i:04d}.png")
    
    return frames_dir


@pytest.fixture
def mock_torch_tensor() -> torch.Tensor:
    """Create a mock torch tensor for testing."""
    return torch.randn(1, 3, 512, 512)


@pytest.fixture
def mock_numpy_array() -> np.ndarray:
    """Create a mock numpy array for testing."""
    return np.random.rand(512, 512, 3).astype(np.float32)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.forward = MagicMock(return_value=torch.randn(1, 3, 16, 512, 512))
    return model


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    # Configure the mock to return the expected result when called
    pipeline.return_value = {"frames": [np.zeros((512, 512, 3)) for _ in range(16)]}
    return pipeline


@pytest.fixture
def env_vars():
    """Set up and clean up environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HOME"] = "/tmp/test_hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/test_transformers_cache"
    
    yield os.environ
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def capture_logs():
    """Capture log messages during tests."""
    import logging
    from io import StringIO
    
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    
    # Get root logger
    logger = logging.getLogger()
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture_string
    
    # Clean up
    logger.removeHandler(ch)


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset torch random seed before each test."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)


@pytest.fixture
def mock_gradio_interface():
    """Create a mock Gradio interface for testing."""
    interface = MagicMock()
    interface.launch = MagicMock(return_value=(None, "http://127.0.0.1:7860", None))
    interface.close = MagicMock()
    return interface


@pytest.fixture
def mock_transformers_model():
    """Create a mock transformers model for testing."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 768
    model.config.num_hidden_layers = 12
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.encode = MagicMock(return_value=[101, 102, 103])
    tokenizer.decode = MagicMock(return_value="test text")
    tokenizer.__call__ = MagicMock(
        return_value={
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )
    return tokenizer


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "network: Tests requiring network access")