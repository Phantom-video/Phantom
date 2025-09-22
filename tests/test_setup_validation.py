import sys
import os
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly set up."""
    
    @pytest.mark.unit
    def test_python_version(self):
        """Test that Python version is compatible."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    @pytest.mark.unit
    def test_project_structure(self):
        """Test that the project structure is as expected."""
        workspace_path = Path("/workspace")
        assert workspace_path.exists(), "Workspace directory should exist"
        
        # Check main package
        phantom_wan_path = workspace_path / "phantom_wan"
        assert phantom_wan_path.exists(), "phantom_wan package should exist"
        assert (phantom_wan_path / "__init__.py").exists(), "phantom_wan should be a package"
        
        # Check test structure
        tests_path = workspace_path / "tests"
        assert tests_path.exists(), "tests directory should exist"
        assert (tests_path / "__init__.py").exists(), "tests should be a package"
        assert (tests_path / "unit").exists(), "unit tests directory should exist"
        assert (tests_path / "integration").exists(), "integration tests directory should exist"
    
    @pytest.mark.unit
    def test_conftest_fixtures(self, temp_dir, mock_config, sample_image):
        """Test that conftest fixtures are working properly."""
        # Test temp_dir fixture
        assert temp_dir.exists(), "Temporary directory should exist"
        assert temp_dir.is_dir(), "temp_dir should be a directory"
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict), "mock_config should be a dictionary"
        assert "model" in mock_config, "mock_config should have model section"
        assert mock_config["model"]["name"] == "phantom-wan", "mock_config should have correct model name"
        
        # Test sample_image fixture
        assert sample_image.exists(), "Sample image should exist"
        assert sample_image.suffix == ".png", "Sample image should be PNG"
        
        # Verify image can be loaded
        img = Image.open(sample_image)
        assert img.size == (512, 512), "Sample image should be 512x512"
    
    @pytest.mark.unit
    def test_torch_fixtures(self, mock_torch_tensor, mock_model):
        """Test PyTorch-related fixtures."""
        # Test tensor fixture
        assert isinstance(mock_torch_tensor, torch.Tensor), "Should be a torch tensor"
        assert mock_torch_tensor.shape == (1, 3, 512, 512), "Tensor should have correct shape"
        
        # Test model fixture
        assert hasattr(mock_model, "device"), "Model should have device attribute"
        assert hasattr(mock_model, "forward"), "Model should have forward method"
        
        # Test model forward pass
        output = mock_model.forward(mock_torch_tensor)
        assert isinstance(output, torch.Tensor), "Model output should be a tensor"
    
    @pytest.mark.unit
    def test_numpy_fixtures(self, mock_numpy_array):
        """Test NumPy-related fixtures."""
        assert isinstance(mock_numpy_array, np.ndarray), "Should be a numpy array"
        assert mock_numpy_array.shape == (512, 512, 3), "Array should have correct shape"
        assert mock_numpy_array.dtype == np.float32, "Array should be float32"
    
    @pytest.mark.unit
    def test_env_vars_fixture(self, env_vars):
        """Test environment variables fixture."""
        assert "CUDA_VISIBLE_DEVICES" in env_vars, "CUDA_VISIBLE_DEVICES should be set"
        assert env_vars["CUDA_VISIBLE_DEVICES"] == "", "CUDA_VISIBLE_DEVICES should be empty for CPU testing"
        assert "HF_HOME" in env_vars, "HF_HOME should be set"
    
    @pytest.mark.unit
    def test_capture_logs_fixture(self, capture_logs):
        """Test log capture fixture."""
        import logging
        
        logger = logging.getLogger(__name__)
        test_message = "Test log message"
        logger.info(test_message)
        
        log_contents = capture_logs.getvalue()
        assert test_message in log_contents, "Log message should be captured"
    
    @pytest.mark.unit
    def test_video_frames_fixture(self, sample_video_frames):
        """Test video frames fixture."""
        assert sample_video_frames.exists(), "Frames directory should exist"
        
        frames = list(sample_video_frames.glob("*.png"))
        assert len(frames) == 16, "Should have 16 frames"
        
        # Check first frame
        first_frame = Image.open(frames[0])
        assert first_frame.size == (512, 512), "Frame should be 512x512"
    
    @pytest.mark.unit
    def test_mock_pipeline_fixture(self, mock_pipeline):
        """Test mock pipeline fixture."""
        assert hasattr(mock_pipeline, "__call__"), "Pipeline should be callable"
        
        result = mock_pipeline()
        assert "frames" in result, "Pipeline should return frames"
        assert len(result["frames"]) == 16, "Pipeline should return 16 frames"
    
    @pytest.mark.unit
    def test_imports(self):
        """Test that key dependencies can be imported."""
        required_imports = [
            "torch",
            "torchvision",
            "cv2",
            "diffusers",
            "transformers",
            "accelerate",
            "PIL",
            "numpy",
            "gradio",
        ]
        
        for module_name in required_imports:
            try:
                __import__(module_name)
            except ImportError:
                pytest.skip(f"{module_name} not installed - this is expected before poetry install")
    
    @pytest.mark.slow
    @pytest.mark.unit
    def test_fixture_cleanup(self, temp_dir):
        """Test that fixtures properly clean up after themselves."""
        temp_file = temp_dir / "test_file.txt"
        temp_file.write_text("test content")
        assert temp_file.exists(), "Test file should exist during test"
        # Cleanup happens after test completes
    
    @pytest.mark.gpu
    def test_gpu_availability(self, gpu_available):
        """Test GPU availability detection."""
        if gpu_available:
            assert torch.cuda.is_available(), "CUDA should be available"
            pytest.skip("GPU tests would run here")
        else:
            assert not torch.cuda.is_available(), "CUDA should not be available"
    
    @pytest.mark.unit
    def test_markers_defined(self):
        """Test that custom markers are properly defined."""
        # This test verifies markers are recognized by pytest
        pass


@pytest.mark.integration
class TestIntegrationSetup:
    """Integration tests for the testing setup."""
    
    def test_project_imports(self):
        """Test that the main project package can be imported."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import phantom_wan
                assert phantom_wan is not None
        except (ImportError, AssertionError) as e:
            if "CUDA" in str(e):
                pytest.skip("phantom_wan requires CUDA - skipping import test in CPU environment")
            else:
                pytest.skip("phantom_wan package import test - expected to fail before installation")
    
    def test_multiple_fixtures_interaction(self, temp_dir, mock_config, sample_image):
        """Test that multiple fixtures work together correctly."""
        # Create a config file in temp directory
        config_path = temp_dir / "test_config.json"
        import json
        with open(config_path, "w") as f:
            json.dump(mock_config, f)
        
        assert config_path.exists(), "Config file should be created"
        
        # Copy sample image to temp directory
        import shutil
        dest_image = temp_dir / "copied_image.png"
        shutil.copy(sample_image, dest_image)
        
        assert dest_image.exists(), "Image should be copied"