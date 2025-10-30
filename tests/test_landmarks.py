import os
import tempfile
import numpy as np
import pytest

from src.landmarks import LandmarkGenerator, ValidationError, ProcessingError


class TestLandmarkGenerator:
    """Test suite for LandmarkGenerator class."""

    def test_valid_initialization(self) -> None:
        """Test valid generator initialization."""
        generator = LandmarkGenerator(
            bounds_x=(-5.0, 5.0),
            bounds_y=(-3.0, 3.0),
            bounds_z=(0.0, 2.0)
        )
        assert generator.bounds_x == (-5.0, 5.0)
        assert generator.bounds_y == (-3.0, 3.0)
        assert generator.bounds_z == (0.0, 2.0)

    @pytest.mark.parametrize(
        "invalid_bounds",
        [
            "not_tuple",  # Wrong type
            (1.0,),  # Wrong length
            (1.0, 2.0, 3.0),  # Wrong length
            (2.0, 1.0),  # Min >= max
            (float("inf"), 1.0),  # Non-finite values
            (1.0, float("nan")),  # Non-finite values
        ],
    )
    def test_invalid_bounds_initialization(self, invalid_bounds: tuple) -> None:
        """Test invalid bounds raise ValidationError."""
        with pytest.raises(ValidationError):
            LandmarkGenerator(bounds_x=invalid_bounds)

        with pytest.raises(ValidationError):
            LandmarkGenerator(bounds_y=invalid_bounds)

        with pytest.raises(ValidationError):
            LandmarkGenerator(bounds_z=invalid_bounds)

    def test_generate_random_valid(self) -> None:
        """Test valid random landmark generation."""
        generator = LandmarkGenerator(
            bounds_x=(-2.0, 2.0),
            bounds_y=(-1.0, 1.0),
            bounds_z=(0.0, 3.0)
        )
        landmarks = generator.generate_random(100, seed=42)

        assert landmarks.shape == (100, 3)
        assert np.all(np.isfinite(landmarks))
        assert np.all(landmarks[:, 0] >= -2.0) and np.all(landmarks[:, 0] <= 2.0)
        assert np.all(landmarks[:, 1] >= -1.0) and np.all(landmarks[:, 1] <= 1.0)
        assert np.all(landmarks[:, 2] >= 0.0) and np.all(landmarks[:, 2] <= 3.0)

    @pytest.mark.parametrize(
        "invalid_num",
        [-1, 0, 1.5, "not_int", 1_000_001]  # Negative, zero, float, string, too large
    )
    def test_generate_random_invalid_num_landmarks(self, invalid_num: int) -> None:
        """Test invalid num_landmarks raises ValidationError."""
        generator = LandmarkGenerator()
        with pytest.raises(ValidationError):
            generator.generate_random(invalid_num)

    def test_generate_random_reproducibility(self) -> None:
        """Test random generation is reproducible with seed."""
        generator = LandmarkGenerator()

        landmarks1 = generator.generate_random(10, seed=123)
        landmarks2 = generator.generate_random(10, seed=123)

        np.testing.assert_array_equal(landmarks1, landmarks2)

    def test_generate_on_ground_plane_valid(self) -> None:
        """Test valid ground plane landmark generation."""
        generator = LandmarkGenerator(
            bounds_x=(-2.0, 2.0),
            bounds_y=(-1.0, 1.0),
            bounds_z=(0.0, 3.0)
        )
        landmarks = generator.generate_on_ground_plane(50, z_height=1.5, seed=42)

        assert landmarks.shape == (50, 3)
        assert np.all(np.isfinite(landmarks))
        assert np.all(landmarks[:, 0] >= -2.0) and np.all(landmarks[:, 0] <= 2.0)
        assert np.all(landmarks[:, 1] >= -1.0) and np.all(landmarks[:, 1] <= 1.0)
        assert np.all(landmarks[:, 2] == 1.5)  # All Z coordinates should be z_height

    @pytest.mark.parametrize(
        "invalid_z_height",
        [float("inf"), float("-inf"), float("nan"), "not_number"]
    )
    def test_generate_on_ground_plane_invalid_z_height(self, invalid_z_height: float) -> None:
        """Test invalid z_height raises ValidationError."""
        generator = LandmarkGenerator()
        with pytest.raises(ValidationError):
            generator.generate_on_ground_plane(10, z_height=invalid_z_height)

    def test_generate_on_ground_plane_z_out_of_bounds(self) -> None:
        """Test z_height out of bounds raises ValidationError."""
        generator = LandmarkGenerator(bounds_z=(0.0, 2.0))
        with pytest.raises(ValidationError):
            generator.generate_on_ground_plane(10, z_height=3.0)  # Above max

        with pytest.raises(ValidationError):
            generator.generate_on_ground_plane(10, z_height=-1.0)  # Below min

    def test_generate_spherical_distribution_valid(self) -> None:
        """Test valid spherical distribution landmark generation."""
        generator = LandmarkGenerator(
            bounds_x=(-5.0, 5.0),
            bounds_y=(-5.0, 5.0),
            bounds_z=(-5.0, 5.0)
        )
        landmarks = generator.generate_spherical_distribution(
            100, center=(0.0, 0.0, 0.0), radius_range=(1.0, 3.0), seed=42
        )

        assert landmarks.shape == (100, 3)
        assert np.all(np.isfinite(landmarks))

        # Check that points are within bounds
        assert np.all(landmarks[:, 0] >= -5.0) and np.all(landmarks[:, 0] <= 5.0)
        assert np.all(landmarks[:, 1] >= -5.0) and np.all(landmarks[:, 1] <= 5.0)
        assert np.all(landmarks[:, 2] >= -5.0) and np.all(landmarks[:, 2] <= 5.0)

        # Check that points are approximately within radius range from center
        distances = np.linalg.norm(landmarks, axis=1)
        assert np.all(distances >= 0.8) and np.all(distances <= 3.2)  # Allow some tolerance

    @pytest.mark.parametrize(
        "invalid_center",
        [
            "not_tuple",  # Wrong type
            (1.0, 2.0),  # Wrong length
            (1.0, 2.0, float("inf")),  # Non-finite
        ],
    )
    def test_generate_spherical_distribution_invalid_center(self, invalid_center: tuple) -> None:
        """Test invalid center raises ValidationError."""
        generator = LandmarkGenerator()
        with pytest.raises(ValidationError):
            generator.generate_spherical_distribution(10, center=invalid_center)

    @pytest.mark.parametrize(
        "invalid_radius_range",
        [
            "not_tuple",  # Wrong type
            (1.0,),  # Wrong length
            (-1.0, 1.0),  # Negative min
            (2.0, 1.0),  # Min > max
            (0.0, 1.0),  # Zero min
        ],
    )
    def test_generate_spherical_distribution_invalid_radius_range(self, invalid_radius_range: tuple) -> None:
        """Test invalid radius_range raises ValidationError."""
        generator = LandmarkGenerator()
        with pytest.raises(ValidationError):
            generator.generate_spherical_distribution(10, radius_range=invalid_radius_range)

    def test_generate_spherical_distribution_clipping_warning(self) -> None:
        """Test that too much clipping raises ProcessingError."""
        # Create generator with tight bounds
        generator = LandmarkGenerator(
            bounds_x=(-0.1, 0.1),
            bounds_y=(-0.1, 0.1),
            bounds_z=(-0.1, 0.1)
        )

        # Try to generate points far from center - should cause clipping
        with pytest.raises(ProcessingError, match="Too many landmarks.*fell outside bounds"):
            generator.generate_spherical_distribution(
                10, center=(0.0, 0.0, 0.0), radius_range=(1.0, 2.0)
            )

    def test_generate_grid_valid(self) -> None:
        """Test valid grid landmark generation."""
        generator = LandmarkGenerator(
            bounds_x=(-2.0, 2.0),
            bounds_y=(-1.0, 1.0),
            bounds_z=(0.0, 3.0)
        )
        landmarks = generator.generate_grid(
            grid_size=(3, 2, 2),
            spacing=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0)
        )

        # Should have 3 * 2 * 2 = 12 landmarks
        assert landmarks.shape == (12, 3)
        assert np.all(np.isfinite(landmarks))

        # Check bounds
        assert np.all(landmarks[:, 0] >= -2.0) and np.all(landmarks[:, 0] <= 2.0)
        assert np.all(landmarks[:, 1] >= -1.0) and np.all(landmarks[:, 1] <= 1.0)
        assert np.all(landmarks[:, 2] >= 0.0) and np.all(landmarks[:, 2] <= 3.0)

        # Check grid structure (should have specific X coordinates)
        expected_x = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(np.sort(landmarks[:, 0]), expected_x)

    @pytest.mark.parametrize(
        "invalid_grid_size",
        [
            "not_tuple",  # Wrong type
            (3, 2),  # Wrong length
            (-1, 2, 3),  # Negative
            (0, 2, 3),  # Zero
            (1.5, 2, 3),  # Float
        ],
    )
    def test_generate_grid_invalid_grid_size(self, invalid_grid_size: tuple) -> None:
        """Test invalid grid_size raises ValidationError."""
        generator = LandmarkGenerator()
        with pytest.raises(ValidationError):
            generator.generate_grid(invalid_grid_size)

    @pytest.mark.parametrize(
        "invalid_spacing",
        [
            "not_tuple",  # Wrong type
            (1.0, 2.0),  # Wrong length
            (-1.0, 2.0, 3.0),  # Negative
            (0.0, 2.0, 3.0),  # Zero
        ],
    )
    def test_generate_grid_invalid_spacing(self, invalid_spacing: tuple) -> None:
        """Test invalid spacing raises ValidationError."""
        generator = LandmarkGenerator()
        with pytest.raises(ValidationError):
            generator.generate_grid((2, 2, 2), spacing=invalid_spacing)

    @pytest.mark.parametrize(
        "invalid_offset",
        [
            "not_tuple",  # Wrong type
            (1.0, 2.0),  # Wrong length
            (1.0, float("inf"), 3.0),  # Non-finite
        ],
    )
    def test_generate_grid_invalid_offset(self, invalid_offset: tuple) -> None:
        """Test invalid offset raises ValidationError."""
        generator = LandmarkGenerator()
        with pytest.raises(ValidationError):
            generator.generate_grid((2, 2, 2), offset=invalid_offset)

    def test_generate_grid_out_of_bounds(self) -> None:
        """Test grid generation out of bounds raises ValidationError."""
        generator = LandmarkGenerator(
            bounds_x=(-1.0, 1.0),
            bounds_y=(-1.0, 1.0),
            bounds_z=(-1.0, 1.0)
        )
        with pytest.raises(ValidationError, match="Generated grid points fall outside"):
            generator.generate_grid(
                grid_size=(3, 3, 3),
                spacing=(2.0, 2.0, 2.0),  # Will go beyond bounds
                offset=(0.0, 0.0, 0.0)
            )

    def test_generate_grid_too_large(self) -> None:
        """Test grid too large raises ValidationError."""
        # Create generator with bounds large enough to fit the grid
        generator = LandmarkGenerator(
            bounds_x=(-100.0, 200.0),
            bounds_y=(-100.0, 200.0),
            bounds_z=(-100.0, 200.0)
        )
        with pytest.raises(ValidationError, match="Grid too large"):
            generator.generate_grid((101, 101, 101))  # 1,061,601 points

    def test_save_landmarks_valid(self) -> None:
        """Test valid landmark saving to CSV."""
        generator = LandmarkGenerator()
        landmarks = generator.generate_random(10, seed=42)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            generator.save_landmarks(landmarks, filepath)

            # Verify file was created and has correct content
            assert os.path.exists(filepath)

            # Read back and verify
            loaded = np.loadtxt(filepath, delimiter=',', skiprows=1)  # Skip header
            np.testing.assert_array_almost_equal(loaded, landmarks, decimal=6)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_landmarks_invalid_array(self) -> None:
        """Test saving with invalid landmarks array raises ValidationError."""
        generator = LandmarkGenerator()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            # Test with non-numpy array
            with pytest.raises(ValidationError, match="landmarks must be a numpy array"):
                generator.save_landmarks([[1.0, 2.0, 3.0]], filepath)

            # Test with wrong shape
            with pytest.raises(ValidationError, match="landmarks must have shape"):
                generator.save_landmarks(np.array([[1.0, 2.0]]), filepath)

            # Test with empty array
            with pytest.raises(ValidationError, match="landmarks cannot be empty"):
                generator.save_landmarks(np.empty((0, 3)), filepath)

            # Test with non-finite values
            with pytest.raises(ValidationError, match="landmarks must contain only finite values"):
                generator.save_landmarks(np.array([[float('inf'), 2.0, 3.0]]), filepath)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_landmarks_invalid_filepath(self) -> None:
        """Test saving with invalid filepath raises ValidationError."""
        generator = LandmarkGenerator()
        landmarks = generator.generate_random(5, seed=42)

        # Test non-string filepath
        with pytest.raises(ValidationError, match="filepath must be a string"):
            generator.save_landmarks(landmarks, 123)

        # Test filepath without .csv extension
        with pytest.raises(ValidationError, match="filepath must end with .csv"):
            generator.save_landmarks(landmarks, "/tmp/test.txt")

        # Test directory that doesn't exist
        with pytest.raises(ValidationError, match="Directory does not exist"):
            generator.save_landmarks(landmarks, "/nonexistent/directory/test.csv")

    def test_load_landmarks_valid(self) -> None:
        """Test valid landmark loading from CSV."""
        generator = LandmarkGenerator()
        original_landmarks = generator.generate_random(10, seed=42)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            # Save landmarks first
            generator.save_landmarks(original_landmarks, filepath)

            # Load them back
            loaded_landmarks = LandmarkGenerator.load_landmarks(filepath)

            # Verify they match
            np.testing.assert_array_almost_equal(loaded_landmarks, original_landmarks, decimal=6)
            assert loaded_landmarks.shape == original_landmarks.shape

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_landmarks_invalid_filepath(self) -> None:
        """Test loading with invalid filepath raises ValidationError."""
        # Test non-string filepath
        with pytest.raises(ValidationError, match="filepath must be a string"):
            LandmarkGenerator.load_landmarks(123)

        # Test filepath without .csv extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            txt_filepath = f.name
            f.write("1.0,2.0,3.0\n")
            f.flush()

        try:
            with pytest.raises(ValidationError, match="filepath must end with .csv"):
                LandmarkGenerator.load_landmarks(txt_filepath)
        finally:
            if os.path.exists(txt_filepath):
                os.unlink(txt_filepath)

        # Test non-existent file
        with pytest.raises(ValidationError, match="File does not exist"):
            LandmarkGenerator.load_landmarks("/nonexistent/file.csv")

        # Test path that is a directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError, match="Path is not a file"):
                LandmarkGenerator.load_landmarks(temp_dir)

    def test_load_landmarks_invalid_format(self) -> None:
        """Test loading invalid CSV format raises ProcessingError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
            # Write invalid CSV content
            f.write("invalid,csv,content\n")
            f.write("1.0,2.0\n")  # Wrong number of columns
            f.flush()

        try:
            with pytest.raises(ProcessingError, match="Invalid landmark data shape"):
                LandmarkGenerator.load_landmarks(filepath)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_landmarks_single_landmark(self) -> None:
        """Test loading a single landmark from CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
            # Write a single landmark
            f.write("x,y,z\n")
            f.write("1.5,2.5,3.5\n")
            f.flush()

        try:
            loaded = LandmarkGenerator.load_landmarks(filepath)
            expected = np.array([[1.5, 2.5, 3.5]])

            np.testing.assert_array_almost_equal(loaded, expected, decimal=6)
            assert loaded.shape == (1, 3)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_load_roundtrip(self) -> None:
        """Test complete save-load roundtrip preserves data."""
        generator = LandmarkGenerator()
        original_landmarks = generator.generate_random(50, seed=123)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            # Save and load
            generator.save_landmarks(original_landmarks, filepath)
            loaded_landmarks = LandmarkGenerator.load_landmarks(filepath)

            # Verify exact match
            np.testing.assert_array_almost_equal(
                loaded_landmarks, original_landmarks, decimal=6
            )
            assert loaded_landmarks.shape == original_landmarks.shape
            assert loaded_landmarks.dtype == original_landmarks.dtype

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
