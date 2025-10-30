import numpy as np
import pytest

from src.camera import PinHoleCamera, ProcessingError, ValidationError


class TestPinHoleCamera:
    """Test suite for PinHoleCamera class."""

    def test_valid_initialization(self) -> None:
        """Test valid camera initialization."""
        camera = PinHoleCamera(fx=500.0, fy=600.0, cx=320.0, cy=240.0)
        assert camera.fx == 500.0
        assert camera.fy == 600.0
        assert camera.cx == 320.0
        assert camera.cy == 240.0

    @pytest.mark.parametrize(
        "invalid_fx", [-500.0, 0.0, float("inf"), float("-inf"), float("nan")]
    )
    def test_invalid_focal_length_fx(self, invalid_fx: float) -> None:
        """Test invalid focal length fx raises ValidationError."""
        with pytest.raises(
            ValidationError, match=r"Focal length fx must be a positive finite number.*"
        ):
            PinHoleCamera(fx=invalid_fx, fy=500.0, cx=320.0, cy=240.0)

    @pytest.mark.parametrize(
        "invalid_fy", [-500.0, 0.0, float("inf"), float("-inf"), float("nan")]
    )
    def test_invalid_focal_length_fy(self, invalid_fy: float) -> None:
        """Test invalid focal length fy raises ValidationError."""
        with pytest.raises(
            ValidationError, match=r"Focal length fy must be a positive finite number.*"
        ):
            PinHoleCamera(fx=500.0, fy=invalid_fy, cx=320.0, cy=240.0)

    @pytest.mark.parametrize("invalid_cx", [float("inf"), float("-inf"), float("nan")])
    def test_invalid_principal_point_cx(self, invalid_cx: float) -> None:
        """Test invalid principal point cx raises ValidationError."""
        with pytest.raises(
            ValidationError, match=r"Principal point cx must be a finite number.*"
        ):
            PinHoleCamera(fx=500.0, fy=500.0, cx=invalid_cx, cy=240.0)

    @pytest.mark.parametrize("invalid_cy", [float("inf"), float("-inf"), float("nan")])
    def test_invalid_principal_point_cy(self, invalid_cy: float) -> None:
        """Test invalid principal point cy raises ValidationError."""
        with pytest.raises(
            ValidationError, match=r"Principal point cy must be a finite number.*"
        ):
            PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=invalid_cy)

    def test_valid_projection(self) -> None:
        """Test valid 3D point projection."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = camera.project(points_3d)

        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))
        # Check projection math: u = (fx * x / z) + cx, v = (fy * y / z) + cy
        expected_u1 = (500.0 * 1.0 / 3.0) + 320.0
        expected_v1 = (500.0 * 2.0 / 3.0) + 240.0
        assert abs(result[0, 0] - expected_u1) < 1e-10
        assert abs(result[0, 1] - expected_v1) < 1e-10

    def test_projection_not_numpy_array(self) -> None:
        """Test projection with non-numpy array raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        with pytest.raises(ValidationError, match="points_3d must be a numpy array"):
            camera.project([[1.0, 2.0, 3.0]])  # type: ignore[arg-type] # Python list instead of numpy array

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (3,),  # 1D array
            (2, 4),  # Wrong number of columns
            (2, 3, 1),  # 3D array
        ],
    )
    def test_projection_invalid_shape(self, bad_shape: tuple) -> None:
        """Test projection with invalid array shapes raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_3d = np.ones(bad_shape)
        with pytest.raises(ValidationError, match="points_3d must have shape"):
            camera.project(points_3d)

    def test_projection_empty_array(self) -> None:
        """Test projection with empty array raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_3d = np.empty((0, 3))
        with pytest.raises(ValidationError, match="points_3d cannot be empty"):
            camera.project(points_3d)

    @pytest.mark.parametrize(
        "invalid_values",
        [
            np.array([[float("inf"), 2.0, 3.0]]),
            np.array([[1.0, float("-inf"), 3.0]]),
            np.array([[1.0, 2.0, float("nan")]]),
        ],
    )
    def test_projection_non_finite_values(self, invalid_values: np.ndarray) -> None:
        """Test projection with non-finite values raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        with pytest.raises(
            ValidationError, match="points_3d must contain only finite values"
        ):
            camera.project(invalid_values)

    def test_projection_zero_z_coordinate(self) -> None:
        """Test projection with zero Z coordinate raises ProcessingError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_3d = np.array([[1.0, 2.0, 0.0]])
        with pytest.raises(
            ProcessingError, match="Cannot project points with zero Z coordinates"
        ):
            camera.project(points_3d)

    def test_valid_unprojection_scalar_depth(self) -> None:
        """Test valid 2D point unprojection with scalar depth."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0], [330.0, 250.0]])
        depths = 2.0
        result = camera.unproject(points_2d, depths)

        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))
        # Check unprojection math: x = ((u - cx) * depth) / fx, y = ((v - cy) * depth) / fy
        expected_x1 = ((320.0 - 320.0) * 2.0) / 500.0
        expected_y1 = ((240.0 - 240.0) * 2.0) / 500.0
        assert abs(result[0, 0] - expected_x1) < 1e-10
        assert abs(result[0, 1] - expected_y1) < 1e-10
        assert abs(result[0, 2] - 2.0) < 1e-10

    def test_valid_unprojection_array_depths(self) -> None:
        """Test valid 2D point unprojection with array depths."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0], [330.0, 250.0]])
        depths = np.array([2.0, 3.0])
        result = camera.unproject(points_2d, depths)

        assert result.shape == (2, 3)
        assert abs(result[0, 2] - 2.0) < 1e-10
        assert abs(result[1, 2] - 3.0) < 1e-10

    def test_valid_unprojection_2d_depths(self) -> None:
        """Test valid 2D point unprojection with 2D column vector depths."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0], [330.0, 250.0]])
        depths = np.array([[2.0], [3.0]])
        result = camera.unproject(points_2d, depths)

        assert result.shape == (2, 3)
        assert abs(result[0, 2] - 2.0) < 1e-10
        assert abs(result[1, 2] - 3.0) < 1e-10

    def test_unprojection_points_2d_not_numpy_array(self) -> None:
        """Test unprojection with points_2d as non-numpy array raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        with pytest.raises(ValidationError, match="points_2d must be a numpy array"):
            camera.unproject([[320.0, 240.0]], 2.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (2,),  # 1D array
            (2, 3),  # Wrong number of columns
            (2, 2, 1),  # 3D array
        ],
    )
    def test_unprojection_invalid_points_2d_shape(self, bad_shape: tuple) -> None:
        """Test unprojection with invalid points_2d shapes raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.ones(bad_shape)
        with pytest.raises(ValidationError, match="points_2d must have shape"):
            camera.unproject(points_2d, 2.0)

    def test_unprojection_empty_points_2d(self) -> None:
        """Test unprojection with empty points_2d raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.empty((0, 2))
        with pytest.raises(ValidationError, match="points_2d cannot be empty"):
            camera.unproject(points_2d, 2.0)

    @pytest.mark.parametrize(
        "invalid_values",
        [
            np.array([[float("inf"), 240.0]]),
            np.array([[320.0, float("-inf")]]),
            np.array([[float("nan"), 240.0]]),
        ],
    )
    def test_unprojection_non_finite_points_2d(
        self, invalid_values: np.ndarray
    ) -> None:
        """Test unprojection with non-finite points_2d values raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        with pytest.raises(
            ValidationError, match="points_2d must contain only finite values"
        ):
            camera.unproject(invalid_values, 2.0)

    @pytest.mark.parametrize(
        "invalid_depths", [-1.0, 0.0, float("inf"), float("-inf"), float("nan")]
    )
    def test_unprojection_invalid_scalar_depths(self, invalid_depths: float) -> None:
        """Test unprojection with invalid scalar depths raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0]])
        if not np.isfinite(invalid_depths):
            with pytest.raises(ValidationError, match="depths must be a finite number"):
                camera.unproject(points_2d, invalid_depths)
        else:
            with pytest.raises(ValidationError, match="depths must be positive"):
                camera.unproject(points_2d, invalid_depths)

    def test_unprojection_array_depths_wrong_shape(self) -> None:
        """Test unprojection with array depths of wrong shape raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0], [330.0, 250.0]])  # 2 points
        depths = np.array([2.0, 3.0, 4.0])  # 3 depths - wrong shape
        with pytest.raises(ValidationError, match="depths array must have shape"):
            camera.unproject(points_2d, depths)

    def test_unprojection_array_depths_non_finite(self) -> None:
        """Test unprojection with non-finite array depths raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0]])
        depths = np.array([float("inf")])
        with pytest.raises(
            ValidationError, match="depths must contain only finite values"
        ):
            camera.unproject(points_2d, depths)

    def test_unprojection_array_depths_negative(self) -> None:
        """Test unprojection with negative array depths raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0]])
        depths = np.array([-1.0])
        with pytest.raises(ValidationError, match="depths must be positive"):
            camera.unproject(points_2d, depths)

    def test_unprojection_invalid_depths_type(self) -> None:
        """Test unprojection with invalid depths type raises ValidationError."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[320.0, 240.0]])
        with pytest.raises(
            ValidationError, match="depths must be a number or numpy array"
        ):
            camera.unproject(points_2d, "invalid_depths")  # type: ignore[arg-type]
