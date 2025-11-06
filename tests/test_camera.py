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
        assert camera.k1 == 0.0
        assert camera.k2 == 0.0
        assert camera.k3 == 0.0

    def test_valid_initialization_with_distortion(self) -> None:
        """Test valid camera initialization with distortion coefficients."""
        camera = PinHoleCamera(
            fx=500.0, fy=600.0, cx=320.0, cy=240.0, k1=-0.1, k2=0.01, k3=-0.001
        )
        assert camera.fx == 500.0
        assert camera.fy == 600.0
        assert camera.cx == 320.0
        assert camera.cy == 240.0
        assert camera.k1 == -0.1
        assert camera.k2 == 0.01
        assert camera.k3 == -0.001

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

    @pytest.mark.parametrize("invalid_k1", [float("inf"), float("-inf"), float("nan")])
    def test_invalid_distortion_k1(self, invalid_k1: float) -> None:
        """Test invalid distortion coefficient k1 raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match=r"Radial distortion coefficient k1 must be a finite number.*",
        ):
            PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=invalid_k1)

    @pytest.mark.parametrize("invalid_k2", [float("inf"), float("-inf"), float("nan")])
    def test_invalid_distortion_k2(self, invalid_k2: float) -> None:
        """Test invalid distortion coefficient k2 raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match=r"Radial distortion coefficient k2 must be a finite number.*",
        ):
            PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, k2=invalid_k2)

    @pytest.mark.parametrize("invalid_k3", [float("inf"), float("-inf"), float("nan")])
    def test_invalid_distortion_k3(self, invalid_k3: float) -> None:
        """Test invalid distortion coefficient k3 raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match=r"Radial distortion coefficient k3 must be a finite number.*",
        ):
            PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, k3=invalid_k3)

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

    def test_projection_with_distortion(self) -> None:
        """Test 3D point projection with radial distortion."""
        # Camera with barrel distortion (k1 > 0)
        camera = PinHoleCamera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.1, k2=0.01, k3=0.001
        )
        points_3d = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        )  # Points at unit distance from center

        projected = camera.project(points_3d)

        # With distortion, points should be pushed outward from center
        # Expected projection without distortion: [[570.0, 240.0], [320.0, 490.0]]
        # With distortion, they should be further from center
        assert projected.shape == (2, 2)
        assert np.all(np.isfinite(projected))

        # Points should be further from optical center due to barrel distortion
        center_u, center_v = 320.0, 240.0
        distances = np.sqrt(
            (projected[:, 0] - center_u) ** 2 + (projected[:, 1] - center_v) ** 2
        )
        expected_distances_no_distortion = np.sqrt(
            (570.0 - 320.0) ** 2 + (490.0 - 240.0) ** 2
        )  # ~250 for both
        assert distances[0] > 250.0  # Should be larger due to distortion
        assert distances[1] > 250.0

    def test_projection_no_distortion_when_coeffs_zero(self) -> None:
        """Test that projection behaves the same with zero distortion coefficients."""
        camera_no_distortion = PinHoleCamera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.0, k2=0.0, k3=0.0
        )
        camera_default = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        points_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        projected_explicit = camera_no_distortion.project(points_3d)
        projected_default = camera_default.project(points_3d)

        np.testing.assert_array_equal(projected_explicit, projected_default)

    def test_undistort_no_distortion(self) -> None:
        """Test undistort method with no distortion coefficients."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        points_2d = np.array([[370.0, 290.0], [420.0, 340.0]])

        undistorted = camera.undistort(points_2d)

        # Should return the same points when no distortion
        np.testing.assert_array_equal(undistorted, points_2d)

    def test_undistort_with_distortion(self) -> None:
        """Test undistort method with distortion coefficients."""
        camera = PinHoleCamera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.1, k2=0.01, k3=0.001
        )

        # Create some distorted points
        points_3d = np.array([[0.5, 0.3, 1.0], [-0.2, 0.4, 1.0]])
        distorted_points = camera.project(points_3d)

        # Undistort them back
        undistorted_points = camera.undistort(distorted_points)

        # Should be close to original (allowing for numerical precision)
        np.testing.assert_allclose(
            undistorted_points,
            points_3d[:, :2] * np.array([500.0, 500.0]) + np.array([320.0, 240.0]),
            rtol=1e-3,
        )

    def test_undistort_invalid_input(self) -> None:
        """Test undistort method with invalid input."""
        camera = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.1)

        # Test with non-numpy array
        with pytest.raises(ValidationError, match="points_2d must be a numpy array"):
            camera.undistort([[320.0, 240.0]])  # type: ignore[arg-type]

        # Test with wrong shape
        with pytest.raises(ValidationError, match="points_2d must have shape"):
            camera.undistort(np.array([[320.0, 240.0, 1.0]]))

        # Test with empty array
        with pytest.raises(ValidationError, match="points_2d cannot be empty"):
            camera.undistort(np.empty((0, 2)))

        # Test with non-finite values
        with pytest.raises(
            ValidationError, match="points_2d must contain only finite values"
        ):
            camera.undistort(np.array([[float("nan"), 240.0]]))

    def test_distortion_affects_projection(self) -> None:
        """Test that radial distortion affects 3D point projection."""
        # Create camera with barrel distortion (k1 > 0)
        camera_distorted = PinHoleCamera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.1, k2=0.01, k3=0.001
        )

        # Create camera without distortion for comparison
        camera_no_distortion = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        # Test points at different distances from center
        test_points_3d = np.array(
            [
                [1.0, 0.0, 1.0],  # Point on x-axis at distance 1
                [0.0, 1.0, 1.0],  # Point on y-axis at distance 1
                [0.707, 0.707, 1.0],  # Point at 45 degrees, distance ~1
            ]
        )

        projected_no_distortion = camera_no_distortion.project(test_points_3d)
        projected_distortion = camera_distorted.project(test_points_3d)

        # For all test points, distorted projection should be further from center
        for i in range(len(test_points_3d)):
            u_no_dist, v_no_dist = projected_no_distortion[i]
            u_dist, v_dist = projected_distortion[i]

            distance_no_dist = np.sqrt(
                (u_no_dist - 320.0) ** 2 + (v_no_dist - 240.0) ** 2
            )
            distance_dist = np.sqrt((u_dist - 320.0) ** 2 + (v_dist - 240.0) ** 2)

            # Barrel distortion (k1 > 0) should push points further from center
            assert distance_dist > distance_no_dist, (
                f"Point {i + 1} should be further from center with distortion"
            )

    def test_distortion_magnitude_increases_with_distance(self) -> None:
        """Test that distortion magnitude increases with distance from optical center."""
        camera_distorted = PinHoleCamera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.1, k2=0.01, k3=0.001
        )

        camera_no_distortion = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        # Test points at different distances from center
        test_points_3d = np.array(
            [
                [0.0, 0.0, 1.0],  # Point at optical center (distance 0)
                [0.5, 0.0, 1.0],  # Point at distance 0.5
                [1.0, 0.0, 1.0],  # Point at distance 1.0
            ]
        )

        projected_no_distortion = camera_no_distortion.project(test_points_3d)
        projected_distortion = camera_distorted.project(test_points_3d)

        # Calculate distortion amounts (difference from undistorted projection)
        distortions = []
        for i in range(len(test_points_3d)):
            u_no_dist, v_no_dist = projected_no_distortion[i]
            u_dist, v_dist = projected_distortion[i]

            distance_no_dist = np.sqrt(
                (u_no_dist - 320.0) ** 2 + (v_no_dist - 240.0) ** 2
            )
            distance_dist = np.sqrt((u_dist - 320.0) ** 2 + (v_dist - 240.0) ** 2)

            distortion_amount = distance_dist - distance_no_dist
            distortions.append(distortion_amount)

        # Distortion should increase with distance from center
        assert distortions[0] < distortions[1], (
            "Distortion should increase with distance from center"
        )
        assert distortions[1] < distortions[2], (
            "Distortion should increase with distance from center"
        )

    def test_undistortion_corrects_distortion(self) -> None:
        """Test that undistortion method correctly reverses distortion."""
        camera_distorted = PinHoleCamera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.1, k2=0.01, k3=0.001
        )

        camera_no_distortion = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        # Test points at different distances from center
        test_points_3d = np.array(
            [
                [1.0, 0.0, 1.0],  # Point on x-axis at distance 1
                [0.0, 1.0, 1.0],  # Point on y-axis at distance 1
                [0.707, 0.707, 1.0],  # Point at 45 degrees, distance ~1
            ]
        )

        # Project without distortion to get "ground truth"
        projected_no_distortion = camera_no_distortion.project(test_points_3d)

        # Project with distortion, then undistort
        projected_distortion = camera_distorted.project(test_points_3d)
        undistorted = camera_distorted.undistort(projected_distortion)

        # Undistorted points should match original undistorted projection
        np.testing.assert_allclose(
            undistorted,
            projected_no_distortion,
            rtol=1e-3,
            err_msg="Undistortion should recover original undistorted projection",
        )

    def test_optical_center_not_affected_by_distortion(self) -> None:
        """Test that the optical center is not affected by radial distortion."""
        camera_distorted = PinHoleCamera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, k1=0.1, k2=0.01, k3=0.001
        )

        camera_no_distortion = PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        # Point at optical center (looking straight ahead)
        optical_center_3d = np.array([[0.0, 0.0, 1.0]])

        projected_no_distortion = camera_no_distortion.project(optical_center_3d)
        projected_distortion = camera_distorted.project(optical_center_3d)

        # Optical center should project to the same point regardless of distortion
        np.testing.assert_allclose(
            projected_distortion,
            projected_no_distortion,
            rtol=1e-10,
            err_msg="Optical center should not be affected by radial distortion",
        )

    def test_camera_matrix_validation(self) -> None:
        """Test the is_valid_camera_matrix method."""
        print("Testing camera matrix validation...")

        # Valid camera matrix
        K_valid = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        assert PinHoleCamera.is_valid_camera_matrix(K_valid), (
            "Valid matrix should be accepted"
        )

        # Invalid: wrong shape
        K_wrong_shape = np.array([[800, 0, 320], [0, 800, 240]])
        assert not PinHoleCamera.is_valid_camera_matrix(K_wrong_shape), (
            "Wrong shape should be rejected"
        )

        # Invalid: non-zero off-diagonal
        K_non_zero_off_diag = np.array([[800, 1, 320], [0, 800, 240], [0, 0, 1]])
        assert not PinHoleCamera.is_valid_camera_matrix(K_non_zero_off_diag), (
            "Non-zero off-diagonal should be rejected"
        )

        # Invalid: wrong bottom row
        K_wrong_bottom = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 2]])
        assert not PinHoleCamera.is_valid_camera_matrix(K_wrong_bottom), (
            "Wrong bottom row should be rejected"
        )

        # Invalid: negative focal length
        K_neg_focal = np.array([[-800, 0, 320], [0, 800, 240], [0, 0, 1]])
        assert not PinHoleCamera.is_valid_camera_matrix(K_neg_focal), (
            "Negative focal length should be rejected"
        )

        # Invalid: not numpy array
        assert not PinHoleCamera.is_valid_camera_matrix(
            [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
        ), "List should be rejected"
