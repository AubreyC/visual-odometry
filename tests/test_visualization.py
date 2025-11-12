import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.camera_pose import CameraPose, TrajectoryGenerator
from src.feature_observation import FeatureObservation, ImageObservations
from src.landmarks import LandmarkGenerator
from src.matplotlib_visualizer import MatplotVisualizer


class TestMatplotVisualizer:
    """Test suite for MatplotVisualizer class."""

    @pytest.fixture
    def visualizer(self) -> MatplotVisualizer:
        """Create a test visualizer."""
        return MatplotVisualizer(figsize=(8, 6))

    @pytest.fixture
    def sample_poses(self) -> list[CameraPose]:
        """Create sample camera poses for testing."""
        generator = TrajectoryGenerator(time_step=0.1)
        poses = generator.generate_circular_trajectory(
            center=np.array([0.0, 0.0]), radius=2.0, height=1.0, num_poses=5
        )
        return poses

    @pytest.fixture
    def sample_landmarks(self) -> np.ndarray:
        """Create sample landmarks for testing."""
        generator = LandmarkGenerator()
        landmarks = generator.generate_random(num_landmarks=10, seed=42)
        return landmarks

    def test_initialization(self) -> None:
        """Test MatplotVisualizer initialization."""
        viz = MatplotVisualizer(figsize=(10, 8))
        assert viz.figsize == (10, 8)

        # Test default figsize
        viz_default = MatplotVisualizer()
        assert viz_default.figsize == (12, 8)

    def test_plot_3d_trajectory(
        self, visualizer: MatplotVisualizer, sample_poses: list[CameraPose]
    ) -> None:
        """Test 3D trajectory plotting."""
        fig = visualizer.plot_3d_trajectory(
            sample_poses, title="Test Trajectory", color="red", show_orientation=True
        )

        assert fig is not None
        # Check that figure has the expected elements
        ax = fig.gca()
        assert ax.get_xlabel() == "X (m)"
        assert ax.get_ylabel() == "Y (m)"
        # For 3D axes, check z-label exists
        assert hasattr(ax, "get_zlabel") and ax.get_zlabel() == "Z (m)"
        assert ax.get_title() == "Test Trajectory"

        plt.close(fig)

    def test_plot_trajectory_comparison(
        self, visualizer: MatplotVisualizer, sample_poses: list[CameraPose]
    ) -> None:
        """Test trajectory comparison plotting."""
        # Create slightly different poses for "estimated"
        estimated_poses = []
        for pose in sample_poses:
            # Add small noise to position
            noisy_pos = pose.position + np.random.normal(0, 0.1, 3)
            estimated_poses.append(
                CameraPose(noisy_pos, pose.quaternion, pose.timestamp)
            )

        fig = visualizer.plot_trajectory_comparison(
            sample_poses, estimated_poses, title="Trajectory Comparison Test"
        )

        assert fig is not None
        ax = fig.gca()
        assert ax.get_title() == "Trajectory Comparison Test"

        plt.close(fig)

    def test_plot_trajectory_comparison_ground_truth_only(
        self, visualizer: MatplotVisualizer, sample_poses: list[CameraPose]
    ) -> None:
        """Test trajectory plotting with ground truth only."""
        fig = visualizer.plot_trajectory_comparison(
            sample_poses, title="Ground Truth Only"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_landmarks(
        self, visualizer: MatplotVisualizer, sample_landmarks: np.ndarray
    ) -> None:
        """Test landmark plotting."""
        fig = visualizer.plot_landmarks(
            sample_landmarks, title="Test Landmarks", color="blue", size=30
        )

        assert fig is not None
        ax = fig.gca()
        assert ax.get_title() == "Test Landmarks"

        plt.close(fig)

    def test_plot_scene_overview(
        self,
        visualizer: MatplotVisualizer,
        sample_landmarks: np.ndarray,
        sample_poses: list[CameraPose],
    ) -> None:
        """Test scene overview plotting."""
        fig = visualizer.plot_scene_overview(
            sample_landmarks, sample_poses, title="Test Scene"
        )

        assert fig is not None
        ax = fig.gca()
        assert ax.get_title() == "Test Scene"

        plt.close(fig)

    def test_plot_synthetic_image(self, visualizer: MatplotVisualizer) -> None:
        """Test synthetic image plotting."""

        observations = [
            FeatureObservation(1, np.array([100.0, 200.0]), np.array([1.0, 2.0, 3.0])),
            FeatureObservation(2, np.array([300.0, 150.0]), np.array([4.0, 5.0, 6.0])),
        ]

        image = ImageObservations(
            0, 0.0, observations, image_width=640, image_height=480
        )

        fig = visualizer.plot_synthetic_image(image, title="Test Synthetic Image")

        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_error(
        self, visualizer: MatplotVisualizer, sample_poses: list[CameraPose]
    ) -> None:
        """Test trajectory error plotting."""
        # Create slightly different poses for "estimated"
        estimated_poses = []
        for pose in sample_poses:
            # Add small noise to position and orientation
            noisy_pos = pose.position + np.random.normal(0, 0.05, 3)
            # Small rotation noise
            noisy_quat = pose.quaternion + np.random.normal(0, 0.01, 4)
            noisy_quat = noisy_quat / np.linalg.norm(noisy_quat)  # Renormalize
            estimated_poses.append(CameraPose(noisy_pos, noisy_quat, pose.timestamp))

        fig1, fig2 = visualizer.plot_trajectory_error(
            sample_poses, estimated_poses, title="Test Error Analysis"
        )

        assert fig1 is not None
        assert fig2 is not None

        # Check titles
        assert "Position Error" in fig1.gca().get_title()
        assert "Orientation Error" in fig2.gca().get_title()

        plt.close(fig1)
        plt.close(fig2)

    def test_save_figure(
        self, visualizer: MatplotVisualizer, sample_poses: list[CameraPose]
    ) -> None:
        """Test figure saving functionality."""
        fig = visualizer.plot_3d_trajectory(sample_poses, title="Test Save")

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_plot.png")

            # This should not raise an exception
            visualizer.save_figure(fig, filepath, dpi=100)

            # Check that file was created
            assert os.path.exists(filepath)

        plt.close(fig)

    def test_trajectory_error_mismatched_lengths(
        self, visualizer: MatplotVisualizer, sample_poses: list[CameraPose]
    ) -> None:
        """Test trajectory error with mismatched pose list lengths."""
        # Create shorter estimated poses list
        estimated_poses = sample_poses[:-1]

        fig1, fig2 = visualizer.plot_trajectory_error(
            sample_poses, estimated_poses, title="Mismatched Lengths Test"
        )

        assert fig1 is not None
        assert fig2 is not None

        plt.close(fig1)
        plt.close(fig2)
