import os
import tempfile

import numpy as np
import pytest

from src.camera_pose import CameraPose, TrajectoryGenerator
from src.landmarks import LandmarkGenerator
from src.visualization import Visualizer


class TestVisualOdometryVisualizer:
    """Test suite for VisualOdometryVisualizer class."""

    @pytest.fixture
    def visualizer(self) -> Visualizer:
        """Create a test visualizer."""
        return Visualizer(figsize=(8, 6))

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
        """Test VisualOdometryVisualizer initialization."""
        viz = Visualizer(figsize=(10, 8))
        assert viz.figsize == (10, 8)

        # Test default figsize
        viz_default = Visualizer()
        assert viz_default.figsize == (12, 8)

    def test_plot_3d_trajectory(
        self, visualizer: Visualizer, sample_poses: list[CameraPose]
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
        assert ax.get_zlabel() == "Z (m)"
        assert ax.get_title() == "Test Trajectory"

        plt.close(fig)

    def test_plot_trajectory_comparison(
        self, visualizer: Visualizer, sample_poses: list[CameraPose]
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
        self, visualizer: Visualizer, sample_poses: list[CameraPose]
    ) -> None:
        """Test trajectory plotting with ground truth only."""
        fig = visualizer.plot_trajectory_comparison(
            sample_poses, title="Ground Truth Only"
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_landmarks(
        self, visualizer: Visualizer, sample_landmarks: np.ndarray
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
        visualizer: Visualizer,
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

    def test_plot_feature_tracks_2d(self, visualizer: Visualizer) -> None:
        """Test 2D feature track plotting."""
        # Create mock feature tracks
        from src.feature_observation import FeatureObservation, FeatureTrack

        pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        tracks = []
        for track_id in range(3):
            track = FeatureTrack(track_id, np.array([track_id, 0.0, 5.0]))

            # Add observations at different positions
            for i in range(3):
                obs = FeatureObservation(
                    track_id,
                    np.array([100 + track_id * 50 + i * 10, 200 + i * 20]),
                    np.array([track_id, 0.0, 5.0]),
                    pose,
                    i * 0.1,
                )
                track.add_observation(obs)

            tracks.append(track)

        fig = visualizer.plot_feature_tracks_2d(
            tracks, title="Test Feature Tracks", max_tracks=5
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_synthetic_image(self, visualizer: Visualizer) -> None:
        """Test synthetic image plotting."""
        from src.feature_observation import FeatureObservation, SyntheticImage

        pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        observations = [
            FeatureObservation(
                1, np.array([100.0, 200.0]), np.array([1.0, 2.0, 3.0]), pose
            ),
            FeatureObservation(
                2, np.array([300.0, 150.0]), np.array([4.0, 5.0, 6.0]), pose
            ),
        ]

        image = SyntheticImage(pose, observations, image_width=640, image_height=480)

        fig = visualizer.plot_synthetic_image(image, title="Test Synthetic Image")

        assert fig is not None
        plt.close(fig)

    def test_plot_trajectory_error(
        self, visualizer: Visualizer, sample_poses: list[CameraPose]
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
        self, visualizer: Visualizer, sample_poses: list[CameraPose]
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

    def test_empty_inputs(self, visualizer: Visualizer) -> None:
        """Test visualization with empty inputs."""
        # Empty trajectory
        fig = visualizer.plot_3d_trajectory([], title="Empty Trajectory")
        assert fig is not None
        plt.close(fig)

        # Empty landmarks
        empty_landmarks = np.empty((0, 3))
        fig = visualizer.plot_landmarks(empty_landmarks, title="Empty Landmarks")
        assert fig is not None
        plt.close(fig)

        # Empty tracks
        fig = visualizer.plot_feature_tracks_2d([], title="Empty Tracks")
        assert fig is not None
        plt.close(fig)

    def test_trajectory_error_mismatched_lengths(
        self, visualizer: Visualizer, sample_poses: list[CameraPose]
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


# Import matplotlib for cleanup
import matplotlib.pyplot as plt
