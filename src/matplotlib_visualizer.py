from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .camera_pose import CameraPose
from .feature_observation import ImageObservations


class MatplotVisualizer:
    """Visualization tools for visual odometry data including trajectories, landmarks, and feature tracks."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer with default figure size.

        Args:
            figsize (Tuple[int, int]): Default figure size for plots.
        """
        self.figsize = figsize

    def plot_3d_trajectory(
        self,
        poses: List[CameraPose],
        title: str = "Camera Trajectory",
        color: str = "blue",
        show_orientation: bool = True,
        orientation_scale: float = 0.1,
    ) -> plt.Figure:
        """Plot 3D camera trajectory.

        Args:
            poses (List[CameraPose]): List of camera poses.
            title (str): Plot title.
            color (str): Trajectory color.
            show_orientation (bool): Whether to show camera orientation arrows.
            orientation_scale (float): Scale factor for orientation arrows.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Handle empty pose list
        if not poses:
            # Plot empty axes with labels
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(title)
            ax.grid(True)
            return fig

        # Extract positions
        positions = np.array([pose.position for pose in poses])

        # Plot trajectory line
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=2,
            label="Trajectory",
        )

        # Plot start and end points
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            color="green",
            s=100,
            label="Start",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            color="red",
            s=100,
            label="End",
        )

        # Show orientation arrows at regular intervals
        if show_orientation and len(poses) > 0:
            step = max(1, len(poses) // 10)  # Show orientation for ~10 poses
            for i in range(0, len(poses), step):
                pose = poses[i]
                pos = pose.position

                # Camera forward direction (X-axis in camera frame)
                x_dir = pose.rotation_matrix @ np.array([1, 0, 0])

                # Plot forward arrow
                ax.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    x_dir[0],
                    x_dir[1],
                    x_dir[2],
                    color="red",
                    length=orientation_scale,
                    normalize=True,
                )

                # Camera forward direction (X-axis in camera frame)
                y_dir = pose.rotation_matrix @ np.array([0, 1, 0])

                # Plot forward arrow
                ax.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    y_dir[0],
                    y_dir[1],
                    y_dir[2],
                    color="green",
                    length=orientation_scale,
                    normalize=True,
                )

                # Camera forward direction (Z-axis in camera frame)
                z_dir = pose.rotation_matrix @ np.array([0, 0, 1])

                # Plot forward arrow
                ax.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    z_dir[0],
                    z_dir[1],
                    z_dir[2],
                    color="blue",
                    length=orientation_scale,
                    normalize=True,
                )

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        # Set equal aspect ratio
        max_range = (
            np.array(
                [
                    positions[:, 0].max() - positions[:, 0].min(),
                    positions[:, 1].max() - positions[:, 1].min(),
                    positions[:, 2].max() - positions[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig

    def plot_trajectory_comparison(
        self,
        ground_truth_poses: List[CameraPose],
        estimated_poses: Optional[List[CameraPose]] = None,
        title: str = "Trajectory Comparison",
    ) -> plt.Figure:
        """Plot comparison between ground truth and estimated trajectories.

        Args:
            ground_truth_poses (List[CameraPose]): Ground truth camera poses.
            estimated_poses (Optional[List[CameraPose]]): Estimated camera poses.
            title (str): Plot title.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot ground truth
        gt_positions = np.array([pose.position for pose in ground_truth_poses])
        ax.plot(
            gt_positions[:, 0],
            gt_positions[:, 1],
            gt_positions[:, 2],
            color="blue",
            linewidth=2,
            label="Ground Truth",
        )

        # Plot estimates if provided
        if estimated_poses is not None:
            est_positions = np.array([pose.position for pose in estimated_poses])
            ax.plot(
                est_positions[:, 0],
                est_positions[:, 1],
                est_positions[:, 2],
                color="red",
                linewidth=2,
                linestyle="--",
                label="Estimated",
            )

            # Plot start points
            ax.scatter(
                gt_positions[0, 0],
                gt_positions[0, 1],
                gt_positions[0, 2],
                color="green",
                s=100,
                label="Start (GT)",
            )
            if len(est_positions) > 0:
                ax.scatter(
                    est_positions[0, 0],
                    est_positions[0, 1],
                    est_positions[0, 2],
                    color="orange",
                    s=100,
                    label="Start (Est)",
                )

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        return fig

    def plot_landmarks(
        self,
        landmarks: np.ndarray,
        title: str = "3D Landmarks",
        color: str = "gray",
        size: int = 20,
        alpha: float = 0.7,
    ) -> plt.Figure:
        """Plot 3D landmarks.

        Args:
            landmarks (np.ndarray): Landmark positions of shape (N, 3).
            title (str): Plot title.
            color (str): Landmark color.
            size (int): Point size.
            alpha (float): Transparency.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Handle empty landmarks
        if len(landmarks) > 0:
            ax.scatter(
                landmarks[:, 0],
                landmarks[:, 1],
                landmarks[:, 2],
                c=color,
                s=size,
                alpha=alpha,
            )

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.grid(True)

        # Set equal aspect ratio for non-empty landmarks
        if len(landmarks) > 0:
            max_range = (
                np.array(
                    [
                        landmarks[:, 0].max() - landmarks[:, 0].min(),
                        landmarks[:, 1].max() - landmarks[:, 1].min(),
                        landmarks[:, 2].max() - landmarks[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )

            if max_range > 0:
                mid_x = (landmarks[:, 0].max() + landmarks[:, 0].min()) * 0.5
                mid_y = (landmarks[:, 1].max() + landmarks[:, 1].min()) * 0.5
                mid_z = (landmarks[:, 2].max() + landmarks[:, 2].min()) * 0.5

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig

    def plot_scene_overview(
        self,
        landmarks: np.ndarray,
        poses: List[CameraPose],
        poses_second: Optional[List[CameraPose]] = None,
        show_orientation: bool = True,
        orientation_scale: float = 0.1,
        title: str = "Scene Overview",
        fig: Optional[plt.Figure] = None,
        ax: Optional[Any] = None,
    ) -> plt.Figure:
        """Plot complete scene with landmarks and camera trajectory.

        Args:
            landmarks (np.ndarray): Landmark positions.
            poses (List[CameraPose]): Camera poses.
            poses_second (Optional[List[CameraPose]]): Optional second set of camera poses.
            show_orientation (bool): Whether to show camera orientation arrows.
            orientation_scale (float): Scale factor for orientation arrows.
            title (str): Plot title.
            fig (Optional[plt.Figure]): Existing matplotlib figure to update. If None, creates new figure.
            ax (Optional[Any]): Existing matplotlib axes to update. If None, creates new axes.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        # Create new figure/axes if not provided, otherwise reuse existing ones
        if fig is None or ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            # Clear existing axes content
            ax.clear()

        # Plot landmarks
        ax.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            landmarks[:, 2],
            c="gray",
            s=10,
            alpha=0.6,
            label="Landmarks",
        )

        # Show orientation arrows at regular intervals
        if show_orientation and len(poses) > 0:
            step = max(1, len(poses) // 10)  # Show orientation for ~10 poses
            for i in range(0, len(poses), step):
                pose = poses[i]
                pos = pose.position

                # Camera forward direction (X-axis in camera frame)
                x_dir = pose.rotation_matrix @ np.array([1, 0, 0])

                # Plot forward arrow
                ax.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    x_dir[0],
                    x_dir[1],
                    x_dir[2],
                    color="red",
                    length=orientation_scale,
                    normalize=True,
                )

                # Camera forward direction (X-axis in camera frame)
                y_dir = pose.rotation_matrix @ np.array([0, 1, 0])

                # Plot forward arrow
                ax.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    y_dir[0],
                    y_dir[1],
                    y_dir[2],
                    color="green",
                    length=orientation_scale,
                    normalize=True,
                )

                # Camera forward direction (Z-axis in camera frame)
                z_dir = pose.rotation_matrix @ np.array([0, 0, 1])

                # Plot forward arrow
                ax.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    z_dir[0],
                    z_dir[1],
                    z_dir[2],
                    color="blue",
                    length=orientation_scale,
                    normalize=True,
                )

        # Plot trajectory
        if poses:
            positions = np.array([pose.position for pose in poses])
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color="blue",
                linewidth=3,
                label="Camera Trajectory",
            )

            # Mark start and end
            ax.scatter(
                positions[0, 0],
                positions[0, 1],
                positions[0, 2],
                color="green",
                s=100,
                label="Start",
            )
            ax.scatter(
                positions[-1, 0],
                positions[-1, 1],
                positions[-1, 2],
                color="red",
                s=100,
                label="End",
            )

        # Show orientation arrows at regular intervals
        if poses_second is not None:
            if show_orientation and len(poses_second) > 0:
                step = max(1, len(poses_second) // 10)  # Show orientation for ~10 poses
                for i in range(0, len(poses_second), step):
                    pose = poses_second[i]
                    pos = pose.position

                    # Camera forward direction (X-axis in camera frame)
                    x_dir = pose.rotation_matrix @ np.array([1, 0, 0])

                    # Plot forward arrow
                    ax.quiver(
                        pos[0],
                        pos[1],
                        pos[2],
                        x_dir[0],
                        x_dir[1],
                        x_dir[2],
                        color="red",
                        length=orientation_scale,
                        normalize=True,
                    )

                    # Camera forward direction (X-axis in camera frame)
                    y_dir = pose.rotation_matrix @ np.array([0, 1, 0])

                    # Plot forward arrow
                    ax.quiver(
                        pos[0],
                        pos[1],
                        pos[2],
                        y_dir[0],
                        y_dir[1],
                        y_dir[2],
                        color="green",
                        length=orientation_scale,
                        normalize=True,
                    )

                    # Camera forward direction (Z-axis in camera frame)
                    z_dir = pose.rotation_matrix @ np.array([0, 0, 1])

                    # Plot forward arrow
                    ax.quiver(
                        pos[0],
                        pos[1],
                        pos[2],
                        z_dir[0],
                        z_dir[1],
                        z_dir[2],
                        color="blue",
                        length=orientation_scale,
                        normalize=True,
                    )

            # Plot trajectory
            if len(poses_second) > 0:
                positions = np.array([pose.position for pose in poses_second])
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    color="purple",
                    linewidth=3,
                    label="Camera Trajectory Second",
                )

                # Mark start and end
                ax.scatter(
                    positions[0, 0],
                    positions[0, 1],
                    positions[0, 2],
                    color="green",
                    s=100,
                    label="Start",
                )
                ax.scatter(
                    positions[-1, 0],
                    positions[-1, 1],
                    positions[-1, 2],
                    color="red",
                    s=100,
                    label="End",
                )

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        ax.set_xlim(-10, 10)  # Set x-axis
        ax.set_ylim(-10, 10)  # Set y-axis
        ax.set_zlim(-10, 10)  # Set z-axis

        return fig

    def plot_synthetic_image(
        self, image: ImageObservations, title: Optional[str] = None
    ) -> plt.Figure:
        """Plot synthetic image with feature observations.

        Args:
            image (SyntheticImage): Synthetic image to plot.
            title (Optional[str]): Plot title.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create blank image
        blank_image = np.ones((image.image_height, image.image_width, 3)) * 0.9

        # Plot blank image
        ax.imshow(blank_image, extent=[0, image.image_width, image.image_height, 0])

        # Plot feature observations
        if image.feature_observations:
            coords = np.array([obs.image_coords for obs in image.feature_observations])
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                color="red",
                s=50,
                alpha=0.8,
                edgecolors="black",
            )

            # Add landmark IDs as text
            for obs in image.feature_observations:
                ax.annotate(
                    f"{obs.landmark_id}",
                    (obs.image_coords[0], obs.image_coords[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="blue",
                )

        if title is None:
            title = f"Synthetic Image - {len(image.feature_observations)} features"

        ax.set_title(title)
        ax.set_xlabel("Image X (pixels)")
        ax.set_ylabel("Image Y (pixels)")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_trajectory_error(
        self,
        ground_truth_poses: List[CameraPose],
        estimated_poses: List[CameraPose],
        title: str = "Trajectory Error Analysis",
    ) -> Tuple[plt.Figure, plt.Figure]:
        """Plot trajectory position and orientation errors.

        Args:
            ground_truth_poses (List[CameraPose]): Ground truth poses.
            estimated_poses (List[CameraPose]): Estimated poses.
            title (str): Plot title.

        Returns:
            Tuple[plt.Figure, plt.Figure]: Position error and orientation error figures.
        """
        # Calculate errors
        min_length = min(len(ground_truth_poses), len(estimated_poses))
        timestamps = []
        position_errors = []
        orientation_errors = []

        for i in range(min_length):
            gt_pose = ground_truth_poses[i]
            est_pose = estimated_poses[i]

            if gt_pose.timestamp is not None:
                timestamps.append(gt_pose.timestamp)
            else:
                timestamps.append(i * 0.1)  # Default time step

            # Position error (Euclidean distance)
            pos_error = np.linalg.norm(gt_pose.position - est_pose.position)
            position_errors.append(pos_error)

            # Orientation error (angle between rotation matrices)
            # Using Frobenius norm of difference
            rot_diff = gt_pose.rotation_matrix - est_pose.rotation_matrix
            ori_error = np.linalg.norm(rot_diff) / np.sqrt(12)  # Normalized
            orientation_errors.append(ori_error)

        # Plot position errors
        fig1, ax1 = plt.subplots(figsize=self.figsize)
        ax1.plot(timestamps, position_errors, "b-", linewidth=2, label="Position Error")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position Error (m)")
        ax1.set_title(f"{title} - Position Error")
        ax1.grid(True)
        ax1.legend()

        # Plot orientation errors
        fig2, ax2 = plt.subplots(figsize=self.figsize)
        ax2.plot(
            timestamps, orientation_errors, "r-", linewidth=2, label="Orientation Error"
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Orientation Error (rad)")
        ax2.set_title(f"{title} - Orientation Error")
        ax2.grid(True)
        ax2.legend()

        return fig1, fig2

    def save_figure(self, fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
        """Save figure to file.

        Args:
            fig (plt.Figure): Matplotlib figure to save.
            filepath (str): Output file path.
            dpi (int): Resolution in dots per inch.
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {filepath}")

    def show(self) -> None:
        """Show all plots."""
        plt.show(block=True)
