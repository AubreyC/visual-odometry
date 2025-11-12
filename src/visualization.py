from typing import Any, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .feature_observation import ImageObservations
from .geometry import GeometryUtils


class Visualizer:
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
        plt.pause(0.1)


class OpenCVSceneVisualizer:
    """OpenCV-based 3D scene visualizer showing camera, landmarks, and coordinate system."""

    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        fx: float = 800.0,
        fy: float = 800.0,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        interactive: bool = False,
    ):
        """Initialize the OpenCV scene visualizer.

        Args:
            image_width (int): Width of the visualization image.
            image_height (int): Height of the visualization image.
            background_color (Tuple[int, int, int]): RGB background color (0-255).
            interactive (bool): Whether to enable interactive camera controls (mouse/keyboard).
                               When False, show_scene displays a static view.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.background_color = background_color
        self.interactive = interactive

        # Camera intrinsic parameters for scene visualization
        self.scene_camera = PinHoleCamera(
            fx=fx,
            fy=fy,
            cx=image_width / 2,
            cy=image_height / 2,
        )

        # Interactive camera control variables
        self.current_scene_camera_pose: Optional[CameraPose] = None
        self.mouse_dragging = False
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        self.move_speed = 0.1  # Movement speed for position controls
        self.rotation_speed = 0.01  # Rotation speed for mouse controls

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        """Handle mouse events for interactive camera control.

        Args:
            event: OpenCV mouse event type
            x, y: Mouse coordinates
            flags: Mouse event flags
            param: Additional parameters
        """
        if not self.interactive or self.current_scene_camera_pose is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start dragging
            self.mouse_dragging = True
            self.last_mouse_pos = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            self.mouse_dragging = False
            self.last_mouse_pos = None

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and self.mouse_dragging
            and self.last_mouse_pos is not None
        ):
            # Handle drag for camera rotation
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]

            # Update camera orientation based on mouse movement
            self._update_camera_orientation(dx, dy)

            self.last_mouse_pos = (x, y)

    def _update_camera_orientation(self, dx: float, dy: float) -> None:
        """Update camera orientation based on mouse movement deltas.

        Args:
            dx: Horizontal mouse movement delta
            dy: Vertical mouse movement delta
        """
        if self.current_scene_camera_pose is None:
            return

        # Create rotation quaternions for yaw (horizontal) and pitch (vertical)
        # Mouse movement maps to camera rotation:
        # - Horizontal movement (dx) -> yaw rotation around world Y axis
        # - Vertical movement (dy) -> pitch rotation around camera X axis

        current_pose = self.current_scene_camera_pose

        # Yaw rotation (around world Y axis)
        if abs(dx) > 0:
            yaw_axis = np.array([0, 1, 0])  # World Y axis
            yaw_angle = -dx * self.rotation_speed
            yaw_quaternion = GeometryUtils.quaternion_from_axis_angle(
                yaw_axis, yaw_angle
            )
            new_orientation = GeometryUtils.quaternion_multiply(
                current_pose.quaternion, yaw_quaternion
            )
            current_pose.orientation_quaternion = new_orientation

        # Pitch rotation (around camera X axis)
        if abs(dy) > 0:
            camera_x_axis = current_pose.rotation_matrix[
                :, 0
            ]  # X axis in world coordinates
            pitch_angle = -dy * self.rotation_speed
            pitch_quaternion = GeometryUtils.quaternion_from_axis_angle(
                camera_x_axis, pitch_angle
            )
            new_orientation = GeometryUtils.quaternion_multiply(
                current_pose.quaternion, pitch_quaternion
            )
            current_pose.orientation_quaternion = new_orientation

    def _update_camera_position(self, key: int) -> None:
        """Update camera position based on keyboard input.

        Args:
            key: OpenCV key code
        """
        if self.current_scene_camera_pose is None:
            return

        current_pose = self.current_scene_camera_pose
        move_vector = np.zeros(3)

        # Movement keys (WASD + QE for up/down)
        if key == ord("w") or key == ord("W"):
            # Move forward (along camera's Z axis)
            move_vector = -current_pose.rotation_matrix[:, 2] * self.move_speed
        elif key == ord("s") or key == ord("S"):
            # Move backward (opposite to camera's Z axis)
            move_vector = current_pose.rotation_matrix[:, 2] * self.move_speed
        elif key == ord("a") or key == ord("A"):
            # Move left (opposite to camera's X axis)
            move_vector = -current_pose.rotation_matrix[:, 0] * self.move_speed
        elif key == ord("d") or key == ord("D"):
            # Move right (along camera's X axis)
            move_vector = current_pose.rotation_matrix[:, 0] * self.move_speed
        elif key == ord("q") or key == ord("Q"):
            # Move down (opposite to world Y axis)
            move_vector = np.array([0, -self.move_speed, 0])
        elif key == ord("e") or key == ord("E"):
            # Move up (along world Y axis)
            move_vector = np.array([0, self.move_speed, 0])

        # Direct axis movement (X, Y, Z)
        elif key == ord("x") or key == ord("X"):
            # Move along world X axis
            move_vector = np.array([self.move_speed, 0, 0])
        elif key == ord("y") or key == ord("Y"):
            # Move along world Y axis
            move_vector = np.array([0, self.move_speed, 0])
        elif key == ord("z") or key == ord("Z"):
            # Move along world Z axis
            move_vector = np.array([0, 0, self.move_speed])

        # Apply movement
        if np.any(move_vector != 0):
            current_pose.position += move_vector

    def project_points_to_image(
        self,
        points_3d: np.ndarray,
        camera_pose: CameraPose,
    ) -> np.ndarray:
        """Project 3D world points to 2D image coordinates using camera pose.

        Args:
            points_3d (np.ndarray): 3D points in world coordinates, shape (N, 3).
            camera_pose (CameraPose): Camera pose defining the view.

        Returns:
            np.ndarray: 2D image coordinates, shape (N, 2). Points behind camera are filtered out.
        """
        # Transform points from world to camera coordinates
        points_camera = camera_pose.transform_points_world_to_camera(points_3d)

        # Filter points in front of camera (positive Z)
        valid_mask = points_camera[:, 2] > 0
        points_camera_valid = points_camera[valid_mask]

        if len(points_camera_valid) == 0:
            return np.empty((0, 2))

        # Project to image plane
        points_2d = self.scene_camera.project(points_camera_valid)

        # Filter points within image bounds
        u_coords, v_coords = points_2d[:, 0], points_2d[:, 1]
        image_mask = (
            (u_coords >= 0)
            & (u_coords < self.image_width)
            & (v_coords >= 0)
            & (v_coords < self.image_height)
        )

        return points_2d[image_mask]

    def draw_coordinate_axes(
        self,
        image: np.ndarray,
        camera_pose: CameraPose,
        axis_length: float = 10.0,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw coordinate axes (X=red, Y=green, Z=blue) in the scene.

        Args:
            image (np.ndarray): Image to draw on.
            camera_pose (CameraPose): Camera pose defining the view.
            axis_length (float): Length of each axis in world units.
            thickness (int): Line thickness.

        Returns:
            np.ndarray: Image with coordinate axes drawn.
        """
        # Define axis endpoints in world coordinates
        origin = np.array([0.0, 0.0, 0.0])
        x_axis = np.array([axis_length, 0.0, 0.0])
        y_axis = np.array([0.0, axis_length, 0.0])
        z_axis = np.array([0.0, 0.0, axis_length])

        axes_points = np.array([origin, x_axis, y_axis, z_axis])

        # Project to image
        projected = self.project_points_to_image(axes_points, camera_pose)

        if len(projected) < 4:
            return image  # Not enough points visible

        # Draw axes
        origin_2d = tuple(map(int, projected[0]))
        x_end_2d = tuple(map(int, projected[1]))
        y_end_2d = tuple(map(int, projected[2]))
        z_end_2d = tuple(map(int, projected[3]))

        # X-axis (red)
        cv2.line(image, origin_2d, x_end_2d, (0, 0, 255), thickness)
        # Y-axis (green)
        cv2.line(image, origin_2d, y_end_2d, (0, 255, 0), thickness)
        # Z-axis (blue)
        cv2.line(image, origin_2d, z_end_2d, (255, 0, 0), thickness)

        # Add axis labels
        cv2.putText(image, "X", x_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Y", y_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "Z", z_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return image

    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        camera_pose: CameraPose,
        cross_size: int = 10,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 1,
        landmark_ids: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Draw landmarks as crosses in the scene.

        Args:
            image (np.ndarray): Image to draw on.
            landmarks (np.ndarray): Landmark positions in world coordinates, shape (N, 3).
            camera_pose (CameraPose): Camera pose defining the view.
            cross_size (int): Size of the cross in pixels.
            color (Tuple[int, int, int]): RGB color for landmarks.
            thickness (int): Line thickness.
            landmark_ids (Optional[List[int]]): Optional landmark IDs to display.

        Returns:
            np.ndarray: Image with landmarks drawn.
        """
        # Project landmarks to image
        projected = self.project_points_to_image(landmarks, camera_pose)

        for i, point_2d in enumerate(projected):
            u, v = map(int, point_2d)

            # Draw cross
            cv2.line(image, (u - cross_size, v), (u + cross_size, v), color, thickness)
            cv2.line(image, (u, v - cross_size), (u, v + cross_size), color, thickness)

            # Draw landmark ID if provided
            if landmark_ids is not None and i < len(landmark_ids):
                cv2.putText(
                    image,
                    str(landmark_ids[i]),
                    (u + cross_size + 5, v - cross_size - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        return image

    def draw_camera_pyramid(
        self,
        image: np.ndarray,
        scene_camera_pose: CameraPose,
        camera_pose: CameraPose,
        pyramid_length: float = 0.5,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw camera pyramid with apex at camera location and base along Z axis.

        Args:
            image (np.ndarray): Image to draw on.
            scene_camera_pose (CameraPose): Scene camera pose defining the viewpoint.
            camera_pose (CameraPose): Camera pose to visualize.
            pyramid_length (float): Length of pyramid from camera along Z axis.
            color (Tuple[int, int, int]): RGB color for pyramid.
            thickness (int): Line thickness.

        Returns:
            np.ndarray: Image with camera pyramid drawn.
        """

        # Camera position (apex of pyramid)
        camera_pos = camera_pose.position

        # Define pyramid base corners in camera coordinates (at pyramid_length distance along Z)
        # Camera looks along +Z axis, base is at pyramid_length
        half_width = pyramid_length * (self.scene_camera.cx / self.scene_camera.fx)
        half_height = pyramid_length * (self.scene_camera.cy / self.scene_camera.fy)

        pyramid_base_corners_camera = np.array(
            [
                [half_width, half_height, pyramid_length],  # top-right
                [-half_width, half_height, pyramid_length],  # top-left
                [-half_width, -half_height, pyramid_length],  # bottom-left
                [half_width, -half_height, pyramid_length],  # bottom-right
            ]
        )

        # Transform base corners to world coordinates
        pyramid_base_corners_world = camera_pose.transform_points_camera_to_world(
            pyramid_base_corners_camera
        )

        # Add camera position (apex) to the points
        all_points = np.vstack([camera_pos.reshape(1, -1), pyramid_base_corners_world])

        # Project to image
        projected = self.project_points_to_image(all_points, scene_camera_pose)

        if len(projected) < 5:
            return image  # Not enough points visible

        camera_pos_2d = tuple(map(int, projected[0]))
        base_corners_2d = [tuple(map(int, pt)) for pt in projected[1:]]

        # Draw pyramid edges from apex (camera position) to base corners
        for corner_2d in base_corners_2d:
            cv2.line(image, camera_pos_2d, corner_2d, color, thickness)

        # Draw pyramid base (rectangle connecting the base corners)
        if len(base_corners_2d) == 4:
            cv2.line(image, base_corners_2d[0], base_corners_2d[1], color, thickness)
            cv2.line(image, base_corners_2d[1], base_corners_2d[2], color, thickness)
            cv2.line(image, base_corners_2d[2], base_corners_2d[3], color, thickness)
            cv2.line(image, base_corners_2d[3], base_corners_2d[0], color, thickness)

        # Draw camera position (apex) as a circle
        cv2.circle(image, camera_pos_2d, 8, color, -1)  # Filled circle
        cv2.circle(image, camera_pos_2d, 8, (255, 255, 255), 2)  # White border

        return image

    def draw_coordinate_frame(
        self,
        image: np.ndarray,
        scene_camera_pose: CameraPose,
        camera_pose: CameraPose,
        axis_length: float = 1.0,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw XYZ coordinate frame axes at camera position with camera orientation.

        Args:
            image (np.ndarray): Image to draw on.
            scene_camera_pose (CameraPose): Scene camera pose defining the viewpoint.
            camera_pose (CameraPose): Camera pose defining the position and orientation of the frame.
            axis_length (float): Length of each axis in world units.
            thickness (int): Line thickness.

        Returns:
            np.ndarray: Image with coordinate frame drawn.
        """
        # Define axis endpoints in camera coordinates
        origin = np.array([0.0, 0.0, 0.0])
        x_axis_end = np.array([axis_length, 0.0, 0.0])
        y_axis_end = np.array([0.0, axis_length, 0.0])
        z_axis_end = np.array([0.0, 0.0, axis_length])

        # Transform points from camera coordinates to world coordinates
        origin_world = camera_pose.transform_points_camera_to_world(
            origin.reshape(1, -1)
        )[0]
        x_end_world = camera_pose.transform_points_camera_to_world(
            x_axis_end.reshape(1, -1)
        )[0]
        y_end_world = camera_pose.transform_points_camera_to_world(
            y_axis_end.reshape(1, -1)
        )[0]
        z_end_world = camera_pose.transform_points_camera_to_world(
            z_axis_end.reshape(1, -1)
        )[0]

        # Combine all points for projection
        frame_points = np.array([origin_world, x_end_world, y_end_world, z_end_world])

        # Project to image
        projected = self.project_points_to_image(frame_points, scene_camera_pose)

        if len(projected) < 4:
            return image  # Not enough points visible

        # Extract 2D coordinates
        origin_2d = tuple(map(int, projected[0]))
        x_end_2d = tuple(map(int, projected[1]))
        y_end_2d = tuple(map(int, projected[2]))
        z_end_2d = tuple(map(int, projected[3]))

        # Draw axes with colors: X=red, Y=green, Z=blue
        cv2.line(image, origin_2d, x_end_2d, (0, 0, 255), thickness)  # X-axis (red)
        cv2.line(image, origin_2d, y_end_2d, (0, 255, 0), thickness)  # Y-axis (green)
        cv2.line(image, origin_2d, z_end_2d, (255, 0, 0), thickness)  # Z-axis (blue)

        # Add axis labels
        cv2.putText(image, "X", x_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Y", y_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "Z", z_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return image

    def render_scene(
        self,
        scene_camera_pose: CameraPose,
        camera_pose: CameraPose,
        landmarks: np.ndarray,
        landmark_ids: Optional[List[int]] = None,
        show_axes: bool = True,
        show_grid: bool = False,
    ) -> np.ndarray:
        """Render the complete 3D scene from the camera's viewpoint.

        Args:
            scene_camera_pose (CameraPose): Camera pose defining the viewpoint.
            camera_pose (CameraPose): Camera pose defining the camera position/orientation.
            landmarks (np.ndarray): Landmark positions in world coordinates, shape (N, 3).
            landmark_ids (Optional[List[int]]): Optional landmark IDs to display.
            show_axes (bool): Whether to show coordinate axes.
            show_grid (bool): Whether to show a reference grid (not implemented yet).

        Returns:
            np.ndarray: Rendered scene image.
        """
        # Create background image
        image = np.full(
            (self.image_height, self.image_width, 3),
            self.background_color,
            dtype=np.uint8,
        )

        # Draw coordinate axes
        if show_axes:
            image = self.draw_coordinate_axes(image, scene_camera_pose)

        # Draw landmarks
        image = self.draw_landmarks(
            image, landmarks, scene_camera_pose, landmark_ids=landmark_ids
        )

        # Draw camera pyramid (representing the viewing camera)
        image = self.draw_camera_pyramid(image, scene_camera_pose, camera_pose)

        # Add info text
        cv2.putText(
            image,
            f"Camera Position: ({camera_pose.position[0]:.2f}, {camera_pose.position[1]:.2f}, {camera_pose.position[2]:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        num_visible_landmarks = len(
            self.project_points_to_image(landmarks, camera_pose)
        )
        cv2.putText(
            image,
            f"Visible Landmarks: {num_visible_landmarks}/{len(landmarks)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return image

    def show_scene_static(
        self,
        scene_camera_pose: CameraPose,
        camera_pose: CameraPose,
        landmarks: np.ndarray,
        landmark_ids: Optional[List[int]] = None,
        show_axes: bool = True,
    ) -> np.ndarray:
        # Static mode: show single image
        image = self.render_scene(
            scene_camera_pose,
            camera_pose,
            landmarks,
            landmark_ids,
            show_axes,
        )

        return image

    def show_scene(
        self,
        scene_camera_pose: CameraPose,
        camera_pose: CameraPose,
        landmarks: np.ndarray,
        landmark_ids: Optional[List[int]] = None,
        window_name: str = "3D Scene Visualizer",
        show_axes: bool = True,
    ) -> None:
        """Render and display the 3D scene.

        When interactive mode is enabled:
        Mouse controls:
        - Click and drag: Rotate camera orientation

        Keyboard controls:
        - WASD: Move camera forward/backward/left/right (relative to camera view)
        - Q/E: Move camera up/down (world coordinates)
        - X/Y/Z: Move camera along world X/Y/Z axes
        - ESC: Exit interactive mode

        When interactive mode is disabled:
        - Press any key to exit the static view

        Args:
            scene_camera_pose (CameraPose): Camera pose defining the viewpoint.
            camera_pose (CameraPose): Camera pose defining the camera position/orientation to visualize.
            landmarks (np.ndarray): Landmark positions in world coordinates, shape (N, 3).
            landmark_ids (Optional[List[int]]): Optional landmark IDs to display.
            window_name (str): Name of the OpenCV window.
            show_axes (bool): Whether to show coordinate axes.
        """
        if self.interactive:
            # Interactive mode: allow camera manipulation
            self.current_scene_camera_pose = CameraPose(
                position=scene_camera_pose.position.copy(),
                orientation=scene_camera_pose.orientation_quaternion.copy(),
                timestamp=scene_camera_pose.timestamp,
            )

            # Create window and set mouse callback
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback)

            print("Interactive Camera Controls:")
            print("Mouse: Click and drag to rotate camera orientation")
            print("WASD: Move forward/backward/left/right (camera-relative)")
            print("Q/E: Move up/down (world coordinates)")
            print("X/Y/Z: Move along world X/Y/Z axes")
            print("ESC: Exit")

            while True:
                # Render current scene
                image = self.render_scene(
                    self.current_scene_camera_pose,
                    camera_pose,
                    landmarks,
                    landmark_ids,
                    show_axes,
                )

                # Add control instructions to the image
                instructions = [
                    "Controls: Mouse drag to rotate | WASD+QE to move | XYZ for axis move | ESC to exit",
                    f"Position: ({self.current_scene_camera_pose.position[0]:.2f}, {self.current_scene_camera_pose.position[1]:.2f}, {self.current_scene_camera_pose.position[2]:.2f})",
                ]

                for i, instruction in enumerate(instructions):
                    cv2.putText(
                        image,
                        instruction,
                        (10, image.shape[0] - 30 - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow(window_name, image)

                # Handle keyboard input
                key = cv2.waitKey(10) & 0xFF

                if key == 27:  # ESC key
                    break
                elif key != 255:  # Any other key
                    self._update_camera_position(key)

        else:
            # Static mode: show single image
            image = self.render_scene(
                scene_camera_pose,
                camera_pose,
                landmarks,
                landmark_ids,
                show_axes,
            )

            cv2.imshow(window_name, image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def save_scene(
        self,
        scene_camera_pose: CameraPose,
        camera_pose: CameraPose,
        landmarks: np.ndarray,
        filepath: str,
        landmark_ids: Optional[List[int]] = None,
        show_axes: bool = True,
    ) -> None:
        """Render and save the 3D scene to a file.

        Args:
            scene_camera_pose (CameraPose): Camera pose defining the viewpoint.
            camera_pose (CameraPose): Camera pose defining the camera position/orientation to visualize.
            landmarks (np.ndarray): Landmark positions in world coordinates, shape (N, 3).
            filepath (str): Path to save the image.
            landmark_ids (Optional[List[int]]): Optional landmark IDs to display.
            show_axes (bool): Whether to show coordinate axes.
        """
        image = self.render_scene(
            scene_camera_pose, camera_pose, landmarks, landmark_ids, show_axes
        )
        cv2.imwrite(filepath, image)
        print(f"Scene saved to: {filepath}")
