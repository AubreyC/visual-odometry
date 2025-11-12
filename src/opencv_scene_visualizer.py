from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .geometry import GeometryUtils


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
        axis_length: float = 5.0,
        thickness: int = 1,
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
        cross_size: int = 5,
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
        cv2.circle(image, camera_pos_2d, 4, color, -1)  # Filled circle
        cv2.circle(image, camera_pos_2d, 4, (255, 255, 255), 2)  # White border

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

    @staticmethod
    def merge_images(
        img_list: List[np.ndarray], interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """Merge a list of images into a single image.

        Args:
            img_list (List[np.ndarray]): List of images to merge.
            interpolation (int): Interpolation method.

        Returns:
            np.ndarray: Merged image.
        """

        # take minimum hights
        h_min = min(img.shape[0] for img in img_list)

        # image resizing
        im_list_resize = [
            cv2.resize(
                img,
                (int(img.shape[1] * h_min / img.shape[0]), h_min),
                interpolation=interpolation,
            )
            for img in img_list
        ]

        # return final image
        return cv2.hconcat(im_list_resize)
