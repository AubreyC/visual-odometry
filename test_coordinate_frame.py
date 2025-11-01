#!/usr/bin/env python3
"""Test script for the new draw_coordinate_frame function."""

import numpy as np

from src.camera_pose import CameraPose
from src.visualization import OpenCVSceneVisualizer


def test_draw_coordinate_frame():
    """Test the draw_coordinate_frame function."""

    # Create visualizer
    visualizer = OpenCVSceneVisualizer(image_width=800, image_height=600)

    # Create scene camera pose (viewpoint)
    scene_camera = CameraPose(
        position=np.array([3.0, 3.0, 3.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity orientation
    )

    # Create test camera pose for coordinate frame
    frame_camera = CameraPose(
        position=np.array([0.0, 0.0, 0.0]),  # At origin
        orientation=np.array(
            [1.0, 0.0, 0.0, 0.0]
        ),  # Identity orientation (aligned with world axes)
    )

    # Create blank image
    image = np.full((600, 800, 3), (255, 255, 255), dtype=np.uint8)

    # Draw coordinate frame
    result_image = visualizer.draw_coordinate_frame(
        image=image,
        scene_camera_pose=scene_camera,
        camera_pose=frame_camera,
        axis_length=1.0,
        thickness=3,
    )

    print("Coordinate frame drawing test completed successfully!")
    print(f"Image shape: {result_image.shape}")

    # Save the test image
    import cv2

    cv2.imwrite("test_coordinate_frame.png", result_image)
    print("Test image saved as 'test_coordinate_frame.png'")


if __name__ == "__main__":
    test_draw_coordinate_frame()
