#!/usr/bin/env python3
"""Test script for the OpenCV Scene Visualizer."""

import numpy as np
from src.camera_pose import CameraPose
from src.geometry import GeometryUtils
from src.visualization import OpenCVSceneVisualizer


def create_test_scene():
    """Create a test scene with landmarks and camera poses."""
    # Create some test landmarks
    landmarks = np.array([
        [2.0, 0.0, 0.0],   # Landmark at origin +X
        [0.0, 2.0, 0.0],   # Landmark at origin +Y
        [0.0, 0.0, 2.0],   # Landmark at origin +Z
        [1.0, 1.0, 1.0],   # Landmark in first octant
        [-1.0, 1.0, 0.5],  # Landmark in different position
        [0.5, -1.5, 1.5],  # Another landmark
    ])

    landmark_ids = list(range(len(landmarks)))

    # Create camera poses looking at the scene from different angles
    poses = []

    # Camera 1: Looking along X-axis towards origin
    quat1 = GeometryUtils.quaternion_from_euler_angles(
        np.array([0.0, np.pi/2, -np.pi/2])  # Rotate to look along +X
    )
    pose1 = CameraPose(
        position=np.array([-3.0, 0.0, 1.0]),
        orientation=quat1,
        timestamp=0.0
    )
    poses.append(pose1)

    # Camera 2: Looking from above
    quat2 = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi/2, 0.0, 0.0])  # Look down along -Z
    )
    pose2 = CameraPose(
        position=np.array([0.0, 0.0, 4.0]),
        orientation=quat2,
        timestamp=1.0
    )
    poses.append(pose2)

    # Camera 3: Angled view
    quat3 = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi/4, np.pi/4, -np.pi/3])
    )
    pose3 = CameraPose(
        position=np.array([-2.0, -2.0, 2.0]),
        orientation=quat3,
        timestamp=2.0
    )
    poses.append(pose3)

    return landmarks, landmark_ids, poses


def main():
    """Main function to test the OpenCV scene visualizer."""
    print("Testing OpenCV Scene Visualizer...")

    # Create test scene
    landmarks, landmark_ids, poses = create_test_scene()

    # Create visualizer
    visualizer = OpenCVSceneVisualizer(
        image_width=1280,
        image_height=720,
        background_color=(30, 30, 30)
    )

    # Test each camera pose
    for i, pose in enumerate(poses):
        print(f"\nRendering scene from camera pose {i+1}...")
        print(".2f")

        # Show the scene
        visualizer.show_scene(
            camera_pose=pose,
            landmarks=landmarks,
            landmark_ids=landmark_ids,
            window_name=f"3D Scene - Camera {i+1}",
            show_axes=True
        )

        # Save the scene
        filename = f"scene_camera_{i+1}.png"
        visualizer.save_scene(
            camera_pose=pose,
            landmarks=landmarks,
            filepath=filename,
            landmark_ids=landmark_ids,
            show_axes=True
        )

    print("\nTest completed! Scene images saved as PNG files.")


if __name__ == "__main__":
    main()
