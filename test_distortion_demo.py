#!/usr/bin/env python3
"""
Demo script showing radial distortion functionality in PinHoleCamera.
"""

import numpy as np
from src.camera import PinHoleCamera

def main():
    print("Radial Distortion Demo")
    print("=" * 40)

    # Create camera with barrel distortion (k1 > 0)
    camera_distorted = PinHoleCamera(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        k1=0.1, k2=0.01, k3=0.001
    )

    # Create camera without distortion for comparison
    camera_no_distortion = PinHoleCamera(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0
    )

    # Test points at different distances from center
    test_points_3d = np.array([
        [1.0, 0.0, 1.0],    # Point on x-axis at distance 1
        [0.0, 1.0, 1.0],    # Point on y-axis at distance 1
        [0.707, 0.707, 1.0], # Point at 45 degrees, distance ~1
        [0.0, 0.0, 1.0],    # Point at optical center
    ])

    print("3D Test Points:")
    for i, point in enumerate(test_points_3d):
        distance_from_center = np.sqrt(point[0]**2 + point[1]**2)
        print(f"  Point {i+1}: {point} (distance from center: {distance_from_center:.3f})")

    print("\nProjection Results:")
    print("-" * 60)

    projected_no_distortion = camera_no_distortion.project(test_points_3d)
    projected_distortion = camera_distorted.project(test_points_3d)

    for i in range(len(test_points_3d)):
        u_no_dist, v_no_dist = projected_no_distortion[i]
        u_dist, v_dist = projected_distortion[i]

        distance_no_dist = np.sqrt((u_no_dist - 320.0)**2 + (v_no_dist - 240.0)**2)
        distance_dist = np.sqrt((u_dist - 320.0)**2 + (v_dist - 240.0)**2)

        print(f"Point {i+1}:")
        print(".3f")
        print(".3f")
        print(".3f")
        print()

    # Test undistortion
    print("Undistortion Test:")
    print("-" * 40)

    # Take the distorted points and undistort them
    undistorted = camera_distorted.undistort(projected_distortion)

    print("Original undistorted points vs undistorted distorted points:")
    for i in range(len(test_points_3d)):
        orig_u, orig_v = projected_no_distortion[i]
        undist_u, undist_v = undistorted[i]

        print(f"Point {i+1}:")
        print(".3f")
        print()

if __name__ == "__main__":
    main()

