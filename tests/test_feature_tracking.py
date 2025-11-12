#!/usr/bin/env python3
"""Pytest tests for feature detection and tracking."""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.feature_tracker as feature_tracker

FeatureTracker = feature_tracker.FeatureTracker


@pytest.fixture
def tracker() -> FeatureTracker:
    """Fixture providing a FeatureTracker instance."""
    return FeatureTracker()


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a simple test image with some features."""
    # Create a blank image
    image = np.zeros((height, width), dtype=np.uint8)

    # Add some geometric shapes to create trackable features
    # Rectangle
    cv2.rectangle(image, (100, 100), (200, 200), 255, 2)

    # Circle
    cv2.circle(image, (400, 150), 50, 255, 2)

    # Lines
    cv2.line(image, (50, 300), (200, 350), 255, 2)
    cv2.line(image, (400, 300), (550, 350), 255, 2)

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    return image


def create_moving_image(
    base_image: np.ndarray, translation: Tuple[int, int] = (5, 2)
) -> np.ndarray:
    """Create a slightly moved version of the base image."""
    # Apply small translation
    rows, cols = base_image.shape
    transformation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    moved_image = cv2.warpAffine(base_image, transformation_matrix, (cols, rows))

    # Add small amount of noise
    noise = np.random.normal(0, 2, moved_image.shape).astype(np.uint8)
    moved_image = cv2.add(moved_image, noise)

    return moved_image


def test_feature_detection(tracker: FeatureTracker) -> None:
    """Test feature detection algorithms."""
    print("Testing Feature Detection")
    print("=" * 40)

    # Create test image
    image = create_test_image()
    print(f"Created test image with shape: {image.shape}")

    # Test ORB detection
    print("\nTesting ORB feature detection...")
    orb_points, orb_descriptors = tracker.detect_orb_features(image)
    print(f"  ✓ ORB detected {len(orb_points)} features")
    print(f"  ✓ ORB descriptors shape: {orb_descriptors.shape}")
    assert len(orb_points) > 0, "ORB should detect some features"
    assert orb_descriptors.shape[0] == len(orb_points), (
        "Descriptors should match number of points"
    )

    # Test Shi-Tomasi detection
    print("\nTesting Shi-Tomasi feature detection...")
    shi_points = tracker.detect_shi_tomasi_features(image)
    print(f"  ✓ Shi-Tomasi detected {len(shi_points)} features")
    assert len(shi_points) > 0, "Shi-Tomasi should detect some features"


def test_feature_tracking(tracker: FeatureTracker) -> None:
    """Test feature tracking with KLT."""
    print("\n\nTesting Feature Tracking (KLT)")
    print("=" * 40)

    # Create test images
    image1 = create_test_image()
    image2 = create_moving_image(image1, translation=(3, 1))

    # Detect features in first image
    print("Detecting features in first image...")
    features1 = tracker.detect_shi_tomasi_features(image1)
    print(f"  ✓ Detected {len(features1)} features in first image")

    assert len(features1) > 0, "Features should be detected for tracking test"

    # Track features to second image
    print("Tracking features to second image...")
    features2, status, err = tracker.track_features_klt(image1, image2, features1)
    successful_tracks: int = np.sum(status)
    print(f"  ✓ Successfully tracked {successful_tracks}/{len(features1)} features")
    print(f"  ✓ Tracking errors range: [{err.min():.3f}, {err.max():.3f}]")

    assert len(features2) == len(features1), (
        "Output should have same number of points as input"
    )
    assert len(status) == len(features1), "Status should match input points"
    assert len(err) == len(features1), "Error should match input points"

    # Some features should be successfully tracked
    assert successful_tracks > 0, (
        "At least some features should be tracked successfully"
    )


def test_feature_filtering(tracker: FeatureTracker) -> None:
    """Test feature filtering functions."""
    print("\n\nTesting Feature Filtering")
    print("=" * 40)

    # Create some test features
    features = np.array(
        [
            [10, 10],  # Too close to boundary
            [50, 50],  # Good
            [100, 100],  # Good
            [600, 400],  # Too close to boundary
            [320, 240],  # Center, good
        ],
        dtype=np.float32,
    )

    print(f"Original features: {len(features)}")

    # Test boundary filtering
    print("Testing boundary filtering...")
    filtered_boundary, boundary_mask = tracker.filter_features_by_region_with_mask(
        features, image_width=640, image_height=480, margin=20
    )
    print(f"  ✓ After boundary filtering: {len(filtered_boundary)} features")

    # Test distance filtering
    print("Testing distance filtering...")
    filtered_distance, distance_mask = tracker.filter_features_by_distance_with_mask(
        filtered_boundary, min_distance=30.0
    )
    print(f"  ✓ After distance filtering: {len(filtered_distance)} features")
