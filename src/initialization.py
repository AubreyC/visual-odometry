"""Initialization utilities for SLAM: pose initialization and landmark triangulation."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import Tracks
from .geometry import se3_exp, se3_inverse
from .validation_error import ValidationError


def initialize_poses_with_noise(
    gt_poses: List[np.ndarray],
    translation_noise: float = 0.1,
    rotation_noise_deg: float = 2.0,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Initialize poses by adding noise to ground truth poses.

    Args:
        gt_poses (List[np.ndarray]): Ground truth SE(3) poses.
        translation_noise (float): Translation noise standard deviation (meters).
        rotation_noise_deg (float): Rotation noise standard deviation (degrees).
        seed (Optional[int]): Random seed.

    Returns:
        List[np.ndarray]: Initialized poses with noise.
    """
    if seed is not None:
        np.random.seed(seed)

    initialized_poses = []
    rotation_noise_rad = np.deg2rad(rotation_noise_deg)

    for T_gt in gt_poses:
        # Add translation noise
        t_noise = np.random.normal(0, translation_noise, 3)
        T_init = T_gt.copy()
        T_init[:3, 3] += t_noise

        # Add rotation noise (small angle approximation)
        omega_noise = np.random.normal(0, rotation_noise_rad, 3)
        xi_noise = np.concatenate([omega_noise, np.zeros(3)])
        T_rot_noise = se3_exp(xi_noise)

        T_init = T_rot_noise @ T_init
        initialized_poses.append(T_init)

    return initialized_poses


def triangulate_landmark(
    observations: List[Tuple[np.ndarray, np.ndarray]], K: np.ndarray
) -> Optional[np.ndarray]:
    """Triangulate 3D landmark position from multiple 2D observations.

    Args:
        observations (List[Tuple[np.ndarray, np.ndarray]]):
            List of (T_w_c, measurement) pairs, where measurement is [u, v].
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        Optional[np.ndarray]: Triangulated 3D point in world coordinates, or None if failed.

    Raises:
        ValidationError: If inputs are invalid.
    """
    if not isinstance(K, np.ndarray) or K.shape != (3, 3):
        raise ValidationError(f"K must be 3x3 matrix, got shape {K.shape}")
    if len(observations) < 2:
        raise ValidationError(f"Need at least 2 observations, got {len(observations)}")

    from .image_renderer import project_point

    # Use iterative least squares triangulation for better accuracy
    # Initialize with a simple midpoint estimate
    if len(observations) == 2:
        # For two views, use simple midpoint triangulation
        T1, m1 = observations[0]
        T2, m2 = observations[1]

        # Get camera centers
        c1 = -T1[:3, :3].T @ T1[:3, 3]
        c2 = -T2[:3, :3].T @ T2[:3, 3]

        # Use the midpoint as initial guess
        X_init = (c1 + c2) / 2.0
    else:
        # For multiple views, use the centroid of camera centers
        centers = []
        for T_w_c, _ in observations:
            c = -T_w_c[:3, :3].T @ T_w_c[:3, 3]
            centers.append(c)
        X_init = np.mean(centers, axis=0)

    # Iterative refinement using Gauss-Newton
    X = X_init.copy()
    max_iterations = 10
    tolerance = 1e-6

    for _ in range(max_iterations):
        A = []
        b = []

        for T_w_c, measurement in observations:
            # Project current estimate
            projected = project_point(K, T_w_c, X)
            residual = projected - measurement

            # Compute Jacobian (projection derivatives)
            # For simplicity, use finite differences
            eps = 1e-6
            J = []
            for i in range(3):
                X_pert = X.copy()
                X_pert[i] += eps
                proj_pert = project_point(K, T_w_c, X_pert)
                J.append((proj_pert - projected) / eps)
            J = np.array(J).T  # 2x3 Jacobian

            A.append(J)
            b.append(-residual)

        if not A:
            return None

        # Build normal equations
        A_mat = np.vstack(A)
        b_vec = np.concatenate(b)

        try:
            # Solve for update
            delta = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]

            # Update estimate
            X_new = X + delta

            # Check convergence
            if np.linalg.norm(delta) < tolerance:
                X = X_new
                break

            X = X_new

        except np.linalg.LinAlgError:
            return None

    # Final validation: check if point is in front of all cameras
    for T_w_c, _ in observations:
        T_c_w = se3_inverse(T_w_c)
        R_c_w = T_c_w[:3, :3]
        t_c_w = T_c_w[:3, 3]
        X_c = R_c_w @ X + t_c_w
        if X_c[2] <= 0.1:  # Require some minimum depth
            return None

    return X


def triangulate_landmarks_from_tracks(
    tracks: Tracks,
    poses: List[np.ndarray],
    K: np.ndarray,
    min_track_length: int = 2,
    max_track_length: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Triangulate landmarks from feature tracks.

    Args:
        tracks (Tracks): Feature tracks.
        poses (List[np.ndarray]): Camera poses T_w_c.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        min_track_length (int): Minimum track length for triangulation.
        max_track_length (Optional[int]): Maximum track length to use.

    Returns:
        Dict[int, np.ndarray]: Landmark ID to 3D position mapping.
    """
    triangulated_landmarks = {}

    for track in tracks.iter_tracks(min_track_length):
        if max_track_length and track.length() > max_track_length:
            # Use only first max_track_length observations
            observations = track.observations[:max_track_length]
        else:
            observations = track.observations

        # Build observation list
        obs_list = []
        for frame_id, u, v in observations:
            if frame_id >= len(poses):
                continue
            measurement = np.array([u, v])
            obs_list.append((poses[frame_id], measurement))

        if len(obs_list) >= 2:
            X = triangulate_landmark(obs_list, K)
            if X is not None:
                triangulated_landmarks[track.landmark_id] = X

    return triangulated_landmarks


def initialize_slam_state(
    gt_poses: List[np.ndarray],
    tracks: Tracks,
    K: np.ndarray,
    pose_noise_translation: float = 0.1,
    pose_noise_rotation_deg: float = 2.0,
    min_track_length: int = 2,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """Initialize complete SLAM state: poses and landmarks.

    Args:
        gt_poses (List[np.ndarray]): Ground truth camera poses.
        tracks (Tracks): Feature tracks with measurements.
        K (np.ndarray): Camera intrinsic matrix.
        pose_noise_translation (float): Translation noise for pose initialization.
        pose_noise_rotation_deg (float): Rotation noise for pose initialization.
        min_track_length (int): Minimum track length for triangulation.
        seed (Optional[int]): Random seed.

    Returns:
        Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
            (initialized_poses, triangulated_landmarks).
    """
    # Initialize poses with noise
    init_poses = initialize_poses_with_noise(
        gt_poses, pose_noise_translation, pose_noise_rotation_deg, seed
    )

    # Triangulate landmarks using ground truth poses (for stability)
    # In real SLAM, you might use estimated poses, but GT gives better initialization
    triangulated_landmarks = triangulate_landmarks_from_tracks(
        tracks, gt_poses, K, min_track_length
    )

    return init_poses, triangulated_landmarks


def refine_initialization_with_ba(
    init_poses: List[np.ndarray],
    init_landmarks: Dict[int, np.ndarray],
    tracks: Tracks,
    K: np.ndarray,
    max_iterations: int = 10,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """Refine initialization with a few bundle adjustment iterations.

    Args:
        init_poses (List[np.ndarray]): Initial poses.
        init_landmarks (Dict[int, np.ndarray]): Initial landmarks.
        tracks (Tracks): Feature tracks.
        K (np.ndarray): Camera intrinsics.
        max_iterations (int): Maximum BA iterations.

    Returns:
        Tuple[List[np.ndarray], Dict[int, np.ndarray]]: Refined poses and landmarks.
    """
    # This would call a simplified BA, but for now just return the initialization
    # The full BA implementation comes later
    return init_poses, init_landmarks
