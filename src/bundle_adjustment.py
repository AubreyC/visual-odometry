from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .feature_observation import Features2D, ImageFeatures, Landmarks3D


class BundleAdjustment:
    """Bundle adjustment class for 3D landmarks and camera poses."""

    def __init__(self) -> None:
        pass

    # Define the residual function
    @staticmethod
    def residual_function(
        params: np.ndarray,
        camera_pose_initial: CameraPose,
        num_poses: int,
        landmark_ids: np.ndarray,
        observations: List[ImageFeatures],
        camera_model: PinHoleCamera,
    ) -> np.ndarray:
        """Residual function for bundle adjustment.

        Args:
            params (np.ndarray): Parameters to optimize: [poses_flat, landmarks_flat] with poses_flat = [pose_0, ..., pose_n] (pose_i = 3 position + 4 quaternion) and landmarks_flat = [landmark_0, ..., landmark_m] (landmark_i = [x, y, z] 3D position).
            camera_pose_initial (CameraPose): Initial camera pose (first camera pose not optimized).
            num_poses (int): Number of poses to optimize.
            landmark_ids (np.ndarray): Landmark IDs.
            observations (List[ImageFeatures]): Observations (image features) size num_poses + 1 (first camera pose not optimized).
            camera_model (PinHoleCamera): Camera model.

        Returns:
            np.ndarray: Residuals (flattened) of size 2 * num_observations [u_1 - u_1_proj, v_1 - v_1_proj, ..., u_n - u_n_proj, v_n - v_n_proj] with n = num_observations.
        """
        # 3 position + 4 quaternion
        pose_size = 7

        # Extract poses and landmarks from params
        landmarks = params[num_poses * pose_size :]
        points_3d = Landmarks3D(landmarks.reshape(-1, 3), landmark_ids)

        # Extract poses and landmarks from params
        residuals = []
        for pose_idx, image_features in enumerate(observations):
            if pose_idx == 0:
                camera_pose = camera_pose_initial
            else:
                # Extract pose parameters
                pose_start = (pose_idx - 1) * pose_size
                pose_end = pose_start + pose_size
                pose_data = params[pose_start:pose_end]

                position = pose_data[:3]
                quaternion = pose_data[3:7]

                # Normalize quaternion to ensure it's valid
                quaternion = quaternion / np.linalg.norm(quaternion)

                camera_pose = CameraPose(position, quaternion, pose_idx)

            # Extract landmarks
            points_3d_selected = points_3d.get_selected_ids(
                image_features.get_points_2d().get_ids()
            )

            landmark_cf = camera_pose.transform_points_world_to_camera(
                points_3d_selected.get_points_3d()
            )

            # Project to image plane
            projected = camera_model.project(landmark_cf)
            projected = projected.flatten()

            residual = (
                image_features.get_points_2d().get_points_2d().flatten()
                - projected.flatten()
            )
            residuals.append(residual)

        result: np.ndarray = np.array(residuals).flatten()
        return result

    @staticmethod
    def bundle_adjustment_sparsity(
        num_poses: int, num_landmarks: int, observations: List[ImageFeatures]
    ) -> lil_matrix:
        """
        Calculate the sparsity pattern for the bundle adjustment problem.
        It assumes state = [poses_flat, landmarks_flat] with poses_flat = [pose_0, ..., pose_n]
        (pose_i = 3 position + 4 quaternion) and landmarks_flat = [landmark_0, ..., landmark_m]
        (landmark_i = [x, y, z] 3D position).

        Args:
            num_poses (int): Number of poses to optimize.
            num_landmarks (int): Number of landmarks.
            observations (List[ImageFeatures]): List of observations for each camera pose.

        Returns:
            scipy.sparse.lil_matrix: Sparsity pattern for the bundle adjustment problem.
        """
        # Count total number of observations (features)
        total_observations = sum(
            len(obs.get_points_2d().get_points_2d()) for obs in observations
        )

        # Parameters: num_poses * 7 (3 pos + 4 quat) + num_landmarks * 3
        m = total_observations * 2  # 2 residuals per observation (u, v)
        n = num_poses * 7 + num_landmarks * 3
        A = lil_matrix((m, n), dtype=int)

        # Build landmark ID to index mapping
        landmark_id_to_idx = {}
        all_landmark_ids: set[int] = set()
        for obs in observations:
            all_landmark_ids.update(obs.get_points_2d().get_ids())
        for idx, landmark_id in enumerate(sorted(all_landmark_ids)):
            landmark_id_to_idx[landmark_id] = idx

        # Fill sparsity pattern
        residual_idx = 0
        for pose_idx, image_features in enumerate(observations):
            observed_ids = image_features.get_points_2d().get_ids()
            observed_points = image_features.get_points_2d().get_points_2d()

            for feature_idx in range(len(observed_points)):
                landmark_id = observed_ids[feature_idx]
                landmark_param_idx = landmark_id_to_idx[landmark_id]

                # Each observation creates 2 residuals (u, v)
                # Residuals depend on the camera pose and the landmark
                for residual_offset in range(2):  # u and v residuals
                    current_residual_idx = residual_idx + residual_offset

                    # Camera pose parameters (7 parameters: 3 position + 4 quaternion)
                    if pose_idx > 0:  # First pose (pose_idx=0) is fixed, not optimized
                        pose_param_start = (pose_idx - 1) * 7
                        for param_offset in range(7):
                            A[current_residual_idx, pose_param_start + param_offset] = 1

                    # Landmark parameters (3 parameters: x, y, z)
                    landmark_param_start = num_poses * 7 + landmark_param_idx * 3
                    for param_offset in range(3):
                        A[current_residual_idx, landmark_param_start + param_offset] = 1

                residual_idx += 2

        print(A.toarray())
        return A

    def optimize(
        self,
        image_features_list: List[ImageFeatures],
        camera_pose_initial: CameraPose,
        camera_model: PinHoleCamera,
        camera_poses_guess: List[CameraPose],
        points3d_guess: Landmarks3D,
    ) -> Tuple[List[ImageFeatures], Landmarks3D, List[CameraPose]]:
        """
        Perform bundle adjustment optimization using scipy.optimize.least_squares.

        Args:
            image_features_list: List of ImageFeatures containing 2D observations
            camera_poses: List of CameraPose objects for each camera
            camera_intrinsics: PinHoleCamera with intrinsic parameters
            initial_points3d: Initial 3D landmark positions

        Returns:
            Tuple of (optimized_image_features, optimized_points3d)
        """

        # Extract data for optimization
        landmark_ids = points3d_guess.get_ids()
        landmark_positions = points3d_guess.get_points_3d()

        # Create parameter vector: [poses_flat, landmarks_flat]
        # Each pose: 3 position + 4 quaternion = 7 parameters
        pose_params: List[float] = []
        n_poses = len(camera_poses_guess)
        for pose in camera_poses_guess:
            pose_params.extend(pose.position)
            pose_params.extend(pose.quaternion)
        landmark_params = landmark_positions.flatten()

        image_width = image_features_list[0].image_width
        image_height = image_features_list[0].image_height

        # Initial parameters: [poses_flat, landmarks_flat]
        initial_params = np.concatenate([pose_params, landmark_params])

        jacobian_sparsity = self.bundle_adjustment_sparsity(
            n_poses, len(landmark_ids), image_features_list
        )

        # Perform optimization
        result = least_squares(
            self.residual_function,
            initial_params,
            jac_sparsity=jacobian_sparsity,
            method="trf",
            x_scale="jac",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=2000,  # Increase max iterations
            verbose=0,
            args=(
                camera_pose_initial,
                n_poses,
                landmark_ids,
                image_features_list,
                camera_model,
            ),
        )

        # Extract optimized parameters
        optimized_params = result.x
        pose_size = 7

        # Update camera poses
        optimized_camera_poses = [camera_pose_initial]
        for i in range(n_poses):
            pose_start = i * pose_size
            pose_end = pose_start + pose_size
            pose_data = optimized_params[pose_start:pose_end]

            position = pose_data[:3]
            quaternion = pose_data[3:7]
            quaternion = quaternion / np.linalg.norm(quaternion)  # Normalize

            # Keep original timestamp
            original_pose = camera_poses_guess[i]
            optimized_pose = CameraPose(position, quaternion, original_pose.timestamp)
            optimized_camera_poses.append(optimized_pose)

        # Update landmark positions
        landmark_start = n_poses * pose_size
        optimized_landmarks = optimized_params[landmark_start:].reshape(-1, 3)
        optimized_points3d = Landmarks3D(optimized_landmarks, landmark_ids)

        # Reproject to create optimized image features
        optimized_image_features = []
        for pose_idx, image_features in enumerate(image_features_list):
            timestamp = image_features.get_timestamp()
            camera_id = image_features.get_camera_id()
            pose = optimized_camera_poses[pose_idx]

            # Get observed landmark IDs for this image
            obs_ids = image_features.get_points_2d().get_ids()

            # Transform and project landmarks
            landmark_positions_opt = optimized_points3d.get_points_3d()
            landmark_camera_coords = pose.transform_points_world_to_camera(
                landmark_positions_opt
            )
            projected_points = camera_model.project(landmark_camera_coords)

            # Create new ImageFeatures with optimized projections
            # Only include landmarks that were observed in this image
            observed_indices = []
            projected_observed = []
            for i, landmark_id in enumerate(optimized_points3d.get_ids()):
                if landmark_id in obs_ids:
                    observed_indices.append(i)
                    projected_observed.append(projected_points[i])

            if projected_observed:
                # Create ImageFeatures with the projected 2D points directly
                # since we already have the projected points
                optimized_features = ImageFeatures(
                    timestamp,
                    camera_id,
                    image_width,
                    image_height,
                    Features2D(np.array(projected_observed), np.array(obs_ids)),
                )
            else:
                # If no valid projections, keep original
                optimized_features = image_features

            optimized_image_features.append(optimized_features)

        return optimized_image_features, optimized_points3d, optimized_camera_poses
