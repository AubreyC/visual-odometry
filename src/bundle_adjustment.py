from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .feature_observation import ImageFeatures, Points2D, Points3D
from .geometry import GeometryUtils
from .matplotlib_visualizer import MatplotVisualizer


class BundleAdjustment:
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
        # Notes:
        # n = Number of poses
        # m = Number of landmarks
        # params = [pose_0, ..., pose_n, landmark_0, ..., landmark_m]

        pose_size = 7  # 3 position + 4 quaternion

        # Extract poses and landmarks from params
        landmarks = params[num_poses * pose_size :]
        points_3d = Points3D(landmarks.reshape(-1, 3), landmark_ids)

        # Extract poses and landmarks from params
        residuals = []
        for pose_idx, image_features in enumerate[ImageFeatures](observations):
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

        result = np.array(residuals).flatten()
        return result

    def optimize(
        self,
        image_features_list: List[ImageFeatures],
        camera_pose_initial: CameraPose,
        camera_model: PinHoleCamera,
        camera_poses_guess: List[CameraPose],
        points3d_guess: Points3D,
    ) -> Tuple[List[ImageFeatures], Points3D, List[CameraPose]]:
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

        # # Collect all observations
        # # Assume image_features_list[i] corresponds to camera_poses[i]
        # observations = []
        # for pose_idx, image_features in enumerate(image_features_list):
        #     points_2d = image_features.get_points_2d().get_points_2d()
        #     obs_ids = image_features.get_points_2d().get_ids()

        #     for point_2d, landmark_id in zip(points_2d, obs_ids):
        #         # Find landmark index
        #         landmark_idx = np.where(landmark_ids == landmark_id)[0]
        #         if len(landmark_idx) == 0:
        #             continue  # Skip if landmark not found
        #         landmark_idx = landmark_idx[0]

        #         observations.append(
        #             {
        #                 "point_2d": point_2d,
        #                 "landmark_idx": landmark_idx,
        #                 "pose_idx": pose_idx,
        #             }
        #         )

        # if not observations:
        #     return image_features_list, points3d_initial

        # Create parameter vector: [poses_flat, landmarks_flat]
        # Each pose: 3 position + 4 quaternion = 7 parameters
        pose_params: List[float] = []
        n_poses = len(camera_poses_guess)
        for pose in camera_poses_guess:
            pose_params.extend(pose.position)
            pose_params.extend(pose.quaternion)

        landmark_params = landmark_positions.flatten()

        # Initial parameters: [poses_flat, landmarks_flat]
        initial_params = np.concatenate([pose_params, landmark_params])

        residuals = self.residual_function(
            initial_params,
            camera_pose_initial,
            n_poses,
            landmark_ids,
            image_features_list,
            camera_model,
        )
        print(f"Initial residual: {residuals}")
        return image_features_list, points3d_guess, camera_poses_guess

        # Perform optimization
        result = least_squares(
            self.residual_function,
            initial_params,
            method="lm",  # Levenberg-Marquardt
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

        print(f"Optimization converged: {result.success}")
        print(f"Final cost: {result.cost}")
        print(f"Number of function evaluations: {result.nfev}")

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
        optimized_points3d = Points3D(optimized_landmarks, landmark_ids)

        # Reproject to create optimized image features
        optimized_image_features = []
        for pose_idx, image_features in enumerate[ImageFeatures](image_features_list):
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
                    Points2D(np.array(projected_observed), np.array(obs_ids)),
                )
            else:
                # If no valid projections, keep original
                optimized_features = image_features

            optimized_image_features.append(optimized_features)

        return optimized_image_features, optimized_points3d, optimized_camera_poses


# Toy exmaple:
def run_bundle_adjustment() -> None:
    # Landmarks:
    landmarks: np.ndarray = np.array(
        [
            [3.0, 0.5, 0.1],
            [3.0, 0.4, 0.2],
            [3.0, -0.3, 0.4],
            [3.0, 0.3, 0.4],
            [3.0, 0.1, 0.0],
            [3.0, -0.2, -0.25],
            [3.0, -0.4, -0.5],
            [3.0, 0.5, -0.2],
        ]
    )

    # landmarks_generator: LandmarkGenerator = LandmarkGenerator(
    #     (1.8, 2.4), (-0.5, 0.5), (-0.5, 0.5)
    # )
    # landmarks: np.ndarray = landmarks_generator.generate_random(
    #     num_landmarks=100, seed=42
    # )

    # Camera poses:
    quat_cam = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )
    camera_poses = [
        CameraPose(
            position=np.array([-0.5, 0.1, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.02, 0.03, 0.0])),
            ),
            timestamp=0.0,
        ),
        CameraPose(
            position=np.array([0.0, -0.2, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.04, -0.1, 0.2])),
            ),
            timestamp=1.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.06, 0.02, 0.4])),
            ),
            timestamp=2.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, -0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.08, 0.3, 0.7])),
            ),
            timestamp=3.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.1, 0.05, -0.2])),
            ),
            timestamp=0.0,
        ),
        CameraPose(
            position=np.array([0.0, -0.2, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.12, 0.0, 0.0])),
            ),
            timestamp=1.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.14, 0.0, 0.0])),
            ),
            timestamp=2.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, -0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.16, 0.0, 0.0])),
            ),
            timestamp=3.0,
        ),
    ]

    # Camera model:
    camera_model = PinHoleCamera(
        fx=1000.0,
        fy=1000.0,
        cx=320.0,
        cy=240.0,
    )

    # Render images:
    camera_id = 0
    image_features = []
    for camera_pose in camera_poses:
        pts3d_camera_frame = camera_pose.transform_points_world_to_camera(landmarks)
        image_feature = ImageFeatures.from_points_3d(
            camera_pose.timestamp,
            camera_id,
            camera_model,
            pts3d_camera_frame,
            np.arange(len(landmarks)),
        )

        # Add noise to the image features
        # noise_pixel_std = 0
        # if noise_pixel_std > 0.0:
        #     image_feature.points_2d.points_2d = (
        #         image_feature.points_2d.points_2d
        #         + np.random.randn(len(image_feature.points_2d.get_points_2d()), 2)
        #         * noise_pixel_std
        #     )

        print("image_feature:", image_feature.get_points_2d().get_points_2d())
        image_features.append(image_feature)

    # Create initial 3D points from the known landmarks
    np.random.seed(42)  # For reproducible results
    landmarks_noisy = landmarks.copy()
    # landmarks_noisy += np.random.randn(len(landmarks), 3) * 1
    points3d_guess = Points3D(landmarks_noisy, np.arange(len(landmarks)))

    # Bundle adjustment:
    # Tranform guess in frame defined by the first camera pose
    camera_poses_first = camera_poses[0].copy()

    points3d_guess.points_3d = camera_poses_first.transform_points_world_to_camera(
        points3d_guess.get_points_3d()
    )

    print("points3d_guess:", points3d_guess.get_points_3d())
    print("landmakrs:", landmarks)

    camera_poses_guess = []
    for camera_pose in camera_poses:
        pose = CameraPose(
            camera_poses_first.transform_points_world_to_camera(camera_pose.position),
            GeometryUtils.quaternion_multiply(
                GeometryUtils.quaternion_inverse(camera_poses_first.quaternion),
                camera_pose.quaternion,
            ),
            camera_pose.timestamp,
        )

        camera_poses_guess.append(pose)

    image_features_bis = []
    for camera_pose in camera_poses_guess:
        pts3d_camera_frame = camera_pose.transform_points_world_to_camera(
            points3d_guess.get_points_3d()
        )
        image_feature = ImageFeatures.from_points_3d(
            camera_pose.timestamp,
            camera_id,
            camera_model,
            pts3d_camera_frame,
            np.arange(len(landmarks)),
        )

        # Add noise to the image features
        # noise_pixel_std = 0
        # if noise_pixel_std > 0.0:
        #     image_feature.points_2d.points_2d = (
        #         image_feature.points_2d.points_2d
        #         + np.random.randn(len(image_feature.points_2d.get_points_2d()), 2)
        #         * noise_pixel_std
        #     )

        image_features_bis.append(image_feature)
        print("image_feature_bis:", image_feature.get_points_2d().get_points_2d())

    camera_pose_initial = camera_poses_guess[0].copy()
    camera_poses_guess = camera_poses_guess[1:].copy()

    bundle_adjustment = BundleAdjustment()
    optimized_image_features, optimized_points3d, optimized_camera_poses = (
        bundle_adjustment.optimize(
            image_features,
            camera_pose_initial,
            camera_model,
            camera_poses_guess,
            points3d_guess,
        )
    )

    print("Original landmarks:")
    print(landmarks)
    # print("Noisy initial landmarks:")
    # print(landmarks_noisy)
    # print("Optimized landmarks:")
    # print(optimized_points3d.get_points_3d())

    # Calculate errors
    landmark_error = np.linalg.norm(
        optimized_points3d.get_points_3d() - landmarks, axis=1
    )
    print(f"Mean landmark error: {np.mean(landmark_error):.6f}")
    print(f"Max landmark error: {np.max(landmark_error):.6f}")

    # scale = np.linalg.norm(
    #     landmarks[0, :] / np.linalg.norm(optimized_points3d.get_points_3d()[0, :])
    # )

    # Rescale solution:
    # optimized_points3d.points_3d = optimized_points3d.get_points_3d() * scale
    # for camera in optimized_camera_poses[1:]:
    #     camera.position = camera.position * scale

    print("points3d_guess landmarks scaled:")
    print(points3d_guess.get_points_3d())

    print("Optimized landmarks scaled:")
    print(optimized_points3d.get_points_3d())

    # Create matplotlib visualizer for 3D plots
    visualizer: MatplotVisualizer = MatplotVisualizer()
    fig = visualizer.plot_scene_overview(
        landmarks=points3d_guess.get_points_3d(),
        landmarks_second=optimized_points3d.get_points_3d(),
        poses=camera_poses_guess,
        poses_second=optimized_camera_poses,
        title="3D Scene",
        show_orientation=True,
        orientation_scale=0.5,
    )
    visualizer.show()


if __name__ == "__main__":
    run_bundle_adjustment()
