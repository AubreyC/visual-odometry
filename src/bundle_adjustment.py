from typing import List, Tuple

import numpy as np

from .camera_pose import CameraPose
from .feature_observation import ImageFeatures, Points3D
from .geometry_utils import GeometryUtils


class BundleAdjustment:
    def __init__(self) -> None:
        pass

    def optimize(
        self, image_features_list: List[ImageFeatures]
    ) -> Tuple[List[ImageFeatures], Points3D]:
        points3d = Points3D(np.empty((0, 3)), np.empty((0,), dtype=int))

        return image_features_list, points3d


# Toy exmaple:
def run_bundle_adjustment() -> None:
    # Landmarks:
    landmarks: np.ndarray = np.array(
        [
            [2.0, 0.5, 0.1],
            [2.0, 0.4, 0.2],
            [2.0, -0.3, 0.4],
            [2.0, 0.3, 0.4],
            [2.0, 0.1, 0.0],
            [2.0, -0.2, -0.25],
            [2.0, -0.4, -0.5],
            [2.0, 0.5, -0.2],
        ]
    )
    # Camera poses:
    quat_cam = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )
    camera_poses = [
        CameraPose(
            position=np.array([0.1, 0.1, 0.0]),
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

    # Ima

    # Bundle adjustment:
    bundle_adjustment = BundleAdjustment()
    bundle_adjustment.optimize(image_features)


if __name__ == "__main__":
    run_bundle_adjustment()
