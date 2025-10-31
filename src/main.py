import sys

import numpy as np

from .camera_pose import TrajectoryGenerator
from .landmarks import LandmarkGenerator
from .visualization import Visualizer


def main(args_input: list[str]) -> None:
    trajectory_generator = TrajectoryGenerator(time_step=0.1)
    poses = trajectory_generator.generate_circular_trajectory(
        center=np.array([0.0, 0.0]), radius=2.0, height=1.0, num_poses=20
    )

    landmarks_generator = LandmarkGenerator()
    landmarks = landmarks_generator.generate_random(num_landmarks=50, seed=42)

    visualizer = Visualizer(figsize=(10, 8))

    visualizer.plot_scene_overview(landmarks, poses, title="3D Trajectory Test")
    visualizer.show()


if __name__ == "__main__":
    main(sys.argv[1:])
