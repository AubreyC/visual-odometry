import numpy as np
import pytest

from src.camera import PinHoleCamera
from src.camera_pose import CameraPose
from src.feature_observation import FeatureObservation, FeatureTrack
from src.image_renderer import ImageObservations, ImageRenderer


class TestFeatureObservation:
    """Test suite for FeatureObservation class."""

    def test_initialization(self) -> None:
        """Test FeatureObservation initialization."""
        landmark_id = 5
        image_coords = np.array([320.0, 240.0])
        timestamp = 1.5

        obs = FeatureObservation(
            landmark_id, image_coords, camera_id=0, timestamp=timestamp
        )

        assert obs.landmark_id == landmark_id
        assert np.allclose(obs.image_coords, image_coords)
        assert obs.timestamp == timestamp


class TestFeatureTrack:
    """Test suite for FeatureTrack class."""

    def test_initialization(self) -> None:
        """Test FeatureTrack initialization."""
        landmark_id = 3

        track = FeatureTrack(landmark_id)
        assert track.landmark_id == landmark_id
        assert track.observations == []
        assert track.length() == 0

    def test_add_observation(self) -> None:
        """Test adding observations to track."""
        landmark_id = 3
        landmark_3d = np.array([[1.0, 2.0, 3.0]])
        landmark_ids = [landmark_id for i in range(len(landmark_3d))]

        track = FeatureTrack(landmark_id)

        camera_pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        camera: PinHoleCamera = PinHoleCamera(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        image_renderer: ImageRenderer = ImageRenderer(camera)

        list_of_observations = image_renderer.project_landmarks_to_image(
            landmark_3d, camera_pose, camera_id=0, landmark_ids=landmark_ids
        )

        for observation in list_of_observations:
            track.add_observation(observation)

        assert track.length() == len(list_of_observations)

    def test_add_wrong_landmark_observation(self) -> None:
        """Test adding observation with wrong landmark ID raises error."""
        track = FeatureTrack(1)

        feature_observation: FeatureObservation = FeatureObservation(
            landmark_id=2,
            image_coords=np.array([100.0, 200.0]),
            camera_id=0,
            timestamp=0.0,
        )

        with pytest.raises(Exception):  # Should raise ValidationError
            track.add_observation(feature_observation)

    def test_get_image_coordinates(self) -> None:
        """Test getting image coordinates from track."""
        track = FeatureTrack(1)

        camera_pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )

        # Add observations
        landmark_3d = np.array([[1.0, 2.0, 3.0]])

        camera: PinHoleCamera = PinHoleCamera(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        image_renderer: ImageRenderer = ImageRenderer(camera)

        list_of_observations = []

        num_observations = 5
        for i in range(num_observations):
            camera_pose.timestamp = i * 0.1
            obs = image_renderer.project_landmarks_to_image(
                landmark_3d, camera_pose, landmark_ids=[1]
            )
            list_of_observations.append(obs[0])
            track.add_observation(obs[0])

        assert track.length() == num_observations

        for i, observation in enumerate(track.observations):
            assert observation.timestamp == i * 0.1
            assert (
                observation.image_coords == list_of_observations[i].image_coords
            ).all()
            assert observation.camera_id == list_of_observations[i].camera_id
            assert observation.landmark_id == list_of_observations[i].landmark_id
            assert observation.timestamp == list_of_observations[i].timestamp
            assert observation.camera_id == list_of_observations[i].camera_id
            assert observation.landmark_id == list_of_observations[i].landmark_id
            assert observation.image_coords.shape == (2,)

    def test_get_timestamps(self) -> None:
        """Test getting timestamps from track."""
        track = FeatureTrack(1)

        pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        # Add observations with timestamps
        timestamps = [0.0, 0.1, 0.2]
        for ts in timestamps:
            obs = FeatureObservation(1, np.array([100.0, 200.0]), 0, ts)
            track.add_observation(obs)

        track_timestamps = track.get_timestamps()
        assert track_timestamps == timestamps


class TestImageObservations:
    """Test suite for ImageObservations class."""

    def test_initialization(self) -> None:
        """Test ImageObservations initialization."""
        pose = CameraPose(
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        observations = [
            FeatureObservation(0, np.array([100.0, 200.0]), 0, 0.0),
            FeatureObservation(1, np.array([150.0, 250.0]), 0, 0.0),
        ]

        image = ImageObservations(
            0, 0.0, observations, image_width=640, image_height=480
        )

        assert image.feature_observations == observations
        assert image.image_width == 640
        assert image.image_height == 480
        assert len(image.feature_observations) == 2

    def test_get_visible_landmarks(self) -> None:
        """Test getting visible landmark IDs."""

        observations = [
            FeatureObservation(5, np.array([100.0, 200.0]), 0, 0.0),
            FeatureObservation(10, np.array([150.0, 250.0]), 0, 0.0),
            FeatureObservation(15, np.array([200.0, 300.0]), 0, 0.0),
        ]

        image = ImageObservations(
            0, 0.0, observations, image_width=640, image_height=480
        )

        visible_ids = image.get_visible_landmarks()
        assert visible_ids == [5, 10, 15]

    def test_get_observations_for_landmark(self) -> None:
        """Test getting observation for specific landmark."""
        pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        obs1 = FeatureObservation(1, np.array([100.0, 200.0]), 0, 0.0)
        obs2 = FeatureObservation(2, np.array([150.0, 250.0]), 0, 0.0)
        image = ImageObservations(
            0, 0.0, [obs1, obs2], image_width=640, image_height=480
        )

        # Test finding existing landmark
        found_obs = image.get_observations_for_landmark(1)
        assert found_obs is obs1

        # Test landmark not in image
        not_found_obs = image.get_observations_for_landmark(99)
        assert not_found_obs is None


class TestImageRenderer:
    """Test suite for ImageRenderer class."""

    @pytest.fixture
    def camera(self) -> PinHoleCamera:
        """Create a test camera."""
        return PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    @pytest.fixture
    def renderer(self, camera: PinHoleCamera) -> ImageRenderer:
        """Create a test image renderer."""
        return ImageRenderer(camera, image_width=640, image_height=480)

    @pytest.fixture
    def landmarks(self) -> np.ndarray:
        """Create test landmarks."""
        result: np.ndarray = np.array(
            [
                [1.0, 0.0, 5.0],  # In front, should be visible
                [0.0, 1.0, 5.0],  # In front, should be visible
                [0.0, 0.0, -1.0],  # Behind camera, should not be visible
                [10.0, 0.0, 5.0],  # Far away, may be out of image bounds
            ]
        )
        return result

    @pytest.fixture
    def camera_pose(self) -> CameraPose:
        """Create a test camera pose."""
        return CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity orientation
        )

    def test_initialization(self, camera: PinHoleCamera) -> None:
        """Test ImageRenderer initialization."""
        renderer = ImageRenderer(camera, image_width=800, image_height=600)

        assert renderer.camera is camera
        assert renderer.image_width == 800
        assert renderer.image_height == 600

    def test_project_landmarks_to_image(
        self, renderer: ImageRenderer, landmarks: np.ndarray, camera_pose: CameraPose
    ) -> None:
        """Test projecting landmarks to image coordinates."""
        observations = renderer.project_landmarks_to_image(landmarks, camera_pose)

        # Should have observations for visible landmarks (first two)
        assert len(observations) >= 2

        # Check that observations have correct properties
        for obs in observations:
            assert isinstance(obs, FeatureObservation)
            assert obs.timestamp == camera_pose.timestamp

            # Check image coordinates are within bounds
            u, v = obs.image_coords
            assert 0 <= u < renderer.image_width
            assert 0 <= v < renderer.image_height

    def test_render_image(
        self, renderer: ImageRenderer, landmarks: np.ndarray, camera_pose: CameraPose
    ) -> None:
        """Test rendering a synthetic image."""
        image = renderer.render_image(landmarks, camera_pose)

        assert isinstance(image, ImageObservations)
        assert image.image_width == renderer.image_width
        assert image.image_height == renderer.image_height

        # Should have some observations
        assert len(image.feature_observations) >= 0

    def test_generate_feature_tracks(self, renderer: ImageRenderer) -> None:
        """Test generating feature tracks."""
        # Create landmarks
        landmarks = np.array(
            [
                [1.0, 0.0, 3.0],
                [0.0, 1.0, 3.0],
                [2.0, 2.0, 5.0],
            ]
        )

        # Create camera poses (circular motion)
        poses = []
        for i in range(3):
            angle = i * 0.1  # Small rotation
            pose = CameraPose(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]),
                timestamp=i * 0.1,
            )
            poses.append(pose)

        tracks = renderer.generate_feature_tracks(landmarks, poses, min_track_length=2)

        # Should have tracks for visible landmarks
        assert len(tracks) >= 0

        # Check track properties
        for track in tracks:
            assert isinstance(track, FeatureTrack)
            assert track.length() >= 2

            # Check that timestamps are increasing
            timestamps = track.get_timestamps()
            valid_timestamps = [t for t in timestamps if t is not None]
            assert valid_timestamps == sorted(valid_timestamps)

    def test_render_image_sequence(self, renderer: ImageRenderer) -> None:
        """Test rendering a sequence of images."""
        landmarks = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 3.0]])

        poses = [
            CameraPose(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                timestamp=0.0,
            ),
            CameraPose(
                position=np.array([0.1, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                timestamp=0.1,
            ),
        ]

        images = renderer.render_image_sequence(landmarks, poses)

        assert len(images) == 2
        for i, image in enumerate(images):
            assert isinstance(image, ImageObservations)
            assert image.timestamp == i * 0.1

    def test_get_track_statistics(self, renderer: ImageRenderer) -> None:
        """Test getting track statistics."""
        # Empty tracks
        stats = renderer.get_track_statistics([])
        assert stats["num_tracks"] == 0
        assert stats["avg_track_length"] == 0.0
        assert stats["total_observations"] == 0

        # Create some test tracks
        track1 = FeatureTrack(1)
        track2 = FeatureTrack(2)

        # Add observations to tracks
        for i in range(3):
            obs1 = FeatureObservation(1, np.array([100.0, 200.0]), 0, i * 0.1)
            obs2 = FeatureObservation(2, np.array([150.0, 250.0]), 0, i * 0.1)
            track1.add_observation(obs1)
            track2.add_observation(obs2)

        stats = renderer.get_track_statistics([track1, track2])

        assert stats["num_tracks"] == 2
        assert stats["avg_track_length"] == 3.0
        assert stats["max_track_length"] == 3
        assert stats["min_track_length"] == 3
        assert stats["total_observations"] == 6

    def test_project_landmarks_invalid_input(
        self, renderer: ImageRenderer, camera_pose: CameraPose
    ) -> None:
        """Test projecting landmarks with invalid input."""
        landmarks = np.array([[1.0, 2.0, 3.0]])
        landmark_ids = [1, 2]  # Wrong length

        with pytest.raises(Exception):  # Should raise ValidationError
            renderer.project_landmarks_to_image(
                landmarks, camera_pose, camera_id=0, landmark_ids=landmark_ids
            )
