from pathlib import Path
from scenedetect import VideoManager, SceneManager, ContentDetector, split_video_ffmpeg


class SceneSpliter:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def split_scenes(self, video_path: str | Path, output_dir: str | Path) -> None:
        if isinstance(video_path, str):
            video_path = Path(video_path)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        video_manager = VideoManager([str(video_path.absolute())])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=15.0, min_scene_len=15))
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        split_video_ffmpeg(video_path, scene_list, output_dir)
