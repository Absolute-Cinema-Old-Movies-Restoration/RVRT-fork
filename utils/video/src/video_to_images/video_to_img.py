from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import threading
from typing import Generator

import cv2
from numpy import ndarray
from tqdm import tqdm

from utils.mp4.src.misc.counter import ThreadSafeCounter
from .abstract_mp4_to_image import (
    AbstractVideoToImage,
)

_logger = logging.getLogger(__name__)


class VideoToImage(AbstractVideoToImage):
    def __init__(
        self,
        frame_name_prefix: str = "frame",
        max_concurrent_writes: int = 10,
        max_queue_length: int = 100,
    ):
        """
        Initialize the MP4ToImg converter.
        :param frame_name_prefix: The prefix for the frame image filenames.
        """
        super().__init__()
        self.frame_name_prefix = frame_name_prefix
        self.executor = ThreadPoolExecutor(max_concurrent_writes)
        self.semaphore = threading.Semaphore(max_queue_length)
        self.counter = ThreadSafeCounter()

    def _get_frame_count(self, video_capture: cv2.VideoCapture) -> int:
        return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _validate_paths(self, video_path: Path, output_dir: Path) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty."
            )
        if not video_path.exists():
            raise FileNotFoundError(f"Video file {video_path} does not exist")
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file {video_path} is not a file")

    def _frame_generator(
        self, video_capture: cv2.VideoCapture
    ) -> Generator[ndarray, None, None]:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            yield frame

    async def video_to_img(
        self, video_path: str | Path, output_dir: str | Path
    ) -> None:
        """
        Processes full video into and saves all frames in the output directory.
        Frames are saved as {frame_name_prefix}_xxxx.png files
        :param video_path: The path to the input video file.
        :param output_dir: The directory where the extracted frames will be saved.
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if isinstance(video_path, str):
            video_path = Path(video_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_paths(video_path, output_dir)

        video_capture = cv2.VideoCapture(str(video_path))
        try:

            total_frames = self._get_frame_count(video_capture)
            num_digits = len(str(total_frames))

            for i, frame in tqdm(
                enumerate(self._frame_generator(video_capture)),
                total=total_frames,
                desc="Extracting frames",
                unit="frame",
            ):
                filename = f"{self.frame_name_prefix}_{i:0{num_digits}d}.png"
                frame_path = output_dir / filename
                self.semaphore.acquire()
                self.executor.submit(self._save_frame, frame_path, frame)

        except Exception:
            _logger.error("An error occurred during video processing.")
            self._cleanup(video_capture, wait=False)
            raise

    def _cleanup(self, video_capture: cv2.VideoCapture, wait=True) -> None:
        try:
            video_capture.release()
        except Exception as e:
            _logger.error(f"Error occurred while releasing video capture: {e}")

        try:
            self.executor.shutdown(wait=wait)
        except Exception as e:
            _logger.critical(f"Error occurred while shutting down executor: {e}")

    def _save_frame(self, filename: Path, frame: ndarray) -> None:
        try:
            is_written = cv2.imwrite(str(filename), frame)
            if not is_written:
                _logger.error(f"Failed to write frame to {filename}")
        except Exception as e:
            _logger.error(f"Error occurred while saving frame to {filename}: {e}")
        finally:
            self.semaphore.release()

    def __del__(self):
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=False)
