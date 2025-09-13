from pathlib import Path
import re
import numpy as np
import torch
import torch.utils.data as data
import decord


class VideoFromVideoFileTestDataset(data.Dataset):
    """Video test dataset (only input Low Quality videos).

    Supports testing dataset with following structures:
    dataroot
    ├── subfolder1
        ├── scene_000
        ├── scene_001
        ├── ...
    ├── subfolder1
        ├── scene_000
        ├── scene_001
        ├── ...
    ├── scene_002
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            name (str): Dataset name.
            max_mem (int): Maximum memory usage (in MB) per video.
    """

    def __init__(self, opt):
        super(VideoFromVideoFileTestDataset, self).__init__()
        decord.bridge.set_bridge("torch")
        self.opt = opt
        self.lq_root = opt["dataroot_lq"]
        self.files = self._find_files(Path(self.lq_root))
        self.max_mem = opt.get("max_mem", None)
        if self.max_mem is not None:
            self.files = list(filter(self._expected_mem_less_than, self.files))

    def _expected_mem_less_than(self, file: Path) -> bool:
        video_reader = decord.VideoReader(str(file), ctx=decord.cpu(0))
        num_frames = len(video_reader)
        frame = video_reader[0]
        mem_bytes = frame.numel() * frame.element_size() * num_frames
        return mem_bytes < self.max_mem * 1024 * 1024

    def _find_files(self, dataroot: Path) -> list[Path]:
        ends_with_digit_re = re.compile(r"^.*\d+\..+$")
        return sorted(
            set(
                path
                for path in dataroot.rglob("*")
                if ends_with_digit_re.match(path.name)
            )
        )

    def _read_video(self, video_path: Path):
        video_reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        video = torch.empty((len(video_reader), *video_reader[0].shape))
        for i, frame in enumerate(video_reader):
            video[i] = frame
        return video.permute(0, 3, 1, 2).divide_(255.0), video_reader.get_avg_fps()

    def __getitem__(self, index):
        file = self.files[index]
        imgs_lq, fps = self._read_video(file)
        return {
            "L": imgs_lq,
            "folder": file.parent.name,
            "lq_path": str(file),
            "filename": file.name,
            "fps": fps,
        }

    def __len__(self):
        return len(self.files)
