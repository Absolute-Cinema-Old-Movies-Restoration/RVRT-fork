from abc import ABC, abstractmethod


class AbstractVideoToImage(ABC):

    @abstractmethod
    async def video_to_img(self, video_path: str, output_dir: str) -> None:
        """
        Convert video to images and save them to the output directory.
        """
        pass
