import asyncio
import logging

from utils.mp4.src.video_to_images.video_to_img import VideoToImage


async def main():
    video_converter = VideoToImage()
    await video_converter.video_to_img(
        "/home/dawid/absolute-cinema/AbsoluteScrapper/data/internet_archive/videos/001WhereIsEverybody.avi",
        "/home/dawid/absolute-cinema/data/video/frames/where_is_everybody",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
