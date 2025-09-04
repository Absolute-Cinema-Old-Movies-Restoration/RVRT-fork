from utils.mp4.src.scene_spliter.scene_spliter import SceneSpliter


def main():
    video_path = "/home/dawid/absolute-cinema/AbsoluteScrapper/data/internet_archive/videos/001WhereIsEverybody.avi"
    output_dir = "/home/dawid/absolute-cinema/data/video/scenes/where_is_everybody"
    scene_spliter = SceneSpliter()
    scene_spliter.split_scenes(video_path, output_dir)


if __name__ == "__main__":
    main()
