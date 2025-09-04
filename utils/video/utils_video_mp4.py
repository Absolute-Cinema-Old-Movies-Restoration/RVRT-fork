import argparse
import glob
import os
import cv2


def images_to_mp4(input_dir: str, output_file: str, fps: int = 30):
    """
    Converts a folder of PNG images into an MP4 video.

    Args:
        input_dir (str): Path to directory containing PNG images.
        output_file (str): Path + filename for the resulting MP4 file (e.g., "output/video.mp4").
        fps (int): Frames per second for the output video.
    """
    images = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not images:
        raise ValueError(f"No PNG files found in {input_dir}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    frame = cv2.imread(images[0])
    if frame is None:
        raise ValueError("First image could not be read.")
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping unreadable file {img_path}")
            continue
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
        out.write(img)

    out.release()
    print(f"Video saved to {output_file}")


def mp4_to_images(input_video: str, output_dir: str, prefix: str = "frame"):
    """
    Extracts frames from an MP4 video and saves them as PNG files.

    Args:
        input_video (str): Path to the MP4 video file.
        output_dir (str): Directory to save the output PNG frames.
        prefix (str): Filename prefix for saved frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frame_path = os.path.join(output_dir, f"{prefix}_{frame_idx:05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")


def arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="Convert images to MP4 or extract frames from MP4."
    )
    arg_parser.add_argument(
        "mode",
        choices=["img_to_mp4", "mp4_to_img"],
        help="Mode of operation: 'img_to_mp4' or 'mp4_to_img'.",
    )
    arg_parser.add_argument("input", help="Input file or directory.")
    arg_parser.add_argument("output", help="Output file or directory.")
    arg_parser.add_argument(
        "--fps",
        type=int,
        help="Frames per second for the output video.",
        default=30,
    )
    return arg_parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    if args.mode == "img_to_mp4":
        images_to_mp4(args.input, args.output, args.fps)
    elif args.mode == "mp4_to_img":
        mp4_to_images(args.input, args.output)
