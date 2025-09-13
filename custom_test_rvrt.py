import argparse
import os
import cv2
import numpy as np
import requests
from models.network_rvrt import RVRT as net
from test_model import test_video
from utils.video.src.dataset.video_dataset import VideoFromVideoFileTestDataset
import torch
from torch.utils.data import DataLoader


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="001_RVRT_videosr_bi_REDS_30frames",
        help="tasks: 001 to 006",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=0,
        help="noise level for denoising: 10, 20, 30, 40, 50",
    )
    parser.add_argument(
        "--folder_lq",
        type=str,
        default="testsets/REDS4/sharp_bicubic",
        help="input low-quality test video folder",
    )
    parser.add_argument(
        "--folder_lq_max_mem",
        default=None,
        type=int,
        help="maximum memory usage (in MB) per video, None for no limit",
    )
    parser.add_argument(
        "--folder_gt",
        type=str,
        default=None,
        help="input ground-truth test video folder",
    )
    parser.add_argument(
        "--tile",
        type=int,
        nargs="+",
        default=[100, 128, 128],
        help="Tile size, [0,0,0] for no tile during testing (testing as a whole)",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        nargs="+",
        default=[2, 20, 20],
        help="Overlapping of different tiles",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="number of workers in data loading"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="directory to save results",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="img",
        choices=["img", "video"],
        help="output format",
    )
    return parser.parse_args()


def get_model(args):
    """prepare model and dataset according to args.task."""
    if args.task == "001_RVRT_videosr_bi_REDS_30frames":
        model = net(
            upscale=4,
            clip_size=2,
            img_size=[2, 64, 64],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[144, 144, 144],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 1, 1, 1, 1, 1],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            cpu_cache_length=100,
        )
        args.scale = 4
        args.window_size = [2, 8, 8]
        args.nonblind_denoising = False

    elif args.task in [
        "002_RVRT_videosr_bi_Vimeo_14frames",
        "003_RVRT_videosr_bd_Vimeo_14frames",
    ]:
        model = net(
            upscale=4,
            clip_size=2,
            img_size=[2, 64, 64],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[144, 144, 144],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 1, 1, 1, 1, 1],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            cpu_cache_length=100,
        )
        args.scale = 4
        args.window_size = [2, 8, 8]
        args.nonblind_denoising = False

    elif args.task in ["004_RVRT_videodeblurring_DVD_16frames"]:
        model = net(
            upscale=1,
            clip_size=2,
            img_size=[2, 64, 64],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[192, 192, 192],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 3, 3, 3, 3, 3],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            cpu_cache_length=100,
        )
        args.scale = 1
        args.window_size = [2, 8, 8]
        args.nonblind_denoising = False

    elif args.task in ["005_RVRT_videodeblurring_GoPro_16frames"]:
        model = net(
            upscale=1,
            clip_size=2,
            img_size=[3, 480, 360],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[192, 192, 192],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 3, 3, 3, 3, 3],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            cpu_cache_length=100,
        )
        args.scale = 1
        args.window_size = [2, 8, 8]
        args.nonblind_denoising = False

    elif args.task == "006_RVRT_videodenoising_DAVIS_16frames":
        model = net(
            upscale=1,
            clip_size=2,
            img_size=[2, 480, 360],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[192, 192, 192],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 3, 4, 6, 8, 4],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            nonblind_denoising=True,
            cpu_cache_length=100,
        )
        args.scale = 1
        args.window_size = [2, 8, 8]
        args.nonblind_denoising = True

    # download model
    model_path = f"model_zoo/rvrt/{args.task}.pth"
    if os.path.exists(model_path):
        print(f"loading model from ./model_zoo/rvrt/{model_path}")
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://github.com/JingyunLiang/RVRT/releases/download/v0.0/{}".format(
            os.path.basename(model_path)
        )
        r = requests.get(url, allow_redirects=True)
        print(f"downloading model {model_path}")
        open(model_path, "wb").write(r.content)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(
        (
            pretrained_model["params"]
            if "params" in pretrained_model.keys()
            else pretrained_model
        ),
        strict=True,
    )

    return model


def get_dataset(args):
    video_from_video_dataset = VideoFromVideoFileTestDataset(
        {
            "dataroot_lq": args.folder_lq,
            "name": "VideoFromVideoFileTestDataset",
            "num_frame": args.window_size[0],
            "max_mem": args.folder_lq_max_mem,
        }
    )
    return video_from_video_dataset


def test_model(model, loader, args, device, save_dir):
    error_count = 0
    for idx, batch in enumerate(loader):
        try:
            print(f"Processing video {idx + 1:03d}: {batch['folder'][0]}")
            lq = batch["L"].to(device)
            timings = []
            # inference
            with torch.inference_mode():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                output = test_video(lq, model, args)
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                print(
                    "Elapsed time: {:.2f} s ({:.2f} FPS)".format(
                        elapsed_time / 1000, 1000 / elapsed_time
                    )
                )
                timings.append(elapsed_time)
                save_output(output, batch, save_dir, args)
        except Exception as e:
            print(f"Error processing video {batch['folder'][0]}: {e}")
            error_count += 1
    average_time = sum(timings) / len(timings)
    print(f"Average elapsed time per video: {average_time:.2f} ms")
    print(f"Total errors encountered: {error_count}")


def save_as_img(output, folder, lq_filename, save_dir):
    max_frames_digits = len(str(output.shape[1]))
    for i in range(output.shape[1]):
        # save image
        img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if img.ndim == 3:
            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
        out_dir = f"{save_dir}/{folder}/{lq_filename}"
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(f"{out_dir}/frame_{i:0{max_frames_digits}d}.png", img)


def save_as_video(output, folder, lq_filename, fps, save_dir):
    out_dir = f"{save_dir}/{folder}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{lq_filename}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # get first frame to infer size
    first = output[:, 0, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if first.ndim == 3:
        first = np.transpose(first[[2, 1, 0], :, :], (1, 2, 0))
    first = (first * 255.0).round().astype(np.uint8)

    H, W = first.shape[:2]  # correct order
    video_writer = cv2.VideoWriter(out_path, fourcc, max(1.0, float(fps)), (W, H))

    for i in range(output.shape[1]):
        img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if img.ndim == 3:
            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # RGB→BGR
        img = (img * 255.0).round().astype(np.uint8)
        video_writer.write(img)

    video_writer.release()
    return out_path


def save_output(output, batch, save_dir, args):
    folder = batch["folder"][0]
    lq_filename = batch["filename"][0]
    batch_fps = batch["fps"][0]
    if args.output_format == "img":
        save_as_img(output, folder, lq_filename, save_dir)
    elif args.output_format == "video":
        save_as_video(output, folder, lq_filename, batch_fps, save_dir)
    else:
        raise NotImplementedError("Only img format is supported.")


def main():
    print("Preparing model ...")
    args = get_argparser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = get_model(args)
    model.eval()
    model = model.to(device)
    print("Model prepared.")
    dataset = get_dataset(args)
    save_dir = args.result_dir
    print(f"Results will be saved to {save_dir}")
    print(f"Number of test videos: {len(dataset)}")
    os.makedirs(save_dir, exist_ok=True)
    test_loader = DataLoader(
        dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False
    )
    test_model(model, test_loader, args, device, save_dir)


if __name__ == "__main__":
    main()
