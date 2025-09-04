import argparse

import torch


def argparser():
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
        "--save_result", action="store_true", help="save resulting image"
    )
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
