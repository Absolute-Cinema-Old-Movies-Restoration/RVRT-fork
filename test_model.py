import torch


def test_video(lq, model, args):
    """test the video as a whole or as clips (divided temporally)."""

    num_frame_testing = args.tile[0]
    if num_frame_testing:
        # test as multiple clips if out-of-memory
        sf = args.scale
        num_frame_overlapping = args.tile_overlap[0]
        not_overlap_border = False
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = num_frame_testing - num_frame_overlapping
        d_idx_list = list(range(0, d - num_frame_testing, stride)) + [
            max(0, d - num_frame_testing)
        ]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros(b, d, 1, 1, 1)

        for d_idx in d_idx_list:
            lq_clip = lq[:, d_idx : d_idx + num_frame_testing, ...]
            out_clip = test_clip(lq_clip, model, args)
            out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    out_clip[:, -num_frame_overlapping // 2 :, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping // 2 :, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, : num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, : num_frame_overlapping // 2, ...] *= 0

            E[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip)
            W[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip_mask)
        output = E.div_(W)
    else:
        # test as one clip (the whole video) if you have enough memory
        window_size = args.window_size
        d_old = lq.size(1)
        d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
        lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
        output = test_clip(lq, model, args)
        output = output[:, :d_old, :, :, :]

    return output


def test_clip(lq, model, args):
    """test the clip as a whole or as patches."""

    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert (
        size_patch_testing % window_size[-1] == 0
    ), "testing patch size should be a multiple of window_size."

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h - size_patch_testing, stride)) + [
            max(0, h - size_patch_testing)
        ]
        w_idx_list = list(range(0, w - size_patch_testing, stride)) + [
            max(0, w - size_patch_testing)
        ]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[
                    ...,
                    h_idx : h_idx + size_patch_testing,
                    w_idx : w_idx + size_patch_testing,
                ]
                out_patch = model(in_patch).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size // 2 :, :] *= 0
                        out_patch_mask[..., -overlap_size // 2 :, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size // 2 :] *= 0
                        out_patch_mask[..., :, -overlap_size // 2 :] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., : overlap_size // 2, :] *= 0
                        out_patch_mask[..., : overlap_size // 2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, : overlap_size // 2] *= 0
                        out_patch_mask[..., :, : overlap_size // 2] *= 0

                E[
                    ...,
                    h_idx * sf : (h_idx + size_patch_testing) * sf,
                    w_idx * sf : (w_idx + size_patch_testing) * sf,
                ].add_(out_patch)
                W[
                    ...,
                    h_idx * sf : (h_idx + size_patch_testing) * sf,
                    w_idx * sf : (w_idx + size_patch_testing) * sf,
                ].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = (
            torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
            if h_pad
            else lq
        )
        lq = (
            torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)
            if w_pad
            else lq
        )

        output = model(lq).detach().cpu()

        output = output[:, :, :, : h_old * sf, : w_old * sf]

    return output
