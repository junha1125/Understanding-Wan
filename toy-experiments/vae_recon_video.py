# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
WAN VAE AE 테스트 스크립트:
- src_video에서 sample_fps 기준으로 frame_num 만큼 프레임 샘플링
- (원본 -> VAE encode -> VAE decode)로 recon 영상 생성
- 원본/복원 영상을 나란히(grid)로 mp4 저장 (왼쪽: orig, 오른쪽: recon)

예)
python vae_recon_video.py \
  --task t2v-1.3B \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --size 832*480 \
  --src_video "/mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/18585469.mp4" \
  --frame_num 65 \
  --sample_fps 4

  --frame_num 41 \
  --sample_fps 4

# /mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/50908595.mp4
# /mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/23987443.mp4
# /mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/68486297.mp4  
# /mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/18585469.mp4
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from PIL import Image

sys.path.append("/mnt/backbone-nfs/junha/video-gen/Wan2.1")
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool  # 그대로 유지(요청)

# WanVAE는 WAN 코드 그대로 사용
from wan.modules.vae import WanVAE

# video io: imageio 우선, 없으면 opencv fallback
try:
    import imageio.v3 as iio  # imageio>=2.28
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _parse_args():
    parser = argparse.ArgumentParser(
        description="WAN VAE로 입력 비디오를 encode->decode 해서 recon 비디오를 저장합니다."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="WAN_CONFIGS 키 (예: t2v-1.3B)"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        required=True,
        help="WAN 체크포인트 디렉토리 (vae checkpoint 포함)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="입력 프레임을 리사이즈할 크기 (width*height)"
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        required=True,
        help="소스 비디오 경로"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="샘플링할 프레임 수(최대). 영상이 길어도 여기까지만 뽑음"
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=4,
        help="소스 비디오에서 샘플링할 FPS"
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="저장할 mp4 경로. 없으면 자동 생성"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="(옵션) seed. 여기선 학습/샘플링 안 해서 사실상 영향 없음"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 또는 cpu"
    )

    args = parser.parse_args()
    return args


def preprocess_image(image, torch_dtype=None, device=None, pattern="B C H W",
                     min_value=-1, max_value=1):
    """
    DiffSynth 코드 스타일을 최대한 유지하면서, 의존성(einops) 없이 구현.
    기본 pattern="B C H W"만 사실상 사용.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("preprocess_image expects a PIL.Image.Image")

    # PIL -> numpy float32 (H,W,C), RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32)  # [H,W,C], 0..255

    x = torch.from_numpy(arr)  # float32
    x = x.to(dtype=torch_dtype or torch.float32, device=device)

    # scale to [min_value, max_value]
    x = x * ((max_value - min_value) / 255.0) + min_value  # [H,W,C]

    # HWC -> CHW
    x = x.permute(2, 0, 1).contiguous()  # [C,H,W]

    if "B" in pattern:
        x = x.unsqueeze(0)  # [B,C,H,W]
    return x


def preprocess_video(video, torch_dtype=None, device=None, pattern="B C T H W",
                     min_value=-1, max_value=1):
    """
    DiffSynth 코드 스타일을 최대한 유지:
    - video: list[PIL.Image]
    - return: torch.Tensor, default [B,C,T,H,W]
    """
    frames = [
        preprocess_image(
            im, torch_dtype=torch_dtype, device=device,
            min_value=min_value, max_value=max_value,
            pattern="B C H W"
        )
        for im in video
    ]  # list of [1,C,H,W]

    # stack along T
    # frames: list([1,C,H,W]) -> [T,1,C,H,W] -> [1,C,T,H,W]
    x = torch.stack(frames, dim=0)          # [T,1,C,H,W]
    x = x.permute(1, 2, 0, 3, 4).contiguous()  # [1,C,T,H,W]
    return x


def _resize_pil(im: Image.Image, size_wh):
    w, h = size_wh
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im.resize((w, h), resample=Image.BICUBIC)


def _load_frames_imageio(video_path, sample_fps, frame_num, resize_wh):
    # imageio v3: 먼저 전체 메타 fps를 얻기 어렵다 → v2 reader fallback 느낌이지만,
    # 간단하게는 v3.imiter로 순회하며 time 기반 샘플링 대신, "대략적인 step"을 위해 v2 방식 사용.
    import imageio

    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 0.0)) if meta else 0.0
    if src_fps <= 0:
        src_fps = 30.0  # 대충 안전 디폴트

    step = max(int(round(src_fps / float(sample_fps))), 1)

    frames = []
    idx = 0
    picked = 0
    try:
        while picked < frame_num:
            try:
                frame = reader.get_data(idx)  # ndarray [H,W,C] uint8
            except Exception:
                break
            im = Image.fromarray(frame)
            im = _resize_pil(im, resize_wh)
            frames.append(im)
            picked += 1
            idx += step
    finally:
        reader.close()

    return frames


def _load_frames_cv2(video_path, sample_fps, frame_num, resize_wh):
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0
    step = max(int(round(src_fps / float(sample_fps))), 1)

    frames = []
    idx = 0
    picked = 0
    while picked < frame_num:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame_rgb)
        im = _resize_pil(im, resize_wh)
        frames.append(im)
        picked += 1
        idx += step

    cap.release()
    return frames


def load_video_frames(video_path, sample_fps, frame_num, resize_wh):
    if _HAS_IMAGEIO:
        return _load_frames_imageio(video_path, sample_fps, frame_num, resize_wh)
    if _HAS_CV2:
        return _load_frames_cv2(video_path, sample_fps, frame_num, resize_wh)
    raise RuntimeError("imageio 또는 opencv-python(cv2) 중 하나가 필요합니다.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = _parse_args()

    cfg = WAN_CONFIGS[args.task]
    cfg.sample_fps = args.sample_fps

    # device
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # dtype: 안전하게 cuda면 fp16, cpu면 fp32
    vae_dtype = torch.float16 if use_cuda else torch.float32

    size_wh = SIZE_CONFIGS[args.size]  # (W,H)
    logging.info(f"Target resize = {size_wh} (W,H), sample_fps={args.sample_fps}, frame_num={args.frame_num}")

    # 1) load frames (PIL)
    frames = load_video_frames(
        video_path=args.src_video,
        sample_fps=args.sample_fps,
        frame_num=args.frame_num,
        resize_wh=size_wh
    )
    if len(frames) == 0:
        raise RuntimeError("프레임을 하나도 로드하지 못했습니다. src_video 경로/코덱을 확인하세요.")
    logging.info(f"Loaded frames: {len(frames)}")

    # 2) preprocess -> tensor [1,C,T,H,W] in [-1,1]
    video_btchw = preprocess_video(frames, torch_dtype=torch.float32, device=device)  # float32 OK
    # WanVAE.encode는 list of [C,T,H,W]를 받는다
    video_cthw = video_btchw[0]  # [C,T,H,W]

    # 3) build VAE
    vae_ckpt = os.path.join(args.ckpt_dir, cfg.vae_checkpoint)
    logging.info(f"Loading VAE from: {vae_ckpt}")
    vae = WanVAE(
        vae_pth=vae_ckpt,
        dtype=vae_dtype,
        device=device
    )

    # 4) encode -> decode (deterministic AE처럼 mu만 사용)
    # WanVAE.encode 내부에서 autocast + model.encode(..., scale) 호출 후 float()로 반환됨
    logging.info("Encoding...")
    z_list = vae.encode([video_cthw])  # list([z_dim, t', h', w'])
    z = z_list[0]
    logging.info(f"Encoded latent shape: {z.shape}") # ex, torch.Size([16, 11, 60, 60])

    logging.info("Decoding...")
    recon_list = vae.decode([z])  # list([C,T,H,W]) in [-1,1]
    recon = recon_list[0]
    logging.info(f"Reconstructed video shape: {recon.shape}") # ex, torch.Size([3, 41, 480, 480])

    # 5) save (orig vs recon side-by-side)
    # cache_video는 generate.py와 동일하게: tensor shape [B,C,T,H,W] 또는 [N,C,T,H,W]로 grid 가능
    # 여기서는 N=2로 원본/복원 나란히 저장
    orig = video_cthw.detach().float().clamp(-1, 1).cpu()
    recon = recon.detach().float().clamp(-1, 1).cpu()
    stacked = torch.stack([orig, recon], dim=0)  # [2,C,T,H,W]

    if args.save_file is None:
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(args.src_video))[0]
        args.save_file = f"vae_recon_{args.task}_{args.size}_{base}_{formatted_time}.mp4"

    logging.info(f"Saving to: {args.save_file}")
    cache_video(
        tensor=stacked,
        save_file=args.save_file,
        fps=args.sample_fps,
        nrow=2,                 # 왼쪽 orig, 오른쪽 recon
        normalize=True,
        value_range=(-1, 1)
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
