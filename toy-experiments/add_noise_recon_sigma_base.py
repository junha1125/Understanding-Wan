#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WAN VAE latent에 특정 timestep 노이즈를 추가해서 decode 결과를 시각화하는 스크립트.

동작:
1) src_video에서 sample_fps 기준으로 frame_num 만큼 프레임 샘플링
2) WAN-VAE encode -> latent (z)
3) timestep(예: 50)에 해당하는 sigma로 latent에 noise 추가
4) clean latent decode / noisy latent decode 해서 비디오 저장 (grid)

예)
python add_noise_recon_sigma_base.py \
  --task t2v-1.3B \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --size 480*480 \
  --src_video "/mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/23987443.mp4" \
  --frame_num 41 \
  --sample_fps 4 \
  --sigma 0.0925 

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
from wan.utils.utils import cache_image, cache_video, str2bool  
from wan.modules.vae import WanVAE

# video io: imageio 우선, 없으면 opencv fallback
try:
    import imageio
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
        description="WAN VAE latent에 특정 timestep 노이즈를 추가 후 decode 결과를 저장합니다."
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
        required=True,
        help="WAN 체크포인트 디렉토리 (vae checkpoint 포함)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="입력 프레임 리사이즈 크기 (width*height)"
    )
    parser.add_argument(
        "--src_video",
        type=str,
        required=True,
        help="소스 비디오 경로"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="샘플링할 프레임 수(최대)"
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=4,
        help="소스 비디오에서 샘플링할 FPS"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0925,
        help="노이즈를 추가할 sigma 값 (0~1 범위 권장, 예: 0.0925)"
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="저장할 mp4 경로. 없으면 자동 생성"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 또는 cpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="난수 시드"
    )
    return parser.parse_args()


def preprocess_image(image, torch_dtype=None, device=None, pattern="B C H W",
                     min_value=-1, max_value=1):
    """
    DiffSynth 스타일을 최대한 유지(의존성 없이 구현).
    기본적으로 pattern="B C H W"만 사용.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32)  # [H,W,C], 0..255

    x = torch.from_numpy(arr).to(device=device, dtype=torch_dtype or torch.float32)
    x = x * ((max_value - min_value) / 255.0) + min_value  # [-1,1]
    x = x.permute(2, 0, 1).contiguous()  # [C,H,W]
    if "B" in pattern:
        x = x.unsqueeze(0)  # [B,C,H,W]
    return x


def preprocess_video(video, torch_dtype=None, device=None, pattern="B C T H W",
                     min_value=-1, max_value=1):
    """
    video: list[PIL.Image]
    return: torch.Tensor [1,C,T,H,W]
    """
    frames = [
        preprocess_image(
            im, torch_dtype=torch_dtype, device=device,
            min_value=min_value, max_value=max_value,
            pattern="B C H W"
        )
        for im in video
    ]  # list of [1,C,H,W]
    x = torch.stack(frames, dim=0)             # [T,1,C,H,W]
    x = x.permute(1, 2, 0, 3, 4).contiguous()  # [1,C,T,H,W]
    return x


def _resize_pil(im: Image.Image, size_wh):
    w, h = size_wh
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im.resize((w, h), resample=Image.BICUBIC)


def _load_frames_imageio(video_path, sample_fps, frame_num, resize_wh):
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 0.0)) if meta else 0.0
    if src_fps <= 0:
        src_fps = 30.0

    step = max(int(round(src_fps / float(sample_fps))), 1)

    frames = []
    idx = 0
    picked = 0
    try:
        while picked < frame_num:
            try:
                frame = reader.get_data(idx)  # ndarray
            except Exception:
                break
            im = Image.fromarray(frame)
            frames.append(_resize_pil(im, resize_wh))
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
        frames.append(_resize_pil(im, resize_wh))
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


def wan_sigma_to_timestep(sigma: float, num_train_timesteps: int = 1000,) -> int:
    """
    sigma -> training timestep 변환 (Wan 스타일)
    """
    sigma = max(sigma, 1000)
    t = (sigma / (1.0 + sigma)) * float(num_train_timesteps)
    t = int(round(t))
    t = max(0, min(t, num_train_timesteps))
    return t
                                     


def add_noise_to_latent(original_samples: torch.Tensor, sigma: float, generator: torch.Generator = None) -> torch.Tensor:
    """
    original_samples: latent z, shape [C,T,H,W] (또는 어떤 shape든 OK)
    return: noisy latent, same shape
    """
    noise = torch.randn(
        original_samples.shape,
        device=original_samples.device,
        dtype=original_samples.dtype,
        generator=generator
    )
    noisy = (1.0 - sigma) * original_samples + sigma * noise
    logging.info(f"add_noise_to_latent: alpha={1.0 - sigma:.6f}, sigma={sigma:.6f}")
    return noisy, noise


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    cfg = WAN_CONFIGS[args.task]
    cfg.sample_fps = args.sample_fps

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    vae_dtype = torch.float16 if use_cuda else torch.float32

    size_wh = SIZE_CONFIGS[args.size]  # (W,H)
    logging.info(f"Resize={size_wh} (W,H), sample_fps={args.sample_fps}, frame_num={args.frame_num}")

    # 1) load frames
    frames = load_video_frames(args.src_video, args.sample_fps, args.frame_num, size_wh)
    if len(frames) == 0:
        raise RuntimeError("프레임 로드 실패. src_video/코덱 확인.")
    logging.info(f"Loaded frames: {len(frames)}")

    # 2) preprocess -> [1,C,T,H,W], then [C,T,H,W]
    video_btchw = preprocess_video(frames, torch_dtype=torch.float32, device=device)
    video_cthw = video_btchw[0]

    # 3) load VAE
    vae_ckpt = os.path.join(args.ckpt_dir, cfg.vae_checkpoint)
    logging.info(f"Loading VAE: {vae_ckpt}")
    vae = WanVAE(vae_pth=vae_ckpt, dtype=vae_dtype, device=device)

    # 4) encode -> latent
    logging.info("Encoding to latent...")
    z = vae.encode([video_cthw])[0]  # [z_dim, t', h', w']

    # (선택) clean decode도 같이 저장하면 비교가 쉬움
    logging.info("Decoding clean latent...")
    recon_clean = vae.decode([z])[0]  # [3,T,H,W], [-1,1]

    # 5) add noise at timestep
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)

    z_noisy, noise = add_noise_to_latent(
        z,
        sigma=args.sigma,
        generator=seed_g
    )
    timestep = wan_sigma_to_timestep(
        sigma=args.sigma,
        num_train_timesteps=cfg.num_train_timesteps
    )

    # 6) decode noisy latent
    logging.info("Decoding noisy latent...")
    recon_noisy = vae.decode([z_noisy])[0]  # [3,T,H,W], [-1,1]
    # recon_noise = vae.decode([noise])[0]  # [3,T,H,W], [-1,1]

    # 7) visualize/save:
    # 한 프레임에 3개를 가로로: [orig | recon_clean | recon_noisy]
    orig = video_cthw.detach().float().clamp(-1, 1).cpu()
    # recon_clean = recon_clean.detach().float().clamp(-1, 1).cpu()
    recon_noisy = recon_noisy.detach().float().clamp(-1, 1).cpu()
    # recon_noise = recon_noise.detach().float().clamp(-1, 1).cpu()

    # stacked = torch.stack([orig, recon_clean, recon_noisy], dim=0)  # [3,C,T,H,W]
    stacked = torch.stack([orig, recon_noisy], dim=0)  # [3,C,T,H,W]

    if args.save_file is None:
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(args.src_video))[0]
        args.save_file = f"vae_noisy_{args.task}_{args.size}_sigma{args.sigma}_{base}_{formatted_time}.mp4"

    logging.info(f"Saving grid video: {args.save_file}")
    cache_video(
        tensor=stacked,
        save_file=args.save_file,
        fps=args.sample_fps,
        nrow=stacked.shape[0],                 # orig | clean | noisy
        normalize=True,
        value_range=(-1, 1)
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
