# ============================================================
# Global config (여기만 바꿔서 원하는 동작 조절)
# ============================================================
VIDEO_PATH = "/mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/68486297.mp4"
# /mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/50908595.mp4
# /mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/23987443.mp4
# /mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/68486297.mp4

EXTRACT_FPS = 1.0          # "추출 fps" (예: 1.0이면 1초 간격으로 샘플링)
NUM_FRAMES = 4             # t, t+Δ, t+2Δ, t+3Δ (총 4장)

SIM_METRIC = "l1"         # "cos" or "l1"
MODEL_NAME = "google/siglip2-giant-opt-patch16-384" # "google/siglip2-so400m-patch16-384"
MODEL_IMAGE_SIZE = 384     # SigLIP 입력 이미지 크기

FORCE_GRID = 24            # 시각화/매칭용 그리드 강제 (24 -> 24x24). None이면 모델 토큰에서 자동 추정
SAVE_FIGS = True
FIG_DIR = "./patch_match_vis_" + VIDEO_PATH.split("/")[-1].split(".")[0]
SAVE_SIM_MATS = False
SIM_DIR = "./sim_mats"

# 라인 너무 많으면 보기 힘들 수 있어서 제한하고 싶으면 값 줄이기 (None이면 전부 그림)
DRAW_STRIDE = 4   # 예: 200, 500, None

ONE_PER_ROW_RANDOM = True
RANDOM_SEED = 0     # 재현 가능하게(원하면 바꾸기)
ROW_STRIDE = 1      # 1이면 모든 row, 2면 격줄만(0,2,4,...)

LINE_WIDTH = 1.0
LINE_ALPHA = 0.6

# ============================================================
# Imports
# ============================================================
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import cv2

from transformers import AutoProcessor, SiglipVisionModel


# ============================================================
# Model load (한 번만)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SiglipVisionModel.from_pretrained(MODEL_NAME).to(device).eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)


# ============================================================
# Utils
# ============================================================
def read_4_frames_middle(video_path, extract_fps=1.0, num_frames=4, resize=384):
    """
    비디오 중간 프레임 t를 기준으로, extract_fps 간격으로 연속된 num_frames를 가져옴.
    반환: (PIL 이미지 리스트, 선택된 frame index 리스트, 원본 fps)
    """

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # extract_fps -> 프레임 간격(원본 fps 기준)
    step = max(1, int(round(src_fps / float(extract_fps))))

    # 중간 t에서 앞으로 4장(t, t+step, t+2step, t+3step)
    t = frame_count // 2
    t = min(t, frame_count - 1 - (num_frames - 1) * step)
    t = max(t, 0)

    idxs = [t + i * step for i in range(num_frames)]

    images = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"프레임 읽기 실패: idx={idx}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb).convert("RGB")
        pil = pil.resize((resize, resize), Image.BICUBIC)
        images.append(pil)

    cap.release()
    return images, idxs, src_fps


def _strip_cls_and_get_grid(tokens_btd):
    """
    tokens_btd: [B, T, D]
    - T가 정사각형이면 그대로 사용
    - T-1이 정사각형이면 CLS 토큰(맨 앞 1개) 제거했다고 가정
    반환: tokens_no_cls [B, G, G, D], G
    """
    B, T, D = tokens_btd.shape
    g = int(round(math.sqrt(T)))
    if g * g == T:
        tok = tokens_btd
        G = g
    else:
        g2 = int(round(math.sqrt(T - 1)))
        if g2 * g2 == (T - 1):
            tok = tokens_btd[:, 1:, :]
            G = g2
        else:
            # 마지막 fallback: 가장 가까운 정사각형으로 자르기 (최소한 돌아가게)
            G = int(math.floor(math.sqrt(T)))
            tok = tokens_btd[:, : G * G, :]

    tok = tok.reshape(B, G, G, D)
    return tok, G


@torch.no_grad()
def extract_patch_features(images_pil):
    """
    images_pil: list of PIL images (길이 4)
    반환: feats [B, N, D], grid_size (N=grid^2)
    """
    inputs = processor(images=images_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    tokens = outputs.last_hidden_state  # [B, T, D]

    grid_tok, G = _strip_cls_and_get_grid(tokens)  # [B, G, G, D]
    if FORCE_GRID is not None and FORCE_GRID != G:
        # (B, D, G, G)로 바꿔서 bilinear interpolate
        x = grid_tok.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(FORCE_GRID, FORCE_GRID), mode="bilinear", align_corners=False)
        grid_tok = x.permute(0, 2, 3, 1).contiguous()
        G = FORCE_GRID

    feats = grid_tok.reshape(grid_tok.shape[0], G * G, grid_tok.shape[-1])
    return feats, G


@torch.no_grad()
def compute_pairwise_similarity(A_nd, B_nd, metric="cos"):
    """
    A_nd: [N, D], B_nd: [N, D]
    반환: sim [N, N]
    """
    A = A_nd.float()
    B = B_nd.float()
    if metric == "cos":
        A = F.normalize(A, dim=-1)
        B = F.normalize(B, dim=-1)
        sim = A @ B.t()  # cosine similarity
    elif metric == "l1":
        # similarity로 쓰기 위해 -L1 distance
        dist = torch.cdist(A, B, p=1)
        sim = -dist
    else:
        raise ValueError(f"SIM_METRIC must be 'cos' or 'l1', got {metric}")
    return sim


def draw_grid(ax, W, H, G):
    """
    ax 위에 GxG 그리드 선을 그림 (픽셀 좌표계 기준)
    """
    # 이미지 좌표: x=[0,W], y=[0,H] (origin upper)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    cell_w = W / G
    cell_h = H / G
    for k in range(1, G):
        x = k * cell_w
        y = k * cell_h
        ax.axvline(x, linewidth=0.5, alpha=0.6)
        ax.axhline(y, linewidth=0.5, alpha=0.6)


def patch_center(patch_idx, G, W, H):
    """
    patch index -> (cx, cy) in pixel coords
    """
    r = patch_idx // G
    c = patch_idx % G
    cell_w = W / G
    cell_h = H / G
    cx = (c + 0.5) * cell_w
    cy = (r + 0.5) * cell_h
    return cx, cy


def visualize_best_matches(imgA_pil, imgB_pil, sim_nn, G, title="", save_path=None):
    """
    sim_nn: [N, N] similarity
    각 patch i in A에 대해 argmax_j sim[i,j] 찾아서 선으로 연결
    """
    imgA = np.array(imgA_pil)
    imgB = np.array(imgB_pil)
    H, W = imgA.shape[0], imgA.shape[1]
    assert imgB.shape[0] == H and imgB.shape[1] == W

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    ax1.imshow(imgA)
    ax2.imshow(imgB)
    draw_grid(ax1, W, H, G)
    draw_grid(ax2, W, H, G)
    ax1.set_title("A")
    ax2.set_title("B")
    fig.suptitle(title)

    # best match in B for each patch in A
    best_j = sim_nn.argmax(dim=1).detach().cpu().numpy()  # [N]
    N = best_j.shape[0]

    # best match in B for each patch in A
    best_j = sim_nn.argmax(dim=1).detach().cpu().numpy()  # [N]
    N = best_j.shape[0]

    if ONE_PER_ROW_RANDOM:
        rng = np.random.default_rng(RANDOM_SEED)
        rows = np.arange(0, G, ROW_STRIDE)
        cols = rng.integers(0, G, size=len(rows))  # 각 row에서 랜덤 col 1개
        draw_indices = rows * G + cols
    else:
        draw_indices = np.arange(N)

    for i in draw_indices:
        j = int(best_j[i])
        x1, y1 = patch_center(i, G, W, H)
        x2, y2 = patch_center(j, G, W, H)
        con = ConnectionPatch(
            xyA=(x1, y1), coordsA=ax1.transData,
            xyB=(x2, y2), coordsB=ax2.transData,
            axesA=ax1, axesB=ax2,
            linewidth=LINE_WIDTH, alpha=LINE_ALPHA,
            color="red",          # <- 빨간색
            linestyle="--"
        )
        fig.add_artist(con)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


# ============================================================
# Run (그냥 이 파일/셀 실행하면 됨)
# ============================================================
images, frame_idxs, src_fps = read_4_frames_middle(
    VIDEO_PATH,
    extract_fps=EXTRACT_FPS,
    num_frames=NUM_FRAMES,
    resize=MODEL_IMAGE_SIZE
)
print(f"src_fps={src_fps:.3f}, selected frame idxs={frame_idxs}")

feats_bnd, G = extract_patch_features(images)  # [4, N, D], grid
print("patch feats:", feats_bnd.shape, "grid:", G)

# 연속된 프레임끼리 similarity 계산 & 저장
sim_mats = []
for k in range(NUM_FRAMES - 1):
    sim = compute_pairwise_similarity(feats_bnd[k], feats_bnd[k + 1], metric=SIM_METRIC)
    sim_mats.append(sim.detach().cpu())

if SAVE_SIM_MATS:
    os.makedirs(SIM_DIR, exist_ok=True)
    for k, sim in enumerate(sim_mats):
        np.save(os.path.join(SIM_DIR, f"sim_{k}_{k+1}_{SIM_METRIC}.npy"), sim.numpy())

# 3개의 조합 시각화: (t,t+Δ), (t+Δ,t+2Δ), (t+2Δ,t+3Δ)
if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)

for k in range(NUM_FRAMES - 1):
    save_path = None
    if SAVE_FIGS:
        save_path = os.path.join(FIG_DIR, f"pair_{k}_{k+1}_{SIM_METRIC}.png")

    visualize_best_matches(
        images[k],
        images[k + 1],
        sim_mats[k],
        G,
        title=f"pair ({k}->{k+1})  | metric={SIM_METRIC} | frames={frame_idxs[k]}->{frame_idxs[k+1]}",
        save_path=save_path
    )



"""region start: GPT prompt 

지금부터 너는 내가 원하는 역할을 하는 python 코드를 만들어줘. 

너는 먼저 다음 스텝을 거칠거야.

1. 주어진 Path 에서 비디오를 하나 가져와. 비디오는 길지 않아 약 20~30초 정도의 비디오 일거야

```
/mnt/backbone-nfs/junha/dataset/PE-Video/test/extracted/50908595.mp4
```


2. 전체 비디오에서 중간정도 timestep t 에서, fps (=fps) 1 로 연속된 4장의 이미지를 뽑아와. 즉, t, t+1xfps, t+2xfps, t+3xfps이미지를 가져오면 돼. (이떄 추출 fps 를 몇으로 할지는 맨 위 Global 변수로 미리 설정해놔줘)

3. 그 모든 것들을 아래의 코드를 참고해서, patch features 를 추출해! 

```
from PIL import Image
import requests
from transformers import AutoProcessor, SiglipVisionModel

model = SiglipVisionModel.from_pretrained(
    "google/siglip2-so400m-patch16-384"
)
processor = AutoProcessor.from_pretrained(
    "google/siglip2-so400m-patch16-384"
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

images = [image, image.copy(), image.copy()]
inputs = processor(images=images, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

print("Last hidden state shape:", last_hidden_state.shape)
#>> Last hidden state shape: torch.Size([3, 729, 1152]) 이건 width, hight 가 일렬로 되어있지만, reshape 해서 정사각형모양으로 나중에는 처리해야할거야. 
```

4. 그 다음이 중요해, 너는 연속된 프레임들끼리 patch wise similarity map 을 먼저 계산해. 예를들어 (t, t+1xfps) 끼리의 patch wise similarity는 729 * 729 행렬로 계산될 수 있겠지. 여기서 similarity 계산은 L1 or Cos 둘중 하나를 맨 위 글로벌 변수로 설정할 수 있게 만들어. (t+1xfps, t+2xfps) ... 그 이외의 연속된 거까지 저장해놔
5. 그 다음 우리는 그림을 그릴거야. 예를 들어서 (t) 이미지 위에 16x16 픽셀 크기를 가지는 패치가 가지도록 총 24x24 그리드를 그려. 그 오른쪽에 (t+1xfps) 이미지를 그려. (이것도 그리드를 가져)
6. 여기서 두 이미지에 대해서 가장 높은 similarity 를 가지는 patch 끼리 선을 그어줘. 예를 들어서 (t) 이미지의 (0,0) 패치가 (t+1xfps) 이미지의 (0,1) 피치와 가장 similarity 가 높았다면 그 두 사이의 선을 그려주면 되는거고, 나머지 패치들도 그렇게 이어주면돼. 
7. 이렇게 총 3개의 조합 (t, t+1xfps)  (t+1xfps, t+2xfps)  (t+2xfps, t+3xfps) 이미지를 그려줘

파이썬 코드는 간단하면 좋겠어. 예외처리도 굳이 많이 하진 말고,,, main 함수 만들지 말고, 내가 생각하기에.. 반복작업이 요구되는 함수만 def 해서 함수로 만들고, 원하는 변수 변경은 global 변수로 맨 위에서 조정할 수 있게끔 하고 싶어.

region end"""  