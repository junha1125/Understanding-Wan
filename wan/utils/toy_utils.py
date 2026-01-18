import itertools
import matplotlib.pyplot as plt
import torch


class NoisePredDiffTracker:
    """
    text2video.py 예시 사용법:
    --------------
    # from .utils.toy_utils import NoisePredDiffTracker # (1)

    # timesteps = tensor([999, 995, 991, 987, 982, 978  ... 302, 241, 172,  92], device='cuda:0') // timesteps.shape = torch.Size([50])
    # diff_tracker = NoisePredDiffTracker(metric="l1_mean") # (2)

    # # noise_pred_uncond.shape = torch.Size([16, 1, 60, 104]) 
    # noise_pred_empty = self.model(
    #     latent_model_input, t=timestep, **arg_empty)[0]
    # noise_pred_zero = self.model(
    #     latent_model_input, t=timestep, **arg_zero)[0] # (3)
    # 
    # diff_tracker.update( t.item(), { "cond": noise_pred_cond, "uncond": noise_pred_uncond, "empty": noise_pred_empty, "zero": noise_pred_zero, }, ) # (4)

    # videos = self.vae.decode(x0)
    # diff_tracker.plot(save_path="noise_pred_diffs.png", show=False, title=f"diffs (guide_scale={guide_scale}, steps={sampling_steps})") # (5)
    --------------

    4개 noise_pred 텐서들 간의 pairwise feature 차이를 timestep별로 기록하고 plot.
    기본 metric: L1 mean (abs diff의 평균)
    """
    def __init__(self, names=("cond", "uncond", "empty", "zero"), metric="l1_mean"):
        self.names = list(names)
        self.metric = metric
        self.ts = []  # timestep 기록 (int)
        # pair key: "cond-uncond" 형태로 저장
        self.data = {f"{a}-{b}": [] for a, b in itertools.combinations(self.names, 2)}

    @torch.no_grad()
    def update(self, t, preds: dict):
        """
        t: int timestep
        preds: {"cond": Tensor, "uncond": Tensor, "empty": Tensor, "zero": Tensor}
               각 텐서 shape: [16, 1, 60, 104] 같은 형태
        """
        self.ts.append(int(t))

        # AMP 영향 줄이려면 float32로 계산하는 게 안전(가볍고 차이 관찰에 유리)
        preds_f = {k: v.detach().float() for k, v in preds.items()}

        for a, b in itertools.combinations(self.names, 2):
            da = preds_f[a]
            db = preds_f[b]
            diff = da - db

            if self.metric == "l1_sum":
                val = diff.abs().sum().item()
            elif self.metric == "l2_mean":
                val = diff.pow(2).mean().sqrt().item()  # RMSE 느낌
            elif self.metric == "cosine":  # 전체를 벡터로 보고 방향 유사도(1-cos)
                va = da.flatten()
                vb = db.flatten()
                val = (1.0 - torch.nn.functional.cosine_similarity(va, vb, dim=0)).item()
            else:
                # default: l1_mean
                val = diff.abs().mean().item()

            self.data[f"{a}-{b}"].append(val)

    def plot(self, save_path=None, show=False, title=None):
        if len(self.ts) == 0:
            return None

        # x축: timestep 그대로 사용
        x = self.ts

        plt.figure(figsize=(8, 5))

        # 선 스타일을 약간씩 다르게
        linestyles = ["-", "--", "-.", ":", "-", "--"]
        markers = [None, None, None, None, None, None]

        for i, (k, y) in enumerate(self.data.items()):
            plt.plot(
                x,
                y,
                label=k,
                linewidth=2.0,
                linestyle=linestyles[i % len(linestyles)],
                alpha=0.85,
            )

        # 핵심 1️⃣: x축을 reverse (왼쪽이 큰 timestep)
        plt.gca().invert_xaxis()

        plt.xlabel("timestep (high noise → low noise)")
        plt.ylabel(self.metric)
        plt.title(title or "noise_pred pairwise diffs over timesteps")

        # 핵심 2️⃣: legend를 그래프 밖으로 빼서 겹침 완화
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

        fig = plt.gcf()
        plt.close()
        return fig
