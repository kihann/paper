import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

# Sinusoidal Embedding 클래스 그대로 복사
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

if __name__ == "__main__":
    embedder = SinusoidalTimeEmbedding(128)  # 128차원으로

    # 시각화할 t 값들
    t_values = [1, 10, 100, 500, 1000]
    colors = ['red', 'orange', 'green', 'blue', 'purple']

    plt.figure(figsize=(12, 6))
    for t, color in zip(t_values, colors):
        t_tensor = torch.tensor([t])
        emb = embedder(t_tensor).detach().cpu().numpy()[0]
        plt.plot(range(len(emb)), emb, label=f"t={t}", color=color)

    plt.title("Sinusoidal Time Embedding at Different t")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()