import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

class NoiseScheduler:
    def __init__(self, timesteps = 1000, beta_start = 1e-4, beta_end = .02, device = device):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise = None):
        if noise is None:
            noise = torch.rand_like(x0)
        
        if isinstance(t, int):
            t = torch.tensor([t], device=x0.device)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1-alpha_bar_t) * noise
    
if __name__ == "__main__":
    scheduler = NoiseScheduler(timesteps=1000, device=device)

    # 가짜 이미지 배치
    x0 = torch.randn(4, 3, 96, 96).to(device)

    # 시간 스텝 t
    t = torch.tensor([10, 50, 100, 500]).to(device)  # 각 샘플마다 다르게

    xt = scheduler.q_sample(x0, t)

    print("x_t shape:", xt.shape, xt)  # [4, 3, 96, 96]