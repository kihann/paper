import torch

match True:
    case _ if torch.cuda.is_available():
        device = "cuda"
    case _ if torch.backends.mps.is_available():
        device = "mps"
    case _:
        device = "cpu"

class NoiseScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = device):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: int | torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor: 
        if noise is None:
            noise = torch.randn_like(x0)
        
        if isinstance(t, int):
            t = torch.tensor([t], device=x0.device)
  
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1-alpha_bar_t) * noise

def _debug():
    scheduler = NoiseScheduler()
    
    x0 = torch.randn(4, 3, 96, 96).to(device)
    t = torch.tensor([10, 50, 100, 500]).to(device)

    xt = scheduler.q_sample(x0, t)

    print(f"{xt.shape=}, {device=}") # [4, 3, 96, 96]

if __name__ == "__main__":
    _debug()