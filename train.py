import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ddpm_unet import UNet
from data import get_stl10_data
from noise import NoiseScheduler, device

def train_teacher_ddpm(epochs: int = None, lr: float = None):
    print(f"{device=}")
    print("TRAIN START")

    model = UNet().to(device)
    scheduler = NoiseScheduler()
    dataloader = get_stl10_data()

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("EPOCH LOOP START")
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)
            t = torch.randint(0, scheduler.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = scheduler.q_sample(x0, t, noise)

            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

if __name__ == "__main__":
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    train_teacher_ddpm(epochs=EPOCHS, lr=LEARNING_RATE)