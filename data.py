import os
import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

BATCH_SIZE = 64
ROOT = "./dataset"

def get_transform():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])

def get_stl10_data(batch_size: int = BATCH_SIZE, root: str = ROOT) -> DataLoader:
    dataset = STL10(
        root=root,
        split="train+unlabeled",
        download=False,
        transform=get_transform()
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True
    )

def _debug():
    dataloader = get_stl10_data()
    print(f"{os.cpu_count()=}")
    
    for image, _ in dataloader:
        print(f"{image.size(0)=} / {image.shape=}")
        break

if __name__ == "__main__":
    _debug()