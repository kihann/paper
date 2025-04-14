from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64
IMAGE_SIZE = 96
ROOT = "./data"

def get_stl10_data(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, root=ROOT):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    dataset = STL10(
        root=root,
        split="train+unlabeled",
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    return dataloader