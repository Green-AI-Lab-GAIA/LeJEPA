import torch

from torchvision.transforms import v2

global_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(32, padding=4, padding_mode='reflect'),
    v2.RandomApply([v2.RandomRotation(10, interpolation=v2.InterpolationMode.BILINEAR)], p=0.5),
    v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.03)
])

local_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(32, padding=4, padding_mode='reflect'),
    v2.RandomApply([v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.8), shear=0.1, interpolation=v2.InterpolationMode.BILINEAR, )], p=0.8),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    v2.RandomGrayscale(p=0.1),
])

common_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

class MultiviewTransform:
    def __init__(self, nglobal=2, nlocal=4):
        self.nglobal = nglobal
        self.nlocal = nlocal

    def __call__(self, x):
        views = []
        views.extend([global_transform(x) for _ in range(self.nglobal)])
        views.extend([local_transform(x) for _ in range(self.nlocal)])
        return views
