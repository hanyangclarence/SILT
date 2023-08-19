import torch
import torchvision.transforms as tfs
import random
import numpy as np
import cv2
import torch.nn.functional as F


def GaussianBlur(image, ratio):
    kernel_size = int(13 * ratio)
    if kernel_size % 2 != 1:
        kernel_size += 1
    image = tfs.GaussianBlur(kernel_size, 2 * ratio)(image)
    return image


def RandomPerspective(image, ratio):
    image = tfs.RandomPerspective(0.6 * ratio, 1)(image)
    return image


def Rotate(image, ratio):
    image = tfs.RandomRotation((-179 * ratio, 179 * ratio))(image)
    return image


def TranslateX(image, ratio):
    image = tfs.RandomAffine(degrees=(0, 0), translate=(0.7 * ratio, 0))(image)
    return image


def TranslateY(image, ratio):
    image = tfs.RandomAffine(degrees=(0, 0), translate=(0, 0.7 * ratio))(image)
    return image


def ShearX(image, ratio):
    image = tfs.RandomAffine(degrees=(0, 0), shear=60 * ratio)(image)
    return image


def ShearY(image, ratio):
    image = tfs.RandomAffine(degrees=(0, 0), shear=(0, 0, -60 * ratio, 60 * ratio))(image)
    return image


def Brightness(image, ratio):
    r = 0.9 * ratio
    image = tfs.ColorJitter(brightness=r)(image)
    return image


def Sharpness(image, ratio):
    image = tfs.RandomAdjustSharpness(10 * ratio, 1)(image)
    return image


def Contrast(image, ratio):
    r = 0.7 * ratio
    image = tfs.ColorJitter(contrast=r)(image)
    return image


def HorizontalFlip(image, ratio):
    return tfs.RandomHorizontalFlip(p=1)(image)


def VerticalFlip(image, ratio):
    return tfs.RandomVerticalFlip(p=1)(image)


def RandomCrop(image, ratio):
    size = image.shape[-1]
    return tfs.RandomResizedCrop(size=(size, size))(image)


def Distraction(image, gts, ratio):
    res = image.shape[-1]
    idx = random.randint(1, 120)
    ranges = int(res * ratio)

    edges = 3 + idx % 5
    points = []
    x_translate = random.randint(0, res - ranges)
    y_translate = random.randint(0, res - ranges)
    for _ in range(edges):
        x = random.randint(0, ranges - 1) + x_translate
        y = random.randint(0, ranges - 1) + y_translate
        points.append([x, y])
    points = np.array(points)
    mask = np.zeros((res, res))
    cv2.fillConvexPoly(mask, points, 1)

    mask = torch.tensor(mask)
    mask = mask.unsqueeze(0).unsqueeze(0)

    kernel = torch.tensor([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], requires_grad=False, dtype=torch.double).unsqueeze(0).unsqueeze(0)
    filtered_mask = F.conv2d(mask, kernel, padding=1)
    mask = filtered_mask > 5

    mask_img = (mask == 0).float() + mask * 0.2
    final_image = image * mask_img
    final_gts = gts
    
    return final_image, final_gts


def RandomNoise(image, ratio):
    res = image.shape[-1]
    max_noise = int(res * res * ratio * 0.3)
    x = random.choices(range(res), k=max_noise)
    y = random.choices(range(res), k=max_noise)
    x = torch.tensor(x)
    y = torch.tensor(y)
    image[:, :, x, y] = 0
    return image


def cat_two(img, gts):
    gts = torch.tile(gts, (1, 3, 1, 1))
    cated = torch.cat((img, gts), dim=0)
    return cated


def separate(cated):
    batch_size = cated.shape[0] // 2
    img = cated[0:batch_size, :, :, :]
    gts = cated[batch_size:2 * batch_size, 0:1, :, :]
    return img, gts


options = [
    GaussianBlur,
    RandomPerspective,
    Rotate,
    TranslateX,
    TranslateY,
    ShearX,
    ShearY,
    Brightness,
    Sharpness,
    Contrast,
    RandomNoise,
    Distraction
]


together = {
    'GaussianBlur': False,
    'RandomPerspective': True,
    'Rotate': True,
    'TranslateX': True,
    'TranslateY': True,
    'ShearX': True,
    'ShearY': True,
    'Brightness': False,
    'Sharpness': False,
    'Contrast': False,
    'RandomNoise': False
}


def data_augmentation(img, gts, num_layer, magnitude):
    selected_aug = random.choices(options, k=num_layer)

    for fn in selected_aug:
        name = fn.__name__
        if name == 'Distraction':
            img, gts = fn(img, gts, magnitude * 1.6)
        else:
            if together[fn.__name__]:
                cated = cat_two(img, gts)
                cated = fn(cated, magnitude)
                img, gts = separate(cated)
            else:
                img = fn(img, magnitude)

    return img, gts

