import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T


def make_dataset(root, mode):
    img_name = []
    if mode == 'train' or mode == 'refine':
        txt_name = 'train.txt'
    elif mode == 'validate':
        txt_name = 'test.txt'
    else:
        ValueError()
    
    img_name = []
    for rt in root:
        img_txt = open(os.path.join(rt, txt_name))

        for img_list in img_txt:
            x = img_list.split()
            img_name.append([os.path.join(rt, x[0]), os.path.join(rt, x[1])])

        img_txt.close()

    return img_name


class ImageFolder(data.Dataset):
    def __init__(self, root, mode, resolution, batch_size=4, refine_resolution=256):
        self.root = root
        self.mode = mode  # mode can be train, validate, refine
        self.imgs = make_dataset(root, mode)
        self.resolution = resolution
        self.batch_size = batch_size
        self.refine_resolution = refine_resolution

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index % len(self.imgs)]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        w, h = target.size
        
        if self.mode != 'refine':
            img = T.Resize((self.resolution, self.resolution))(img)
            target = T.Resize((self.resolution, self.resolution))(target)
        else:
            img = T.Resize((self.refine_resolution, self.refine_resolution))(img)
            target = T.Resize((self.refine_resolution, self.refine_resolution))(target)

        img = T.ToTensor()(img)
        target = T.ToTensor()(target)

        if self.mode == 'train' or self.mode == 'validate':
            return img, target
        else:
            return img, target, gt_path, (h, w)

    def __len__(self):
        return len(self.imgs) + self.batch_size - (len(self.imgs) % self.batch_size)
