import os
from os.path import join as pjoin
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from kornia import rgb_to_ycbcr
import numpy as np
import skimage.io as io
from omegaconf import OmegaConf
import datetime

from data.dataset import ImageFolder


def initialize_project(refined_img_path, refinement_visualization_path, save_ckpt_path, dataset_path, args):
    print('Initializing the project')

    os.makedirs(refined_img_path, exist_ok=True)
    os.makedirs(refinement_visualization_path, exist_ok=True)
    os.makedirs(save_ckpt_path, exist_ok=True)
    assert os.path.exists(dataset_path)

    print('copying %s dataset to %s' % (args['training_config']['dataset'], refined_img_path))

    img_txt = open(pjoin(dataset_path, 'train.txt'))
    for idx, line in enumerate(img_txt):
        img = Image.open(pjoin(dataset_path, line.split()[0]))
        gts = Image.open(pjoin(dataset_path, line.split()[1]))

        os.makedirs(pjoin(refined_img_path, *((line.split()[0]).split('/')[:-1])), exist_ok=True)
        os.makedirs(pjoin(refined_img_path, *((line.split()[1]).split('/')[:-1])), exist_ok=True)

        img.save(pjoin(refined_img_path, line.split()[0]))
        gts.save(pjoin(refined_img_path, line.split()[1]))

        refine_resolution = args['training_config']['refine']['refine_resolution']
        img = transforms.Resize((refine_resolution, refine_resolution))(img)
        gts = transforms.Resize((refine_resolution, refine_resolution))(gts)
        img = transforms.ToTensor()(img)
        gts = transforms.ToTensor()(gts)

        img_save = torch.zeros((3, refine_resolution * 4, refine_resolution * 2 + 100))
        img_save[:, 0:refine_resolution, 0:refine_resolution] = img.squeeze(0)
        img_save[:, refine_resolution:refine_resolution * 2, 0:refine_resolution] = img.squeeze(0)
        img_save[:, 0:refine_resolution, refine_resolution + 100:refine_resolution * 2 + 100] = torch.tile(
            gts.squeeze(0), (3, 1, 1))
        img_save[:, refine_resolution:refine_resolution * 2,
        refine_resolution + 100:refine_resolution * 2 + 100] = torch.tile(gts.squeeze(0), (3, 1, 1))
        img_save = transforms.ToPILImage()(img_save)
        img_save.save('%s/%d.jpg' % (refinement_visualization_path, idx + 1))

        print('copying: %d' % (idx + 1))
    img_txt.close()

    os.system('cp %s %s' % (pjoin(dataset_path, 'train.txt'), refined_img_path))
    assert os.path.exists(pjoin(refined_img_path, 'train.txt'))


def build_dataloader(refined_img_path, dataset_path, args):
    additional_dataset_path = args['data_dir']['additional_dataset']
    args_train = args['training_config']
    train_on_ucf = args_train['dataset'] == 'UCF'

    if not args_train['data']['use_additional_dataset']:
        additional_dataset_path = refined_img_path

    print('Building dataset')

    train_set = ImageFolder(
        root=list(set([refined_img_path, additional_dataset_path])),
        mode='train',
        resolution=args_train['data']['resolution'],
        batch_size=args_train['train']['batch_size'],
        refine_resolution=args_train['refine']['refine_resolution']
    )

    test_set = ImageFolder(
        root=[args['data_dir']['SBU']] if train_on_ucf else [dataset_path],
        mode='validate',
        resolution=args_train['data']['resolution'],
        batch_size=args_train['train']['batch_size'],
        refine_resolution=args_train['refine']['refine_resolution']
    )

    refine_set = ImageFolder(
        root=[refined_img_path],
        mode='refine',
        resolution=args_train['data']['resolution'],
        batch_size=1,
        refine_resolution=args_train['refine']['refine_resolution']
    )

    train_loader = DataLoader(train_set, batch_size=args_train['train']['batch_size'], num_workers=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1)
    refine_loader = DataLoader(refine_set, batch_size=1, num_workers=1, shuffle=False)

    return train_loader, test_loader, refine_loader


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = None

    def update(self, val, n=1):
        if self.avg is None:
            self.avg = val * n
        else:
            self.avg = 0.99 * self.avg + 0.01 * val * n


def check_refine(inputs, res, gts, idx, res_global, res_local, args_refine, refinement_visualization_path=None):
    YCbCr = rgb_to_ycbcr(inputs)
    luminosity = YCbCr[0:1]

    lum_min = torch.min(luminosity)
    lum_max = torch.max(luminosity)
    delta = (args_refine['final_threshold_end'] - args_refine['final_threshold_begin']) / (lum_max - lum_min)

    threshold_map = torch.ones_like(res) * args_refine['final_threshold_end'] - (luminosity - lum_min) * delta

    gts_bool = gts > 0.49
    res_bool = res > threshold_map
    pos = ((gts_bool == 1) & (res_bool == 1)).sum().item()
    neg = ((gts_bool == 1) & (res_bool == 0)).sum().item()
    if pos + neg > 0:
        correctness = pos / (pos + neg)
    else:
        correctness = 0

    if correctness > args_refine['correctness_threshold']:
        refined = res
        refined = (refined > threshold_map).float()
        to_save = torch.tile(res, (3, 1, 1)).cpu()
        is_refined = True
    else:
        refined = gts
        to_save = torch.tile(gts, (3, 1, 1)).cpu()
        is_refined = False

    if refinement_visualization_path is not None and os.path.exists('%s/%d.jpg' % (refinement_visualization_path, idx)):
        prev = Image.open('%s/%d.jpg' % (refinement_visualization_path, idx))
        prev = transforms.ToTensor()(prev)
        w = prev.shape[-1]

        img_save = torch.zeros((3, args_refine['refine_resolution'] * 4, w + 100 + args_refine['refine_resolution']))
        img_save[:, :, 0:w] = prev
        img_save[:, 0:args_refine['refine_resolution'], w + 100:w + 100 + args_refine['refine_resolution']] = to_save
        img_save[:, args_refine['refine_resolution']:args_refine['refine_resolution'] * 2,
        w + 100:w + 100 + args_refine['refine_resolution']] = torch.tile(refined.cpu(), (3, 1, 1))
        img_save[:, args_refine['refine_resolution'] * 2:args_refine['refine_resolution'] * 3,
        w + 100:w + 100 + args_refine['refine_resolution']] = torch.tile(res_global.cpu(), (3, 1, 1))
        img_save[:, args_refine['refine_resolution'] * 3:args_refine['refine_resolution'] * 4,
        w + 100:w + 100 + args_refine['refine_resolution']] = torch.tile(res_local.cpu(), (3, 1, 1))
        img_save = transforms.ToPILImage()(img_save)
        img_save.save('%s/%d.jpg' % (refinement_visualization_path, idx))

    return refined, is_refined


def save_mask(img, result, gts, curr_iter, project_path, size):
    batch_size = result.shape[0]
    bool_result = result > 0.5

    img = (img.numpy() * 255).astype('uint8')
    result = (result.numpy() * 255).astype('uint8')
    gts = (gts.numpy() * 255).astype('uint8')
    bool_result = (bool_result.numpy() * 255).astype('uint8')

    gts = np.tile(gts, (1, 3, 1, 1))
    result = np.tile(result, (1, 3, 1, 1))
    bool_result = np.tile(bool_result, (1, 3, 1, 1))

    h, w = size, size

    img_list = np.array([img, gts, bool_result, result])
    img_list = np.transpose(img_list, (0, 1, 3, 4, 2))

    img = np.zeros((batch_size * h, 4 * w, 3))
    for i in range(batch_size):
        for j in range(4):
            img[i * h:i * h + h, j * w:j * w + w] = img_list[j][i]

    save_path = pjoin(project_path, 'training_visualization')
    os.makedirs(save_path, exist_ok=True)

    io.imsave(pjoin(save_path, '%d_iter.jpg' % (curr_iter + 1)), img)


def get_model_args(args_from_parser):
    save_project_dir = args_from_parser.save_project_dir
    project_name = args_from_parser.project_name

    if project_name is None:
        project_name = datetime.datetime.now().strftime(
            "%Y-%m-%dT%H-%M-%S") + '_' + args_from_parser.backbone + '_on_' + args_from_parser.dataset

    os.makedirs(pjoin(save_project_dir, project_name), exist_ok=True)

    config = OmegaConf.load(args_from_parser.config)
    config['project_config'] = {
        'save_project_dir': save_project_dir,
        'project_name': project_name
    }
    config['training_config']['dataset'] = args_from_parser.dataset
    config['training_config']['backbone'] = args_from_parser.backbone

    return config

