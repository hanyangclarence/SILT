import os
import time
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from os.path import join as pjoin
import argparse
from omegaconf import OmegaConf

from model.network import Network

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2019)
torch.cuda.set_device(0)


def main(args_from_parser):
    args_infer = OmegaConf.load(args_from_parser.inference_config)
    checkpoint = torch.load(args_from_parser.ckpt)
    args_model = checkpoint['configs']

    inference_save_dir = args_from_parser.save_dir
    if inference_save_dir is None:
        if 'project_config' in args_model and os.path.exists(
                pjoin(args_model['project_config']['save_project_dir'], args_model['project_config']['project_name'])):
            inference_save_dir = pjoin(args_model['project_config']['save_project_dir'],
                                       args_model['project_config']['project_name'], 'infer_results')
        else:
            inference_save_dir = 'infer_results'

    transform = transforms.Compose([
        transforms.Resize([args_infer['resolution'], args_infer['resolution']]),
        transforms.ToTensor()
    ])

    to_test = {
        args_from_parser.dataset: args_model['data_dir'][args_from_parser.dataset]
    }

    os.makedirs(inference_save_dir, exist_ok=True)

    net = Network(
        input_resolution=args_model['training_config']['data']['resolution'],
        backbone=args_model['training_config']['backbone'],
        backbone_ckpt=args_model['pretrained_encoder_dir'][args_model['training_config']['backbone']]
    ).cuda()
    net.load_state_dict(checkpoint['state_dict'])

    net.eval()
    total_time = 0

    total_Tp = 0
    total_Tn = 0
    total_P = 0
    total_N = 0

    with torch.no_grad():
        for name, root in to_test.items():

            img_txt = open(pjoin(root, 'test.txt'))
            img_name = []
            mask_name = []

            for img_list in img_txt:
                x = img_list.split()
                img_name.append(x[0])
                mask_name.append(x[1])

            for idx, image_name in enumerate(img_name):
                img = Image.open(pjoin(root, image_name))
                mask = Image.open(pjoin(root, mask_name[idx]))

                w, h = img.size
                img_var = transform(img).unsqueeze(0)
                img_var = img_var.cuda()

                start_time = time.time()
                if args_infer['crop']:
                    crop_num = args_infer['crop_size'] * args_infer['crop_size']
                    img_sep = torch.zeros((crop_num + 1, 3, args_infer['resolution'], args_infer['resolution'])).cuda()

                    img_sep[-1] = img_var.squeeze(0).clone()

                    length = args_infer['resolution'] // args_infer['crop_size']
                    for i in range(args_infer['crop_size']):
                        for j in range(args_infer['crop_size']):
                            img_sep[i * args_infer['crop_size'] + j] = transforms.Resize(args_infer['resolution'])(
                                img_var[:, :, \
                                i * length:(i + 1) * length, j * length:(j + 1) * length].squeeze(0))

                    res_ini = net(img_sep)
                    res_global = res_ini[-1].unsqueeze(0).clone()

                    res_ini = transforms.Resize(length)(res_ini)

                    res_local = torch.zeros((1, 1, args_infer['resolution'], args_infer['resolution'])).cuda()
                    for i in range(args_infer['crop_size']):
                        for j in range(args_infer['crop_size']):
                            res_local[:, :, i * length:(i + 1) * length, j * length:(j + 1) * length] = \
                                res_ini[i * args_infer['crop_size'] + j:i * args_infer['crop_size'] + j + 1]

                    res_local = res_local * (res_global > args_infer['filter_threshold'])
                    res = torch.maximum(res_global, res_local)

                else:
                    res = net(img_var)

                res = (res > args_infer['threshold_for_result']).float()

                torch.cuda.synchronize()
                total_time = total_time + time.time() - start_time
                print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_name), total_time / (idx + 1)))

                res = transforms.Resize((h, w))(res.squeeze(0))
                res = (res > args_infer['threshold_for_result']).float()
                res = transforms.ToPILImage()(res)
                sub_name = image_name.split('/')
                res.save(pjoin(inference_save_dir, sub_name[-1]))

                result_for_BER = np.array(res)
                result_for_BER = result_for_BER > 255 * 0.5
                mask = np.array(mask)
                mask = mask > 0.5

                total_P += np.sum(mask == True).item()
                total_N += np.sum((mask == False)).item()
                total_Tp += np.sum((mask == True) & (result_for_BER == True)).item()
                total_Tn += np.sum((mask == False) & (result_for_BER == False)).item()

    total_BER = 0.5 * (2 - total_Tp / total_P - total_Tn / total_N) * 100

    pos = (1 - total_Tp / total_P) * 100
    neg = (1 - total_Tn / total_N) * 100

    print('%.2f, %.2f, %.2f' % (total_BER, pos, neg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inference_config",
        type=str,
        required=False,
        default='/home/wangty/Projects/Shadow/shadow_detection/SILT/configs/inference_config.yaml',
        help="The yaml file for the inference configuration",
    )

    parser.add_argument(
        '-c',
        "--ckpt",
        type=str,
        required=True,
        help="the checkpoint to test",
    )

    parser.add_argument(
        '-s',
        "--save_dir",
        type=str,
        required=False,
        default=None,
        help="dir to save the model",
    )

    parser.add_argument(
        '-d',
        "--dataset",
        type=str,
        required=False,
        default='SBU',
        help="dataset to test",
        choices=['SBU', 'ISTD']
    )

    args_from_parser = parser.parse_args()
    main(args_from_parser)
