import os
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import argparse

from torch.backends import cudnn

cudnn.benchmark = True
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from model.network import Network, Weighted_BCE_Loss
from data.data_augmentation import data_augmentation
from utils import initialize_project, build_dataloader, AvgMeter, check_refine, save_mask, get_model_args

history_BER = None
history_epoch = None


def main(args):
    global history_BER, history_epoch  # to record the best ckpt in training
    args_train = args['training_config']

    project_dir = pjoin(args['project_config']['save_project_dir'], args['project_config']['project_name'])

    refined_img_path = pjoin(project_dir, 'refined_data_%s' % args_train['dataset'])
    refinement_visualization_path = pjoin(project_dir, 'refinement_visualize')
    dataset_path = args['data_dir'][args_train['dataset']]
    save_ckpt_path = pjoin(project_dir, 'weights')

    # prepare data, directories and visualizations
    initialize_project(refined_img_path, refinement_visualization_path, save_ckpt_path, dataset_path, args)

    net = Network(
        input_resolution=args_train['data']['resolution'],
        backbone=args_train['backbone'],
        backbone_ckpt=args['pretrained_encoder_dir'][args_train['backbone']]
    ).cuda()

    tensorboard_path = pjoin(project_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_path)

    for T in range(args_train['round']):

        train_loader, test_loader, refine_loader = build_dataloader(refined_img_path, dataset_path, args)
        loss_fn = Weighted_BCE_Loss(mu=0.7, eps=1e-5)
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, net.parameters()), args_train['train']['lr'])

        history_BER = 9999
        history_epoch = 0

        train(net, optimizer, loss_fn, writer, train_loader, test_loader, T + 1, project_dir, args)

        if T != args_train['round'] - 1:
            refine(net, refine_loader, T + 1, refinement_visualization_path, args)

    writer.close()


def train(net, optimizer, loss_fn, writer, train_loader, test_loader, round, project_path, args):
    net.train()
    train_net_loss_record = AvgMeter()
    total_iter = 0
    args_train = args['training_config']

    for epoch in range(args_train['train']['total_epoch']):
        print('total parameters: %d' % (sum(p.numel() for p in net.parameters())))

        for data in train_loader:
            total_iter += 1

            inputs, gts = data
            if args_train['data']['data_augmentation']:
                inputs, gts = data_augmentation(inputs, gts, args_train['data']['layers'],
                                                args_train['data']['augmentation_magnitude'])
            inputs = inputs.cuda()
            gts = gts.cuda()

            optimizer.zero_grad()
            result = net(inputs)
            loss_net = loss_fn(result, gts)
            loss_net.backward()
            optimizer.step()

            writer.add_scalar('Loss', loss_net, total_iter)
            train_net_loss_record.update(loss_net.data, inputs.size(0))

            log = '[round %d] [epoch %d] [iter %d] [train loss %.5f] [curr_optimal %.4f]' % \
                  (round, epoch + 1, total_iter, train_net_loss_record.avg, history_BER)
            print(log)

            if (total_iter + 1) % args_train['train']['save_mask_freq'] == 0:
                with torch.no_grad():
                    save_mask(inputs.cpu(), result.cpu(), gts.cpu(), total_iter, project_path,
                              args_train['data']['resolution'])

        validate(net, epoch + 1, test_loader, round, writer, args)


def validate(net, epoch, test_loader, round, writer, args):
    print('validating...')
    net.eval()

    total_Tp = 0
    total_Tn = 0
    total_P = 0
    total_N = 0

    global history_BER, history_epoch

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, gts = data
            inputs = inputs.cuda()
            gts = gts.cuda()

            res = net(inputs)

            gts_map = gts > 0.5
            res_map = res > 0.5

            total_P += torch.sum(gts_map == True).item()
            total_N += torch.sum((gts_map == False)).item()
            total_Tp += torch.sum((gts_map == True) & (res_map == True)).item()
            total_Tn += torch.sum((gts_map == False) & (res_map == False)).item()

            print('validating: %d/%d' % (i + 1, len(test_loader)))

    BER = 0.5 * (2 - total_Tp / total_P - total_Tn / total_N) * 100
    writer.add_scalar('BER on validation, round %d' % (round), BER, epoch + 1)

    if (BER < history_BER):
        save_ckpt_path = pjoin(args['project_config']['save_project_dir'], args['project_config']['project_name'],
                               'weights')

        # remove previous ckpt
        old_weight_path = pjoin(save_ckpt_path, '%.4f_round%d_epoch%d.pth' % (history_BER, round, history_epoch))
        if os.path.exists(old_weight_path):
            os.system('rm %s' % old_weight_path)

        history_BER = BER
        history_epoch = epoch
        new_weight_path = pjoin(save_ckpt_path, '%.4f_round%d_epoch%d.pth' % (history_BER, round, history_epoch))
        checkpoint = {
            'state_dict': net.state_dict(),
            'configs': args
        }
        torch.save(checkpoint, new_weight_path)

    net.train()


def refine(net, refine_loader, round, refinement_visualization_path, args):
    print('refining...')

    # load the best ckpt
    params_path = pjoin(args['project_config']['save_project_dir'], args['project_config']['project_name'], 'weights',
                        '%.4f_round%d_epoch%d.pth' % (history_BER, round, history_epoch))
    net.load_state_dict(torch.load(params_path, map_location=lambda storage, loc: storage.cuda(0))['state_dict'])

    net.eval()

    args_refine = args['training_config']['refine']
    with (torch.no_grad()):
        for idx, data in enumerate(refine_loader):
            if idx == len(refine_loader) - 1:  # this is a problem with the dataloader
                break

            img, gts, gts_path, size = data
            img_var = img.cuda()

            if args_refine['crop']:
                crop_num = args_refine['crop_size'] * args_refine['crop_size']
                img_sep = torch.zeros(
                    (crop_num + 1, 3, args_refine['refine_resolution'], args_refine['refine_resolution'])).cuda()

                img_sep[-1] = img_var.squeeze(0).clone()

                length = args_refine['refine_resolution'] // args_refine['crop_size']
                for i in range(args_refine['crop_size']):
                    for j in range(args_refine['crop_size']):
                        img_sep[i * args_refine['crop_size'] + j] = transforms.Resize(args_refine['refine_resolution'])(
                            img_var[:, :, \
                            i * length:(i + 1) * length, j * length:(j + 1) * length].squeeze(0))

                res_ini = net(img_sep)
                res_global = res_ini[-1].unsqueeze(0).clone()

                res_ini = transforms.Resize(length)(res_ini)

                res_local = torch.zeros(
                    (1, 1, args_refine['refine_resolution'], args_refine['refine_resolution'])).cuda()
                for i in range(args_refine['crop_size']):
                    for j in range(args_refine['crop_size']):
                        res_local[:, :, i * length:(i + 1) * length, j * length:(j + 1) * length] = \
                            res_ini[i * args_refine['crop_size'] + j:i * args_refine['crop_size'] + j + 1]

                res_local = res_local * (res_global > args_refine['filter_threshold'])
                res = torch.maximum(res_global, res_local)

            else:
                res = net(img_var)
                res_global = res
                res_local = res

            gts_refined, is_refined = check_refine(  # check whether to accept the refined mask
                img.squeeze(0).cuda(), res.squeeze(0), gts.squeeze(0).cuda(),
                idx + 1, res_global.squeeze(0), res_local.squeeze(0),
                args_refine,
                refinement_visualization_path)
            gts_refined = transforms.Resize(size)(transforms.ToPILImage()(gts_refined))
            if is_refined:
                gts_refined.save(gts_path[0])

            print('refining: %d/%d' % (idx + 1, len(refine_loader)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        "--config",
        type=str,
        required=False,
        default='configs/silt_training_config.yaml',
        help="The yaml file for the model's configuration",
    )

    parser.add_argument(
        '-s',
        "--save_project_dir",
        type=str,
        required=False,
        default='ckpt',
        help="path to save the project results",
    )

    parser.add_argument(
        '-n',
        "--project_name",
        type=str,
        required=False,
        default=None,
        help="name of the project",
    )

    parser.add_argument(
        '-d',
        "--dataset",
        type=str,
        required=False,
        default='SBU',
        help="training dataset",
        choices=['SBU', 'UCF', 'ISTD']
    )

    parser.add_argument(
        '-b',
        "--backbone",
        type=str,
        required=False,
        default='PVT-b5',
        help="network backbone name",
        choices=['ResNeXt', 'efficientnet-b3', 'efficientnet-b7', 'efficientnet-b8',
                 'convnext-small', 'convnext-base', 'PVT-b3', 'PVT-b5']
    )

    args_from_parser = parser.parse_args()
    args = get_model_args(args_from_parser)

    main(args)
