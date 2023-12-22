from model.utils import *


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(
        description='FLSCNet: full-level supervision complementary network for infrared small target detection')

    # choose model
    parser.add_argument('--model', type=str, default='FLSCNet',
                        help='model name: FLSCNet')

    # parameter for FLSCNet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--fullLevel_supervision', type=bool, default=True, help='True or False (model==FLSCNet)')

    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name: NUAA-SIRST, IRSTD-1k')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name: TXT')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='80_20',
                        help='split method of dataset')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 1500)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        training (default: 4)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        testing (default: 4)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        help='default: CosineAnnealingLR scheduler')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    args = parser.parse_args()

    # make dir for save result
    args.save_dir = make_dir(args.dataset, args.model)

    # save training log
    save_train_log(args, args.save_dir)

    args.epoch_weight_path = './result/%s_FLSCNet_wDS/mIoU__FLSCNet_%s_epoch.pth.tar' % (args.dataset, args.dataset)
    args.best_weight_path = './result/%s_FLSCNet_wDS/Best_IoU__FLSCNet_%s_epoch.pth.tar' % (args.dataset, args.dataset)

    # the parser
    return args
