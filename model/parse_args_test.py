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

    #  hyper params for testing
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')

    args = parser.parse_args()

    args.epoch_weight_path = './result/%s_FLSCNet_wDS/mIoU__FLSCNet_%s_epoch.pth.tar' % (args.dataset, args.dataset)
    args.best_weight_path = './result/%s_FLSCNet_wDS/Best_IoU__FLSCNet_%s_epoch.pth.tar' % (args.dataset, args.dataset)

    # the parser
    return args
