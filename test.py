# torch and visualization
from model.load_param_data import load_dataset, load_param
from model.loss import *
from model.metric import *

# model
from model.model import FLSCNet
from model.parse_args_test import parse_args

# metric, loss .etc
from model.utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.mIoU = mIoU(1)
        self.nIoU = nIoUMetric(1)
        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, args.ROC_thr)
        filters = load_param(args.channel_size)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            self.train_img_ids, self.val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset = TestSetLoader(dataset_dir, img_id=self.val_img_ids, base_size=args.base_size,
                                crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'FLSCNet':
            model = FLSCNet(num_classes=1, input_channels=args.in_channels, filters=filters,
                            fullLevel_supervision=args.fullLevel_supervision)
        model = model.cuda()
        weights = torch.load(args.best_weight_path)['net_state_dict']
        model.load_state_dict(weights)
        self.model = model
        print('Start evaluating')

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.nIoU.reset()
        self.ROC.reset()
        self.PD_FA.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.fullLevel_supervision:
                    preds = self.model(data)
                    pred = preds[-1]
                    loss = SoftIoULoss(pred, labels)
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.mIoU.update(pred, labels)
                self.nIoU.update(pred, labels)
                self.ROC.update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IoU = self.mIoU.get()
                nIoU = self.nIoU.get()
                true_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                FA, PD = self.PD_FA.get(len(self.val_img_ids))
                tbar.set_description(
                    'Epoch %d, test loss %.4f, mean_IoU: %.4f, nIoU: %.4f, Recall: %.4f, Precision: %.4f, PD: %.10f, FA: %.10f' % (
                        epoch, losses.avg, mean_IoU, nIoU, recall[5], precision[5], PD[0], FA[0]))
        print('True positive rate:')
        print(true_positive_rate)
        print('False positive rate:')
        print(false_positive_rate)


def main(args):
    trainer = Trainer(args)
    trainer.testing(torch.load(args.best_weight_path)['epoch'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
