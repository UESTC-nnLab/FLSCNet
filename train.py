# torch and visualization
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import load_dataset, load_param

# model
from model.model import FLSCNet


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.mIoU = mIoU(1)
        self.ROC = ROCMetric(1, 10)
        self.save_prefix = '_'.join([args.model, args.dataset])
        filters = load_param(args.channel_size)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                  transform=input_transform, suffix=args.suffix)
        testset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                transform=input_transform, suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'FLSCNet':
            model = FLSCNet(num_classes=1, input_channels=args.in_channels, filters=filters,
                            fullLevel_supervision=args.fullLevel_supervision)
        model = model.cuda()
        if args.start_epoch == 0:
            print('Init start')
            model.apply(weights_init_xavier)
            self.best_IoU = 0
        else:
            print('Continue')
            weights = torch.load(args.epoch_weight_path)['net_state_dict']
            model.load_state_dict(weights)
        self.model = model

        # Optimizer and lr scheduling
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        if args.start_epoch != 0:
            self.best_IoU = torch.load(args.best_weight_path)['mean_IoU']
            self.optimizer.load_state_dict(torch.load(args.epoch_weight_path)['optimizer_state_dict'])
            self.scheduler.load_state_dict(torch.load(args.epoch_weight_path)['scheduler_state_dict'])
            # print(torch.load(args.epoch_weight_path)['optimizer_state_dict']['param_groups'][0]['lr'])

    # Training
    def training(self, epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        if args.fullLevel_supervision and epoch <= (args.epochs - 300):
            factor = 0.1 * (1 - ((epoch - 1) // 300) * 300 / (args.epochs - 300))
        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            if args.fullLevel_supervision:
                preds = self.model(data)
                pred = preds[-1]
                loss = SoftIoULoss(pred, labels)
                if epoch <= (args.epochs - 300):
                    preds_rest = preds[:-1]
                    labels_rest = fullLevel_supervision_labels(labels)
                    for j, (predict, label) in enumerate(zip(preds_rest, labels_rest)):
                        loss += (0.5 ** (5 - j)) * SoftIoULoss(predict, label) * factor
            else:
                pred = self.model(data)
                loss = SoftIoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.scheduler.step()
        self.train_loss = losses.avg

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.ROC.reset()
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
                self.ROC.update(pred, labels)
                _, mean_IoU = self.mIoU.get()
                true_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IoU))
        test_loss = losses.avg
        # save high-performance model
        save_model(mean_IoU, self.best_IoU, args.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict(),
                   self.optimizer.state_dict(), self.scheduler.state_dict())
        if self.best_IoU < mean_IoU:
            self.best_IoU = mean_IoU


def main(args):
    if os.path.exists(args.epoch_weight_path):
        args.start_epoch = torch.load(args.epoch_weight_path)['epoch']
    trainer = Trainer(args)
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        trainer.training(epoch)
        trainer.testing(epoch)


def fullLevel_supervision_labels(labels):
    label1 = labels.clone()
    label2 = compute_label(labels, 2)
    label3 = compute_label(labels, 4)
    label4 = compute_label(labels, 8)
    label5 = compute_label(labels, 16)
    label6 = compute_label(labels, 32)
    return [label1, label2, label3, label4, label5, label6]


def compute_label(labels_demo, scale):
    labels_copy = labels_demo.clone()
    input_shape = labels_copy.size()
    indexs = torch.nonzero(labels_copy)
    for i in range(indexs.shape[0]):
        start_h = max(0, indexs[i][2] - scale // 2)
        end_h = min(input_shape[2], indexs[i][2] + scale // 2 + 1)
        start_w = max(0, indexs[i][3] - scale // 2)
        end_w = min(input_shape[3], indexs[i][3] + scale // 2 + 1)
        labels_copy[indexs[i][0], indexs[i][1], start_h:end_h, start_w:end_w] = 1

    return labels_copy


def set_seed(seed=6):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    # set_seed(6)
    args = parse_args()
    main(args)
