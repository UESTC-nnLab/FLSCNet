# Basic module
from tqdm import tqdm
from model.parse_args_test import parse_args

# Torch and visualization
from torchvision import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.loss import *
from model.load_param_data import load_dataset, load_param

# Model
from model.model import FLSCNet


class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args = args
        filters = load_param(args.channel_size)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, self.val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

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
        self.model.eval()
        print('Start visualizing')

        visualization_path = dataset_dir + '/' + 'visualization_result' + '/' + args.model + '_visualization_result'
        visualization_fuse_path = dataset_dir + '/' + 'visualization_result' + '/' + args.model + '_visualization_fuse'

        make_visualization_dir(visualization_path, visualization_fuse_path)

        # Visualizing
        tbar = tqdm(self.test_data)
        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.fullLevel_supervision:
                    preds = self.model(data)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                save_Pred_GT(pred, labels, visualization_path, self.val_img_ids, i, args.suffix)

        total_visualization_generation(dataset_dir, test_txt, args.suffix, visualization_path,
                                          visualization_fuse_path)


def main(args):
    Trainer(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
