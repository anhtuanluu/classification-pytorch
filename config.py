import argparse
from models.resnet import resnet18, resnet50
from models.efficientnet import EfficientNetB0
import torch
from torchsummary import summary
import torch.backends.cudnn as cudnn

def opt():
    # Parameters
    parser = argparse.ArgumentParser(description='Image Classification.')

    parser.add_argument('--img_rows', default = 112, type=int, help='image rows')
    parser.add_argument('--img_cols', default = 112, type=int, help='image cols')
    parser.add_argument('--img_size', default = 112, type=int, help='image size')
    parser.add_argument('--batch_size', default = 16, type=int, help='batch size')
    parser.add_argument('--num_workers', default = 4, type=int, help='num workers to prepare data')
    parser.add_argument('--num_classes', default = 4, type=int, required=False, help='num classes')
    parser.add_argument('--epoch', default = 100, type=int, help='epochs')
    parser.add_argument('--train_path', type=str, required=False, help='train path') # not using
    parser.add_argument('--test_path', type=str, required=False, help='test path to visualize')
    parser.add_argument('--model', default = 'resnet18', type=str, required=False, help='model. resnet18|resnet50|efficientnet')
    parser.add_argument('--checkpoint_test', default = './checkpoint/ckpt_best.pth', type=str, required=False, help='check point for test')
    parser.add_argument('--data_dir', type=str, required=False, help='data dir must contain train, test folder')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--eval', action='store_true', help='only eval')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from best checkpoint')
    parser.add_argument('--save_checkpoint', default = 10, type=int, required=False, help='save every save_checkpoint epoch')
    args = parser.parse_args()
    return args

def model(args):
    print('Building model..')
    if args.model == 'resnet18':
        net = resnet18(num_classes=args.num_classes)
    if args.model == 'resnet50':
        net = resnet50(num_classes=args.num_classes)
    if args.model == 'efficientnet':
        net = EfficientNetB0(num_classes=args.num_classes)
    return net