from models.resnet import resnet18, resnet50
from models.efficientnetv2 import efficientnet_b0
from models.examplenet import examplenet
from models.resnetv2 import resnet18v2
import torch
from torchsummary import summary
import torch.backends.cudnn as cudnn

def opt(parser):
    # Parameters
    parser.add_argument('--img_rows', default = 112, type=int, help='image rows')
    parser.add_argument('--img_cols', default = 112, type=int, help='image cols')
    parser.add_argument('--img_size', default = 112, type=int, help='image size')
    parser.add_argument('--batch_size', default = 16, type=int, help='batch size')
    parser.add_argument('--num_workers', default = 4, type=int, help='num workers to prepare data')
    parser.add_argument('--num_classes', default = 4, type=int, required=False, help='num classes')
    parser.add_argument('--epoch', default = 100, type=int, help='epochs')
    parser.add_argument('--train_path', type=str, required=False, help='train path') # not using
    parser.add_argument('--test_path', type=str, required=False, help='test path to visualize')
    parser.add_argument('--model', default = 'resnet18', type=str, required=False, help='model. resnet18|resnet50|efficientnetb0|...')
    parser.add_argument('--checkpoint', default = './checkpoint/ckpt_best.pth', type=str, required=False, 
                        help='checkpoint for resuming training, testing and converting to onnx')
    parser.add_argument('--data_dir', type=str, required=False, help='data dir must contain train, test folder')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--step_size', default=20, type=int, help='decay learning rate step size')
    parser.add_argument('--gamma', default=0.1, type=int, help='factor learning rate decay')
    parser.add_argument('--adam', action='store_true', help='use adam, default is SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--eval', action='store_true', help='no training, only evaluating')
    parser.add_argument('--balance', action='store_true', 
                        help='imbalanced dataset sampler. Source: https://github.com/ufoym/imbalanced-dataset-sampler#Usage')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save_checkpoint', default = 10, type=int, required=False, help='save every save_checkpoint epoch')
    args = parser.parse_args()
    return args

def model(args):
    print('Building model..')
    if args.model == 'resnet18':
        net = resnet18(num_classes=args.num_classes)
    if args.model == 'resnet50':
        net = resnet50(num_classes=args.num_classes)
    if args.model == 'efficientnetb0':
        net = efficientnet_b0(num_classes=args.num_classes)
    if args.model == 'examplenet':
        net = examplenet(num_classes=args.num_classes)
    if args.model == 'resnet18v2':
        net = resnet18v2(3, n_classes=args.num_classes)
    # add new model here
    return net