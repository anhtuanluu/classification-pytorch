import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from config import opt, model
import os
from utils import count_parameters, progress_bar, imshow
from sklearn import metrics
from torchsummary import summary
import time
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'
''' Inference in a folder 'test'. Show data dir training to get class name '''

# config
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--check_wrong', action='store_true', help='show wrong images, set false to turn off')
args = opt(parser)

# test image in folder and show
data_dir = args.data_dir # get class names
test_dir = args.test_path # test path

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
test_dataset = datasets.ImageFolder(os.path.join(test_dir, 'test'),
                                          data_transforms['test'])
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=False, num_workers=args.num_workers)
class_names = train_dataset.classes
print("Classes: {}".format(class_names))
# model
net = model(args)
net = net.to(device)
# print parameters
summary(net, (3, args.img_size, args.img_size))
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

assert os.path.isfile(args.checkpoint), 'Error: no checkpoint found!'
# load checkpoint.
print('Resuming from checkpoint {}'.format(args.checkpoint))
checkpoint = torch.load(args.checkpoint)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
net.eval()

if not check_wrong:
    print("Showing image...")
    for image, target in testloader:
        image = image.to(device)
        start_time = time.time()
        # predict
        output = net(image)
        _, predicted = output.max(1)
        index = predicted[0]
        # get label
        label = class_names[index]
        end_time = time.time()
        image = image.squeeze(dim = 0)
        print("Predict: {}. Time infer: {:.4f}".format(label, end_time-start_time))
        imshow(image, delay=5)
        # break
else:
    print("Showing wrong predicted image...")
    for image, target in testloader:
        image, target = image.to(device), target.to(device).tolist()
        start_time = time.time()
        # predict
        output = net(image)
        _, predicted = output.max(1)
        index = predicted[0]
        # get label
        label = class_names[index]
        label_ = class_names[target[0]]
        end_time = time.time()
        image = image.squeeze(dim = 0)
        if index != target[0]:
            print("Predict: {}. Truth: {}. Time infer: {:.4f}".format(label, label_, end_time-start_time))
            imshow(image, delay=5)
        # break