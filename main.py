import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from config import opt, model
import os
from utils import count_parameters, progress_bar, compute_class_weights
from sklearn import metrics
from torchsummary import summary
from dataset.imbalanced import ImbalancedDatasetSampler
import argparse
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# config
parser = argparse.ArgumentParser(description='Image classification')
args = opt(parser)
# writer
writer = SummaryWriter('checkpoint/tensorboard')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('Preparing data...')
# data augment
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# prepare data
data_dir = args.data_dir # must include train, test folder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
if not args.balance:
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, pin_memory=True,
                                                shuffle=True, num_workers=args.num_workers)
                for x in ['train', 'test']}
else:
    print("Use imbalanced dataset sampler...")
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], sampler=ImbalancedDatasetSampler(image_datasets[x]), pin_memory=True,
                                                batch_size=args.batch_size, num_workers=args.num_workers)
                for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
trainloader = dataloaders['train']
testloader = dataloaders['test']

# compute class weights
class_weights = compute_class_weights(image_datasets['train'].imgs, len(image_datasets['train'].classes))
class_weights = torch.FloatTensor(class_weights).to(device)

print("Found {} training images".format(len(image_datasets['train'])))
print("Found {} evaluating images".format(len(image_datasets['test'])))
print("Classes: {}".format(class_names))

# model
net = model(args)
net = net.to(device)
# print(net)
# net parameters
# count_parameters(net)
summary(net, (3, args.img_size, args.img_size))

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # load checkpoint.
    print('Resuming from checkpoint {}'.format(args.checkpoint))
    assert os.path.isfile(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if not args.balance:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()
# Optimizer
if args.adam:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# Decay LR by a factor of gamma every step_size epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    acc = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), acc, correct, total))
    # write acc & loss
    writer.add_scalar('loss/training', train_loss/(len(trainloader)), epoch * len(trainloader))
    writer.add_scalar('acc/training', acc / 100, epoch * len(trainloader))

    # Save checkpoint.
    if epoch != 0 and epoch % args.save_checkpoint == 0:
        print('Saving checkpoint every {} epoch..'.format(args.save_checkpoint))
        state = {
            'net': net.state_dict(),
            'acc' : 100. * correct/total,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{}.pth'.format(epoch))

# testing
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # write acc & loss
    writer.add_scalar('loss/testing', test_loss/(len(testloader)), epoch * len(trainloader))
    writer.add_scalar('acc/testing', acc / 100, epoch * len(trainloader))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('New accuracy {:.2f}, saving best checkpoint..'.format(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_best.pth')
        best_acc = acc
    else:
        print('Accuracy still {:.2f}'.format(best_acc))

# eval
def eval():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc = 0
    print('Evaluating from checkpoint {}'.format(args.checkpoint))
    assert os.path.isfile(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    target_total = []
    predicted_total = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            target_total = target_total + targets.tolist()
            predicted_total = predicted_total + predicted.tolist()
            acc = 100.*correct/total
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), acc, correct, total))

    confusion_matrix = metrics.confusion_matrix(target_total, predicted_total)
    print("confusion matrix: \n{}".format(confusion_matrix))
    report = metrics.classification_report(target_total, predicted_total)
    print(report)

# train and eval
if __name__ == '__main__':
    if args.eval:
        print("Start evaluating...")
        eval()
    else:
        print("Start training...")
        for epoch in range(start_epoch, start_epoch + args.epoch):
            train(epoch)
            test(epoch)
            scheduler.step()
            writer.flush()
        writer.close()
