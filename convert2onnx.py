import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.onnx as torch_onnx
import onnx
import torch.backends.cudnn as cudnn
from config import opt, model
import os
import cv2 as cv
import collections
import onnxruntime as rt
from onnxruntime.datasets import get_example
import argparse
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
'''convert checkpoint to onnx.'''

# config
parser = argparse.ArgumentParser(description='Convert2onnx')
parser.add_argument('--image_test', default = 'test.jpg', type=str, help='image for testing onnx model')
parser.add_argument('--model_onnx_path', default = 'classification.onnx', type=str, help='path to save onnx model')
args = opt(parser)

# example testing image
image_test = args.image_test
# save model dir
model_onnx_path = args.model_onnx_path

# model
net = model(args)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# load checkpoint
assert os.path.isfile(args.checkpoint), 'Error: no checkpoint found!'
print('Resuming from checkpoint {}'.format(args.checkpoint))
checkpoint = torch.load(args.checkpoint)

# new state
new_state_dict = collections.OrderedDict()
for k, v in checkpoint['net'].items():
    name = k.replace('module.', '') # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
# net.load_state_dict(checkpoint['net'])
net.eval()

# convert 2 onnx
dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
torch_onnx.export(net, dummy_input, model_onnx_path,
                    verbose=True, input_names=["image"],
                    output_names=["output"], opset_version=11)
print("Export of {} complete!".format(model_onnx_path))

print("Testing...")
model_onnx = onnx.load(model_onnx_path)
sess = rt.InferenceSession(model_onnx_path)

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
input_type = sess.get_inputs()[0].type
output_name = sess.get_outputs()[0].name
output_shape = sess.get_outputs()[0].shape
output_type = sess.get_outputs()[0].type
print("input name: {}, input shape: {}, input type: {}".format(input_name, input_shape, input_type))
print("output name: {}, output shape: {}, output type: {}".format(output_name, output_shape, output_type))

if image_test is not None:
    # read image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = cv.imread(image_test, -1)
    image = cv.resize(image, (args.img_size, args.img_size))
    image = image[:, :, [2, 1, 0]] / 255 # BGR2RGB
    image = (image - mean)/std
    image = np.rollaxis(image, 2, 0)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    # predict
    start_time = time.time()
    preds = sess.run([output_name], {input_name: (image)})
    end_time = time.time()
    preds = preds[0]
    preds_index = np.argmax(preds, axis=1)
    print("\nPredict: {}. Time infer: {:.4f}".format(preds_index, end_time-start_time))