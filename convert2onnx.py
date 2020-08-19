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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

checkpoint_dir = './checkpoint/ckpt_best.pth'
# example image
image_test = 'test.jpg'
# save model dir
model_onnx_path = "model.onnx"

# config
args = opt()
# model
net = model(args)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# load checkpoint
assert os.path.isfile(args.checkpoint_test), 'Error: no checkpoint found!'
print('Resuming from checkpoint {}'.format(checkpoint_dir))
checkpoint = torch.load(checkpoint_dir)

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
model_out = net(dummy_input)
torch_onnx.export(net, dummy_input, model_onnx_path,
                    verbose=True, input_names=["image"],
                    output_names=["output"], opset_version=11)
print("Export of torch_model.onnx complete!")

print("Testing...")
model_onnx = onnx.load(model_onnx_path)
print("Load model onnx success\n")

sess = rt.InferenceSession(model_onnx_path)

input_name = sess.get_inputs()[0].name
print("input name:", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape:", input_shape)
input_type = sess.get_inputs()[0].type
print("input type:", input_type)

output_name = sess.get_outputs()[0].name
print("output name:", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape:", output_shape)
output_type = sess.get_outputs()[0].type
print("output type:", output_type)

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