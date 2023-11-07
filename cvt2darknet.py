import numpy as np
import torch.nn as nn
from detect import model_init

def load_conv_model(module, f):
    conv_layer = module.conv
    if isinstance(module.bn, nn.BatchNorm2d):
        bn_layer = module.bn
        # bn bias
        num_b = bn_layer.bias.numel()
        bn_b = bn_layer.bias.data.view(num_b).numpy()
        bn_b.tofile(f)
        # bn weights
        num_w = bn_layer.weight.numel()
        bn_w = bn_layer.weight.data.view(num_w).numpy()
        bn_w.tofile(f)
        # bn running mean
        num_rm = bn_layer.running_mean.numel()
        bn_rm = bn_layer.running_mean.data.view(num_rm).numpy()
        bn_rm.tofile(f)
        # bn running var
        num_rv = bn_layer.running_var.numel()
        bn_rv = bn_layer.running_var.data.view(num_rv).numpy()
        bn_rv.tofile(f)
    else:
        # conv bias
        num_b = conv_layer.bias.numel()
        conv_b = conv_layer.bias.data.view(num_b).numpy()
        conv_b.tofile(f)
    # conv weights
    num_w = conv_layer.weight.numel()
    conv_w = conv_layer.weight.data.view(num_w).numpy()
    conv_w.tofile(f)

print('load pytorch model ... ')
checkpoint_path = 'weights/yolov4_final.pth'
B, C = 3, 20
model = model_init(checkpoint_path, B, C)

print('convert to darknet ... ')
with open('weights/yolov4-tiny-final.weights', 'wb') as f:
    np.asarray([0, 2, 0, 32013312, 0], dtype=np.int32).tofile(f)
    for name, module in model.named_children():
        if 'conv' in name:
            load_conv_model(module, f)
        elif 'block' in name:
            for subname, submodule in module.named_children():
                load_conv_model(submodule, f)


print('done!')