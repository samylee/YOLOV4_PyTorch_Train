import numpy as np
import torch
import torch.nn as nn
import math


def nms(boxes, iou_thres):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []

    index = np.argsort(scores)[::-1]

    while(index.size):
        i = index[0]
        keep.append(index[0])

        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        inter_area = np.maximum(inter_x2 - inter_x1 + 1, 0) * np.maximum(inter_y2 - inter_y1 + 1, 0)
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        ids = np.where(iou <= iou_thres)[0]
        index = index[ids + 1]

    return boxes[keep]


def xywh2xyxy(x, w, h):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 1] = w * (x[:, 1] - x[:, 3] / 2)  # top left x
    y[:, 2] = h * (x[:, 2] - x[:, 4] / 2)  # top left y
    y[:, 3] = w * (x[:, 1] + x[:, 3] / 2)  # bottom right x
    y[:, 4] = h * (x[:, 2] + x[:, 4] / 2)  # bottom right y
    return y


def xyxy2xywh(x, w, h):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2 / w  # x center
    y[:, 2] = (x[:, 2] + x[:, 4]) / 2 / h  # y center
    y[:, 3] = (x[:, 3] - x[:, 1]) / w  # width
    y[:, 4] = (x[:, 4] - x[:, 2]) / h  # height
    return y


def bbox_iou(box1, box2, x1y1x2y2=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.T
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
    else:
        return iou


def load_conv_layer(module, ptr, weights):
    conv_layer = module.conv
    if isinstance(module.bn, nn.BatchNorm2d):
        # Load BN bias, weights, running mean and running variance
        bn_layer = module.bn
        num_b = bn_layer.bias.numel()  # Number of biases
        # Bias
        bn_b = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_b)
        ptr += num_b
        # Weight
        bn_w = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_w)
        ptr += num_b
        # Running Mean
        bn_rm = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_rm)
        ptr += num_b
        # Running Var
        bn_rv = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_rv)
        ptr += num_b
    else:
        # Load conv. bias
        num_b = conv_layer.bias.numel()
        conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
        conv_layer.bias.data.copy_(conv_b)
        ptr += num_b
    # Load conv. weights
    num_w = conv_layer.weight.numel()
    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
    conv_layer.weight.data.copy_(conv_w)
    ptr += num_w
    return ptr


def load_darknet_pretrain_weights(model, weights_path):
    # Open the weights file
    with open(weights_path, 'rb') as f:
        # First five are header values
        header = np.fromfile(f, dtype=np.int32, count=5)
        seen = header[3]  # number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
        weights_len = len(weights)

    ptr = 0
    # feature
    for name, module in model.named_children():
        # for pretrain
        if name == 'conv6':
            break
        if 'conv' in name:
            ptr = load_conv_layer(module, ptr, weights)
        elif 'block' in name:
            for subname, submodule in module.named_children():
                ptr = load_conv_layer(submodule, ptr, weights)

    print('Load pretrain done!')