import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        print('PoseNetFeat: x shape after conv layer: ', x.shape)
        emb = F.relu(self.e_conv1(emb))
        print('PoseNetFeat: emb shape after conv layer: ', emb.shape)
        pointfeat_1 = torch.cat((x, emb), dim=1)
        print('after concatenation shape of pointnet1: ', pointfeat_1.shape)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)
        print('after concatenation shape of pointnet2: ', pointfeat_2.shape)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)
        print('apx shape: ', ap_x.shape)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        print('apx shape after change: ', ap_x.shape)
        print('final shape after concat of pointnetfeat_1 & pointnetfeat_2: ', torch.cat([pointfeat_1, pointfeat_2, ap_x], 1).shape)
        
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        
        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)
        print('\n\n\n\nout_img.size: ', out_img.size())
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        print('emb::::',emb.shape)
        print('choose.shape before: ', choose.shape)
        choose = choose.repeat(1, di, 1)
        print('choose: ', choose.shape)
        emb = torch.gather(emb, 2, choose).contiguous()
        print('emb after gather:: ', emb.shape)
        print('pointcloud shape before transpose: ', x.shape)

        
        x = x.transpose(2, 1).contiguous()
        print('pointcloud after transpose: ', x.shape)
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        print('rx shape after 1st iteration: ', rx.shape)
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        print('rx shape after 2nd iteration: ', rx.shape)
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        print('rx shape after 3rd iteration: ', rx.shape)
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        print('rx after shape change: ', rx.shape)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        print('out__rx shape before contiguous: ', out_rx.shape)        
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        print('out__rx shape after contiguous: ', out_rx.shape)        
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        print('\n\nemb at the end: ', emb.shape, '\n')
        
        return out_rx, out_tx, out_cx, emb.detach()
 


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
