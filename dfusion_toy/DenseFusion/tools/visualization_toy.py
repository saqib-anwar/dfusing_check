import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import matplotlib.pyplot as plt 
import cv2
import matplotlib.image as mpimg
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset/', help='dataset root dir')
parser.add_argument('--model', type=str, default = '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset/trained_models/pose_model_10_0.033466376511690515.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
# cam_cx = 312.9869
# cam_cy = 241.3109
# cam_fx = 1066.778
# cam_fy = 1067.487
cam_scale = 1.00000
num_obj = 1
img_width = 480
img_length = 640
num_points = 500
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset/'
ycb_toolbox_dir = 'YCB_Video_toolbox'########################################
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'#################################
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'#################################

dist = np.array([0.0, 0.0, 0.0, 0.0])

def get_bbox(label):
    print('\n\n*************now we are in get_bbox function block************\n\n')
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    
    c_b = cmax - cmin
    
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path, 'rb')
    assert f.readline().decode('ascii').strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    print('here is the value of N:', N)
    while f.readline().decode('ascii').strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)


def cam_props(id):

    #NP1_rgb_k

    NP1_cam_cx = 627.93334
    NP1_cam_cy = 494.46569
    NP1_cam_fx = 1080.2331
    NP1_cam_fy = 1080.2312

    #NP2_rgb_k

    NP2_cam_cx = 628.32147
    NP2_cam_cy = 492.13550
    NP2_cam_fx = 1074.3439
    NP2_cam_fy = 1074.3447

    #NP3_rgb_k

    NP3_cam_cx = 618.76531
    NP3_cam_cy = 484.4791
    NP3_cam_fx = 1075.8584
    NP3_cam_fy = 1075.8538

    #NP4_rgb_k

    NP4_cam_cx = 637.6103
    NP4_cam_cy = 495.49479
    NP4_cam_fx = 1077.2459
    NP4_cam_fy = 1077.2495

    #NP5_rgb_k

    NP5_cam_cx = 626.34743
    NP5_cam_cy = 490.52912
    NP5_cam_fx = 1075.5821
    NP5_cam_fy = 1075.5776

    
    #all_cams = [1,2,3,4,5]
    NP1 = [627.93334*1.05, 494.46569*.95, 1080.2331, 1080.2312]
    NP2 = [628.32147*1.2, 492.13550, 1074.3439, 1074.3447]
    NP3 = [618.76531*1.15, 484.4791 * 1.1 , 1075.8584, 1075.8538]
    NP4 = [637.6103*1.15 , 495.49479* 1.2, 1077.2459, 1077.2495]
    NP5 = [626.34743, 490.52912* 1.2, 1075.5821, 1075.5776]
    all_cams = [0,NP1, NP2, NP3, NP4, NP5]
    cam_id = 1
    cam_dict = {}

    # for cams in enumerate(all_cams):
    #     for num, val in enumerate(cams):
    #         #cam_dict[f'NP{cams}'] = [f'{'NP'+f'{cams}'+'_cam_cx'}']
    #         print(num, val)
    #         cam_dict[cams]= []
    #         cam_dict[cams].append(val)

    #     #cam_dict[].append(cams)
    
    while cam_id < 6:
        cam_dict[cam_id] = []
        cam_dict[cam_id].append(all_cams[cam_id])
        cam_id += 1
    
    return (cam_dict[id][0])


# estimator = PoseNet(num_points = num_points, num_obj = num_obj)
# estimator.cuda()
# estimator.load_state_dict(torch.load(opt.model))
# estimator.eval()

# refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
# refiner.cuda()
# refiner.load_state_dict(torch.load(opt.refine_model))
# refiner.eval()

testlist = []
input_file = open('{0}/validation.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

#class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
# while 1:
#     # class_input = class_file.readline()
#     # if not class_input:
#     #     break
#     # class_input = class_input[:-1]

pointcloud_file =dataset_config_dir + 'model_pointcloud/merged_cloud_updated.ply'
print (pointcloud_file)
cld[class_id] = []
# while 1:
#     input_line = input_file.readline()
#     if not input_line:
#         break
#     input_line = input_line[:-1]
#     input_line = input_line.split(' ')
    #cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
cld[class_id].append(ply_vtx(pointcloud_file))
# input_file.close()
# cld[class_id] = np.array(cld[class_id])
# class_id += 1

print('here is the dictionary for model pointlcoud\n\n', cld)
print(np.array(cld[class_id][0]).shape)
model_points = cld

values = cam_props(2)
print(values[0])

for img in testlist:
    #imgplt = Image.open('{0}/rgbd/{1}.jpg'.format(dataset_config_dir,img))
    imgplt = Image.open('/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/072-a_toy_airplane_rgbd/'+img+'.jpg')

    f = h5py.File(dataset_config_dir + 'poses2/{0}_pose.h5'.format(img), 'r')
    posemat = np.array(f['poses'])
    #print('posemat:\n', posemat)
    
    target_r = np.array(posemat)[:3,:3]
    target_t = np.array(posemat)[:3,3]
    print('target_t value: ',target_t)

    cam = img.split('_')[0]
    id = cam.split('P')[-1]
    id = int(id)
    print('printing camera id here: ', id)
    # target = np.dot(model_points, target_r.T)
    # target = np.add(target, target_t)
    # out_t = target_t
    #id = 4
    values = cam_props(id)
    #cam_mat = np.matrix([[values[2], 0, values[0]*1.2],[0, values[3], values[1]*1.17],[0, 0, 1]])
    cam_mat = np.matrix([[values[2], 0, values[0]],[0, values[3], values[1]],[0, 0, 1]])
    #homrot_mat = quaternion_matrix(my_r)[0:3,0:3]
    #rot_vect= cv2.Rodrigues(homrot_mat)
    #print('\n\nhomrot::::', homrot_mat)


    imgpts, jac = cv2.projectPoints(cld[1][0], target_r, target_t, cam_mat, dist)
    
        
    #print('\n\n**********IMGPTS*******', imgpts)
    d = np.array(imgpts)
    #d = d.reshape(2621,2)
    d = d.reshape(-1,2)
    print('\n\n**********IMGPTS*******', d)
    print('\n\n**********IMGPTS*******', d[0][0])
    print('\n\n**********IMGPTS.shape*******', d.shape)

    
    
    scatter_x = []
    scatter_y = []
    for e in range(985648):
        scatter_x.append(d[e][0])
        scatter_y.append(d[e][1])
    
    plt.scatter(scatter_x, scatter_y, s= .1)

    img_plot = plt.imshow(imgplt)
    plt.show()



    #plot the by pointclouds by projecting them in 2d







# for now in range(0, 2949):
#     img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     imgplt = mpimg.imread('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
#     posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
#     label = np.array(posecnn_meta['labels'])
#     posecnn_rois = np.array(posecnn_meta['rois'])

#     lst = posecnn_rois[:, 1:2].flatten()
#     my_resulfor img in testlist:

    # f = h5py.File(dataset_config_dir + 'poses2/{0}_pose.h5'.format(img), 'r')
    # posemat = np.array(f['poses'])
    # #print('posemat:\n', posemat)
    
    # target_r = np.array(posemat)[:3,:3]
    # target_t = np.array(posemat)[:3,3]

    # target = np.dot(model_points, target_r.T)
    # target = np.add(target, target_t)
    # out_t = target_t

    # cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0, cam_fy, cam_cy],[0, 0, 1]])
    # homrot_mat = quaternion_matrix(my_r)[0:3,0:3]
    # #rot_vect= cv2.Rodrigues(homrot_mat)
    # print('\n\nhomrot::::', homrot_mat)


    # imgpts, jac = cv2.projectPoints(cld[itemid], homrot_mat, my_t, cam_mat, dist)
    
        
    # #print('\n\n**********IMGPTS*******', imgpts)
    # d = np.array(imgpts)
    # #d = d.reshape(2621,2)
    # d = d.reshape(-1,2)
    # print('\n\n**********IMGPTS*******', d)
    # print('\n\n**********IMGPTS*******', d[0][0])
    # print('\n\n**********IMGPTS.shape*******', d.shape)
    
    # scatter_x = []
    # scatter_y = []
    # for e in range(2620):
    #     scatter_x.append(d[e][0])
    #     scatter_y.append(d[e][1])
    
    # plt.scatter(scatter_x, scatter_y, s= .1)

    # img_plot = plt.imshow(imgplt)
    # plt.show()



    #plot the by pointclouds by projecting them in 2d







# for now in range(0, 2949):
#     img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     imgplt = mpimg.imread('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
#     posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
#     label = np.array(posecnn_meta['labels'])
#     posecnn_rois = np.array(posecnn_meta['rois'])

#     lst = posecnn_rois[:, 1:2].flatten()
#     my_result_wo_refine = []
#     my_result = []
    
#     for idx in range(len(lst)):
#         itemid = lst[idx]
#         try:
#             rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

#     my_resulfor img in testlist:

    # f = h5py.File(dataset_config_dir + 'poses2/{0}_pose.h5'.format(img), 'r')
    # posemat = np.array(f['poses'])
    # #print('posemat:\n', posemat)
    
    # target_r = np.array(posemat)[:3,:3]
    # target_t = np.array(posemat)[:3,3]

    # target = np.dot(model_points, target_r.T)
    # target = np.add(target, target_t)
    # out_t = target_t

    # cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0, cam_fy, cam_cy],[0, 0, 1]])
    # homrot_mat = quaternion_matrix(my_r)[0:3,0:3]
    # #rot_vect= cv2.Rodrigues(homrot_mat)
    # print('\n\nhomrot::::', homrot_mat)


    # imgpts, jac = cv2.projectPoints(cld[itemid], homrot_mat, my_t, cam_mat, dist)
    
        
    # #print('\n\n**********IMGPTS*******', imgpts)
    # d = np.array(imgpts)
    # #d = d.reshape(2621,2)
    # d = d.reshape(-1,2)
    # print('\n\n**********IMGPTS*******', d)
    # print('\n\n**********IMGPTS*******', d[0][0])
    # print('\n\n**********IMGPTS.shape*******', d.shape)
    
    # scatter_x = []
    # scatter_y = []
    # for e in range(2620):
    #     scatter_x.append(d[e][0])
    #     scatter_y.append(d[e][1])
    
    # plt.scatter(scatter_x, scatter_y, s= .1)

    # img_plot = plt.imshow(imgplt)
    # plt.show()



    #plot the by pointclouds by projecting them in 2d







# for now in range(0, 2949):
#     img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     imgplt = mpimg.imread('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
#     posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
#     label = np.array(posecnn_meta['labels'])
#     posecnn_rois = np.array(posecnn_meta['rois'])

#     lst = posecnn_rois[:, 1:2].flatten()
#     my_result_wo_refine = []
#     my_result = []
    
#     for idx in range(len(lst)):
#         itemid = lst[idx]
#         try:
#             rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

    
#     for idx for img in testlist:

#     f = h5py.File(dataset_config_dir + 'poses2/{0}_pose.h5'.format(img), 'r')
#     posemat = np.array(f['poses'])
#     #print('posemat:\n', posemat)
    
#     target_r = np.array(posemat)[:3,:3]
#     target_t = np.array(posemat)[:3,3]

#     target = np.dot(model_points, target_r.T)
#     target = np.add(target, target_t)
#     out_t = target_t

#     cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0, cam_fy, cam_cy],[0, 0, 1]])
#     homrot_mat = quaternion_matrix(my_r)[0:3,0:3]
#     #rot_vect= cv2.Rodrigues(homrot_mat)
#     print('\n\nhomrot::::', homrot_mat)


#     imgpts, jac = cv2.projectPoints(cld[itemid], homrot_mat, my_t, cam_mat, dist)
    
        
#     #print('\n\n**********IMGPTS*******', imgpts)
#     d = np.array(imgpts)
#     #d = d.reshape(2621,2)
#     d = d.reshape(-1,2)
#     print('\n\n**********IMGPTS*******', d)
#     print('\n\n**********IMGPTS*******', d[0][0])
#     print('\n\n**********IMGPTS.shape*******', d.shape)
    
#     scatter_x = []
#     scatter_y = []
#     for e in range(2620):
#         scatter_x.append(d[e][0])
#         scatter_y.append(d[e][1])
    
#     plt.scatter(scatter_x, scatter_y, s= .1)

#     img_plot = plt.imshow(imgplt)
#     plt.show()



#     #plot the by pointclouds by projecting them in 2d







# # for now in range(0, 2949):
# #     img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
# #     imgplt = mpimg.imread('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
# #     depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
# #     posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
# #     label = np.array(posecnn_meta['labels'])
# #     posecnn_rois = np.array(posecnn_meta['rois'])

# #     lst = posecnn_rois[:, 1:2].flatten()
# #     my_result_wo_refine = []
# #     my_result = []
    
# #     for idx in range(len(lst)):
# #         itemid = lst[idx]
# #         try:
# #             rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

# #         itemfor img in testlist:

#     f = h5py.File(dataset_config_dir + 'poses2/{0}_pose.h5'.format(img), 'r')
#     posemat = np.array(f['poses'])
#     #print('posemat:\n', posemat)
    
#     target_r = np.array(posemat)[:3,:3]
#     target_t = np.array(posemat)[:3,3]

#     target = np.dot(model_points, target_r.T)
#     target = np.add(target, target_t)
#     out_t = target_t

#     cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0, cam_fy, cam_cy],[0, 0, 1]])
#     homrot_mat = quaternion_matrix(my_r)[0:3,0:3]
#     #rot_vect= cv2.Rodrigues(homrot_mat)
#     print('\n\nhomrot::::', homrot_mat)


#     imgpts, jac = cv2.projectPoints(cld[itemid], homrot_mat, my_t, cam_mat, dist)
    
        
#     #print('\n\n**********IMGPTS*******', imgpts)
#     d = np.array(imgpts)
#     #d = d.reshape(2621,2)
#     d = d.reshape(-1,2)
#     print('\n\n**********IMGPTS*******', d)
#     print('\n\n**********IMGPTS*******', d[0][0])
#     print('\n\n**********IMGPTS.shape*******', d.shape)
    
#     scatter_x = []
#     scatter_y = []
#     for e in range(2620):
#         scatter_x.append(d[e][0])
#         scatter_y.append(d[e][1])
    
#     plt.scatter(scatter_x, scatter_y, s= .1)

#     img_plot = plt.imshow(imgplt)
#     plt.show()



#     #plot the by pointclouds by projecting them in 2d







# # for now in range(0, 2949):
# #     img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
# #     imgplt = mpimg.imread('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
# #     depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
# #     posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
# #     label = np.array(posecnn_meta['labels'])
# #     posecnn_rois = np.array(posecnn_meta['rois'])

# #     lst = posecnn_rois[:, 1:2].flatten()
# #     my_result_wo_refine = []
# #     my_result = []
    
# #     for idx in range(len(lst)):
# #         itemid = lst[idx]
# #         try:
# #             rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

# #         try:for img in testlist:

#     f = h5py.File(dataset_config_dir + 'poses2/{0}_pose.h5'.format(img), 'r')
#     posemat = np.array(f['poses'])
#     #print('posemat:\n', posemat)
    
#     target_r = np.array(posemat)[:3,:3]
#     target_t = np.array(posemat)[:3,3]

#     target = np.dot(model_points, target_r.T)
#     target = np.add(target, target_t)
#     out_t = target_t

#     cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0, cam_fy, cam_cy],[0, 0, 1]])
#     homrot_mat = quaternion_matrix(my_r)[0:3,0:3]
#     #rot_vect= cv2.Rodrigues(homrot_mat)
#     print('\n\nhomrot::::', homrot_mat)


#     imgpts, jac = cv2.projectPoints(cld[itemid], homrot_mat, my_t, cam_mat, dist)
    
        
#     #print('\n\n**********IMGPTS*******', imgpts)
#     d = np.array(imgpts)
#     #d = d.reshape(2621,2)
#     d = d.reshape(-1,2)
#     print('\n\n**********IMGPTS*******', d)
#     print('\n\n**********IMGPTS*******', d[0][0])
#     print('\n\n**********IMGPTS.shape*******', d.shape)
    
#     scatter_x = []
#     scatter_y = []
#     for e in range(2620):
#         scatter_x.append(d[e][0])
#         scatter_y.append(d[e][1])
    
#     plt.scatter(scatter_x, scatter_y, s= .1)

#     img_plot = plt.imshow(imgplt)
#     plt.show()



    #plot the by pointclouds by projecting them in 2d







# for now in range(0, 2949):
#     img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     imgplt = mpimg.imread('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
#     depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
#     posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
#     label = np.array(posecnn_meta['labels'])
#     posecnn_rois = np.array(posecnn_meta['rois'])

#     lst = posecnn_rois[:, 1:2].flatten()
#     my_result_wo_refine = []
#     my_result = []
    
#     for idx in range(len(lst)):
#         itemid = lst[idx]
#         try:
#             rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

#             rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

#             mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
#             mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
#             mask = mask_label * mask_depth

#             choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
#             if len(choose) > num_points:
#                 c_mask = np.zeros(len(choose), dtype=int)
#                 c_mask[:num_points] = 1
#                 np.random.shuffle(c_mask)
#                 choose = choose[c_mask.nonzero()]
#             else:
#                 choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

#             depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
#             xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
#             ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
#             choose = np.array([choose])

#             pt2 = depth_masked / cam_scale
#             pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
#             pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
#             cloud = np.concatenate((pt0, pt1, pt2), axis=1)

#             img_masked = np.array(img)[:, :, :3]
#             img_masked = np.transpose(img_masked, (2, 0, 1))
#             img_masked = img_masked[:, rmin:rmax, cmin:cmax]

#             cloud = torch.from_numpy(cloud.astype(np.float32))
#             choose = torch.LongTensor(choose.astype(np.int32))
#             img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
#             index = torch.LongTensor([itemid - 1])

#             cloud = Variable(cloud).cuda()
#             choose = Variable(choose).cuda()
#             img_masked = Variable(img_masked).cuda()
#             index = Variable(index).cuda()

#             cloud = cloud.view(1, num_points, 3)
#             img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

#             pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
#             pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

#             pred_c = pred_c.view(bs, num_points)
#             how_max, which_max = torch.max(pred_c, 1)
#             pred_t = pred_t.view(bs * num_points, 1, 3)
#             points = cloud.view(bs * num_points, 1, 3)

#             my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
#             my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
#             my_pred = np.append(my_r, my_t)
#             my_result_wo_refine.append(my_pred.tolist())

#             for ite in range(0, iteration):
#                 T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
#                 my_mat = quaternion_matrix(my_r)
#                 R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
#                 my_mat[0:3, 3] = my_t
                
#                 new_cloud = torch.bmm((cloud - T), R).contiguous()
#                 pred_r, pred_t = refiner(new_cloud, emb, index)
#                 pred_r = pred_r.view(1, 1, -1)
#                 pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
#                 my_r_2 = pred_r.view(-1).cpu().data.numpy()
#                 my_t_2 = pred_t.view(-1).cpu().data.numpy()
#                 #print('\nmy_t_2::::', my_t_2)
#                 my_mat_2 = quaternion_matrix(my_r_2)
#                 #print('\nmy_mat_2 before my_t_2:::::', my_mat_2)
#                 my_mat_2[0:3, 3] = my_t_2
#                 #print('\nmy_myt_2 after my_t_2:::::', my_mat_2)

#                 my_mat_final = np.dot(my_mat, my_mat_2)
#                 #print('\nmy_mat_final:::',my_mat_final)
#                 my_r_final = copy.deepcopy(my_mat_final)
#                 #print('\nmy_r_final after decopy::::', my_r_final)
#                 my_r_final[0:3, 3] = 0
#                 #print('\nmy_r_final after setting last coloumn to zero::', my_r_final)
#                 my_r_final = quaternion_from_matrix(my_r_final, True)
#                 #print('\nmy_r_final in quaternion form:::', my_r_final)
#                 my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
#                 #print('\nmy_t_final::::', my_t_final)

#                 my_pred = np.append(my_r_final, my_t_final)
#                 #print('\n\nmy_pred*******', my_pred)
#                 my_r = my_r_final
#                 my_t = my_t_final

# #************************************************************************************************************
# #************************************************************************************************************
# #******************this part causes the problem and turns the values below in my_result to zero *************
# #************************************************************************************************************
# #************************************************************************************************************

#                 # cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0, cam_fy, cam_cy],[0, 0, 1]])
#                 # mat_r=quaternion_matrix(my_r)[0:3,0:3]
#                 # imgpts, jac = cv2.projectPoints(cld[itemid], my_r, my_t, cam_mat, dist)
                
#                 # print('**********IMGPTS*******', imgpts)

#                 # print('here is my_r_Final at the end of loop', my_r_final)
#                 # print('here is my_t_Final at the end of loop', my_t_final)
#                 # print('here is cld[itemid] at the end of loop', cld[itemid])

#             my_result.append(my_pred.tolist())
#             # print('\n\nmy_r:::::::::::', my_r)
#             # print('\n\nmy_t:::::::::::', my_t)

#             cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0, cam_fy, cam_cy],[0, 0, 1]])
#             homrot_mat = quaternion_matrix(my_r)[0:3,0:3]
#             #rot_vect= cv2.Rodrigues(homrot_mat)
#             print('\n\nhomrot::::', homrot_mat)


#             imgpts, jac = cv2.projectPoints(cld[itemid], homrot_mat, my_t, cam_mat, dist)
            
                
#             #print('\n\n**********IMGPTS*******', imgpts)
#             d = np.array(imgpts)
#             #d = d.reshape(2621,2)
#             d = d.reshape(-1,2)
#             print('\n\n**********IMGPTS*******', d)
#             print('\n\n**********IMGPTS*******', d[0][0])
#             print('\n\n**********IMGPTS.shape*******', d.shape)
            
#             scatter_x = []
#             scatter_y = []
#             for e in range(2620):
#                 scatter_x.append(d[e][0])
#                 scatter_y.append(d[e][1])
            
#             plt.scatter(scatter_x, scatter_y, s= .1)

#             img_plot = plt.imshow(imgplt)
#             plt.show()

#                 # print('here is my_r_Final at the end of loop', my_r_final)
#                 # print('here is my_t_Final at the end of loop', my_t_final)
#                 # print('here is cld[itemid] at the end of loop', cld[itemid])


#             # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
#             #print('\n\nmy_pred after for loop', my_pred)
            
#             #print('\n\nmy_result____appended::::', my_result)
#         except:
#             print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
#             my_result_wo_refine.append([0.0 for i in range(7)])
#             my_result.append([0.0 for i in range(7)])

#     print('\n\nhere the end of second for loop')
#     print('saved mat; my_result $$$$$$$$$$$$$$$$$$$$$$',my_result)
#     print('\nsaved mat; my_result_wo_refine $$$$$$$$$$$$$$$$$$$$$$',my_result_wo_refine)
#     a = np.array(my_result) ## same 5x7 result as we get in the resulting pose matlab files
#     print('\n\nsize for my_result', a.shape)

#     scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now), {'poses':my_result_wo_refine})
#     scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % now), {'poses':my_result})
#     print("Finish No.{0} keyframe".format(now))