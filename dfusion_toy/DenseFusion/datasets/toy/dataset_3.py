# resize all images in the orignal directory
# get bbox function from ycb and apply it here
# setup target_r and target based on what is inside meta
# pose file/meta is setup a bit earlier
# in calibration i need to setup if else
# setup model pointcloud ******need to work on it on extracting data from ply file

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# root = /run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        #self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.objlist = [1]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        #self.list_obj = []
        self.list_poses = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        #item_count = 0
        #for item in self.objlist:
        if self.mode == 'train':
            input_file = open(self.root + '/train.txt')
        else:
            input_file = open(self.root+'/eval.txt')
        while 1:
            #item_count += 1
            input_line = input_file.readline()
            # if self.mode == 'test' and item_count % 10 != 0:
            #     continue
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            #self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
            self.list_rgb.append('{0}/rgbd/{1}.jpg'.format(self.root, input_line))
            self.list_depth.append('{0}/rgbd/{1}.h5'.format(self.root, input_line))
            self.list_poses.append('{0}/poses2/{1}_pose.h5'.format(self.root, input_line))  #additional line here,,,Check************************
            
            # posefile = '{0}/poses1/{1}_pose.h5'.format(self.root, input_line)
            # f = h5py.File(posefile, 'r')
            # posemat = np.array(f['poses'])
            # self.meta[input_file] = posemat



            # for k in f.keys():
            #     rot_tra = np.array(f[k])
            #     self.meta[input_line].append(rot_tra)
            #     break
            if self.mode == 'eval':
                self.list_label.append('{0}/resized_masks/{1}_mask.pbm'.format(self.root, input_line))
            else:
                self.list_label.append('{0}/resized_masks/{1}_mask.pbm'.format(self.root, input_line))
            
            
            #self.list_obj.append(item)
            self.list_rank.append((input_line))
        print('length of all lists: ', len (self.list_rgb), len(self.list_depth), len(self.list_poses), len(self.list_label), len(self.list_rank))
            
        #print('\n\nhere printing self.list_rank to check if dictionary is setup alrigth: \n', (self.list_rank), '\n\n',len(self.list_rank))

        # meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r') # pose file setup understood but need to setup a function to get pose from root through calibration file
        # self.meta[item] = yaml.load(meta_file)
        self.pt = ply_vtx('{0}/model_pointcloud/merged_cloud_updated.ply'.format(self.root))  # modelpointcloud yet to setup the function for listing
        #print('\n\nhere printing self.pt to check if dictionary is setup alrigth: \n', self.pt)
        
        print("Object buffer loaded")

        self.length = len(self.list_rgb)


        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [9999, 9991]

    def __getitem__(self, index):
        print('\n\n\n***********************************ITERATION*******************************************\n\n\n')
        print('value of index here:\n\n', index, '\n\n')
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = h5py.File(self.list_depth[index], 'r')    # depth is h5 file
        a_group_key = list(depth.keys())[0]
        data = list(depth[a_group_key])
        d_img = np.array(data)
        label = np.array(Image.open(self.list_label[index]))    # label is pbm file
        #obj = self.list_obj[index]
        rank = self.list_rank[index]        

        # if obj == 2:
        #     for i in range(0, len(self.meta[obj][rank])):
        #         if self.meta[obj][rank][i]['obj_id'] == 2:
        #             meta = self.meta[obj][rank][i]
        #             break
        # else:
        #meta = self.meta[rank][0]
        #print('printing meta here to see if its working alright:\n', meta)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(d_img, 0))
        #plt.figure(8)
        #plt.title('mask_depth')
        #plt.imshow(np.array(mask_depth), cmap= cm.gray)
        #plt.show()


        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            #mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
            #mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
            mask_label = ma.getmaskarray(ma.masked_not_equal(label, 0))
            #plt.figure(9)
            #plt.title('mask_label')
            #plt.imshow(np.array(mask_label), cmap= cm.gray)
            #plt.show()
        
        mask = mask_label * mask_depth
        #plt.figure(10)
        #plt.title('mask')
        #plt.imshow(np.array(mask), cmap= cm.gray)
        #plt.show()

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img
        # plt.figure(11)
        # plt.title('img_mased')
        # plt.imshow(np.array(img_masked), cmap= cm.gray)
        # plt.show()
        bbox_mask = ma.getmaskarray(ma.masked_equal(label, 0))
        # print('here is the name of label that we are generating:\n', self.list_label[index])
        # plt.figure(11)
        # plt.title('bbox_mask')
        # plt.imshow(bbox_mask, cmap= cm.gray)
        # plt.show()
        rmin, rmax, cmin, cmax = get_bbox(bbox_mask)
        print('values of bounding box: ',rmin, rmax, cmin, cmax)

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        # changed = np.transpose(img_masked, (1,2,0))
        # plt.figure(12)
        # plt.title('img_applied_mask')
        # plt.imshow(np.array(changed))#, cmap= cm.gray)
        # plt.show()
        # #p_img = np.transpose(img_masked, (1, 2, 0))
        # #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        f = h5py.File(self.list_poses[index], 'r')
        posemat = np.array(f['poses'])
        print('posemat:\n', posemat)
        
        target_r = np.array(posemat)[:3,:3]
        target_t = np.array(posemat)[:3,3]
        print('traget_r:\n', target_r)
        print('target_t:\n', target_t)
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        depth_masked = d_img[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        camera = self.list_rank[index].split('_')[0]
        print('first: ',camera)
        #camera = camera.split('p')[0]
        camera = int(camera[-1])
        print('second: ', camera)
        print(cam_props(camera)[0])
        cam_scale = 1.0
        # pt2 = depth_masked / cam_scale
        # pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        # pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy

        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_props(camera)[0]) * pt2 / cam_props(camera)[2]
        pt1 = (xmap_masked - cam_props(camera)[1]) * pt2 / cam_props(camera)[3]

        print('\n\n\n***********************************ITERATION MID*******************************************\n\n\n')



        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = np.add(cloud, -1.0 * target_t) / 1000.0
        cloud = np.add(cloud, target_t / 1000.0)

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        # #for it in cloud:
        # #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # #fw.close()

        model_points = self.pt #/ 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)
        print('\n\nhere are the MODEL POINTS: \n', model_points)        

        # #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        # #for it in model_points:
        # #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # #fw.close()

        target = np.dot(model_points, target_r.T)
        # for once try without dividing by 1000
        if self.add_noise:
            # target = np.add(target, target_t / 1000.0 + add_t)
            # out_t = target_t / 1000.0 + add_t
            target = np.add(target, target_t + add_t)
            out_t = target_t + add_t
        else:
            target = np.add(target, target_t)
            out_t = target_t

        # #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        # #for it in target:
        # #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # #fw.close()

        print('\n\nprinting here the cloud: \n',torch.from_numpy(cloud.astype(np.float32)))
        print('\n\nprinting here the length of cloud: \n',(torch.from_numpy(cloud.astype(np.float32))).shape)
        print('\n\n\nprinting here the choose: \n',torch.LongTensor(choose.astype(np.int32)))
        print('\n\nprinting here the length of choose: \n',(torch.LongTensor(choose.astype(np.int32))).shape)
        print('\n\n\nprinting here the img_masked: \n',self.norm(torch.from_numpy(img_masked.astype(np.float32))))
        print('\n\nprinting here the length of img_masked: \n',(self.norm(torch.from_numpy(img_masked.astype(np.float32)))).shape)
        print('\n\n\nprinting here the target: \n',torch.from_numpy(target.astype(np.float32)))
        print('\n\nprinting here the length of target: \n',(torch.from_numpy(target.astype(np.float32))).shape)
        print('\n\n\nprinting here the model_points: \n',torch.from_numpy(model_points.astype(np.float32)))
        print('\n\nprinting here the length of model_points: \n',(torch.from_numpy(model_points.astype(np.float32))).shape)

        ids = int(self.list_rank[index].split('_')[1])
        print(ids)
        






        # print('\n\nprinting here the long tensor: \n',torch.LongTensor([1]))
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([ids])

    def __len__(self):
        return self.length

    def get_sym_list(self):
       return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    #print(rows)
    cols = np.any(label, axis=0)
    #print(cols)
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
    NP1 = [627.93334, 494.46569, 1080.2331, 1080.2312]
    NP2 = [628.32147, 492.13550, 1074.3439, 1074.3447]
    NP3 = [618.76531, 484.4791, 1075.8584, 1075.8538]
    NP4 = [637.6103, 495.49479, 1077.2459, 1077.2495]
    NP5 = [626.34743, 490.52912, 1075.5821, 1075.5776]
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


# dataset_root = '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset'
# experiment = PoseDataset('train', 500, True, dataset_root, 0.03, True)
# experiment.addition(20)