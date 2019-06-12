import numpy as np 
import h5py


class Posecreation:
    def __init__(self, root, output_poses):
        self.root = root
        self.output_poses = output_poses
        f = h5py.File(self.root+'/rgbd/calibration.h5', 'r')
        cal_list= list(f.keys())

        # NP1_from_NP5 = np.array(f['H_NP1_from_NP5'])
        # NP2_from_NP5 = np.array(f['H_NP2_from_NP5'])
        # NP3_from_NP5 = np.array(f['H_NP3_from_NP5'])
        # NP4_from_NP5 = np.array(f['H_NP4_from_NP5'])

        train_file = open(self.root+'/toy_dataset.txt')
        while 1:
            input_file = train_file.readline()
            if not input_file:
                break
            input_file = input_file[:-1]
            posename = input_file
            input_file = input_file.split('_')
            print(input_file)
            e = h5py.File(self.root+'/poses/NP5_{}_pose.h5'.format(input_file[1]), 'r')
            val = np.array(e['H_table_from_reference_camera'])
            pose = np.dot(np.array(f['H_{}_from_NP5'.format(input_file[0])]), val)
            hf = h5py.File(self.output_poses+'/'+posename+'_pose.h5', 'w')
            hf.create_dataset('poses', data=pose)
            hf.close()

        train_file.close()


root = '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset'
output_poses = '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset/poses2'

experiment = Posecreation(root, output_poses)
