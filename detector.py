import os
import numpy as np
import sys
import time
import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch

VN_DIR = '/home/azav/objectnav/votenet'  # votenet directory
sys.path.append(os.path.join(VN_DIR, 'utils'))
sys.path.append(os.path.join(VN_DIR, 'models'))
sys.path.append(os.path.join(VN_DIR, 'sunrgbd'))
demo_dir = os.path.join(VN_DIR, 'demo_files') 
checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')

from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions, softmax
from sunrgbd_detection_dataset import DC


MODEL = importlib.import_module('votenet') # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_point = 20000
num_input_channel = 1

HFOV = 79 * np.pi / 180
VFOV = HFOV * 3 / 4
XYZ = np.ones((480, 640, 3))
for i in range(480):
    for j in range(640):
        XYZ[i, j, 0] = (j - 320 + 0.5) / 320 * np.tan(HFOV / 2)
        XYZ[i, j, 2] = (240 - i + 0.5) / 240 * np.tan(VFOV / 2)
        
SUN_TO_HAB = {0:6, 1:1, 2:5, 3:0, 4:10, 5:1, 6:7, 7:1, 8:3, 9:15}


class GoalDetector():
    """Habitat ObjectNav goal object detector."""
    def __init__(self, device=device):
        self.device = device
        self.net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
                                 sampling='seed_fps', num_class=DC.num_class,
                                 num_heading_bin=DC.num_heading_bin,
                                 num_size_cluster=DC.num_size_cluster,
                                 mean_size_arr=DC.mean_size_arr).to(self.device)
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        
        self.eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
                                 'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
                                 'conf_thresh': 0.5, 'dataset_config': DC}

        
    def depth_to_pc(self, dep):
        """Depth map to point cloud network input."""
        point_map = np.zeros((480, 640, 3))
        for i in range(3):
            point_map[:, :, i] = dep * XYZ[:, :, i]
        point_cloud = point_map.reshape(480 * 640, 3)
        floor_height = np.percentile(point_cloud[:,2],0.99)
        height = point_cloud[:,2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
        point_cloud, choices = random_sampling(point_cloud,
            num_point, return_choices=True)
        pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,20000,4)
        return pc
    
    def get_detections(self, depth):
        """
        Detection output after applying confidence threshold and non-maximum suppression
        Format is (batch size, # of preds) array of tuples of 
        (semantic class, bbox corners, objectness).
        """
        pc = self.depth_to_pc(depth)
        inputs = {'point_clouds': torch.from_numpy(pc).to(self.device)}
        tic = time.time()
        with torch.no_grad():
            end_points = self.net(inputs)
        toc = time.time()
        obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
        obj_prob = softmax(obj_logits)[:,:,1] # (B,K)
        if np.sum(obj_prob > self.eval_config_dict['conf_thresh']) == 0:
            return [[]]
        #print('Inference time: %f'%(toc-tic))
        end_points['point_clouds'] = inputs['point_clouds']
        pred_map_cls = parse_predictions(end_points, self.eval_config_dict)
        return pred_map_cls
        
    def detect_goal(self, depth, goal_class):
        """
        Returns center location of closest goal object if found, None otherwise.
        Coordinates are (x, y) where x is rightwards from agent's view and y is forwards.
        """
        preds = self.get_detections(depth)
        min_dist = np.inf
        closest_center = None
        for sem_cls, corners, conf in preds[0]:
            if SUN_TO_HAB[sem_cls] == goal_class:
                center = np.mean(corners, axis=0)[[0, 2]]
                dist = np.linalg.norm(center)
                if dist < min_dist:
                    min_dist = dist
                    closest_center = center
        return closest_center
