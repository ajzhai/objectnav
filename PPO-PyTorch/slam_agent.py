import argparse
import os
import random
import sys
import time
from math import pi

import numpy as np
import matplotlib.pyplot as plt
import orbslam2
import PIL
import cv2
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

import habitat
from habitat.config import Config as CN
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat_baselines.slambased.mappers import DirectDepthMapper
from habitat_baselines.slambased.monodepth import MonoDepthEstimator
from habitat_baselines.slambased.path_planners import DifferentiableStarPlanner
from habitat_baselines.slambased.reprojection import (
    get_direction,
    get_distance,
    habitat_goalpos_to_mapgoal_pos,
    homogenize_p,
    planned_path2tps,
    project_tps_into_worldmap,
    angle_to_pi_2_minus_pi_2 as norm_ang,
)
from habitat_baselines.slambased.utils import generate_2dgrid

GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
#GOAL_SENSOR_UUID = "objectgoal"


class RandomAgent(object):
    r"""Simplest agent, which returns random actions,
    until reach the goal
    """

    def __init__(self, config):
        super(RandomAgent, self).__init__()
        self.num_actions = config.NUM_ACTIONS
        self.dist_threshold_to_stop = config.DIST_TO_STOP
        self.reset()
        return

    def reset(self):
        self.steps = 0
        return

    def update_internal_state(self, habitat_observation):
        self.obs = habitat_observation
        self.steps += 1
        return

    def is_goal_reached(self):
        dist = self.obs[GOAL_SENSOR_UUID][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, habitat_observation=None, random_prob=1.0):
        self.update_internal_state(habitat_observation)
        # Act
        # Check if we are done
        if self.is_goal_reached():
            action = HabitatSimActions.STOP
        else:
            action = random.randint(0, self.num_actions - 1)
        return {"action": action}
    
    
class ORBSLAM2Agent(RandomAgent):
    def __init__(self, config, device=torch.device("cuda:0"), print_attrs=False):
        self.num_actions = config.NUM_ACTIONS
        self.dist_threshold_to_stop = config.DIST_TO_STOP
        self.slam_vocab_path = config.SLAM_VOCAB_PATH
        assert os.path.isfile(self.slam_vocab_path)
        self.slam_settings_path = config.SLAM_SETTINGS_PATH
        assert os.path.isfile(self.slam_settings_path)
        self.slam = orbslam2.System(
            self.slam_vocab_path, self.slam_settings_path, orbslam2.Sensor.RGBD
        )
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.device = device
        self.map_size_meters = config.MAP_SIZE
        self.map_cell_size = config.MAP_CELL_SIZE
        self.pos_th = config.DIST_REACHED_TH
        self.next_wp_th = config.NEXT_WAYPOINT_TH
        self.angle_th = config.ANGLE_TH
        self.obstacle_th = config.MIN_PTS_IN_OBSTACLE
        self.depth_denorm = config.DEPTH_DENORM
        self.planned_waypoints = []
        self.mapper = DirectDepthMapper(
            camera_height=config.CAMERA_HEIGHT,
            near_th=config.D_OBSTACLE_MIN,
            far_th=config.D_OBSTACLE_MAX,
            h_min=config.H_OBSTACLE_MIN,
            h_max=config.H_OBSTACLE_MAX,
            map_size=config.MAP_SIZE,
            map_cell_size=config.MAP_CELL_SIZE,
            device=device,
        )
        self.planner = DifferentiableStarPlanner(
            max_steps=config.PLANNER_MAX_STEPS,
            preprocess=config.PREPROCESS_MAP,
            beta=config.BETA,
            device=device,
        )
        self.slam_to_world = 1.0
        self.timestep = 0.1
        self.timing = False
        self.reset()
        self.needs_inspection = True
        self.located_true_goal = False
        self.start_location = None
        self.start_rotation = None
        self.obstacle_ray_map = None
        
        self.curr_map_info = np.zeros(3)  # loc and rot
        
        
        # print('self', self, dir(self))
        self.init_time = int(time.time())
        if print_attrs:
            for attr_name in dir(self):
                if '__' not in attr_name:
                    print(attr_name, getattr(self, attr_name), type(getattr(self, attr_name)))
                    type_name = str(type(getattr(self, attr_name)))
                    if 'numpy' in type_name or 'ensor' in type_name:
                        print(getattr(getattr(self, attr_name), 'shape'))
        return

    def reset(self, map_info=np.zeros(3)):
        super(ORBSLAM2Agent, self).reset()
        self.offset_to_goal = None
        self.tracking_is_OK = False
        self.waypointPose6D = None
        self.unseen_obstacle = False
        self.action_history = []
        self.planned_waypoints = []
        self.map2DObstacles = self.init_map2d()
        n, ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, width, False).to(
            self.device
        )
        self.pose6D = self.init_pose6d()
        self.action_history = []
        self.pose6D_history = []
        self.position_history = []
        self.planned2Dpath = torch.zeros((0))
        self.slam.reset()
        self.cur_time = 0
        self.to_do_list = []
        self.waypoint_id = 0
        
        self.curr_map_info = map_info
        
        if self.device != torch.device("cpu"):
            torch.cuda.empty_cache()
        return

    def update_internal_state(self, habitat_observation):
        # sets self.obs = observation
        super(ORBSLAM2Agent, self).update_internal_state(habitat_observation)

        if self.start_location is None:
            self.start_location = habitat_observation['gps']
        if self.start_rotation is None:
            self.start_rotation = habitat_observation['compass']

        self.cur_time += self.timestep
        rgb, depth = self.rgb_d_from_observation(habitat_observation)
        t = time.time()
        try:
            self.slam.process_image_rgbd(rgb, depth, self.cur_time)
            if self.timing:
                print(time.time() - t, "ORB_SLAM2")
            self.tracking_is_OK = str(self.slam.get_tracking_state()) == "OK"
        except BaseException as e:
            #print("Warning!!!! ORBSLAM processing frame error")
            #print("orbslam error:", e)
            self.tracking_is_OK = False
        if not self.tracking_is_OK:
            #print("\n\n\n\nRESETTING MYSELF BC TRACKING NOT OKAY\n\n\n\n")
            #print("SLAM TRACKING STATE:", self.slam.get_tracking_state())
            self.reset(np.append(habitat_observation['gps'], habitat_observation['compass']))
        t = time.time()
        #self.set_offset_to_goal(habitat_observation)
        if self.tracking_is_OK:
            trajectory_history = np.array(self.slam.get_trajectory_points())
            self.pose6D = homogenize_p(
                torch.from_numpy(trajectory_history[-1])[1:]
                .view(3, 4)
                .to(self.device)
            ).view(1, 4, 4)
            self.trajectory_history = trajectory_history
            if len(self.position_history) > 1:
                previous_step = get_distance(
                    self.pose6D.view(4, 4),
                    torch.from_numpy(self.position_history[-1])
                    .view(4, 4)
                    .to(self.device),
                )
                if self.action_history[-1] == HabitatSimActions.MOVE_FORWARD:
                    self.unseen_obstacle = (
                        previous_step.item() <= 0.001
                    )  # hardcoded threshold for not moving
        current_obstacles = self.mapper(
            torch.from_numpy(depth).to(self.device).squeeze(), self.pose6D
        ).to(self.device)
        self.current_obstacles = current_obstacles
        self.map2DObstacles = torch.max(
            self.map2DObstacles, current_obstacles.unsqueeze(0).unsqueeze(0)
        )
        if self.timing:
            print(time.time() - t, "Mapping")
        return True

    def init_pose6d(self):
        return torch.eye(4).float().to(self.device)

    def map_size_in_cells(self):
        return int(self.map_size_meters / self.map_cell_size)

    def init_map2d(self):
        return (
            torch.zeros(
                1, 1, self.map_size_in_cells(), self.map_size_in_cells()
            )
            .float()
            .to(self.device)
        )

    def get_orientation_on_map(self):
        self.pose6D = self.pose6D.view(1, 4, 4)
        return torch.tensor(
            [
                [self.pose6D[0, 0, 0], self.pose6D[0, 0, 2]],
                [self.pose6D[0, 2, 0], self.pose6D[0, 2, 2]],
            ]
        )

    def get_position_on_map(self, do_floor=True):
        return project_tps_into_worldmap(
            self.pose6D.view(1, 4, 4),
            self.map_cell_size,
            self.map_size_meters,
            do_floor,
        )

    def act(self, habitat_observation, random_prob=0.1):
        # Update internal state
        t = time.time()
        cc = 0
        update_is_ok = self.update_internal_state(habitat_observation)
        while not update_is_ok:
            update_is_ok = self.update_internal_state(habitat_observation)
            cc += 1
            if cc > 2:
                break
        if self.timing:
            print(time.time() - t, " s, update internal state")
        self.position_history.append(
            self.pose6D.detach().cpu().numpy().reshape(1, 4, 4)
        )
        success = self.is_goal_reached()
        if success:
            #print("\n\n\nGOAL IS BELIEVED TO BE REACHED before planning\n\n\n")
            action = HabitatSimActions.STOP
            self.action_history.append(action)
            return {"action": action}
        # Plan action
        t = time.time()

        if len(self.to_do_list) > 0:
            # pop from to do list queue
            action = self.to_do_list.pop(0)
            self.action_history.append(action)
            return {"action": action}

        
        if self.needs_inspection:
            # do a left turn 360
            # each turn is 10 degrees
            self.to_do_list.extend(
                [HabitatSimActions.TURN_LEFT] * 36
            )
            self.needs_inspection = False

        # find a goal and then go to it
        # after reaching the goal, get inspection again

        self.planned2Dpath, self.planned_waypoints = self.plan_path()
        if self.timing:
            print(time.time() - t, " s, Planning")
        t = time.time()
        # Act
        if self.waypointPose6D is None:
            self.waypointPose6D = self.get_valid_waypoint_pose6d()
        if (
            self.is_waypoint_reached(self.waypointPose6D)
            or not self.tracking_is_OK
        ):
            self.waypointPose6D = self.get_valid_waypoint_pose6d()
            self.waypoint_id += 1
        action = self.decide_what_to_do()
        # May be random? ALEX: LETS NOT
#         random_action = random.randint(0, self.num_actions - 1)
#         what_to_do = np.random.uniform(0, 1, 1)
#         if what_to_do < random_prob:
#             action = random_action
        if self.timing:
            print(time.time() - t, " s, get action")
        self.action_history.append(action)
        if action == 0:
            print("\n\n\nGOAL IS BELIEVED TO BE REACHED after planning\n\n\n")
        return {"action": action}

    def is_waypoint_good(self, pose6d):
        p_init = self.pose6D.squeeze()
        dist_diff = get_distance(p_init, pose6d)
        valid = dist_diff > self.next_wp_th
        return valid.item()

    def is_waypoint_reached(self, pose6d):
        p_init = self.pose6D.squeeze()
        dist_diff = get_distance(p_init, pose6d)
        reached = dist_diff <= self.pos_th
        return reached.item()

    def get_waypoint_dist_dir(self):
        angle = get_direction(
            self.pose6D.squeeze(), self.waypointPose6D.squeeze(), 0, 0
        )
        dist = get_distance(
            self.pose6D.squeeze(), self.waypointPose6D.squeeze()
        )
        return torch.cat(
            [
                dist.view(1, 1),
                torch.sin(angle).view(1, 1),
                torch.cos(angle).view(1, 1),
            ],
            dim=1,
        )

    def get_valid_waypoint_pose6d(self):
        p_next = self.planned_waypoints[0]
        while not self.is_waypoint_good(p_next):
            if len(self.planned_waypoints) > 1:
                self.planned_waypoints = self.planned_waypoints[1:]
                p_next = self.planned_waypoints[0]
            else:
                p_next = self.estimatedGoalPos6D.squeeze()
                break
        return p_next

    def is_goal_reached(self):
        if self.obs.get("pointgoal_with_gps_compass") is None:
            dist = self.obs["explorationgoal"][0]
        else:
            dist = self.obs["pointgoal_with_gps_compass"][0]
        return dist <= self.dist_threshold_to_stop

    
    def find_object_goal(self, observation):
        """
        call the object detector and see if the goal is in view. if so, lock it down!
        """
        # return just none for now
        return None


    def plan_exploration(self, observation):
        """
        If just exploring, shoot rays outwards from current location, and go to the ray
        that can go the farthest without hitting and object
        """
        num_rays = 72
        curr_rays = [2 * np.pi / num_rays * i for i in range(num_rays)]
        dist_step = 0.1
        max_dist = 10
        location_in_obs_map = (observation['gps'] - self.start_location).astype(int) * 10 + 200

        if 'numpy' in str(type(self.map2DObstacles)): 
            np_obs_map = (self.map2DObstacles > self.obstacle_th).astype(np.uint8)
        else:
            np_obs_map = (self.map2DObstacles[0,0].cpu().numpy() > self.obstacle_th).astype(np.uint8)

        # kernel for image closing
        kernel = np.ones((5,5),np.uint8)            
        np_obs_map = cv2.morphologyEx(np_obs_map, cv2.MORPH_CLOSE, kernel)

        best_ray = None
        best_coord = None
        best_dist = None

        for n_steps in range(int(max_dist / dist_step)):
            curr_dist = n_steps * dist_step
            next_rays = []
            for ray in curr_rays:
                y_coord = int(np.cos(ray) * curr_dist * 10 + location_in_obs_map[0])
                x_coord = int(-np.sin(ray) * curr_dist * 10 + location_in_obs_map[1])
                if np_obs_map[y_coord, x_coord] != 1:
                    next_rays.append(ray)
                    # color in map to visualize
                    np_obs_map[y_coord, x_coord] = 2

                    if curr_dist == best_dist:
                        curr_ray_angle_delta = norm_ang(ray - float(observation['compass']))
                        best_ray_angle_delta = norm_ang(best_ray - float(observation['compass']))
                        if abs(curr_ray_angle_delta) > abs(best_ray_angle_delta):
                            continue
                                            
                    best_dist = curr_dist
                    best_coord = [y_coord, x_coord]
                    best_ray = ray
            
            if len(next_rays) > 0:
                curr_rays = next_rays
            else:
                break

        if best_dist is None:
            return None

        # color in best ray:
        for n_steps in range(int(best_dist // dist_step)):
            curr_dist = n_steps * dist_step
            y_coord = int(np.cos(best_ray) * curr_dist * 10 + location_in_obs_map[0])
            x_coord = int(-np.sin(best_ray) * curr_dist * 10 + location_in_obs_map[1])
            np_obs_map[y_coord, x_coord] = 4

        self.obstacle_ray_map = np_obs_map
        
        # translate coord to absolute and relative location
        best_gps_coord = np.array([
            (y_coord - 200) / 10,
            (x_coord - 200) / 10,
        ]) + self.start_location

        best_relative_coord = best_gps_coord - observation['gps']

        rotation_from_start = observation['compass'] - self.start_rotation
        # assuming compass is counterclockwise from y+
        best_relative_polar_coord = np.array([
            np.linalg.norm(best_relative_coord),
            # substract 90 deg because arctan is from x+ axis
            float((np.arctan2(best_relative_coord[0], best_relative_coord[1]) - np.pi / 2)
                  - observation['compass'])
        ])
        print('start location {}, start rotation {}'.format(
            self.start_location, self.start_rotation
        ))
        print(
            '''ray {}, dist {}, coord {}, gps coord {}, my gps {},
               my compass {}, relative coord {}, rel polar coord {}'''.format(
                best_ray,
                max_dist,
                best_coord,
                best_gps_coord,
                observation['gps'],
                observation['compass'],
                best_relative_coord,
                best_relative_polar_coord,
            )
        )
        return best_relative_polar_coord
        

    def set_offset_to_goal(self, observation):
        """ ID mappings"""
        obj_to_id = {'chair': 0, 'table': 1, 'picture': 2, 'cabinet': 3, 'cushion': 4, 'sofa': 5, 'bed': 6, 'chest_of_drawers': 7, 'plant': 8, 'sink': 9, 'toilet': 10, 'stool': 11, 'towel': 12, 'tv_monitor': 13, 'shower': 14, 'bathtub': 15, 'counter': 16, 'fireplace': 17, 'gym_equipment': 18, 'seating': 19, 'clothes': 20}
        id_to_obj = {obj_to_id[key]: key for key in obj_to_id}
#         if GOAL_SENSOR_UUID == 'objectgoal':
#             # [distance to goal in metres, angle to goal in radians]
#             # Take the existing goal and find your offset from it
#             self.obs["pointgoal_with_gps_compass"] = self.find_object_goal(observation)
#             if self.obs["pointgoal_with_gps_compass"] is None:
#                 self.obs["explorationgoal"] = self.plan_exploration(observation)
                
#             # doesn't see viable rays. this is not good
#             if self.obs["explorationgoal"] is None:
#                 print("\n\nWarning: no viable rays found\n\n")
#                 self.obs["explorationgoal"] = HabitatSimActions.TURN_RIGHT
#             print('class observation from goal', id_to_obj[observation[GOAL_SENSOR_UUID][0]])

            
        if self.obs["pointgoal_with_gps_compass"] is not None:
            self.located_true_goal = True
            self.offset_to_goal = (
                    torch.from_numpy(self.obs["pointgoal_with_gps_compass"])
                    .float()
                    .to(self.device)
            )
        else:
            self.offset_to_goal = (
                    torch.from_numpy(self.obs["explorationgoal"])
                    .float()
                    .to(self.device)
            )
        #print('ostensible observation from goal', self.offset_to_goal)
        self.estimatedGoalPos2D = habitat_goalpos_to_mapgoal_pos(
            self.offset_to_goal,
            self.pose6D.squeeze(),
            self.map_cell_size,
            self.map_size_meters,
        )
        self.estimatedGoalPos6D = planned_path2tps(
            [self.estimatedGoalPos2D],
            self.map_cell_size,
            self.map_size_meters,
            1.0,
        ).to(self.device)[0]
        # {'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2, 'TURN_RIGHT': 3, 'LOOK_UP': 4, 'LOOK_DOWN': 5}
        # https://github.com/facebookresearch/habitat-api/blob/master/habitat/sims/habitat_simulator/actions.py
        log_attrs = [
            'pose6D', 'estimatedGoalPos2D', 'estimatedGoalPos6D', 'action_history', 'cur_time', 'steps'
        ]
#         for attr in log_attrs:
#             print(attr, getattr(self, attr))
#         if len(self.position_history) > 0:
#             print("most recent position:", self.position_history[-1])
#         print('num obstacles:', self.map2DObstacles.sum())
#         print('gps and compass:', observation['gps'], observation['compass'])
        
        # Log the 2D obstacles map and observations
        if self.map2DObstacles.sum() > 0:
            map_dir = 'maps/' + GOAL_SENSOR_UUID + '_' + str(self.init_time)
            if not os.path.exists(map_dir):
                os.mkdir(map_dir)
            suffix = str(int(time.time())) + '_' + str(int(self.map2DObstacles.sum()))
            np.savez_compressed(map_dir + '/obstacle_map_' +
                       suffix + '.npz',
                       self.map2DObstacles[0,0].cpu().numpy())
            if self.obstacle_ray_map is not None:
                np.savez_compressed(map_dir + '/obstacle_ray_map_' +
                           suffix + '.npz',
                           self.obstacle_ray_map)
            PIL.Image.fromarray(observation['rgb']).save(map_dir + '/rgb_' + suffix + '.png')
            PIL.Image.fromarray(
                (observation['depth'] * 255).astype(np.uint8).squeeze(2), 'L'
            ).save(map_dir + '/depth_' + suffix + '.png')
        return

    def rgb_d_from_observation(self, habitat_observation):
        rgb = habitat_observation["rgb"]
        depth = None
        if "depth" in habitat_observation:
            depth = self.depth_denorm * habitat_observation["depth"]
        return rgb, depth

    def prev_plan_is_not_valid(self):
        if len(self.planned2Dpath) == 0:
            return True
        pp = torch.cat(self.planned2Dpath).detach().cpu().view(-1, 2)
        binary_map = self.map2DObstacles.squeeze().detach() >= self.obstacle_th
        obstacles_on_path = (
            binary_map[pp[:, 0].long(), pp[:, 1].long()]
        ).long().sum().item() > 0
        return obstacles_on_path  # obstacles_nearby or  obstacles_on_path

    def rawmap2_planner_ready(self, rawmap, start_map, goal_map):
        map1 = (rawmap / float(self.obstacle_th)) ** 2
        map1 = (
            torch.clamp(map1, min=0, max=1.0)
            - start_map
            - F.max_pool2d(goal_map, 3, stride=1, padding=1)
        )
        return torch.relu(map1)
    
    def plan_path(self, overwrite=False):
        t = time.time()
        if (
            (not self.prev_plan_is_not_valid())
            and (not overwrite)
            and (len(self.planned_waypoints) > 0)
        ):
            return self.planned2Dpath, self.planned_waypoints
        self.waypointPose6D = None
        current_pos = self.get_position_on_map()
        start_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        start_map[
            0, 0, current_pos[0, 0].long(), current_pos[0, 1].long()
        ] = 1.0
        goal_map = torch.zeros_like(self.map2DObstacles).to(self.device)
        goal_map[
            0,
            0,
            self.estimatedGoalPos2D[0, 0].long(),
            self.estimatedGoalPos2D[0, 1].long(),
        ] = 1.0
        path, cost = self.planner(
            self.rawmap2_planner_ready(
                self.map2DObstacles, start_map, goal_map
            ).to(self.device),
            self.coordinatesGrid.to(self.device),
            goal_map.to(self.device),
            start_map.to(self.device),
        )
        if len(path) == 0:
            return path, []
        if self.timing:
            print(time.time() - t, " s, Planning")
        t = time.time()
        planned_waypoints = planned_path2tps(
            path, self.map_cell_size, self.map_size_meters, 1.0, False
        ).to(self.device)
        return path, planned_waypoints

    def planner_prediction_to_command(self, p_next):
        command = HabitatSimActions.STOP
        p_init = self.pose6D.squeeze()
        d_angle_rot_th = self.angle_th
        pos_th = self.pos_th
        if get_distance(p_init, p_next) <= pos_th:
            return command
        d_angle = norm_ang(
            get_direction(p_init, p_next, ang_th=d_angle_rot_th, pos_th=pos_th)
        )
        if abs(d_angle) < d_angle_rot_th:
            command = HabitatSimActions.MOVE_FORWARD
        else:
            if (d_angle > 0) and (d_angle < pi):
                command = HabitatSimActions.TURN_LEFT
            elif d_angle > pi:
                command = HabitatSimActions.TURN_RIGHT
            elif (d_angle < 0) and (d_angle > -pi):
                command = HabitatSimActions.TURN_RIGHT
            else:
                command = HabitatSimActions.TURN_LEFT
        return command

    def decide_what_to_do(self):
        action = None
        if self.is_goal_reached():
            if self.located_true_goal:
                action = HabitatSimActions.STOP
            else:
                action = HabitatSimActions.TURN_LEFT
                self.needs_inspection = True
            return {"action": action}
        if self.unseen_obstacle:
            command = HabitatSimActions.TURN_RIGHT
            return command
        command = HabitatSimActions.STOP
        command = self.planner_prediction_to_command(self.waypointPose6D)
        return command
    
    
config = get_config("../habitat-api/configs/tasks/objectnav_mp3d_fast.yaml")
config.defrost()
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
config.ORBSLAM2 = CN()
config.ORBSLAM2.SLAM_VOCAB_PATH = "../habitat-api/habitat_baselines/slambased/data/ORBvoc.txt"
config.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "../habitat-api/habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
config.ORBSLAM2.MAP_CELL_SIZE = 0.1
config.ORBSLAM2.MAP_SIZE = 40
config.ORBSLAM2.CAMERA_HEIGHT = config.SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
config.ORBSLAM2.BETA = 100
config.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * config.ORBSLAM2.CAMERA_HEIGHT
config.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * config.ORBSLAM2.CAMERA_HEIGHT
config.ORBSLAM2.D_OBSTACLE_MIN = 0.1
config.ORBSLAM2.D_OBSTACLE_MAX = 4.0
config.ORBSLAM2.PREPROCESS_MAP = True
config.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    config.SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
config.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
config.ORBSLAM2.DIST_REACHED_TH = 0.15
config.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
config.ORBSLAM2.NUM_ACTIONS = 3
config.ORBSLAM2.DIST_TO_STOP = 0.05
config.ORBSLAM2.PLANNER_MAX_STEPS = 500
config.ORBSLAM2.DEPTH_DENORM = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH


def make_good_config_for_orbslam2(config):
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.RGB_SENSOR.WIDTH = 640
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 480
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
    config.ORBSLAM2.CAMERA_HEIGHT = config.SIMULATOR.DEPTH_SENSOR.POSITION[
        1
    ]
    config.ORBSLAM2.H_OBSTACLE_MIN = (
        0.3 * config.ORBSLAM2.CAMERA_HEIGHT
    )
    config.ORBSLAM2.H_OBSTACLE_MAX = (
        1.0 * config.ORBSLAM2.CAMERA_HEIGHT
    )
    config.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
        config.SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
    )
    return

make_good_config_for_orbslam2(config)