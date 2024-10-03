# -*-coding:utf-8-*-
# python3.7
# @Time    : 2023/9/17 下午4:51
# @Author  : Shuo yang
# @Software: PyCharm


import math
import matplotlib.pyplot as plt
import gym
import numpy as np

import pandas as pd
from gym import spaces
from tools.modules import *
from agents.local_planner.frenet_optimal_trajectory_lon import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.low_level_controller.controller import PIDLongitudinalController
from agents.low_level_controller.controller import PIDLateralController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel

from datas.data_log import data_collection

from agents.local_planner.frenet_optimal_trajectory_lon import velocity_inertial_to_frenet, \
    get_obj_S_yaw

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
MODULE_ADV_SCENARIO = 'ADV_SCENARIO'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN']
MODULE_SCENARIO = 'SCENARIO'


# ==============================================================================
# -- Util -----------------------------------------------------------
# ==============================================================================

def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    """
    given the ego_state and frenet_path this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index


def lamp(v, x, y):
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0] + 1e-10)


def cal_lat_error(waypoint1, waypoint2, vehicle_transform):
    """
    Estimate the steering angle of the vehicle based on the PID equations
    :param waypoint: target waypoint [x, y]
    :param vehicle_transform: current transform of the vehicle
    :return: lat_error
    """
    v_begin = vehicle_transform.location
    v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                     y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
    v_vec_0 = np.array(
        [math.cos(math.radians(vehicle_transform.rotation.yaw)), math.sin(math.radians(vehicle_transform.rotation.yaw)),
         0.0])
    v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
    w_vec = np.array([waypoint2[0] -
                      waypoint1[0], waypoint2[1] -
                      waypoint1[1], 0.0])
    lat_error = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                  (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

    return lat_error


class CarlagymEnv(gym.Env):

    # metadata = {'render.modes': ['human']}
    def __init__(self):

        self.acc_last = 0.0
        self.replay_scenario_para = None
        self.scenario_module = None
        self.adversarial_scenario_module = None

        self.du_lon_last = None
        self.Input_lon = None
        self.lat_error = None
        self.__version__ = "9.9.2"

        # simulation
        self.verbosity = 0
        self.auto_render = False  # automatically render the environment
        self.n_step = 0
        try:
            self.global_route = np.load(
                'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
            # 1520 *  3

        except IOError:
            self.global_route = None

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.minSpeed = float(cfg.GYM_ENV.MIN_SPEED)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        #self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)
        self.N_SPAWN_CARS = 2


        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)
        self.d_max_s = int(cfg.CARLA.D_MAX_S)
        self.obj_max_vs = float(cfg.TRAFFIC_MANAGER.MAX_SPEED)
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.lanechange = False
        self.is_first_path = True

        # RL
        self.collision_penalty = int(cfg.RL.COLLISION)
        self.low_speed_reward = float(cfg.RL.Low_SPEED_REWARD)
        self.middle_speed_reward = float(cfg.RL.Middle_SPEED_REWARD)
        self.high_speed_reward = float(cfg.RL.High_SPEED_REWARD)
        # self.lane_change_reward = float(cfg.RL.LANE_CHANGE_REWARD)
        # self.lane_change_penalty = float(cfg.RL.LANE_CHANGE_PENALTY)
        # self.off_the_road_penalty = int(cfg.RL.OFF_THE_ROAD)

        # instances
        self.ego = None
        self.ego_los_sensor = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.acceleration_ = 0
        self.eps_rew = 0
        self.u_lon_last = 0.0
        self.u_lon_llast = 0.0
        self.df_ego = 0.0
        self.fig, self.ax = plt.subplots()
        self.x = []
        self.y = []

        self.motionPlanner = None
        self.vehicleController = None
        self.PIDLongitudinalController = None
        self.PIDLateralController = None

        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

        action_low = -2  ###
        action_high = 2  ###
        self.action_dim = (1, 1)  ####
        self.action_space = spaces.Box(-np.inf, np.inf, shape=self.action_dim, dtype='float32')  ####
        self.obs_dim = (1, 5)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.obs_dim, dtype='float32')

        self.log_flag = cfg.GYM_ENV.LOG_FLAG
        self.rt_plot_flag = cfg.GYM_ENV.RT_PLOT_FLAG
        # data_record
        if self.log_flag:
            self.log = data_collection()
        else:
            pass

    def get_vehicle_ahead(self, ego_s, ego_d, ego_init_d, ego_target_d):
        """
        This function returns the values for the leading actor in front of the ego vehicle. When there is lane-change
        it is important to consider actor in the current lane and target lane. If leading actor in the current lane is
        too close than it is considered to be vehicle_ahead other wise target lane is prioritized.
        """
        distance = self.effective_distance_from_vehicle_ahead
        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.adversarial_scenario_module.actors_batch):
            act_s, act_d = actor['Frenet State'][0][-1], actor['Frenet State'][1]
            others_s[i] = act_s
            others_d[i] = act_d

        init_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 1.75) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        init_lane_strict_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 0.4) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        target_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 3.3) * (abs(np.array(others_d) - ego_target_d) < 1))[0]

        if len(init_lane_d_idx) and len(target_lane_d_idx) == 0:
            return None  # no vehicle ahead
        else:
            init_lane_s = np.array(others_s)[init_lane_d_idx]
            init_s_idx = np.concatenate(
                (np.array(init_lane_d_idx).reshape(-1, 1), (init_lane_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_s_idx = init_s_idx[init_s_idx[:, 1].argsort()]

            init_lane_strict_s = np.array(others_s)[init_lane_strict_d_idx]
            init_strict_s_idx = np.concatenate(
                (np.array(init_lane_strict_d_idx).reshape(-1, 1), (init_lane_strict_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_strict_s_idx = init_strict_s_idx[init_strict_s_idx[:, 1].argsort()]

            target_lane_s = np.array(others_s)[target_lane_d_idx]
            target_s_idx = np.concatenate((np.array(target_lane_d_idx).reshape(-1, 1),
                                           (target_lane_s - ego_s).reshape(-1, 1),), axis=1)
            sorted_target_s_idx = target_s_idx[target_s_idx[:, 1].argsort()]

            if any(sorted_init_s_idx[:, 1][sorted_init_s_idx[:, 1] <= 10] > 0):
                vehicle_ahead_idx = int(sorted_init_s_idx[:, 0][sorted_init_s_idx[:, 1] > 0][0])
            elif any(sorted_init_strict_s_idx[:, 1][sorted_init_strict_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_init_strict_s_idx[:, 0][sorted_init_strict_s_idx[:, 1] > 0][0])
            elif any(sorted_target_s_idx[:, 1][sorted_target_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_target_s_idx[:, 0][sorted_target_s_idx[:, 1] > 0][0])
            else:
                return None

            return self.adversarial_scenario_module.actors_batch[vehicle_ahead_idx]['Obj_Frenet_state']

    def obj_info(self):
        """
        Frenet:  [s,d,v_s, v_d, phi_Frenet]
        """
        others_s = np.zeros(self.N_SPAWN_CARS)
        others_d = np.zeros(self.N_SPAWN_CARS)
        others_v_S = np.zeros(self.N_SPAWN_CARS)
        others_v_D = np.zeros(self.N_SPAWN_CARS)
        others_phi_Frenet = np.zeros(self.N_SPAWN_CARS)

        for i, actor in enumerate(self.adversarial_scenario_module.actors_batch):
            act_s, act_d, act_v_S, act_v_D, act_psi_Frenet = actor['Obj_Frenet_state']
            others_s[i] = act_s
            others_d[i] = act_d
            others_v_S[i] = act_v_S
            others_v_D[i] = act_v_D
            others_phi_Frenet[i] = act_psi_Frenet
        obj_info_Mux = np.vstack((others_s, others_d, others_v_S, others_v_D, others_phi_Frenet))
        return obj_info_Mux

    def state_input_vector(self, v_S, ego_s, ego_d, current_acc):
        # Paper: Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning
        obj_mat = self.obj_info()
        state_vector = np.zeros(5)

        # No normalized
        state_vector[0] = obj_mat[0][0] - ego_s  # 障碍车辆与自车纵向相对距离
        state_vector[1] = obj_mat[2][0]- v_S
        state_vector[2] = v_S
        state_vector[3] = current_acc
        state_vector[4] = obj_mat[1][0] - ego_d

        # Normalized
        #state_vector[0] = (obj_mat[0] - ego_s) / 100
        #state_vector[1] = np.clip(lamp(obj_mat[2] - v_S, [-20, 10], [0, 1]), 0, 1)
        #state_vector[2] = np.clip(v_S / 20, 0, 1)

        return state_vector

    def sensor_info(self):

        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]
        vx_ego = self.ego.get_velocity().x
        vy_ego = self.ego.get_velocity().y
        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
        v_S, v_D = velocity_inertial_to_frenet(ego_s, vx_ego, vy_ego, self.motionPlanner.csp)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                     math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]
        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s

        direct_x = self.ego.get_transform().get_forward_vector().x
        direct_y = self.ego.get_transform().get_forward_vector().y
        normal_dir_x = direct_x / math.sqrt(direct_x ** 2 + direct_y ** 2)
        normal_dir_y = direct_y / math.sqrt(direct_x ** 2 + direct_y ** 2)
        current_acc = acc_vec.x * normal_dir_x + acc_vec.y * normal_dir_y
        obj_mat = self.obj_info()

        return obj_mat, v_S, ego_s, current_acc

    def action_normal(self, action):

        k = self.action_space.high[0] - (self.action_space.high[0] + self.action_space.low[0]) / 2.0
        action = action * k + (self.action_space.high[0] + self.action_space.low[0]) / 2.0
        return action

    def step(self, a):

        #action = self.action_normal(action[0])
        #self.adversarial_scenario_module.update_adv_action(self.replay_scenario_para[self.n_step * 3])  ###因为场景参数选用了3个，排序是action、车距、相对位移
        #self.adversarial_scenario_module.update_adv_action(action)
        action = a[0]
        self.n_step += 1
        state = np.zeros_like(self.observation_space.sample())

        '''******   Action Design    ******'''

        acc_input = self.acc_last + action
        acc_input = np.clip(acc_input, -4, 3)
        # print('acc_input=',acc_input)

        df_n = 0
        #if self.n_step >= 200:
        #   df_n = 3.5

        """
                **********************************************************************************************************************
                ************************************************* Planner *********************************************************
                **********************************************************************************************************************
        """

        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]
        fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state, self.f_idx,
                                                                                       df_n, Tf=5,
                                                                                       Vf_n=0)
        wps_to_go = len(fpath.t) - 3  # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]

        '''******   Initialize flags    ******'''
        collision = False
        self.f_idx = 1

        '''******   birds-eye view    ******'''
        spectator = self.world_module.world.get_spectator()
        transform = self.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=100),
                                                carla.Rotation(pitch=-90)))

        """
        # third person view
        spectator = self.world.get_spectator()
        transform = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(x=0, y=7, z=5),
                                                carla.Rotation(pitch=-20, yaw=-90)))
        """

        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """

        vx_ego = self.ego.get_velocity().x
        vy_ego = self.ego.get_velocity().y
        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
        ego_d = fpath.d[self.f_idx]
        v_S, v_D = velocity_inertial_to_frenet(ego_s, vx_ego, vy_ego, self.motionPlanner.csp)

        ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                     math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]

        self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)

        # print(self.f_idx)
        cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
        cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

        vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
        self.df_ego = closest([self.LANE_WIDTH * lane_n for lane_n in range(-1, 3)], ego_d)

        direct_x = self.ego.get_transform().get_forward_vector().x
        direct_y = self.ego.get_transform().get_forward_vector().y
        normal_dir_x = direct_x / math.sqrt(direct_x ** 2 + direct_y ** 2)
        normal_dir_y = direct_y / math.sqrt(direct_x ** 2 + direct_y ** 2)
        act_acc = acc_vec.x * normal_dir_x + acc_vec.y * normal_dir_y

        # acc_input = self.IDM_model(v_S, ego_s, obj_mat)

        '''******  speed control method 1******'''
        # cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)
        # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control

        '''******  speed control method 2******'''
        # cmdSpeed = get_speed(self.ego) + float(acc_input) * self.dt
        # control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control()

        '''****** acc control ******'''
        control = self.vehicleController.run_step_acc_2_wp(acc_input, act_acc, cmdWP,
                                                           cmdWP2)  # calculate control

        self.ego.apply_control(control)  # apply control

        obj_mat, v_S, ego_s, current_acc = self.sensor_info()
        self.adversarial_scenario_module.update_obj_input([obj_mat, v_S, ego_s, current_acc])
        # other vehicle for against agent

        '''******   Update carla world ******'''
        self.module_manager.tick()
        if self.auto_render:
            self.render()

        """
                **********************************************************************************************************************
                ************************************************* Sensor *********************************************************
                **********************************************************************************************************************
        """

        self.world_module.collision_sensor.reset()
        collision_hist = self.world_module.get_collision_history()
        if any(collision_hist):
            collision = True

        obj_info = self.obj_info()
        """
                **********************************************************************************************************************
                *********************************************** Draw Waypoints *******************************************************
                **********************************************************************************************************************
        """

        if self.world_module.args.play_mode != 0:
            for i in range(len(fpath.x)):
                self.world_module.points_to_draw['path wp {}'.format(i)] = [
                    carla.Location(x=fpath.x[i], y=fpath.y[i]),
                    'COLOR_ALUMINIUM_0']

            self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
            self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
            self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

            """
               **********************************************************************************************************************
               *********************************************** Reinforcement Learning *******************************************************
               **********************************************************************************************************************
            """

        '''******   State Design    ******'''

        state_vector = self.state_input_vector(v_S, ego_s, ego_d, current_acc)

        # print(len(state_vector))
        for i in range(len(state_vector)):
            state[0][i] = state_vector[i]

        '''******   Reward Design   ******'''
        #k_attr = 1/4
        #k_detra = 1500
        #s_goal = 15
        #speed_reward = 4 * math.e ** (-0.09 * (state_vector[2] - 12) ** 2)
        #collision_reward = 0
#
        #if collision:
        #    collision_reward = -200

        #attra_reward = 3 - k_attr * (state_vector[0])

        #if state_vector[0] < 15:
        #    detra_reward = - k_detra * (1 / state_vector[0] - 1 / s_goal)*( 1 / state_vector[0] ** 2)
        #else:
        #    detra_reward = 0

        #reward = speed_reward + collision_reward + attra_reward+detra_reward


        s_len = 5.037
        d_width = 2.077
        k_f = 0.001
        deta_1 = 8
        deta_2 = 10
        reward_factor = -18
        scale = math.exp(((0.8-d_width)**2/deta_1 **2)+((20-s_len)**2/deta_2 **2))
        k = -reward_factor / (k_f * scale -k_f)
        b = reward_factor-k * k_f
        obj_mat = self.obj_info()
        reward_dis = 0
        if state_vector[0]>=4:
            dd = abs(obj_mat[1][0] - ego_d)
            ds = abs(obj_mat[0][0] - ego_s)
            r = min((math.exp((min(max(dd-d_width,0.0),150)**2/deta_1**2)+((min(max(ds-s_len,0.0),150))**2/deta_2**2))*k_f*k+b),0)
            reward_dis += r*0.15

        scaled_speed_l = lamp(state_vector[2],[0,self.minSpeed],[0,1])
        scaled_speed_h = lamp(state_vector[2], [self.minSpeed, self.maxSpeed], [0, 1])
        reward_hs_l = 0.5
        reward_hs_h = 4.0
        reward_speed = reward_hs_l * np.clip(scaled_speed_l,0,1)+reward_hs_h * np.clip(scaled_speed_h,0,1)

        if collision:
            reward_cl = -200.0
        else:
            reward_cl = 0.0

        reward = -(reward_dis+reward_speed-reward_cl)
        done = False

        if collision or self.n_step >= 400:
            done = True


        '''******   Data Record   ******'''

        if done:
            self.module_manager.get_module(MODULE_SCENARIO).clear_memory()
        '''******   Info Process   ******'''
        info = {'reserved': 0}
        obs = state[0, :]
        self.acc_last = acc_input

        return obs, reward, done, info

    def reset(self):
        self.module_manager.get_module(MODULE_SCENARIO).clear_memory()

        self.vehicleController.reset()
        self.PIDLongitudinalController.reset()
        self.PIDLateralController.reset()
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        self.init_d = self.world_module.init_d
        self.adversarial_scenario_module.reset(self.init_s, self.init_d)
        self.motionPlanner.reset(self.init_s, self.init_d)

        self.adversarial_scenario_module.update_global_route_csp(self.motionPlanner.csp)  ####

        self.f_idx = 0

        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.is_first_path = True

        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=False)

        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)

        return np.zeros_like(self.observation_space.sample()[0, :])

    def update_against_agent(self, against_agent):

        self.replay_scenario_para = against_agent.replay_scenario_para

    # def update_scenario_para(self, replay_scenario_para):
    #     self.replay_scenario_para = replay_scenario_para

    def begin_modules(self, args):
        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=100.0, module_manager=self.module_manager,
                                        width=width, height=height)

        self.adversarial_scenario_module = Adversarial_scenario_manager(MODULE_ADV_SCENARIO,
                                                                        module_manager=self.module_manager)
        self.scenario_module = Scenario_manager(MODULE_SCENARIO, module_manager=self.module_manager)

        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.adversarial_scenario_module)
        self.module_manager.register_module(self.scenario_module)

        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        # generate and save global route if it does not exist in the road_maps folder
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            self.global_route = np.vstack([self.global_route, self.global_route[0, :]])
            np.save('road_maps/global_route_town04', self.global_route)
            # plt.plot(self.global_route[:, 0], self.global_route[:, 1])
            # plt.show()

        self.motionPlanner = MotionPlanner()

        # Start Modules
        self.motionPlanner.start(self.global_route)
        # solve Spline
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.adversarial_scenario_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()

        self.ego = self.world_module.hero_actor
        self.ego_los_sensor = self.world_module.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.PIDLongitudinalController = PIDLongitudinalController(self.ego, K_P=40.0, K_D=0.1, K_I=4.0)
        self.PIDLateralController = PIDLateralController(self.ego, K_P=1.5, K_D=0.0, K_I=0.0)
        self.IDM = IntelligentDriverModel(self.ego)

        self.module_manager.tick()  # Update carla world 传参
        self.init_transform = self.ego.get_transform()

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human', close=False):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
