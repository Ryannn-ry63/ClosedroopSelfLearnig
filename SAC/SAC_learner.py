# -*-coding:utf-8-*-
# python3.7
# @Time    : 2023/9/17 下午4:51
# @Author  : Shuo yang
# @Software: PyCharm

import os
import sys
import time
import Carla_gym

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
root_path = os.getcwd()
sys.path.append(parent_path)  # 添加路径到系统路径
import re
import gym
import torch
import datetime
# from SoftActorCritic.env_wrapper import NormalizedActions
from SAC.env import NormalizedActions
from SAC.agent import SAC
from common.utils import save_results, make_dir
from common.plot import plot_rewards

import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def get_dir_path(path):
    """
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    """

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        # os.makedirs(path)
        return path, None
    else:
        path, path_origin = directory_check(path)
        # os.makedirs(path)
        return path + '/', path_origin + '/'


def directory_check(directory_check):
    temp_directory_check = directory_check
    i = 1
    while i:

        if os.path.exists(temp_directory_check):
            search = '_'
            numList = [m.start() for m in re.finditer(search, temp_directory_check)]
            numList[-1]
            temp_directory_check = temp_directory_check[0:numList[-1] + 1] + str(i)
            i = i + 1
        else:
            return temp_directory_check, temp_directory_check[0:numList[-1] + 1] + str(i - 2)


class SACConfig:
    def __init__(self, mode_name='self-learning', train_name='default', con_learning_flag=True, total_timesteps=100000):

        self.mode_name = mode_name
        if self.mode_name == 'self-learning':
            self.env_name = 'gym_env-v0'
        elif self.mode_name == 'self-adversarial':
            self.env_name = 'gym_env_scenario-v0'

        self.algo = 'SAC'
        self.train_name = train_name
        self.con_learning_flag = con_learning_flag
        self.model_save_path = root_path + "/Results/RL_Results/" + "model_save"
        self.log_save_path = root_path + "/Results/RL_Results/runs_info/" + 'runs'
        self.eval_path = root_path + "/Results/RL_Results/eval_info/" + 'runs'
        self.ma_window = 10
        self.train_eps = 200
        self.train_steps = 650
        self.eval_eps = 200
        self.eval_steps = 650
        self.total_steps = total_timesteps
        self.gamma = 0.99
        self.soft_tau = 5e-3
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000

        self.eval_total_steps = 60000
        self.hidden_dim = 256
        self.batch_size = 128
        self.alpha_lr = 3e-4
        self.AUTO_ENTROPY = True
        self.DETERMINISTIC = False
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")

class against_agent:
    def __init__(self, SAC_cfg):
        self.state_dim = None
        self.action_dim = None
        if SAC_cfg.mode_name == 'self-learning':
            self.agent_name = 'adv_agent'
        else:
            self.agent_name = 'ego_agent'
        self.SAC_cfg = SAC_cfg
        self.replay_scenario_para = None
        self.agent_model = None
        # self.load_against_agent()

    def load_against_agent(self):
        # 加载对抗智能体

        if self.agent_name == 'adv_agent':
            # if self.adv_scenario_data_path is not None:
            #     self.env.scenario_module.update_result_path(self.adv_scenario_data_path)
            #     replay_scenario_para = self.env.scenario_module.replay_scenario()
            # else:
            #     replay_scenario_para = 0.5 * np.ones(1200)
            self.replay_scenario_para = 0.5 * np.ones(1200)
            print("load adv_agent successfully label " + self.SAC_cfg.train_name)
        else:
            env = gym.make('gym_env_scenario-v0')
            self.action_dim = env.action_space.shape[0]  ###need to be modified into 3？
            self.state_dim = env.observation_space.shape[1]
            # 此版本中SAC算法的参数是一致的
            self.agent_model = SAC(self.state_dim, self.action_dim, self.SAC_cfg)

            _, model_path = get_dir_path(
                self.SAC_cfg.model_save_path + '/' + 'self-learning_model' + '/' + self.SAC_cfg.train_name + '/')
            # model_path = (self.SAC_cfg.model_save_path + '/' + 'self-learning_model' + '/' + 'train_1' + '/')
            print('load' + model_path)
            # print("load ego_agent successfully label " + self.SAC_cfg.train_name)
            self.agent_model.load(model_path)


class SAC_Learner:

    def __init__(self, SAC_cfg, env_cfg, args):

        self.writer = None
        self.rewards = None
        self.ma_rewards = None
        self.SAC_cfg = SAC_cfg
        self.model_save_path = self.SAC_cfg.model_save_path

        self.model_path = self.model_save_path + '/' + self.SAC_cfg.mode_name + '_model' + '/' + self.SAC_cfg.train_name + '/'
        self.log_path = self.SAC_cfg.log_save_path + '/' + self.SAC_cfg.mode_name + '_runs' + '/' + self.SAC_cfg.train_name + '/'
        self.eval_path = self.SAC_cfg.eval_path + '/' + self.SAC_cfg.mode_name + '_runs' + '/' + self.SAC_cfg.train_name + '/'
        if self.SAC_cfg.con_learning_flag:
            # 是否继续进行闭环学习
            self.model_path, _ = get_dir_path(self.model_path)
        else:
            self.model_path = self.model_path

        self.adv_scenario_data_path = None
        self.env = None
        self.cfg = env_cfg
        self.args = args
        self.agent = None

        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        self.aga_agent = against_agent(self.SAC_cfg)
        self.aga_agent.load_against_agent()

    def env_agent_initialize(self, seed=1):

        self.env = gym.make(self.SAC_cfg.env_name)
        self.env.seed(seed)
        print('Env is starting')
        if self.args.play_mode:
            self.env.enable_auto_render()
        self.env.begin_modules(self.args)
        action_dim = self.env.action_space.shape[1]  #########
        state_dim = self.env.observation_space.shape[1]

        self.agent = SAC(state_dim, action_dim, self.SAC_cfg)
        print(self.SAC_cfg.algo + ' algorithm is starting')
        # tensorboard
        if self.args.train_or_eval == 1:
            self.log_path, _ = get_dir_path(self.log_path)
            self.writer = SummaryWriter(self.log_path)  # default at runs folder if not sepecify path
        else:
            self.eval_path, _ = get_dir_path(self.eval_path)
            self.writer = SummaryWriter(self.eval_path)  # default at runs folder if not sepecify path

    def load(self, model_path):
        self.agent.load(model_path)

    def train(self):

        print('Start to train !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        rewards = []
        ma_rewards = []  # moving average reward
        total_nums = 0

        for i_ep in range(self.SAC_cfg.train_eps):

            state = self.env.reset()

            self.env.update_against_agent(self.aga_agent)
            # 将对抗智能体传入gym env中
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action[0])

            eps_reward = 0.0
            for i_step in range(self.SAC_cfg.train_steps):

                total_nums = total_nums + 1
                action = self.agent.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.update(reward_scale=1., auto_entropy=self.SAC_cfg.AUTO_ENTROPY,
                                  target_entropy=-1. * self.env.action_space.shape[0], gamma=self.SAC_cfg.gamma,
                                  soft_tau=self.SAC_cfg.soft_tau)
                state = next_state
                eps_reward += reward
                if i_ep == 0 or done:
                    break
            # mean_reward = eps_reward / i_step
            if (i_ep + 1) % 1 == 0:
                rewards.append(eps_reward)
                print(f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{eps_reward:.3f}")
                print(f'总步数：{total_nums}')
                self.writer.add_scalar("Reward", eps_reward, total_nums)

            if total_nums >= self.SAC_cfg.total_steps:
                break
        print('Complete training！')
        self.env.destroy()
        return rewards, ma_rewards

    def eval(self):
        print('Start to eval !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        rewards = []
        total_nums = 0
        ma_rewards = []  # moveing average reward
        for i_ep in range(self.SAC_cfg.eval_eps):

            state = self.env.reset()

            self.env.update_against_agent(self.aga_agent)
            # 将对抗智能体传入gym env中

            next_state, reward, done, _ = self.env.step([0])

            eps_reward = 0.0
            for i_step in range(self.SAC_cfg.train_steps):

                total_nums = total_nums + 1
                action = self.agent.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.update(reward_scale=1., auto_entropy=self.SAC_cfg.AUTO_ENTROPY,
                                  target_entropy=-1. * self.env.action_space.shape[0], gamma=self.SAC_cfg.gamma,
                                  soft_tau=self.SAC_cfg.soft_tau)
                state = next_state
                eps_reward += reward
                if i_ep == 0 or done:
                    break
            # mean_reward = eps_reward / i_step
            if (i_ep + 1) % 1 == 0:
                rewards.append(eps_reward)
                print(f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{eps_reward:.3f}")
                print(f'总步数：{total_nums}')
                self.writer.add_scalar("Reward", eps_reward, total_nums)

            if total_nums >= self.SAC_cfg.total_steps:
                break
        print('Complete evaluating')
        self.env.destroy()
        return rewards, ma_rewards

    def save(self):
        make_dir(self.model_path)
        self.agent.save(path=self.model_path)
