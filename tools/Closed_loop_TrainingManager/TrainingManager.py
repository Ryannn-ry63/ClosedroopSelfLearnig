# -*-coding:utf-8-*-
# python3.7
# @Time    : 2023/9/17 下午4:51
# @Author  : Shuo yang
# @Software: PyCharm


from SAC.SAC_learner import SAC_Learner, SACConfig, against_agent


MODULE_SCENARIO = 'SCENARIO'


class TrainingManager:
    def __init__(self, SAC_cfg, SAC_cfg_adv, env_cfg, args):
        self.adv_agent = None
        self.ego_agent = None
        self.S_train = 15000  # 训练阶段步长
        self.S_adversarial = 1000  # 场景对抗阶段步长
        self.closed_loop_num = 2  # 闭环场景循环次数
        self.agent_learner = None
        self.adv_scenario_learner = None
        self.SAC_cfg = SAC_cfg
        self.SAC_cfg_adv = SAC_cfg_adv
        self.env_cfg = env_cfg
        self.args = args

        self.against_agent = None

    def Closed_loop_Training_Manager(self):

        state_num = 1
        self.SAC_cfg.total_steps = self.S_train
        self.SAC_cfg_adv.total_steps = self.S_adversarial

        for state in range(self.closed_loop_num):  # 循环次数参数根据预训练步数决定
            # 训练对抗时，将agent_learner的train和save注掉，修改load agent

            self.agent_learner = SAC_Learner(self.SAC_cfg, self.env_cfg, self.args)

            print("====================")
            print("====================")
            print("State num =", state_num)
            print("====================")
            print("====================")

            # ...................................

            self.agent_learner.env_agent_initialize()
            print("====================")
            print("agent_learner_initialize")
            print("====================")

            self.agent_learner.train()
            self.agent_learner.save()

            # ...................................
            #print("====================")
            #print("adv_scenario_stage")
            #print("====================")
            # #
            #self.adv_scenario_learner = SAC_Learner(self.SAC_cfg_adv, self.env_cfg, self.args)
            #self.adv_scenario_learner.env_agent_initialize()
            #print("====================")
            #print("adv_scenario_initialize ")
            #print("====================")
            # # #
            #self.adv_scenario_learner.train()
            #self.adv_scenario_learner.save()
            #
            state_num = state_num + 1
