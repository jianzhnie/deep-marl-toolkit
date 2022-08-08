import copy
import importlib

from env.environment.interface import Environment


class ScenarioEnv():
    def __init__(self, arglist, sampler_id):
        self.sim_time = arglist.sim_time
        self.scenario = arglist.scenario
        self.mpts_sitation = arglist.mpts_sitation
        if arglist.mpts_sitation in ['k8s']:
            if sampler_id in [-1]:
                self.base = Environment(
                    self.sim_time,
                    mpts_sitation='docker',
                    sample_id=sampler_id,
                    mpts_version=arglist.mpts_config['mpts_version'])
            else:
                self.base = Environment(
                    self.sim_time,
                    mpts_sitation=arglist.mpts_sitation,
                    k8s_svc_mpts_host=arglist.mpts_config['k8s_svc_mpts_host'],
                    k8s_svc_mpts_port=arglist.mpts_config['k8s_svc_mpts_port'])
        elif arglist.mpts_sitation in ['docker']:
            self.base = Environment(
                self.sim_time,
                mpts_sitation=arglist.mpts_sitation,
                sample_id=sampler_id,
                mpts_version=arglist.mpts_config['mpts_version'])
        else:
            self.base = Environment(
                self.sim_time,
                mpts_sitation=arglist.mpts_sitation,
                hosts_data=arglist.mpts_config['hosts'][sampler_id])

        self.sim_run_id = self.base.sim_run_id
        self.entities = self.base.entities
        self.entities_red = self.base.entities_red
        self.entities_blue = self.base.entities_blue
        self.fly_missiles = self.base.fly_missiles
        self.fly_missiles_red = self.base.fly_missiles_red
        self.fly_missiles_blue = self.base.fly_missiles_blue

        self.entities_num = self.base.entities_num
        self.red_num = self.base.red_entity_num
        self.blue_num = self.base.blue_entity_num
        self.step_cnt = self.base.step_cnt

        scenario = importlib.import_module('env.scenarios.{}.scenario'.format(
            self.scenario)).Scenario()
        self.area_points = scenario.area_points

        self.is_win = importlib.import_module('env.scenarios.{}.is_win'.format(
            self.scenario)).IsWin(self)

        self.top_chg = []

        self.group_information = self.base.kg_handel.fetch_group_init()
        self.group_actions = self.base.group_actions
        self.isReadyTime = 'true'

    def init_agent(self, agent_one, agent_two):
        agent_one.setup(self.entities_red, self.area_points,
                        self.group_actions)
        agent_two.setup(self.entities_blue, self.area_points,
                        self.group_actions)
        self.agent_one = agent_one
        self.agent_two = agent_two

    def get_done(self, apf=None):
        if apf is not None:
            win_flag, stop_flag, red_score, blue_score = self.is_win.is_win(
                apf)
        else:
            win_flag, stop_flag, red_score, blue_score = self.is_win.is_win()

        if self.base.step_cnt >= self.sim_time or stop_flag > 0:
            done = True
        else:
            done = False
        is_win = -1
        if done == True:
            is_win = win_flag
        return done, is_win, red_score, blue_score

    def step(self,
             step_cnt,
             red_campaign_action=[],
             red_tactic_action=[],
             blue_campaign_action=[],
             blue_tactic_action=[]):
        self.step_cnt = step_cnt
        RespData = None
        if len(red_campaign_action) > 0:
            RespData = self.base.campaign_step(self.agent_one.groups,
                                               red_campaign_action)
        if len(blue_campaign_action) > 0:
            RespData = self.base.campaign_step(self.agent_two.groups,
                                               blue_campaign_action)
        if len(red_tactic_action) > 0:
            for item in red_tactic_action:
                if item['mark'] == 1 and 'SwitchMount' in item[
                        'missionSeqName']:
                    item['missionSeqName'] = 'SwitchMountRunning'
            RespData = self.base.tactic_step(red_tactic_action)
        if len(blue_tactic_action) > 0:
            for item in blue_tactic_action:
                if item['mark'] == 1 and 'SwitchMount' in item[
                        'missionSeqName']:
                    item['missionSeqName'] = 'SwitchMountRunning'
            RespData = self.base.tactic_step(blue_tactic_action)

        self.top_chg = self.base.combat_effect(self.step_cnt)
        return RespData

    def get_obs(self, step_cnt):
        self.update_state_global(step_cnt)
        self.update_state_red(step_cnt)
        self.update_state_blue(step_cnt)
        self.update_state_fly_missile(step_cnt)
        return self.entities_red, self.entities_blue

    def update_state(self, step_cnt):
        self.update_state_global(step_cnt)
        self.update_state_red(step_cnt)
        self.update_state_blue(step_cnt)
        if self.agent_one != None:
            for i in range(len(self.agent_one.groups)):
                for j in range(len(self.agent_one.groups[i].group_entity)):
                    for entity in self.entities:
                        if self.agent_one.groups[i].group_entity[
                                j].identity_id == entity.identity_id:
                            self.agent_one.groups[i].group_entity[j] = entity
                            break

    def update_state_global(self, step_cnt):
        '''
        Args:
            - step_cnt:
        '''
        self.base.update_entities(step_cnt)
        self.entities = self.base.entities

    def update_state_red(self, step_cnt):
        '''
        更新红方态势，实体
        Args:
            - step_cnt: 仿真引擎的步数
        '''
        self.base.update_entities_red(step_cnt)
        self.entities_red = self.base.entities_red

    def update_state_blue(self, step_cnt):
        '''
        更新蓝方态势，实体
        Args:
            - step_cnt: 仿真引擎的步数
        '''
        self.base.update_entities_blue(step_cnt)
        self.entities_blue = self.base.entities_blue

    def update_state_fly_missile(self, step_cnt):
        self.base.update_fly_missiles(step_cnt)
        self.base.update_fly_missiles_red(step_cnt)
        self.base.update_fly_missiles_blue(step_cnt)
        self.fly_missiles = self.base.fly_missiles
        self.fly_missiles_red = self.base.fly_missiles_red
        self.fly_missiles_blue = self.base.fly_missiles_blue

    def group_excute_action(self):
        """下达编组动作指令."""
        ice_connect = self.base.group_excute_action()
        return ice_connect

    def excute_action(self):
        """下达单装飞机的指令."""
        ice_connect = self.base.excute_action()
        return ice_connect

    def reset(self):
        '''
        重启引擎
        Return:
            - obs: 重启后的状态空间
        '''
        self.base.reset()

        self.sim_run_id = self.base.sim_run_id
        self.entities = copy.deepcopy(self.base.entities)
        self.entities_red = copy.deepcopy(self.base.entities_red)
        self.entities_blue = copy.deepcopy(self.base.entities_blue)
        self.step_cnt = self.get_step_cnt()
        self.update_state(self.step_cnt)

    def start_simulation(self):
        self.base.start_simulation()
        self.isReadyTime = 'false'

    def clear_redis(self):
        self.base.redis_handel.clear()

    def stop(self):
        """停止引擎."""
        self.base.ice_handel.set_speed_skip_server(-2)

    def close(self):
        if self.mpts_sitation in ['docker', 'k8s']:
            self.base.get_host_handel.close()

    def get_step_cnt(self):
        """获取仿真引擎的步数."""
        return self.base.redis_handel.get_step_cnt()

    def pause_enginee(self):
        """暂停引擎."""
        self.base.ice_handel.set_speed_skip_server(-1)

    def unpause_enginee(self):
        """取消暂停引擎."""
        self.base.ice_handel.set_speed_skip_server(self.base.run_speed)

    def step_control(self, step_cnt=20):
        """控制算法步数与平台步数保持一致."""
        self.base.step_control(step_cnt)

    def set_position(self, entities):
        """
        改变实体的位置
        Args:
            - ships: 实体的经纬度，<class 'list'>: [{'identity_id': '01010002_225', 'longitude': 120.5, 'latitude': 14.62}, {'identity_id': '01010002_226', 'longitude': 120.5, 'latitude': 14.629999999999999}, {'identity_id': '01010002_227', 'longitude': 120.5, 'latitude': 14.639999999999999}, {'identity_id': '01010002_228', 'longitude': 120.5, 'latitude': 14.649999999999999}, {'identity_id': '01010002_229', 'longitude': 120.5, 'latitude': 14.659999999999998}, {'identity_id': '01010002_230', 'longitude': 120.5, 'latitude': 14.67}, {'identity_id': '01010002_231', 'longitude': 120.5, 'latitude': 14.68}, {'identity_id': '01010002_232', 'longitude': 120.5, 'latitude': 14.69}, {'identity_id': '01010002_233', 'longitude': 120.5, 'latitude': 14.7}, {'identity_id': '01010002_234', 'longitude': 120.5, 'latitude': 14.709999999999999}, {'identity_id': '01170003_16152', 'longitude': 120.5, 'latitude': 14.719999999999999}, {'identity_id': '02010002_61688', 'longitude': 120.5, 'latitude': 14.729999999999999}, {'identity_id': '02030002_4089', 'longitude': 120.5, 'latitude': 14.739999999999998}, {'identity_id': '01070001_60124', 'longitude': 120.5, 'latitude': 14.75}, {'identity_id': '01070001_60125', 'longitude': 120.5, 'latitude': 14.76}, {'identity_id': 'CMD_CENTER8', 'longitude': 120.5, 'latitude': 14.77}, {'identity_id': 'CMD_CENTER9', 'longitude': 120.5, 'latitude': 14.78}, {'identity_id': 'CMD_CENTER10', 'longitude': 120.5, 'latitude': 14.79}, {'identity_id': 'CMD_CENTER11', 'longitude': 120.5, 'latitude': 14.799999999999999}, {'identity_id': 'CMD_CENTER12', 'longitude': 120.5, 'latitude': 14.809999999999999}, {'identity_id': 'CMD_CENTER13', 'longitude': 120.5, 'latitude': 14.819999999999999}, {'identity_id': '01070001_19721', 'longitude': 120.5, 'latitude': 14.83}, {'identity_id': '01070001_19722', 'longitude': 120.5, 'latitude': 14.84}, {'identity_id': '01010002_10487', 'longitude': 120.5, 'latitude': 14.85}, {'identity_id': '01010002_10488', 'longitude': 120.5, 'latitude': 14.86}, {'identity_id': '01010002_10489', 'longitude': 120.5, 'latitude': 14.87}, {'identity_id': '01010002_10490', 'longitude': 120.5, 'latitude': 14.879999999999999}, {'identity_id': '01010002_10491', 'longitude': 120.5, 'latitude': 14.889999999999999}, {'identity_id': '01010002_10492', 'longitude': 120.5, 'latitude': 14.899999999999999}, {'identity_id': '01010002_10493', 'longitude': 120.5, 'latitude': 14.909999999999998}, {'identity_id': '01010002_10494', 'longitude': 120.5, 'latitude': 14.92}, {'identity_id': '01010002_10495', 'longitude': 120.5, 'latitude': 14.93}, {'identity_id': '01010002_10496', 'longitude': 120.5, 'latitude': 14.94}, {'identity_id': '01010002_10497', 'longitude': 120.5, 'latitude': 14.95}, {'identity_id': '01010002_10498', 'longitude': 120.5, 'latitude': 14.959999999999999}, {'identity_id': '01010002_10499', 'longitude': 120.5, 'latitude': 14.969999999999999}, {'identity_id': '01010002_10500', 'longitude': 120.5, 'latitude': 14.979999999999999}]
        """
        self.base.set_position(entities)

    def get_position(self, ship_identity_id):
        '''
        从引擎上获取船的经纬度
        Args:
            - ship_identity_id：船的id, <class 'list'>: ['02010002_61688']
        Return:
            - positions：船的经纬度, <class 'list'>: [{'identity_id': '02010002_61688', 'longitude': 116.51798421316465, 'latitude': 15.819806093192428}]
        '''
        positions = self.base.read_position(ship_identity_id)
        return positions

    def set_ship_position(self, ship_positions):
        '''
        ship_positions: 船的经纬度，<class 'list'>: [{'identity_id': '02020001_17519', 'longitude': '112.771', 'latitude': '16.98'}, {'identity_id': '02020001_17520', 'longitude': '112.78172222222223', 'latitude': '16.628805555555555'}, {'identity_id': '02020001_17521', 'longitude': '115.46050925925925', 'latitude': '16.24837962962963'}, {'identity_id': '02020001_17522', 'longitude': '114.37', 'latitude': '16.2'}]
        '''
        if ship_positions != None and ship_positions != []:
            for index, item in enumerate(ship_positions):
                if item['identity_id'] == '304fee48-aad6-42f9-89fa-46c40d4d2001':
                    ship_positions.pop(index)
                    break
            self.base.set_position(ship_positions)

    def set_weapon(self, switchout_actions):
        """初始化挂载."""
        for item in switchout_actions:
            if item['mark'] == 1 and 'SwitchMount' in item['missionSeqName']:
                item['missionSeqName'] = 'SwitchMountReady'
        RespData = self.base.tactic_step(switchout_actions)
