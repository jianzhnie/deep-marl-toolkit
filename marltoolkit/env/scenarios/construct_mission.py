import copy
from typing import List


class ConstructMission(object):
    '''
    任务名称(missionSeqName:
        StrikeMission: 进攻任务

        任务类型(missionCmd):
            AirIntercept:空中拦截
            LandStrike:对地攻击
            MaritimeStrike:对舰攻击
            Sub_Strike:反潜攻击

    任务名称(missionSeqName):
        PatrolMission: 巡逻任务

        任务类型(missionCmd):
            ASW:反潜战斗巡逻
            ASuWNaval:对舰战斗巡逻
            AAW:对空巡逻
            ASuWLand:对地战斗巡逻
            ASuWMixed:对地/对舰战斗巡逻
            SEAD:对敌防空压制
            SeaControl:制海巡逻

    任务名称(missionSeqName):
        SupportMission: 支撑任务

        任务类型(missionCmd):无
    '''

    # 构建命令(也就是 构建战役命令)
    def constrcut_campaign_mission(self,
                                   group,
                                   mission_name: str,
                                   missionCmd: str = None,
                                   patrol_time: float = None,
                                   areaVertex: List = [],
                                   forceSide: str = 'Red',
                                   outsideArea: bool = False,
                                   targets: List = [],
                                   protect_target=None,
                                   sonarOperationStatus: bool = False,
                                   ecmOperationStatus: bool = False,
                                   radarOperationStatus: bool = True,
                                   oneTimeOnly: bool = False,
                                   activeTime=0,
                                   weaponRange: bool = False,
                                   cmdFrom: str = '',
                                   packageName: str = ''):
        """构建命令(也就是 构建战役命令)

        :param group:
        :param mission_name:  必填,共有,任务名称 str
        :param missionCmd:    必填,共有,任务类型 str
        :param patrol_time:   选填,共有,任务持续时间 float
        :param areaVertex:    必填,巡逻和支援,任务区域 list
        :param forceSide:     必填,共有,下达任务的阵营,str
        :param outsideArea:   选填,巡逻,调查巡逻区域外的目标,bool
        :param targets:       必填,打击任务,打击目标的所有仿真ID list
        :param protect_target:选填,巡逻任务,护航目标 None
        :param sonarOperationStatus: 选填,共有,声呐运行状态 bool
        :param ecmOperationStatus:   选填,共有,电子战传感器运行状态 bool
        :param radarOperationStatus: 选填,共有,雷达传感器运行状态 bool
        :param oneTimeOnly:          选填,支援任务,任务仅执行一次的开关 bool
        :return:
        """

        mission_tmp = {}
        for item in group.group_action_list:
            mission_tmp = copy.deepcopy(item)
            if mission_tmp['missionSeqName'] == mission_name:
                group.last_mission_name = mission_name
                mission_tmp['mark'] = 1
                mission_tmp['commandUnit']['cmdFrom'] = cmdFrom
                mission_tmp['commandUnit']['packageName'] = packageName
                mission_tmp['controlCmd'][
                    'formationType'] = group.formation_type
                mission_tmp['controlCmd']['groupName'] = group.group_name
                mission_tmp['controlCmd']['entityList'] = [
                    e.identity_id for e in group.group_entity
                ]
                mission_tmp['controlCmd']['escortList'] = [
                    e.identity_id for e in group.group_escort
                ]
                mission_tmp['controlCmd'][
                    'patrolTime'] = 0 if not patrol_time else patrol_time
                mission_tmp['controlCmd']['forceSide'] = forceSide
                mission_tmp['controlCmd']['sonarOperationStatus'] = str(
                    sonarOperationStatus).lower()
                mission_tmp['controlCmd']['ecmOperationStatus'] = str(
                    ecmOperationStatus).lower()
                mission_tmp['controlCmd']['radarOperationStatus'] = str(
                    radarOperationStatus).lower()
                mission_tmp['activeTime'] = activeTime
                if mission_name in ['StrikeMission']:
                    mission_tmp['missionCmd'] = missionCmd
                    mission_tmp['controlCmd']['target'] = targets

                elif mission_name in ['PatrolMission']:
                    mission_tmp['missionCmd'] = missionCmd
                    mission_tmp['controlCmd']['target'] = protect_target
                    mission_tmp['controlCmd']['outsideArea'] = str(
                        outsideArea).lower()
                    mission_tmp['controlCmd']['oneThirdRule'] = 'False'.lower()
                    mission_tmp['controlCmd']['weaponRange'] = str(
                        weaponRange).lower()
                    if len(areaVertex):
                        assert ('The areaVertex should not be empty!')
                    mission_tmp['controlCmd']['areaVertex'] = areaVertex

                elif mission_name in ['SupportMission']:
                    mission_tmp['missionCmd'] = missionCmd
                    if len(areaVertex):
                        assert ('The areaVertex should not be empty!')
                    mission_tmp['controlCmd']['areaVertex'] = areaVertex
                    mission_tmp['controlCmd']['oneTimeOnly'] = str(
                        oneTimeOnly).lower()
                    mission_tmp['controlCmd']['oneThirdRule'] = 'False'.lower()
                break
        return mission_tmp

    # 构建命令(也就是 构建战术命令)
    def constrcut_tactic_mission(self,
                                 agent,
                                 mission_name: str,
                                 missionCmd: str = None,
                                 wayPoint: List = [],
                                 missileType: str = None,
                                 missileNum: int = 0,
                                 speed: float = None,
                                 altitude: float = None,
                                 targetID: str = None,
                                 radarOperationStatus: bool = None,
                                 sonarOperationStatus: bool = None,
                                 ecmOperationStatus: bool = None,
                                 passiveOrActive: bool = None,
                                 shallowOrDeep: bool = None,
                                 mountId=None,
                                 cmdFrom: str = '',
                                 packageName: str = ''):
        '''
        :param agent:
        :param mission_name:
        :param missionCmd:
        :param wayPoint:
        :param missileType:
        :param missileNum:
        :param speed:
        :param altitude:
        :param targetID:
        :param radarOperationStatus:
        :param sonarOperationStatus:
        :param ecmOperationStatus:
        :param passiveOrActive:
        :param shallowOrDeep:
        :return:
        '''

        mission = {}
        mission_tmp = {}
        for item in agent.action_list:
            mission_tmp = copy.deepcopy(item)
            if mission_tmp['missionSeqName'] == mission_name:
                agent.last_mission_name = mission_name
                mission_tmp['mark'] = 1
                mission_tmp['identity_id'] = agent.identity_id
                mission_tmp['missionCmd'] = missionCmd
                mission_tmp['commandUnit']['cmdFrom'] = cmdFrom
                mission_tmp['commandUnit']['packageName'] = packageName

                if mission_name in ['AircraftTakeOffAction']:
                    mission_tmp['controlCmd']['wayPoint'] = wayPoint
                if mission_name in ['WayPointMoveAction']:
                    if wayPoint is None:
                        assert (
                            'wayPoint should not be None for WayPointMoveAction'
                        )
                    else:
                        mission_tmp['controlCmd']['wayPoint'] = wayPoint
                elif mission_name in ['AdjustSpeedAlt']:
                    if speed is not None:
                        mission_tmp['controlCmd']['speed'] = str(speed)
                    if altitude is not None:
                        mission_tmp['controlCmd']['altitude'] = str(altitude)
                elif mission_name in ['AttackConditionJudge']:
                    if targetID is None or missileType is None:
                        assert (
                            'targetID or missileType should not be none for AttackConditionJudge'
                        )
                    else:
                        mission_tmp['controlCmd']['targetID'] = targetID
                        mission_tmp['controlCmd']['missileType'] = missileType
                elif mission_name in ['AttackTargetAction']:
                    if targetID is None:
                        assert (
                            'targetID should not be None for AttackTargetAction'
                        )
                    else:
                        mission_tmp['controlCmd']['targetID'] = targetID
                    if missileType is not None:
                        mission_tmp['controlCmd']['missileType'] = missileType
                        mission_tmp['controlCmd']['missileNum'] = missileNum
                        mission_tmp['controlCmd']['wayPoint'] = wayPoint
                elif mission_name in ['SensorControlAction']:
                    if radarOperationStatus is not None:
                        mission_tmp['controlCmd'][
                            'radarOperationStatus'] = str(
                                radarOperationStatus).lower()
                    if sonarOperationStatus is not None:
                        mission_tmp['controlCmd'][
                            'sonarOperationStatus'] = str(
                                sonarOperationStatus).lower()
                    if ecmOperationStatus is not None:
                        mission_tmp['controlCmd']['ecmOperationStatus'] = str(
                            ecmOperationStatus).lower()
                elif mission_name in ['DelpoySonobuoy']:
                    if passiveOrActive is None:
                        assert (
                            'passiveOrActive should not be None for DelpoySonobuoy!'
                        )
                    else:
                        mission_tmp['controlCmd']['passiveOrActive'] = str(
                            passiveOrActive).lower()
                    if shallowOrDeep is None:
                        assert (
                            'shallowOrDeep should not be None for DelpoySonobuoy!'
                        )
                    else:
                        mission_tmp['controlCmd']['shallowOrDeep'] = str(
                            shallowOrDeep).lower()
                elif mission_name in ['SwitchMount']:
                    if mountId is None:
                        assert ('mountId should not be None for SwitchMount!')
                    else:
                        mission_tmp['controlCmd']['mountId'] = mountId

                mission = copy.deepcopy(mission_tmp)
                break

        return mission
