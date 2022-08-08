#  @Time    : 2022/5/12
#  @Author  : Yuan Li


#  装备战术任务名称
class TacticsMissions():
    def __init__(self):
        self.task_one = 'AircraftTakeOffAction'  # 起飞
        self.task_two = 'ReturnToBase'  # 返航
        self.task_three = 'WayPointMoveAction'  # 航路机动
        self.task_four = 'AdjustSpeedAlt'  # 调整速度和高度
        self.task_five = 'AttackConditionJudge'  # 打击可行性判断
        self.task_six = 'AttackTargetAction'  # 目标打击
        self.task_seven = 'SensorControlAction'  # 传感器控制
        self.task_eight = 'DelpoySonobuoy'  # 部署声纳浮标
        self.task_nine = 'SwitchMountReady'  # 切换飞机挂载方案(准备阶段)
        self.task_ten = 'SwitchMountRunning'  # 切换飞机挂载方案(推演过程中)
        self.task_eleven = 'CancelAttack'  # 放弃打击
        self.task_twelve = 'FormationSetting'  # 阵型组成


# 编队战役任务名称
class CampaignMissions():
    def __init__(self):
        self.mission_one = 'StrikeMission'  # 打击任务
        self.mission_two = 'PatrolMission'  # 巡逻任务
        self.mission_three = 'SupportMission'  # 支援任务


# 编队打击任务对应子任务名称
class StrikeMission():
    def __init__(self):
        self.missioncmd_one = 'AirIntercept'  # 对空拦截
        self.missioncmd_two = 'LandStrike'  # 对地攻击
        self.missioncmd_three = 'MaritimeStrike'  # 对舰攻击
        self.missioncmd_four = 'Sub_Strike'  # 反潜攻击


# 编队巡逻任务对应子任务名称
class PatrolMission():
    def __init__(self):
        self.missioncmd_one = 'ASW'  # 反潜巡逻
        self.missioncmd_two = 'ASuWNaval'  # 对舰巡逻
        self.missioncmd_three = 'AAW'  # 对空巡逻
        self.missioncmd_four = 'ASuWLand'  # 对地巡逻
        self.missioncmd_five = 'ASuWMixed'  # 对地/对舰巡逻
        self.missioncmd_six = 'SEAD'  # 对敌防空压制
        self.missioncmd_seven = 'SeaControl'  # 制海巡逻


class SupportMission():
    def __init__(self):
        self.missioncmd_one = ''  # 支援任务


class EquipmentID():
    def __init__(self):
        self.RED_CMD_CENTER = '09010001'  # 指挥所
        self.RED_FYS_RADAR = '09040001'  # 反隐身雷达
        self.RED_DKJY_RADAR = '09040002'  # 对空警戒引导雷达
        self.RED_DKJJ_RADAR = '09040003'  # 对空警戒雷达
        self.RED_DHJJ_RADAR = '09040005'  # 对海警戒雷达
        self.RED_JCFK_WEAPON = '09050001'  # 近程防空武器系统
        self.RED_ZCFH_WEAPON = '09050002'  # 中远程防空导弹武器系统
        self.RED_HP_WEAPON = '09050003'  # 火炮
        self.RED_AIRPORT = '09060001'  # 机场

        self.RED_ZDJ_ONE_FLIGHT = '01010003'  # 战斗机1
        self.RED_ZDJ_TWO_FLIGHT = '01010004'  # 战斗机2
        self.RED_YJJ_ONE_FLIGHT = '01170005'  # 预警机1
        self.RED_YJJ_TWO_FLIGHT = '01170001'  # 预警机2
        self.RED_FFXLJ_FLIGHT = '01160001'  # 反潜巡逻机
        self.RED_JZZSJ_FLIGHT = '01150001'  # 舰载直升机
        self.RED_DZZJ_FLIGHT = '01070002'  # 电子战飞机
        self.RED_HZJ_FLIGHT = '01040001'  # 轰炸机
        self.RED_XLJ_SHIP = '02040001'  # 巡逻舰
        self.RED_SLJ_SHIP = '02060001'  # 扫雷舰
        self.RED_SHJCC_SHIP = '02070001'  # 水声监视船
        self.RED_QZJ_ONE_SHIP = '02010001'  # 驱逐舰1
        self.RED_QZJ_TWO_SHIP = '02010003'  # 驱逐舰2
        self.RED_HWJ_SHIP = '02020001'  # 护卫舰
        self.RED_MBHWJ_SHIP = '02020002'  # 目标护卫舰
        self.RED_HDL_SUBMRINE = '02030004'  # 核动力攻击型潜艇
        self.RED_CH_SUBMRINE = '02030005'  # 常规潜艇
        self.RED_DDFSC = '09100001'  # 弹道导弹发射车

        self.BLUE_ZLYJ_RADAR = '09040004'  # 战略预警雷达
        self.BLUE_JZZDJ_FLIGHT = '01010005'  # 舰载战斗机
        self.BLUE_YSZDJ_FLIGHT = '01010006'  # 隐身战斗机
        self.BLUE_ZDJ_FLIGHT = '01010002'  # 战斗机
        self.BLUE_YJJ_FLIGHT = '01170002'  # 预警机
        self.BLUE_HZJ_FLIGHT = '01040002'  # 轰炸机
        self.BLUE_JZZSJ_FLIGHT = '01150002'  # 舰载直升机
        self.BLUE_DZZZCJ_FLIGHT = '01180001'  # 电子战侦察机
        self.BLUE_UAV_FLIGHT = '01140001'  # 无人机
        self.BLUE_FFXLJ_SHIP = '01160002'  # 反潜巡逻机
        self.BLUE_HKMJ_SHIP = '02100001'  # 航空母舰
        self.BLUE_HYJCC_SHIP = '02070002'  # 海洋监测船
        self.BLUE_QZJ_SHIP = '02010002'  # 驱逐舰
        self.BLUE_XYJ_SHIP = '02110001'  # 巡洋舰
        self.BLUE_HDL_SUBMARINE = '02030002'  # 核动力潜艇
        self.BLUE_GJHDL_SUBMARINE = '02030004'  # 攻击型核潜艇
        self.BLUE_AIRPORT = '09060001'  # 机场
        self.BLUE_ANTIMISSILE = '04030004'  # 地空导弹
