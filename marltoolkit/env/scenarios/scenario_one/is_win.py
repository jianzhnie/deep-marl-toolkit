class IsWin():
    def __init__(self, env):
        self.env = env

    def is_win(self):
        '''
        :return: 1： 为蓝方赢 0： 为红方赢, 2 为平局
        '''
        stop_flag = 0
        win_flag = 0
        # 蓝方岛屿周边驱逐舰和防空系统被被消灭
        dead_num = 0
        red_score = 0
        blue_score = 0
        for item in self.env.entities:
            if item.identity_id == '65s9y7-00001fp3gjs3d' and item.alive is False:
                dead_num += 1
            if item.identity_id == '65s9y7-00001fp3gjsai' and item.alive is False:
                dead_num += 1
            if item.identity_id == '65s9y7-00001fp3gjsqm' and item.alive is False:
                dead_num += 1
        if dead_num == 3:
            stop_flag = 1
            win_flag = 1

        step_cnt = self.env.step_cnt
        if step_cnt >= self.env.sim_time:
            stop_flag = 1

        if stop_flag == 1 and win_flag == 1:
            red_score = 1
        if stop_flag == 1 and win_flag == 0:
            red_score = 1

        return win_flag, stop_flag, red_score, blue_score
