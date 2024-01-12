import os
from typing import Dict, Iterable, List, Union

import numpy as np
from pywargame import (K_INVALID_GOBJECT_ID, AlertStatus, GameAction,
                       GameActionCode, GameParams, ObjectType, PathLoopMode,
                       PlayerID, SceneInfo, SupplyStatus, UnitRuntimeInfo,
                       UnitStaticInfo, Vector3, WarGame,
                       create_agent_from_file)

MIN_SUPPLY_RATIO = 0.3
LIFE_SCALE_FACTOR = 500.0

EPSILON = 1e-10


def get_number_feat_dim(max_val: float, n_decimals: int) -> int:
    return len(str(int(max_val))) + 2 + n_decimals


def encode_large_positive_int(v: int, size: int) -> List[float]:
    assert v >= 0
    old_v = v
    v = int(v)
    feat = [0.0] * size
    k = size - 1
    while v > 0:
        if k < 0:
            raise RuntimeError(
                f'Int number {old_v} overflow of feature size {size}')
        digit = v % 10
        feat[k] = digit * 0.1
        v = v // 10
        k -= 1
    return feat


def encode_large_float(v: float,
                       max_val: float,
                       n_decimal: int = 2) -> List[float]:
    assert max_val > 0
    flag = 1.0 if v >= 0 else -1.0
    vp = abs(v)
    v_int_part = int(vp)
    v_decimal_part = int((vp - v_int_part) * pow(10, n_decimal))
    int_max_size = len(str(int(max_val)))
    return [v / max_val, flag] + encode_large_positive_int(
        v_int_part, int_max_size) + encode_large_positive_int(
            v_decimal_part, n_decimal)


def get_opposite_id(player_id: int) -> int:
    """Get the opposite player id, 0 -> 1, 1 -> 0.

    Args:
        player_id (int):  player id, 0 for ally, 1 for enemy

    Raises:
        RuntimeError: _description_

    Returns:
        int: _description_
    """
    if player_id == 0:
        return 1
    elif player_id == 1:
        return 0
    raise RuntimeError(f'Invalid player_id {player_id}')


def _simple_one_hot(index: Union[int, List[int]], size: int) -> List[float]:
    """One hot encoding for index.

    Args:
        index (Union[int, List[int]]): _description_
        size (int): _description_

    Returns:
        List[float]: _description_
    """
    if size <= 0:
        raise RuntimeError(f'Invalid one hot feature size: {size}')
    oh_feat = [0.0] * size
    if isinstance(index, int):
        index = [
            index,
        ]
    for i in index:
        oh_feat[i] = 1.0
    return oh_feat


def preprocess_data(x, mean, std) -> np.ndarray:
    """Preprocess data, normalize to [-1, 1]

    Args:
        x (np.ndarray): The data to be normalized
        mean (np.ndarray): mean value
        std (_type_): std value

    Returns:
        np.ndarray: normalized data
    """
    x_new = x.copy()
    mask = std >= EPSILON
    if len(x.shape) == 2:
        x_new[:, mask] = (x[:, mask] - mean[mask]) / std[mask]
    elif len(x.shape) == 1:
        x_new[mask] = (x[mask] - mean[mask]) / std[mask]
    else:
        raise RuntimeError(f'Expected 1d or 2d feature array, given {x.shape}')

    min_v = np.min(x_new)
    max_v = np.max(x_new)
    mean_v = np.mean(x_new)
    if np.abs(mean_v) > 1000 or np.abs(min_v) > 1000 or np.abs(max_v) > 1000:
        print('Unexpected feature data!')

    return x_new


class WarGameSAWrapper:

    def __init__(self, params: GameParams, stats_file: str = None) -> None:
        self._params = params
        self._ally_player_id = PlayerID.RedTeam
        self._enemy_player_id = PlayerID.BlueTeam
        self._env = WarGame(params)
        self._enemy_agent = None
        self._map_size = -1
        self._map_width = -1
        self._map_height = -1

        self._n_decimal = 2
        self._coord_max_v = -1
        self._coord_dim = -1
        self._speed_max_v = 340.0 * 8
        self._speed_dim = get_number_feat_dim(self._speed_max_v,
                                              self._n_decimal)
        self._height_max_v = 100000
        self._height_dim = get_number_feat_dim(self._height_max_v,
                                               self._n_decimal)

        self._stats = None
        if stats_file is not None and os.path.exists(stats_file):
            import pickle
            with open(stats_file, 'rb') as fd:
                self._stats = pickle.load(fd)
                print(f'Load obs/state mean/std from {stats_file}')

        self.reset()

    def get_obs(self, player_id: int = 0) -> List[np.ndarray]:
        if self._stats is not None:
            return [
                preprocess_data(x, self._stats['obs_mean'],
                                self._stats['obs_std'])
                for x in self._team_agent_features[player_id]
            ]
        else:
            return self._team_agent_features[player_id]

    def get_state(self) -> np.ndarray:
        if self._stats is not None:
            return preprocess_data(self._global_state,
                                   self._stats['state_mean'],
                                   self._stats['state_std'])
        else:
            return self._global_state

    def close(self):
        if self._env is not None:
            self._env.close()

    def __del__(self):
        self.close()

    def reset(self, seed: int = 0) -> None:
        """Reset the Env.

        Args:
            seed (int, optional): _description_. Defaults to 0.

        Attribute:

        - self._team_scene_info : Dict[PlayerID, SceneInfo]
        - self._team_n_units: Dict[PlayerID, Num_of_unit]
        - self._team_usid: Dict[PlayerID, UnitStaticInfo]
        - self._team_wsid: Dict[PlayerID, WeaponStaticInfo]
        - self._team_unit_id2index: Dict[PlayerID, unit_id2_index]
        - self._team_unit_index2id: Dict[PlayerID, unit_index2id]
        - self._team_squad_id2index: Dict[PlayerID, squad_id2index]
        - self._team_squad_id2index: Dict[PlayerID, squad_index2id]
        - self._team_agent_features: Dict[PlayerID, List[None, None, ..., None]]
        """
        if self._env is None:
            self._env = WarGame(self._params)

        self._env.reset(seed)

        self._team_scene_info = {}
        self._team_n_units = {}
        self._team_usid = {}
        self._team_wsid = {}
        self._team_unit_id2index = {}
        self._team_unit_index2id = {}
        self._team_squad_id2index = {}
        self._team_squad_index2id = {}
        self._team_obs = {}
        self._team_agent_features = {}
        self._team_has_alive_supply_unit = {
            PlayerID.RedTeam: False,
            PlayerID.BlueTeam: False,
        }

        si = self._env.get_player_scene_info(PlayerID.Neutral)
        self._map_width = si.map_width
        self._map_height = si.map_height
        self._map_size = max(si.map_width, si.map_height)
        self._coord_max_v = self._map_size
        self._coord_dim = get_number_feat_dim(self._coord_max_v,
                                              self._n_decimal)

        for player_id in [PlayerID.RedTeam, PlayerID.BlueTeam]:
            # self._team_scene_info : Dict[PlayerID, SceneInfo]
            self._team_scene_info[player_id] = self._env.get_player_scene_info(
                player_id)
            # self._team_n_units: Dict[PlayerID, Num_of_unit]
            self._team_n_units[player_id] = sum(
                self._team_scene_info[player_id].
                team_units_num_info[player_id].values())
            # self._team_usid: Dict[PlayerID, UnitStaticInfo]
            self._team_usid[
                player_id] = self._env.get_player_unit_static_info_dict(
                    player_id)
            # self._team_wsid: Dict[PlayerID, WeaponStaticInfo]
            self._team_wsid[
                player_id] = self._env.get_player_weapon_static_info_dict(
                    player_id)
            # self._team_unit_id2index: Dict[PlayerID, unit_id2_index]
            self._team_unit_id2index[player_id] = dict([
                (v, i) for i, v in enumerate(
                    sorted(self._team_usid[player_id].keys()))
            ])
            # self._team_unit_index2id: Dict[PlayerID, unit_index2id]
            self._team_unit_index2id[player_id] = dict([
                (v, k) for k, v in self._team_unit_id2index[player_id].items()
            ])
            # self._team_squad_id2index: Dict[PlayerID, squad_id2index]
            self._team_squad_id2index[player_id] = dict([
                (v, i) for i, v in enumerate(
                    sorted(self._team_scene_info[player_id].
                           team_squads[player_id].keys()))
            ])
            # self._team_squad_id2index: Dict[PlayerID, squad_index2id]
            self._team_squad_index2id[player_id] = dict([
                (v, k)
                for k, v in self._team_squad_id2index[player_id].items()
            ])
            # self._team_agent_features: Dict[PlayerID, List[None, None, ..., None]] Num_of_unit
            self._team_agent_features[player_id] = [
                None,
            ] * self._team_n_units[player_id]

        # 最大运行时间
        self._max_episode_time = self._team_scene_info[
            PlayerID.RedTeam].max_time
        # 最大运行步数
        self._max_episode_steps = int(
            self._team_scene_info[PlayerID.RedTeam].max_time /
            self._params.delta_time / self._params.step_mul)
        # 创建敌方智能体
        self._enemy_agent = create_agent_from_file(
            'random_move', self._team_scene_info[self._enemy_player_id],
            self._team_usid[self._enemy_player_id],
            self._team_wsid[self._enemy_player_id], self._enemy_player_id)
        self._enemy_agent.reset()

        for player_id in [PlayerID.RedTeam, PlayerID.BlueTeam]:
            # self._team_obs: Dict[PlayerID, player_obs]
            self._team_obs[player_id] = self._env.get_player_obs(player_id)
            # update for player_id
            # 更新每个 unit agent 的观测
            for i in range(self._team_n_units[player_id]):
                self._update_obs_agent(i, player_id)

            # 更新 self._team_has_alive_supply_unit, 如果当前阵营存在活的实体，则设置为 True
            for unit_id, usi in self._team_usid[player_id].items():
                uri = self._team_obs[player_id].unit_runtime_info[unit_id]
                if usi.is_able_to_supply and uri.cur_life > 0:
                    self._team_has_alive_supply_unit[player_id] = True

        global_unit_static_ally = self.encode_team_units_static_feat(
            self._ally_player_id)
        global_unit_static_enemy = self.encode_team_units_static_feat(
            self._enemy_player_id)
        global_unit_runtime_ally = self.encode_team_units_runtime_feat_v1(
            self._ally_player_id)
        global_unit_runtime_enemy = self.encode_team_units_runtime_feat_v1(
            self._enemy_player_id)

        self._global_state = np.array(global_unit_static_ally +
                                      global_unit_runtime_ally +
                                      global_unit_static_enemy +
                                      global_unit_runtime_enemy).astype(
                                          np.float32)

        return self._global_state

    def get_env_info(self) -> Dict:
        return {
            'n_actions': 13 + self._team_n_units[self._enemy_player_id],
            'n_agents': self._team_n_units[self._ally_player_id],
            'state_shape': self._global_state.shape[0],
            'obs_shape': self.get_obs_size(),
            'episode_limit': self._max_episode_steps,
            'episode_time': self._max_episode_time,
        }

    def step(self, actions: List[int], player_id: int = 0):
        """"""
        ally_unit_actions = {}
        assert len(actions) == self._team_n_units[player_id]
        for unit_index, action in enumerate(actions):
            unit_id = self._team_unit_index2id[player_id][unit_index]
            ally_unit_actions[unit_id] = self.decode_action(
                action, self._team_usid[player_id][unit_id],
                self._team_obs[player_id].unit_runtime_info[unit_id],
                player_id)

        enemy_action = self._enemy_agent.step(
            self._team_obs[self._enemy_player_id])

        # player_unit_actions
        # player_squad_actions
        self._env.step(
            {
                self._ally_player_id: ally_unit_actions,
                self._enemy_player_id: enemy_action[0],
            }, {
                self._ally_player_id: {},
                self._enemy_player_id: enemy_action[1],
            })

        for player_id in [PlayerID.RedTeam, PlayerID.BlueTeam]:
            self._team_obs[player_id] = self._env.get_player_obs(player_id)

            for i in range(self._team_n_units[player_id]):
                self._update_obs_agent(i, player_id)

            for unit_id, usi in self._team_usid[player_id].items():
                if usi.is_able_to_supply and \
                        self._team_obs[player_id].unit_runtime_info[unit_id].cur_life > 0:
                    self._team_has_alive_supply_unit[player_id] = True

        global_unit_static_ally = self.encode_team_units_static_feat(
            self._ally_player_id)
        global_unit_static_enemy = self.encode_team_units_static_feat(
            self._enemy_player_id)
        global_unit_runtime_ally = self.encode_team_units_runtime_feat_v1(
            self._ally_player_id)
        global_unit_runtime_enemy = self.encode_team_units_runtime_feat_v1(
            self._enemy_player_id)

        self._global_state = np.array(global_unit_static_ally +
                                      global_unit_runtime_ally +
                                      global_unit_static_enemy +
                                      global_unit_runtime_enemy).astype(
                                          np.float32)

        ally_reward = self._env.get_player_cur_reward(self._ally_player_id)
        enemy_reward = self._env.get_player_cur_reward(self._enemy_player_id)
        done = self._env.is_game_terminated()
        info = {
            'battle_won': done and ally_reward > enemy_reward,
        }
        return ally_reward, done, info

    def get_player_summed_reward(self, player_id: int = 0) -> float:
        return self._env.get_player_accumulated_reward(self._ally_player_id)

    def is_unit_idle(self, usi, uri) -> bool:
        if uri.action_in_processing.code == GameActionCode.NO_OP_ACTION:
            return True
        else:
            if usi.min_speed == 0:
                return False
            if uri.action_in_processing.loop_mode == PathLoopMode.LOOP_FROM_START and \
                    uri.action_in_processing.code in (GameActionCode.MOVE_2D_ACTION, GameActionCode.MOVE_3D_ACTION):
                return True
            return False

    def get_avail_agent_actions(self,
                                agent_id: int,
                                player_id: int = 0) -> List[int]:
        unit_id = self._team_unit_index2id[player_id][agent_id]
        usi = self._team_usid[player_id][unit_id]
        uri = self._team_obs[player_id].unit_runtime_info[unit_id]
        ssr = self._team_obs[player_id].sensor_scan_result
        # if uri.cur_life <= 0 or uri.cur_fuel <= 0 or (not self.is_unit_idle(usi, uri)):
        if uri.cur_life <= 0 or uri.cur_fuel <= 0:
            avail_actions = [0] * (13 +
                                   self._team_n_units[self._enemy_player_id])
            avail_actions[0] = 1
            return avail_actions
        else:
            avail_actions = [1] * (13 +
                                   self._team_n_units[self._enemy_player_id])
            if usi.max_speed == 0:
                # disable move actions
                avail_actions[5] = 0
                avail_actions[6] = 0
                avail_actions[7] = 0
                avail_actions[8] = 0
                avail_actions[9] = 0
                avail_actions[10] = 0
                avail_actions[11] = 0
                avail_actions[12] = 0
            if usi.sensors is None or len(usi.sensors) == 0:
                # disable sensor action
                avail_actions[3] = 0
            if uri.alert_status == AlertStatus.ALERT_STATUS_ALERT:
                avail_actions[2] = 0
            if len(uri.cur_weapons) == 0 or sum(uri.cur_weapons.values()) == 0:
                # disable attack actions
                for i in range(
                        13,
                        13 + self._team_n_units[get_opposite_id(player_id)]):
                    avail_actions[i] = 0

            locked_enemy_unit_ids = set([ssdi.unit_id for ssdi in ssr])
            if K_INVALID_GOBJECT_ID in locked_enemy_unit_ids:
                locked_enemy_unit_ids.remove(K_INVALID_GOBJECT_ID)

            for i in range(self._team_n_units[self._enemy_player_id]):
                enemy_unit_id = self._team_unit_index2id[
                    self._enemy_player_id][i]
                if enemy_unit_id in locked_enemy_unit_ids:
                    avail_actions[i + 13] = 1
                else:
                    avail_actions[i + 13] = 0

            # debug: force to turn on sensors
            if uri.is_sensor_on:
                avail_actions[3] = 0

            if uri.action_in_processing.code == GameActionCode.NO_OP_ACTION:
                avail_actions[1] = 0

            # disable get supply
            avail_actions[4] = 0
            if not usi.is_able_to_supply and self._team_has_alive_supply_unit[
                    self._ally_player_id]:
                if uri.cur_life < usi.total_life * MIN_SUPPLY_RATIO:
                    avail_actions = [0] * (
                        13 + self._team_n_units[self._enemy_player_id])
                    avail_actions[4] = 1
                    return avail_actions
                for k, v in uri.cur_weapons.items():
                    if v < usi.weapons[k] * MIN_SUPPLY_RATIO:
                        avail_actions[4] = 1

            return avail_actions

    def _update_obs_agent(self, unit_index: int, player_id: int) -> None:
        """unit_id+si_feat+team_static_feat+team_runtime_feat+sensor_scaned_fea
        t."""
        unit_id = self._team_unit_index2id[player_id][unit_index]
        if self._team_obs[player_id].unit_runtime_info[unit_id].cur_life <= 0:
            self._team_agent_features[player_id][unit_index][:] = 0
            return
        if self._team_agent_features[player_id][unit_index] is None:
            feat_list = self.encode_unit_id(unit_id, self._team_unit_id2index[player_id]) + \
                self.encode_unit_runtime_info(
                    self._team_obs[player_id].unit_runtime_info[unit_id], player_id) + \
                self.encode_team_destroyed_feat(player_id)
            self._team_agent_features[player_id][unit_index] = np.array(
                feat_list).astype(np.float32)
        else:
            # 复用之前的静态特征，只更新unit runtime特征和探测器扫描结果
            # self.encode_team_units_runtime_feat(player_id) + \
            rt_feat = self.encode_unit_runtime_info(
                self._team_obs[player_id].unit_runtime_info[unit_id], player_id) + \
                self.encode_team_destroyed_feat(player_id)
            self._team_agent_features[player_id][unit_index][-len(rt_feat
                                                                  ):] = rt_feat

    def encode_team_units_static_feat(self, player_id: int) -> List[float]:
        """Encoding the team units static feature map.

        Args:
            player_id (int): player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        feat_list = []
        for unit_index in range(self._team_n_units[player_id]):
            unit_id = self._team_unit_index2id[player_id][unit_index]
            feat_list += self.encode_unit_static_info(
                self._team_usid[player_id][unit_id], player_id)
        return feat_list

    def encode_team_units_runtime_feat_v1(self, player_id: int) -> List[float]:
        """Encoding the team units runtime feature map.

        Args:
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        feat_list = []
        for unit_index in range(self._team_n_units[player_id]):
            unit_id = self._team_unit_index2id[player_id][unit_index]
            feat_list += self.encode_unit_runtime_info(
                self._team_obs[player_id].unit_runtime_info[unit_id],
                player_id)
        return feat_list

    def encode_team_units_runtime_feat_v2(self, player_id: int) -> List[float]:
        """Encoding the team units runtime feature map version 2, add sensor
        scan result and destroyed enemy units.

        Args:
            player_id (int): player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        feat_list = []
        for unit_index in range(self._team_n_units[player_id]):
            unit_id = self._team_unit_index2id[player_id][unit_index]
            feat_list += self.encode_unit_runtime_info(
                self._team_obs[player_id].unit_runtime_info[unit_id],
                player_id)
            feat_list += self.encode_team_sensor_scan_feat_per_unit(
                unit_id, player_id)
            feat_list += self.encode_ally_info(unit_id, player_id)
        return feat_list

    def encode_team_sensor_scan_feat_per_unit(self, unit_id: int,
                                              player_id: int) -> List[float]:
        """Encoding the team sensor scan feature map per unit.

        Args:
            unit_id (int): unit id
            player_id (int): player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        uri = self._team_obs[player_id].unit_runtime_info[unit_id]
        n_enemy_units = self._team_n_units[get_opposite_id(player_id)]
        n_max = n_enemy_units
        n_feat_ssdi = 2 + 2 * self._coord_dim

        feat = [0.0] * n_feat_ssdi * n_max
        for ssdi in self._team_obs[player_id].sensor_scan_result:
            # 目前只对精确锁定的单位进行编码
            if ssdi.unit_id == K_INVALID_GOBJECT_ID:
                continue

            if ssdi.unit_id not in self._team_unit_id2index[get_opposite_id(
                    player_id)]:
                # 如果探测到的是导弹，直接忽略
                continue
            i = self._team_unit_id2index[get_opposite_id(player_id)][
                ssdi.unit_id]

            feat[i * n_feat_ssdi] = ssdi.confidence
            is_critical_v = -1.0
            if ssdi.is_critical:
                is_critical_v = float(ssdi.is_critical)
            feat[i * n_feat_ssdi + 1] = is_critical_v

            delta_x = ssdi.center.x - uri.cur_pos.x
            delta_y = ssdi.center.y - uri.cur_pos.y

            k = i * n_feat_ssdi + 2

            feat[k:k + self._coord_dim] = encode_large_float(
                delta_x, self._coord_max_v)
            k += self._coord_dim

            feat[k:k + self._coord_dim] = encode_large_float(
                delta_y, self._coord_max_v)
            k += self._coord_dim

        return feat

    def encode_team_sensor_scan_feat(self, player_id: int) -> List[float]:
        """Encoding the team sensor scan feature map.

        Args:
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        n_enemy_units = self._team_n_units[get_opposite_id(player_id)]
        n_max = n_enemy_units
        n_feat_ssdi = 2 + n_enemy_units + 2 * self._coord_dim

        feat = [0.0] * n_feat_ssdi * n_max
        for index, ssdi in enumerate(
                self._team_obs[player_id].sensor_scan_result):
            # 目前只对精确锁定的单位进行编码
            if ssdi.unit_id == K_INVALID_GOBJECT_ID:
                continue

            if ssdi.unit_id not in self._team_unit_id2index[get_opposite_id(
                    player_id)]:
                # 如果探测到的是导弹，直接忽略
                continue
            i = self._team_unit_id2index[get_opposite_id(player_id)][
                ssdi.unit_id]

            feat[i * n_feat_ssdi] = ssdi.confidence
            is_critical_v = -1.0
            if ssdi.is_critical:
                is_critical_v = float(ssdi.is_critical)
            feat[i * n_feat_ssdi + 1] = is_critical_v
            feat[i * n_feat_ssdi + 2:i * n_feat_ssdi + 2 +
                 n_enemy_units] = self.encode_unit_id(
                     ssdi.unit_id,
                     self._team_unit_id2index[get_opposite_id(player_id)])

            k = i * n_feat_ssdi + 2 + n_enemy_units

            feat[k:k + self._coord_dim] = encode_large_float(
                ssdi.center.x, self._coord_max_v)
            k += self._coord_dim

            feat[k:k + self._coord_dim] = encode_large_float(
                ssdi.center.y, self._coord_max_v)
            k += self._coord_dim

        return feat

    def encode_team_destroyed_feat(self, player_id: int) -> List[float]:
        """Encode destroyed enemy units, info from `runtime observation`
        support 2 types:

            - 0 for not destroyed
            - 1 for destroyed

        Args:
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feat list
        """
        feat = [
            0,
        ] * self._team_n_units[get_opposite_id(player_id)]
        for unit_id in self._team_obs[player_id].enemy_units_destroyed:
            feat[self._team_unit_id2index[get_opposite_id(player_id)]
                 [unit_id]] = 1.0
        return feat

    def encode_game_mode(self,
                         game_mode: str,
                         mode_size: int = 3) -> List[float]:
        """Currently support 3 game modes:

            - destroy_all
            - destroy_critical_target
            - search_critical_target

        Args:
            game_mode (str): game mode
            size (int, optional): Size of the game_mode, Currently set it to 3.

        Returns:
            List[float]: _description_
        """
        if game_mode == 'destroy_all':
            return _simple_one_hot(index=0, size=mode_size)
        elif game_mode == 'destroy_critical_target':
            return _simple_one_hot(index=1, size=mode_size)
        elif game_mode == 'search_critical_target':
            return _simple_one_hot(index=2, size=mode_size)
        return _simple_one_hot(-1, 3)

    def encode_unit_id(self, unit_id: int, unit_id2index: Dict) -> List[float]:
        """Encode the unit id,   unit_id from the `UnitStaticInfo`

        Args:
            unit_id (int): _description_
            unit_id2index (Dict): _description_

        Returns:
            List[float]: _description_
        """
        if unit_id == K_INVALID_GOBJECT_ID:
            index = -1
        else:
            if unit_id not in unit_id2index:
                raise RuntimeError(
                    f'unit_id {unit_id} not in record: {unit_id2index}')
            index = unit_id2index[unit_id]
        return _simple_one_hot(index, size=len(unit_id2index))

    def encode_squad_id(self, squad_id: int,
                        squad_id2index: Dict) -> List[float]:
        """Encode Squad id, info from `UnitStateInfo`

        Args:
            squad_id (int): _description_
            squad_id2index (Dict): _description_

        Returns:
            List[float]: _description_
        """
        if squad_id not in squad_id2index:
            raise RuntimeError(
                f'squad_id {squad_id} not in record: {squad_id2index}')
        return _simple_one_hot(
            index=squad_id2index[squad_id], size=len(squad_id2index))

    def encode_object_type(
            self, object_type: Union[ObjectType,
                                     List[ObjectType]]) -> List[float]:
        """Encode object type, support multiple object types, Currently support
        6 types:

            - OBJECT_GROUND
            - OBJECT_AIR
            - OBJECT_SHIP
            - OBJECT_MISSILE
            - OBJECT_SUBMARINE
            - OBJECT_OTHER_WEAPON

        Args:
            object_type (Union[ObjectType, List[ObjectType]]): _description_

        Returns:
            List[float]: _description_
        """
        type_list = object_type
        if not isinstance(object_type, Iterable):
            type_list = [object_type]
        type_list = list(type_list)

        for i, object_type in enumerate(type_list):
            if object_type == ObjectType.OBJECT_GROUND:
                type_list[i] = 0
            elif object_type == ObjectType.OBJECT_AIR:
                type_list[i] = 1
            elif object_type == ObjectType.OBJECT_SHIP:
                type_list[i] = 2
            elif object_type == ObjectType.OBJECT_MISSILE:
                type_list[i] = 3
            elif object_type == ObjectType.OBJECT_SUBMARINE:
                type_list[i] = 4
            elif object_type == ObjectType.OBJECT_OTHER_WEAPON:
                type_list[i] = 5

        return _simple_one_hot(type_list, 6)

    def encode_alert_status(self, alert_status: AlertStatus) -> List[float]:
        """Encode alert status, support 2 types:

           - 0 for silent
           - 1 for alert

        Args:
            alert_status (AlertStatus): _description_

        Returns:
            List[float]: _description_
        """
        if alert_status == AlertStatus.ALERT_STATUS_SILENT:
            return _simple_one_hot(index=0, size=2)
        else:
            return _simple_one_hot(index=1, size=2)

    def encode_supply_status(self, status: SupplyStatus) -> List[float]:
        """Encode supply status, support 4 types:

        Args:
            status (SupplyStatus): _description_

        Returns:
            List[float]: _description_
        """
        if status == SupplyStatus.UNIT_LEFT_SUPPLY:
            return _simple_one_hot(index=0, size=4)
        elif status == SupplyStatus.UNIT_MOVING_TO_SUPPLY:
            return _simple_one_hot(index=1, size=4)
        elif status == SupplyStatus.UNIT_IN_SUPPLY:
            return _simple_one_hot(index=2, size=4)
        elif status == SupplyStatus.UNIT_SUPPLY_FINISHED:
            return _simple_one_hot(index=3, size=4)
        return _simple_one_hot(-1, 4)

    def encode_scene_info(self, player_id: int) -> List[float]:
        """Encode scene info for player_id.

        Args:
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        si: SceneInfo = self._team_scene_info[player_id]
        si_feat = []
        si_feat += self.encode_game_mode(si.game_mode)
        si_feat += [
            # si.max_time,
            # float(si.map_width) / DISTANCE_SCALE_FACTOR,
            # float(si.map_height) / DISTANCE_SCALE_FACTOR,
            float(si.is_able_to_know_enemy_info),
            si.default_detection_reward,
            si.default_destruction_reward,
            si.final_reward,
            # self._team_n_units[player_id],
            # self._team_n_units[get_opposite_id(player_id)],
        ]
        return si_feat

    def encode_unit_static_info(self, usi: UnitStaticInfo,
                                player_id: int) -> List[float]:
        """Encode unit static info for player_id.

        Args:
            usi (UnitStaticInfo): unit static info
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        usi_feat = [
            usi.total_life / LIFE_SCALE_FACTOR,
            float(usi.is_critical),
            usi.detection_reward,
            usi.destruction_reward,
            *encode_large_float(usi.birth_pos.x, self._coord_max_v),
            *encode_large_float(usi.birth_pos.y, self._coord_max_v),
            *encode_large_float(usi.birth_pos.z, self._height_max_v),
            *encode_large_float(usi.min_speed, self._speed_max_v),
            *encode_large_float(usi.default_speed, self._speed_max_v),
            *encode_large_float(usi.max_speed, self._speed_max_v),
            *encode_large_float(usi.min_height, self._height_max_v),
            *encode_large_float(usi.max_height, self._height_max_v),
            # usi.max_range_min_speed / DISTANCE_SCALE_FACTOR,
            # usi.max_range_max_speed / DISTANCE_SCALE_FACTOR,
            *encode_large_float(usi.alert_range, self._coord_max_v),
            usi.dodge_rate,
            usi.sensor_scan_stealth,
            float(usi.is_able_to_supply),
            usi.supply_time / 3600.0,
        ]
        usi_feat += self.encode_unit_id(usi.unit_id,
                                        self._team_unit_id2index[player_id])
        usi_feat += self.encode_object_type(usi.object_type)
        usi_feat += self.encode_squad_id(usi.squad_id,
                                         self._team_squad_id2index[player_id])
        usi_feat += self.encode_object_type(usi.supply_object_types)

        # TODO: unit_type/model/weapons/sensors/supply_object_types

        return usi_feat

    def encode_ally_info(self, unit_id: int, player_id: int) -> List[float]:
        """Encode 装备（实体）及编组信息， infor form the `unit_id` and `player_id`

        Args:
            unit_id (int):  编队 id
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: 返回编组和装备信息
        """
        n_ally_units = self._team_n_units[player_id]
        uri = self._team_obs[player_id].unit_runtime_info[unit_id]
        n_ally_info_dim = n_ally_units + 1 + self._coord_dim * 2 + self._height_dim
        feat_list = [0] * (n_ally_info_dim * (n_ally_units - 1))

        i = -1
        for ally_unit_id, ally_uri in self._team_obs[
                player_id].unit_runtime_info.items():
            if ally_unit_id == unit_id:
                continue

            i += 1

            k = i * n_ally_info_dim
            feat_list[k:k + n_ally_units] = self.encode_unit_id(
                ally_unit_id, self._team_unit_id2index[player_id])
            k += n_ally_units

            if ally_uri.cur_life <= 0:
                continue

            feat_list[k] = max(0.0, ally_uri.cur_life / LIFE_SCALE_FACTOR)
            k += 1

            delta_x = ally_uri.cur_pos.x - uri.cur_pos.x
            delta_y = ally_uri.cur_pos.y - uri.cur_pos.y
            delta_z = ally_uri.cur_pos.z - uri.cur_pos.z

            feat_list[k:k + self._coord_dim] = encode_large_float(
                delta_x, self._coord_max_v)
            k += self._coord_dim
            feat_list[k:k + self._coord_dim] = encode_large_float(
                delta_y, self._coord_max_v)
            k += self._coord_dim
            feat_list[k:k + self._height_dim] = encode_large_float(
                delta_z, self._height_max_v)

        return feat_list

    def encode_unit_runtime_info(self, uri: UnitRuntimeInfo,
                                 player_id: int) -> List[float]:
        """Encode unit runtime info for player_id.

        Args:
            uri (UnitRuntimeInfo): unit runtime info
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            List[float]: feature list
        """
        uri_feat = [0] * (1 + self._coord_dim * 2 + self._height_dim)
        uri_feat += [
            *encode_large_float(uri.desired_speed, self._speed_max_v),
            float(uri.is_sensor_on),
            uri.cur_life / LIFE_SCALE_FACTOR,
            *encode_large_float(uri.cur_pos.x, self._coord_max_v),
            *encode_large_float(uri.cur_pos.y, self._coord_max_v),
            *encode_large_float(uri.cur_pos.z, self._height_max_v),
            *encode_large_float(uri.cur_speed, self._speed_max_v),
            uri.cur_direction.x,
            uri.cur_direction.y,
            uri.cur_direction.z,
            float(uri.is_sensor_scanned_high_conf),
            uri.cur_fuel,
        ]
        if uri.cur_move_point_index >= 0 and len(uri.cur_move_path):
            p = uri.cur_move_path[uri.cur_move_point_index]
            uri_feat[0] = 1
            k = 1
            uri_feat[k:k + self._coord_dim] = encode_large_float(
                p.x, self._coord_max_v)
            k += self._coord_dim
            uri_feat[k:k + self._coord_dim] = encode_large_float(
                p.y, self._coord_max_v)
            k += self._coord_dim
            uri_feat[k:k + self._height_dim] = encode_large_float(
                p.z, self._height_max_v)

        uri_feat += self.encode_alert_status(uri.alert_status)
        uri_feat += self.encode_supply_status(uri.supply_status)

        # TODO: cur_weapons

        if uri.cur_life <= 0:
            return [0.0 for _ in range(len(uri_feat))]
        return uri_feat

    def get_obs_size(self):
        """Get observation size."""
        if self._team_agent_features[self._ally_player_id] is None:
            raise RuntimeError('Can not get obs feature size before reset.')
        return self._team_agent_features[self._ally_player_id][0].shape[0]

    def decode_action(self, action: int, usi: UnitStaticInfo,
                      uri: UnitRuntimeInfo, player_id: int) -> GameAction:
        """Decode action from action index.

        Args:
            action (int): action index
            usi (UnitStaticInfo): unit static info
            uri (UnitRuntimeInfo): unit runtime info
            player_id (int):  player id, 0 for ally, 1 for enemy

        Returns:
            GameAction: game action, see `GameAction` for details
        """
        raw_act = GameAction()

        # 若当前单位已经死亡, 则不执行任何动作
        if uri.cur_life <= 0:
            raw_act.code = GameActionCode.NO_OP_ACTION
            return raw_act

        # 若当前单位动作 ID 为 0, 则不执行任何动作
        if action == 0:
            raw_act.code = GameActionCode.NO_OP_ACTION
        # 若当前单位动作 ID 为 1, 则停止当前动作
        elif action == 1:
            # stop action
            raw_act.code = GameActionCode.STOP_ACTION
        # 若当前单位动作 ID 为 2, 则切换警戒状态
        elif action == 2:
            # alert status
            raw_act.code = GameActionCode.SET_ALERT_STATUS_ACTION
            if uri.alert_status == AlertStatus.ALERT_STATUS_ALERT:
                raw_act.alert_status = AlertStatus.ALERT_STATUS_SILENT
            else:
                raw_act.alert_status = AlertStatus.ALERT_STATUS_ALERT
        # 若当前单位动作 ID 为 3, 则打开探测器
        elif action == 3:
            raw_act.code = GameActionCode.SENSOR_SCAN_ACTION
        # 若当前单位动作 ID 为 4, 则补给
        elif action == 4:
            raw_act.code = GameActionCode.GET_SUPPLY_ACTION
            raw_act.desired_speed_ratio = 1.0
            raw_act.target_unit_id = K_INVALID_GOBJECT_ID
        # 若当前单位动作介于 5~12 之间, 则执行移动动作
        elif action >= 5 and action < 13:
            # move action
            cur_pos = uri.cur_pos
            raw_act.code = GameActionCode.MOVE_2D_ACTION
            raw_act.desired_speed_ratio = 1.0
            delta_d = usi.max_speed * 3
            # 当前动作为 5 时, 向北移动
            if action == 5:
                # north
                raw_act.move_path = [
                    Vector3(cur_pos.x, cur_pos.y + delta_d),
                ]
            # 当前动作为 6 时, 向南移动
            elif action == 6:
                # south
                raw_act.move_path = [
                    Vector3(cur_pos.x, cur_pos.y - delta_d),
                ]
            # 当前动作为 7 时, 向西移动
            elif action == 7:
                # west
                raw_act.move_path = [
                    Vector3(cur_pos.x - delta_d, cur_pos.y),
                ]
            # 当前动作为 8 时, 向东移动
            elif action == 8:
                # east
                raw_act.move_path = [
                    Vector3(cur_pos.x + delta_d, cur_pos.y),
                ]
            # 当前动作为 9 时, 向东北移动
            elif action == 9:
                # north-east
                raw_act.move_path = [
                    Vector3(cur_pos.x + delta_d * 0.707,
                            cur_pos.y + delta_d * 0.707),
                ]
            # 当前动作为 10 时, 向西北移动
            elif action == 10:
                # north-west
                raw_act.move_path = [
                    Vector3(cur_pos.x - delta_d * 0.707,
                            cur_pos.y + delta_d * 0.707),
                ]
            # 当前动作为 11 时, 向西南移动
            elif action == 11:
                # south-west
                raw_act.move_path = [
                    Vector3(cur_pos.x - delta_d * 0.707,
                            cur_pos.y - delta_d * 0.707),
                ]
            # 当前动作为 12 时, 向东南移动
            elif action == 12:
                # south-east
                raw_act.move_path = [
                    Vector3(cur_pos.x + delta_d * 0.707,
                            cur_pos.y - delta_d * 0.707),
                ]
        # 若当前动作大于 13
        elif action >= 13 and action < 13 + self._team_n_units[get_opposite_id(
                player_id)]:
            # 执行智能攻击指令， 自动跟踪和射击对方
            raw_act.code = GameActionCode.ATTACK_AND_FOLLOW_ACTION
            raw_act.target_unit_id = self._team_unit_index2id[get_opposite_id(
                player_id)][action - 13]
            raw_act.desired_ammo_num = -1
        else:
            raise RuntimeError(
                f'Invalid action {action}, max action value '
                f'is {12 + self._team_n_units[get_opposite_id(player_id)]}')

        return raw_act


if __name__ == '__main__':
    params = GameParams()
    params.scene = 'destroy_all0'
    params.enable_rendering = False
    env = WarGameSAWrapper(params)

    env_info = env.get_env_info()
    print(f'env_info={env_info}')
    n_actions = env_info['n_actions']
    n_agents = env_info['n_agents']

    agent_obs_list = env.get_obs(0)

    env.step([np.random.randint(0, n_actions) for _ in range(n_agents)])

    env.close()
