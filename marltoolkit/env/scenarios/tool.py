import numpy as np


class Coordinate():
    def __init__(self):
        self.d = 0
        self.m = 0
        self.s = 0


def set_area(area_points):
    '''
    conver points to a retangle
    Return:
        - rectangle：retangle
    '''
    points = area_points
    rectangle = [Coordinate() for i in range(4)]
    rectangle[0].d = points[0]['d']
    rectangle[0].m = points[0]['m']
    rectangle[0].s = points[0]['s']

    rectangle[1].d = points[1]['d']
    rectangle[1].m = points[1]['m']
    rectangle[1].s = points[1]['s']

    rectangle[2].d = points[2]['d']
    rectangle[2].m = points[2]['m']
    rectangle[2].s = points[2]['s']

    rectangle[3].d = points[3]['d']
    rectangle[3].m = points[3]['m']
    rectangle[3].s = points[3]['s']

    return rectangle


def fighter_patrol_area(area_points, xn, yn):
    rectangle = set_area(area_points)

    def trans(coordinate):
        return coordinate.d + (coordinate.m + coordinate.s / 60) / 60

    x = np.linspace(trans(rectangle[0]), trans(rectangle[1]), xn)
    y = np.linspace(trans(rectangle[2]), trans(rectangle[3]), yn)

    area = []
    for i in range(xn - 1):
        for j in range(yn - 1):
            area_sub = {}
            area_sub['points'] = [
                {
                    'longitude': x[i],
                    'latitude': y[j],
                    'altitude': 6000,
                    'velocity': 500
                },
                {
                    'longitude': x[i + 1],
                    'latitude': y[j],
                    'altitude': 6000,
                    'velocity': 500
                },
                {
                    'longitude': x[i + 1],
                    'latitude': y[j + 1],
                    'altitude': 6000,
                    'velocity': 500
                },
                {
                    'longitude': x[i],
                    'latitude': y[j + 1],
                    'altitude': 6000,
                    'velocity': 500
                },
            ]
            area_sub['center'] = {
                'longitude': 0.5 * (x[i] + x[i + 1]),
                'latitude': 0.5 * (y[j] + y[j + 1])
            }
            area.append(area_sub)

    return area


def inverse_trans(item):
    d = int(item)
    f = int((item - d) * 60)
    s = ((item - d) * 60 - f) * 60
    return d, f, s


def groups_action_verrify(force_side, entities, groups):

    curr_groups = groups
    actions_mask = [0] * len(curr_groups)
    for group_inx, g in enumerate(curr_groups):
        for g_e in g.group_entity:
            for e in entities:
                if e.forceSide == force_side and e.identity_id == g_e.identity_id:
                    if e.alive and e.AirStatus in [
                            0, 1, 2, 4, 8, 9, 12, 18, 20, 21
                    ]:
                        actions_mask[group_inx] += 1
                    break
    return actions_mask
