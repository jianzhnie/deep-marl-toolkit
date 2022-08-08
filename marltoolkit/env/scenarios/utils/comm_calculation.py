import math

import matplotlib.path as plt_path
import numpy as np
#import matplotlib.pyplot as plt
from env.scenarios.utils import coord_convert


def calculateGaussDistance(p1Lng, p1Lat, p1Alt, p2Lng, p2Lat, p2Alt):
    """计算两点间的高斯距离.

    :param p1Lon: 点1的经度
    :param p1Lat: 点1的纬度
    :param p1Alt: 点1的高度
    :param p2Lon: 点2的经度
    :param p2Lat: 点2的纬度
    :param p2Alt: 点2的高度
    :return: 高斯距离
    """
    x, y, z = coord_convert.World2Local(p1Lng, p1Lat, p1Alt, p2Lng, p2Lat,
                                        p2Alt)
    return math.sqrt(x**2 + y**2 + z**2)


def pointInPolygon(x, y, polygon=()):
    """判断单个点是否在多边形内(高斯坐标)

    :param x: 点横坐标
    :param y: 点纵坐标
    :param polygon: 多边形，例：polygon=([-1,-1], [2,-1], [2,1], [-2,2])
    :return: True or False
    """
    path = plt_path.Path(polygon)
    inside = path.contains_point([x, y])
    return inside


def pointsInPolygon(points=(), polygon=()):
    """判断点集是否在多边形内(高斯坐标)

    :param points: 点集，例：points=([0,0], [1,0], [1,1])
    :param polygon: 多边形，例：polygon=([-1,-1], [2,-1], [2,1], [-2,2])
    :return: bool值列表，例：[False, True, True]
    """
    path = plt_path.Path(polygon)
    inside = path.contains_points(points)
    return list(inside)


def getPolygonCentre(latLngPolygon=()):
    """计算多边形的重心（经纬度坐标）

    :param latLngPolygon: 多边形（经纬度坐标），例：latLongPolygon=([114, 10], [114, 8], [111, 8], [112, 10])
    :return: 重心（经纬度坐标）
    """
    area = 0
    x, y = 0, 0
    for i in range(len(latLngPolygon)):
        lat = latLngPolygon[i][0]
        lng = latLngPolygon[i][1]
        if i == 0:
            lat1 = latLngPolygon[-1][0]
            lng1 = latLngPolygon[-1][1]
        else:
            lat1 = latLngPolygon[i - 1][0]
            lng1 = latLngPolygon[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    x = x / area
    y = y / area
    return x, y


def getIntersection(line0P0, line0P1, line1P0, line1P1):
    """计算两条直线的交点.

    :param p1Lng:
    :param p1Lat:
    :param p2Lng:
    :param p2Lat:
    :param p3Lng:
    :param p3Lat:
    :param p4Lng:
    :param p4Lat:
    :return:
    """
    a1 = line0P0[1] - line0P1[1]
    b1 = line0P1[0] - line0P0[0]
    c1 = line0P0[0] * line0P1[1] - line0P1[0] * line0P0[1]
    a2 = line1P0[1] - line1P1[1]
    b2 = line1P1[0] - line1P0[0]
    c2 = line1P0[0] * line1P1[1] - line1P1[0] * line1P0[1]
    d = a1 * b2 - a2 * b1
    if d == 0:
        return None
    x = round((b1 * c2 - b2 * c1) * 1.0 / d, 2)
    y = round((c1 * a2 - c2 * a1) * 1.0 / d, 2)
    # if inSegment((x, y), line0P0, line0P1, line1P0, line1P1):
    #     return x, y
    # else:
    #     return None
    return x, y


def inSegment(p, line0P0, line0P1, line1P0, line1P1):
    """判断点是否在线段上（暂时没用）

    :param p:
    :param line0P0:
    :param line0P1:
    :param line1P0:
    :param line1P1:
    :return:
    """
    if line0P0[0] == line0P1[0]:
        if p[1] > min(line0P0[1], line0P1[1]) and p[1] < max(
                line0P0[1], line0P1[1]):
            if p[0] >= min(line1P0[0], line1P1[0]) and p[0] <= max(
                    line1P0[0], line1P1[0]):
                return True
    elif line0P0[1] == line0P1[1]:
        if p[0] > min(line0P0[0], line0P1[0]) and p[0] < max(
                line0P0[0], line0P1[0]):
            if p[1] >= min(line1P0[1], line1P1[1]) and p[1] <= max(
                    line1P0[1], line1P1[1]):
                return True
    else:
        if p[0] > min(line0P0[0], line0P1[0]) and p[0] < max(
                line0P0[0], line0P1[0]):
            if p[1] >= min(line1P0[1], line1P1[1]) and p[1] <= max(line1P0[1], line1P1[1]) \
                and p[0] >= min(line1P0[0], line1P1[0]) and p[0] <= max(line1P0[0], line1P1[0]):
                return True
    return False


def isNearTarget(entity_longitude: float, entity_latitude: float,
                 entity_altitude, target_longitude: float,
                 target_latitude: float, target_altitude, threshold: int):
    """判断敌方单位是否接近我方单位.

    :param entity_longitude:
    :param entity_latitude:
    :param entity_altitude:
    :param target_longitude:
    :param target_latitude:
    :param target_altitude:
    :param threshold:
    :return:
    """
    distance = calculateGaussDistance(entity_longitude, entity_latitude,
                                      entity_altitude, target_longitude,
                                      target_latitude, target_altitude)
    if distance < threshold:
        return True
    else:
        return False


def point_in_area(longitude, latitude, poly):
    """判断单个点是否在多边形内(经纬度坐标)

    :param x: 点横坐标
    :param y: 点纵坐标
    :param polygon: 多边形
    :return: True or False
    """
    in_flag = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l - 1:
        i += 1
        if ((poly[i]['longitude'] <= longitude
             and longitude < poly[j]['longitude'])
                or (poly[j]['longitude'] <= longitude
                    and longitude < poly[i]['longitude'])):
            if (latitude < (poly[j]['latitude'] - poly[i]['latitude']) *
                (longitude - poly[i]['longitude']) /
                (poly[j]['longitude'] - poly[i]['longitude']) +
                    poly[i]['latitude']):
                in_flag = not in_flag
        j = i
    return in_flag


def point_in_rectangle(longitude, latitude, which_area):
    """判断单个点（矩形）是否在矩形内.

    :param longitude: 经度
    :param latitude:  纬度
    :param which_area: 矩形
    :return:
    """
    in_flag = False
    max_latitude = which_area[2]['latitude']
    min_latitude = which_area[0]['latitude']
    max_longitude = which_area[2]['longitude']
    min_longitude = which_area[0]['longitude']

    if min_longitude <= longitude <= max_longitude and min_latitude <= latitude <= max_latitude:
        in_flag = True

    return in_flag


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

    def inverse_trans(item):
        d = int(item)
        f = int((item - d) * 60)
        s = ((item - d) * 60 - f) * 60
        return d, f, s

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


# --------------------------------------------------------------------------------------------------
# ----------------------------------          测试函数      ------------------------------------------
# --------------------------------------------------------------------------------------------------


def test_inPolygon():
    """
    测试函数：点是否在多边形内
    :return:
    """
    lenPoly = 100  # 多边形
    polygon = [[np.sin(x) + 1, np.cos(x) + 1]
               for x in np.linspace(0, 2 * np.pi, lenPoly)[:-1]]
    polygonNP = np.array(polygon)
    print('polygonNP:', polygonNP)

    n = 1000  # 点
    points = list(zip(np.random.random(n), np.random.random(n)))
    pointsNP = np.array(points)
    print('points NP:', pointsNP)

    # 调用函数判断
    inside = pointsInPolygon(points=points, polygon=polygon)
    print('inside:', inside)

    # 画图
    #plt.figure()
    #plt.plot(polygonNP[:,0], polygonNP[:,1])
    #for i in range(len(pointsNP)):
    #if inside[i]:
    #plt.scatter(pointsNP[i][0], pointsNP[i][1], c='r')
    #else:
    #plt.scatter(pointsNP[i][0], pointsNP[i][1], c='b')
    #plt.show()


def test_getPolygonCentre():
    """
    测试：计算多边形重心
    :return:
    """
    polygon = ([114, 10], [114, 8], [111, 8], [112, 10])
    print(getPolygonCentre(polygon))


def test_calculateGaussDistance():
    """
    测试：计算两点间高斯距离
    :return:
    """
    a = (101, 10, 1000)
    b = (100, 10, 1000)
    print(calculateGaussDistance(a[0], a[1], a[2], b[0], b[1], b[2]))


def test_getIntersection():
    x, y = getIntersection((0, 5), (4, 1), (4, 4), (5, -2))
    print(x, y)


if __name__ == '__main__':
    # test_inPolygon()
    # test_getPolygonCentre()
    # test_calculateGaussDistance()
    test_getIntersection()
