import copy
from math import *

import numpy as np

PI = 3.1415926
_earth_r_a = 6378137.000
_earth_r_b = 6356752.3141
_earth_e = 1 - pow(_earth_r_b, 2) / pow(_earth_r_a, 2)


def LonLat2XYZ(dLon, dLat, dAlt):
    x = 0
    y = 0
    z = 0
    # if (dLon > 180 or dLon < -180 or dLat > 90 or dLat < -90 or dAlt > 180 or dAlt < -180):
    # return x,y,z

    L = copy.deepcopy(dLon)
    B = copy.deepcopy(dLat)
    L = L * PI / 180.0
    B = B * PI / 180.0
    N = _earth_r_a / sqrt(1 - _earth_e * sin(B) * sin(B))

    x = (N + dAlt) * cos(B) * cos(L)
    y = (N + dAlt) * cos(B) * sin(L)
    z = ((1 - _earth_e) * N + dAlt) * sin(B)
    return x, y, z


def XYZ2LonLat(x, y, z):
    nSpace = 0
    if (y >= 0):
        if (x >= 0):
            nSpace = 1
        else:
            nSpace = 2
    else:
        if (x >= 0):
            nSpace = 4
        else:
            nSpace = 3

    L = atan(abs(y / x)) * 180.0 / PI

    if nSpace == 2:
        L = 180.0 - L
    elif nSpace == 3:
        L = L - 180.0
    elif nSpace == 4:
        L = -L

    t0 = z / sqrt(x * x + y * y)
    t1 = t0
    K = 1 + _earth_e
    P = _earth_e * _earth_r_a * _earth_r_a / (_earth_r_b * sqrt(x * x + y * y))
    t2 = t0 + P * t1 / sqrt(K + t1 * t1)

    while (abs(t2 - t1) > pow(10.0, -6)):
        t1 = t2
        t2 = t0 + P * t1 / sqrt(K + t1 * t1)

    B = atan(t1)
    if (z >= 0):
        N = _earth_r_a / sqrt(1 - _earth_e * sin(B) * sin(B))
        H = sqrt(x * x + y * y) / cos(B) - N
    else:
        N = _earth_r_a / sqrt(1 - _earth_e * sin(-B) * sin(-B))
        H = sqrt(x * x + y * y) / cos(-B) - N

    dLon = L
    dLat = B * 180.0 / PI
    dAlt = H

    return dLon, dLat, dAlt


def CreateTransferMatrix(dLon, dLat):
    B = dLat
    if dLon >= 0:
        L = dLon
    else:
        L = 360.0 + dLon

    B = B * PI / 180.0
    L = L * PI / 180.0

    matrix = [[0 for i in range(3)] for j in range(3)]
    matrix[0][0] = -sin(B) * cos(L)
    matrix[0][1] = -sin(B) * sin(L)
    matrix[0][2] = cos(B)

    matrix[1][0] = -sin(L)
    matrix[1][1] = cos(L)
    matrix[1][2] = 0

    matrix[2][0] = cos(B) * cos(L)
    matrix[2][1] = cos(B) * sin(L)
    matrix[2][2] = sin(B)

    return matrix


def World2Local(dLon, dLat, dAlt, dLon1, dLat1, dAlt1):

    # x_North, y_East, z_Sky = 0,0,0
    m = CreateTransferMatrix(dLon, dLat)

    xo, yo, zo = LonLat2XYZ(dLon, dLat, dAlt)
    xp, yp, zp = LonLat2XYZ(dLon1, dLat1, dAlt1)
    x_ = xp - xo
    y_ = yp - yo
    z_ = zp - zo

    x_North = m[0][0] * x_ + m[0][1] * y_ + m[0][2] * z_
    y_East = m[1][0] * x_ + m[1][1] * y_ + m[1][2] * z_
    z_Sky = m[2][0] * x_ + m[2][1] * y_ + m[2][2] * z_

    return x_North, y_East, z_Sky


def CreateReverseMatrix(dLon, dLat):
    B = dLat
    if dLon >= 0:
        L = dLon
    else:
        L = 360.0 + dLon

    B = B * PI / 180.0
    L = L * PI / 180.0

    matrix = [[0 for i in range(3)] for j in range(3)]
    matrix[0][0] = -sin(B) * cos(L)
    matrix[1][0] = -sin(B) * sin(L)
    matrix[2][0] = cos(B)

    matrix[0][1] = -sin(L)
    matrix[1][1] = cos(L)
    matrix[2][1] = 0

    matrix[0][2] = cos(B) * cos(L)
    matrix[1][2] = cos(B) * sin(L)
    matrix[2][2] = sin(B)

    return matrix


def Local2World(dLon, dLat, dAlt, x, y, z):
    dLon1, dLat1, dAlt1 = 0, 0, 0
    m = CreateReverseMatrix(dLon, dLat)

    xo, yo, zo = LonLat2XYZ(dLon, dLat, dAlt)

    x_ = m[0][0] * x + m[0][1] * y + m[0][2] * z
    y_ = m[1][0] * x + m[1][1] * y + m[1][2] * z
    z_ = m[2][0] * x + m[2][1] * y + m[2][2] * z

    x_ += xo
    y_ += yo
    z_ += zo

    dLon1, dLat1, dAlt1 = XYZ2LonLat(x_, y_, z_)

    return dLon1, dLat1, dAlt1


def dms2num(d, m, s):
    return d + (m + s / 60.0) / 60.0


def build_discrete_area(x_min, x_max, x_num, y_min, y_max, y_num):
    area = []

    x = np.linspace(x_min, x_max, x_num + 1)
    y = np.linspace(y_min, y_max, y_num + 1)

    for i in range(x_num):
        for j in range(y_num):
            sub_area = {}
            sub_area['points'] = [
                {
                    'longitude': x[i],
                    'latitude': y[j],
                    'altitude': 6000,
                    'velocity': 300
                },
                {
                    'longitude': x[i + 1],
                    'latitude': y[j],
                    'altitude': 6000,
                    'velocity': 300
                },
                {
                    'longitude': x[i + 1],
                    'latitude': y[j + 1],
                    'altitude': 6000,
                    'velocity': 300
                },
                {
                    'longitude': x[i],
                    'latitude': y[j + 1],
                    'altitude': 6000,
                    'velocity': 300
                },
            ]
            sub_area['center'] = [
                {
                    'longitude': 0.5 * (x[i] + x[i + 1]),
                    'latitude': 0.5 * (y[j] + y[j + 1]),
                    'altitude': 6000,
                    'velocity': 300
                },
            ]
            area.append(sub_area)
    return area


def point_in_area(longitude, latitude, poly):
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
