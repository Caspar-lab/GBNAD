import math

import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance, KDTree
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import check_consistent_length, column_or_1d
import Tool
import GB
from Granular_ball import GrainBall
import mk_figure
from NNSearch import *

warnings.filterwarnings('ignore')
'''
TODO:
    1. 为考虑包含不同个数数据点的粒球，为粒球设置重量G
        G = GrainBall.weight
    2. 每个粒球的K可根据G和半径变化，受引力范围启发，G越大，K越大
        K取决于粒球的半径和重量, 即密度: ρ = G / V
    3. 基于该重量和近邻数量，计算gravitational force
    4. 基于此和每个数据点到粒球中心计算的力f2计算每个点的异常分数S
'''

class MS:
    def __init__(self, data, trandata, index):
        self.data = data
        self.trandata = trandata
        self.index = index
        self.iteration_number = 3

    # 通过KNN算法获取每个点的K个最近邻
    def k_nearest_neighbor(self, point, nbrs, points):
        distances, indices = nbrs.kneighbors([point])
        k_smallest = []
        index_smallest = indices[:, 1:]
        index_smallest = np.array(index_smallest, dtype=int)
        # indices为二维数组
        for i, p in enumerate(indices[0]):
            k_smallest.append(points[p])
        # 不包含自己
        del k_smallest[0]
        return k_smallest, index_smallest


def getDataPointAnomalyScore(centers, gb_list, dis_index, k, index, trandata, nbG_propotion, alpha):
    n = trandata.shape[0]
    feature_num = trandata.shape[1]
    GB_num = len(centers)
    m = len(centers[0])
    dataPointAnomalyScore = np.zeros((trandata.shape[0], m))
    for i in range(GB_num):
        # 算法1：使用矩阵广播
        index[i] = np.asarray(index[i], dtype=int)
        dps_location = np.asarray(trandata[index[i], :])
        # 获取粒球j的索引
        z_indices = dis_index[i][1][1:k]
        centers_z = centers[z_indices]
        Q_j = np.array([len(gb_list[z]) for z in z_indices]).reshape(-1, 1)

        # 计算粒球j与数据点的距离和方向向量
        GB_distances = np.linalg.norm(dps_location[:, np.newaxis, :] - centers_z, axis=2)
        directions = (dps_location[:, np.newaxis, :] - centers_z) / GB_distances[:, :, np.newaxis]

        # 计算得分并累加
        forces = (GB_distances[:, :, np.newaxis] * directions) ** (m-1) * Q_j
        dataPointAnomalyScore[index[i], :] += np.sum(forces, axis=1)

    dataPointAnomalyScore = np.linalg.norm(dataPointAnomalyScore, axis=1)
    scaler = MinMaxScaler()
    dataPointAnomalyScore = scaler.fit_transform(dataPointAnomalyScore.reshape(-1, 1))
    dataPointAnomalyScore = dataPointAnomalyScore.flatten()

    for i in range(GB_num):
        for dp in index[i]:
            dataPointAnomalyScore[int(dp)] = dataPointAnomalyScore[int(dp)] * alpha + nbG_propotion[i] * (1 - alpha)
    return dataPointAnomalyScore
