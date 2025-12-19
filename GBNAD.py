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

# 根据粒球所受场强大小计算粒界的异常分数
def GBFOD(ExternalElectricity, InternalElectricity, index, row):
    # row = 0.5
    GB_num = len(ExternalElectricity)
    AnomalyScore_GB = np.zeros(GB_num)
    AnomalyScore_Datapoint = np.zeros(np.sum([arr.size for arr in index]))
    for i in range(GB_num):
        AnomalyScore_GB[i] = np.linalg.norm(ExternalElectricity[i])
    # i遍历粒球,j遍历粒球中的数据点
    for i in range(len(index)):
        for j in index[i]:
            # 分数可以乘以和粒球密度相关的权重
            Score = np.linalg.norm(ExternalElectricity[i]*(1-row) + row*InternalElectricity[int(j)])
            AnomalyScore_Datapoint[int(j)] = Score
    return AnomalyScore_Datapoint
    # # 1. 根据S1对粒球GB进行排序，得到升序排序后的粒球索引
    # sorted_gb_indices = np.argsort(np.linalg.norm(ExternalElectricity, axis=1))
    # # 2. 初始化最终的排名列表
    # rankings = []
    # # 3. 遍历排序后的粒球
    # for i in sorted_gb_indices:
    #     # 获取当前粒球的异常分数（S2_i）
    #     gb_points = np.array(index[i], dtype=int)
    #     # 以gb_points为索引，获取当前粒球内的数据点
    #     point_scores = InternalElectricity[gb_points]
    #     # 4. 根据当前粒球内的异常分数S2_i对数据点进行排序
    #     sorted_point_indices = np.argsort(np.linalg.norm(point_scores, axis=1))
    #     # 5. 将排序后的数据点索引添加到排名列表中
    #     for j in sorted_point_indices:
    #         # 6. 返回最终的排名结果，按异常分数从小到大排序
    #         rankings.append((i, gb_points[j]))
    # rankings = np.array(rankings)
    # Rank = rankings[:, 1]
    # Result = np.zeros(len(Rank), dtype=int)
    # for i, j in enumerate(Rank):
    #     Result[j] = i + 1
    # return Result

# 根据粒球所受场强大小计算粒界的异常分数
def GBPOD(ExternalElectricity, InternalElectricity, index, density, row):
    # row = 0.5
    AnomalyScore_Datapoint = np.zeros(len(InternalElectricity))
    # GB_num = len(ExternalElectricity)
    # AnomalyScore_GB = np.zeros(GB_num)
    # for i in range(GB_num):
    #     AnomalyScore_GB[i] = np.linalg.norm(ExternalElectricity[i])
    # i遍历粒球,j遍历粒球中的数据点
    for i in range(len(index)):
        for j in index[i]:
            # 分数可以乘以和粒球密度相关的权重
            if density[i] == 0:
                Score = np.linalg.norm(ExternalElectricity[i] * (1 - row) + row * InternalElectricity[int(j)])
            else:
                ExternalElectricity_normalized = ExternalElectricity[i] / (density[i] ** 3)
                # TODO：是否需要内部力
                # 捕捉局部信息，即结合局部异常的方法
                # TODO：计算目标到邻域粒球的距离并计算库仑力
                # TODO：更改模糊相似性的计算方式，建立库仑力模型
                Score = np.linalg.norm(ExternalElectricity_normalized * (1 - row) + row * InternalElectricity[int(j)])
            AnomalyScore_Datapoint[int(j)] = Score
            has_nan = np.isnan(AnomalyScore_Datapoint).any()
            if has_nan:
                print()
        has_nan = np.isnan(AnomalyScore_Datapoint)
        if has_nan.any():
            print()
    return AnomalyScore_Datapoint

def adaptive_k(GBs, density, start, end):
    K_list = np.zeros(len(GBs), dtype=int)
    density = np.array(density)
    density = density.reshape(-1, 1)
    scaler = MinMaxScaler()
    density = scaler.fit_transform(density)
    # 根据密度计算每个粒球的K
    for i in range(len(GBs)):
        K_list[i] = start + int(density[i] * (end - start))
    return K_list


# 计算粒球单位的集成力/电场强度
def get_ExternalElectricity(GBs, gb_list):
    GB_num = len(GBs)
    m = len(GBs[0])
    IntegratedForce = {}
    ExternalElectricity = np.zeros((GB_num, m))
    for i in range(len(GBs)):
        # 维度
        m = len(GBs[i])
        Q_i = len(gb_list[i])
        IntegratedForce[i] = np.zeros(m)
        ExternalElectricity[i, :] = np.zeros(m)
        # 计算每个粒球的集成力
        for j in range(len(GBs)):
            if j == i:
                continue
            # 计算粒球i和粒球j之间的距离
            GB_distance = np.linalg.norm(GBs[i] - GBs[j])
            powered_distance = GB_distance ** (m-1)
            Q_j = len(gb_list[j])
            # 计算方向向量
            direction = (GBs[i] - GBs[j]) / GB_distance
            # 计算粒球i和粒球j之间的集成力
            IntegratedForce[i] += (Q_i * Q_j / powered_distance * direction)
            # 计算电场强度分量
            ExternalElectricity[i, :] += Q_j * powered_distance * direction
    return IntegratedForce, ExternalElectricity


def getDataPointAnomalyScore(centers, gb_list, dis_index, k, index, trandata, nbG_propotion, alpha):
    n = trandata.shape[0]
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
        forces = (GB_distances[:, :, np.newaxis] * directions) * Q_j
        # forces = (GB_distances[:, :, np.newaxis] * directions) / Q_j
        dataPointAnomalyScore[index[i], :] += np.sum(forces, axis=1) * (1 - (len(gb_list[i]) / n) ** (1 / 3))
        # dataPointAnomalyScore[index[i], :] += np.sum(forces, axis=1)
        # # 算法2：优化前
        # for dp in index[i]:
        #     dp_location = np.asarray(trandata[int(dp), :])
        #     dp_location = dp_location.reshape(1, -1)
        #     for j in range(index_smallest.size):
        #         # 粒球j的索引
        #         z = index_smallest[0, j]
        #         # GB_distance = Tool.KernelSimilarity_s(dp_location, centers[z], Bandwidth)
        #         GB_distance = np.linalg.norm(dp_location - KNearestNeighbor[j])
        #         # 电荷-实验20
        #         # Q_j = len(gb_list[z]) * len(gb_list[i])
        #         Q_j = len(gb_list[z])
        #         # 计算方向向量
        #         direction = (dp_location - KNearestNeighbor[j]) / np.linalg.norm(dp_location - KNearestNeighbor[j])
        #         direction = np.reshape(direction, (1, m))
        #         # 实验22 - （* GB_distance + 距离计算）
        #         score = Q_j * direction * GB_distance
        #         score = score.flatten()
        #         # 计算电场强度分量
        #         if GB_distance != 0:
        #             dataPointAnomalyScore[int(dp), :] += score
        #         else:
        #             continue

    dataPointAnomalyScore = np.linalg.norm(dataPointAnomalyScore, axis=1)
    scaler = MinMaxScaler()
    dataPointAnomalyScore = scaler.fit_transform(dataPointAnomalyScore.reshape(-1, 1))
    dataPointAnomalyScore = dataPointAnomalyScore.flatten()

    for i in range(GB_num):
        # index_GB = np.asarray(index[i], dtype=int)
        # dataPointAnomalyScore[index_GB] = dataPointAnomalyScore[index_GB] * nbG_propotion[i]
        for dp in index[i]:
            # 实验21：探索最优权重 nbG_propotion/2-不行；试试0.001-1000
            dataPointAnomalyScore[int(dp)] = dataPointAnomalyScore[int(dp)] * alpha + nbG_propotion[i] * (1 - alpha)
            # dataPointAnomalyScore[int(dp)] = dataPointAnomalyScore[int(dp)]
            # dataPointAnomalyScore[int(dp)] = nbG_propotion[i]
    return dataPointAnomalyScore


# GBNOD算法
def getAnomalyScoreByNbor(centers, index, trandata, nbG_propotion):
    GB_num = len(centers)
    m = len(centers[0])
    dataPointAnomalyScore = np.zeros((trandata.shape[0], 1))
    for i in range(GB_num):
        # index_GB = np.asarray(index[i], dtype=int)
        # dataPointAnomalyScore[index_GB] = dataPointAnomalyScore[index_GB] * nbG_propotion[i]
        for dp in index[i]:
            # 实验21：探索最优权重 nbG_propotion/2-不行；试试0.001-1000
            dataPointAnomalyScore[int(dp)] = nbG_propotion[i]  # for dp in index[i]:
    #         # 线性加权融合
    # if K_list[0] == 16:
    #     print(dataPointAnomalyScore)
    return dataPointAnomalyScore

def KList_get_ExternalElectricity(detector, centers, gb_list, K_list):
    GB_num = len(centers)
    m = len(centers[0])
    ExternalElectricity = np.zeros((GB_num, m))
    # 用于缓存不同k值的邻居信息
    k_neighbors_cache = {}

    # 改动2.2 使用核函数计算粒球之间的距离
    Bandwidth = Tool.Bandwidth(centers)
    KernelSimilarity = Tool.KernelSimilarity(centers, Bandwidth)
    KernelDistance = Tool.KernelDistance(centers, KernelSimilarity)

    for i in range(GB_num):
        ExternalElectricity[i, :] = np.zeros(m)
        # 计算每个粒球的集成力
        k = K_list[i]
        if k not in k_neighbors_cache:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(centers)
            k_neighbors_cache[k] = nbrs
        # 通过已缓存的k邻居信息进行查询
        KNearestNeighbor, index_smallest = MS.k_nearest_neighbor(detector, centers[i], k_neighbors_cache[k], centers)
        for j in range(index_smallest.size):
            # 版本1
            # 计算粒球i和粒球j之间的距离
            # GB_distance = np.linalg.norm(centers[i] - KNearestNeighbor[j])
            # powered_distance = GB_distance ** (m-1)
            # z = index_smallest[0, j]
            # Q_j = len(gb_list[z])
            # # 计算方向向量
            # direction = (centers[i] - KNearestNeighbor[j]) / GB_distance
            # # 计算电场强度分量
            # ExternalElectricity[i, :] += Q_j * powered_distance * direction

            # 版本2.1
            # 计算粒球i和粒球j之间的距离
            # 加入i的电荷量Q_i
            # GB_distance = np.linalg.norm(centers[i] - KNearestNeighbor[j])
            # powered_distance = GB_distance
            # z = index_smallest[0, j]
            # Q_i = len(gb_list[i])
            # Q_j = len(gb_list[z])
            # # 计算方向向量
            # direction = (centers[i] - KNearestNeighbor[j]) / GB_distance
            # # 计算电场强度分量
            # ExternalElectricity[i, :] += Q_i * Q_j * direction / powered_distance

            # 版本2.2
            # 核函数计算粒球i和粒球j之间的距离
            # TODO：距离公式的选择
            GB_distance = KernelDistance[i, j]
            # 电荷
            Q_i = len(gb_list[i])
            z = index_smallest[0, j]
            Q_j = len(gb_list[z])
            # 计算方向向量
            direction = (centers[i] - KNearestNeighbor[j]) / np.linalg.norm(centers[i] - KNearestNeighbor[j])
            # 计算电场强度分量
            if GB_distance != 0:
                ExternalElectricity[i, :] += Q_i * Q_j * direction / GB_distance
            else:
                continue

    return ExternalElectricity

# 计算粒球内部电场强度
def get_InternalElectricity(index, trandata):
    m = trandata.shape[1]
    InternalElectricity = np.zeros((trandata.shape[0], m), dtype=np.float64)
    # i遍历粒球,j遍历粒球中的数据点
    # for i in range(len(index)):
    #     for j in index[i]:
    #         InternalElectricity[int(j), :] = np.zeros(m)
    #         for k in index[i]:
    #             if j == k:
    #                 continue
    #             p_distance = np.linalg.norm(trandata[int(j), :] - trandata[int(k), :])
    #             powered_distance = p_distance
    #             if p_distance == 0:
    #                 continue
    #             direction = (trandata[int(j), :] - trandata[int(k), :]) / p_distance
    #             InternalElectricity[int(j), :] += powered_distance * direction

    # 使用核函数计算粒球内部电场强度
    for i in range(len(index)):
        indices = np.array(index[i], dtype=int)
        data = trandata[indices, :]
        Bandwidth = Tool.Bandwidth(data)
        # 当Bandwidth==0时，说明数据集中的数据完全相同，此时无法计算核函数，直接将KernelSimilarity设置为1
        if Bandwidth==0:
            KernelSimilarity = np.ones((len(indices), len(indices)))
            KernelDistance = Tool.KernelDistance(data, KernelSimilarity)
        else:
            KernelSimilarity = Tool.KernelSimilarity(data, Bandwidth)
            KernelDistance = Tool.KernelDistance(data, KernelSimilarity)
        for index_j, j in enumerate(indices):
            InternalElectricity[j, :] = np.zeros(m)
            for index_k, k in enumerate(indices):
                if j == k:
                    continue
                kernel_distance = KernelDistance[index_j, index_k]
                if kernel_distance == 0:
                    # TODO: 为了避免出现nan值，暂时将kernel_distance设置为一个很小的值
                    kernel_distance = 10e-6
                    # 相同数据的互相作用力为0
                    InternalElectricity[j, :] += np.zeros(m)
                else:
                    direction = (trandata[j, :] - trandata[k, :]) / np.linalg.norm(trandata[j, :] - trandata[k, :])
                    InternalElectricity[j, :] += direction * 1 / kernel_distance
            has_nan = np.isnan(InternalElectricity[j, :]).any()
            if has_nan:
                print()
    return InternalElectricity

if __name__ == '__main__':

    # data_path = "../datasets/hepatitis_2_9_variant1.mat"
    data_path = "datasets\\syntheticData\\data_01.csv"
    # 获取datasetName的后缀名
    suffix = data_path.split('.')[-1]
    trandata, label = mk_figure.loadData(data_path, suffix)

    oridata = trandata.copy()
    trandata = trandata.astype(float)

    scaler = MinMaxScaler()
    trandata[:] = scaler.fit_transform(trandata[:])
    outliers = trandata[label == 1]

    X = trandata[:]
    # 返回每个粒球的中心，粒球的数据点信息，粒球的权重
    centers, gb_list, gb_weight, radius = GB.getGranularBall(X)
    GBs = []
    density = []
    for i in range(len(gb_list)):
        GBs.append(GrainBall(centers[i], radius[i], gb_weight[i]))
    # index[i]为第i个粒球包含的数据点索引
    index = []
    weight = []
    for gb in gb_list:
        index.append(gb[:, -1])
        weight.append(gb.shape[0])
    mk_figure.picture2d(trandata, GBs, outliers)

    detector = MS(centers, X, index)

    Ball_Bandwidth = Tool.Bandwidth(centers)
    Ball_Similarity = Tool.KernelSimilarity(centers, Ball_Bandwidth)
    Ball_Similarity = scaler.fit_transform(Ball_Similarity)
    NNtool = NNSearch(Ball_Similarity)
    t, nn, rnn, dis_index = NNtool.natural_search()
    # 通过自然搜索得到的邻居关系，计算每个粒球的邻居粒球
    nbGroup = NNtool.get_nb_group(nn, rnn, centers)
    set_sizes = np.array([len(s) for s in nbGroup])
    normalized_set_sizes = 1 / (set_sizes + 1e-6)
    nbG_propotion = scaler.fit_transform(normalized_set_sizes.reshape(-1, 1)).reshape(-1)
    k = 25
    out_scores = getDataPointAnomalyScore(detector, centers, gb_list, k, index, X, nbG_propotion, 0.7)
    if np.isnan(out_scores).any():
        print("nan")
    # 根据邻居粒球计算粒球间的引力
    # scores = NNgetAnomalyScore(centers, gb_list, nbGroup, trandata)


    # scaler2 = MinMaxScaler(feature_range=(-1, 1))
    # ExternalElectricity2_scalared = scaler2.fit_transform(ExternalElectricity2)
    # InternalElectricity_scalared = scaler2.fit_transform(InternalElectricity)


    # 画图
    mk_figure.picture2d(X, GBs)

    # 先建库 再分类

    # out_scores = detector.GBMOD()
    # anomaly_scores = GBFOD(ExternalElectricity2_scalared, InternalElectricity_scalared, index, 0.5)
    # anomaly_scores2 = GBPOD(ExternalElectricity2_scalared, InternalElectricity_scalared, index, density, 0.5)
    # 对每一行求向量的模
    # anomaly_scores3 = np.linalg.norm(dataPointScore, axis=1)
    # anomaly_scores = np.array(anomaly_scores2)
    # anomaly_scores3 = column_or_1d(anomaly_scores3)
    # for score in anomaly_scores3:
    #     print(score)
