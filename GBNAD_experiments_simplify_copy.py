import numpy as np
from scipy import io
import warnings
import os
import GB
import GBNAD
from Nbr_MGNR import *
from NNSearch import *

warnings.filterwarnings('ignore')
# si
class GB1:
    def __init__(self, data,
                 index):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data[:, :-1]
        self.index = index
        self.center = self.data.mean(0)
        self.xuhao = list(data[:, -1])
        self.score = 0
        self.radius = self.calculate_radius()

    def calculate_radius(self):
        distances = np.mean(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)
        return distances

def Wrap_class(gb_list):
    gb_dist = []
    for i in range(0, len(gb_list)):
        gb = GB1(gb_list[i], i)
        gb_dist.append(gb)
    return gb_dist
# si
# Calculate the distance between GBs for fuzzy similarity.
def get_Dist(center1, center2, radius1, radius2):
    dis_GB = np.linalg.norm(center1 - center2) + radius1 + radius2
    return dis_GB
def gaussian_kernel(Dis, sigma):
    Dis = Dis ** 2
    return np.exp(-Dis / (sigma))

# def GBNAD_experiment(X, k, alpha, delta):
def GBNAD_experiment(X, k, alpha):
    dataPointAnomalyScore = np.zeros(len(X))

    # def cal_entropy(data, delta):
    # si
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(X)
    # gb_list = GBshengcheng.getGranularBall(data)
    # gb_dist = Wrap_class(gb_list)
    # gb_len = len(gb_list)

    # def cal_force(data, delta):
    centers, gb_list, gb_weight, radius = GB.getGranularBall(data)  # 得到粒球
    # k = centers.shape[0]
    # 考虑半径为0的情况
    Max_Radius = max(radius)
    Min_Radius = min(radius)
    density = np.zeros(len(gb_list))
    for i in range(len(gb_list)):
        if radius[i] == 0:
            radius[i] = Max_Radius
            density[i] = 0
            continue
        density[i] = gb_weight[i] / (radius[i])
    density = scaler.fit_transform(density.reshape(-1, 1)).flatten()
    index = []
    for gb in gb_list:
        index.append(gb[:, -1])  # 获取在原始数据中的index

    Ball_Dis = distance.squareform(distance.pdist(centers, 'euclidean'))
    # 计算模糊相似关系
    sigma = np.median(Ball_Dis)
    Sim = np.exp(-(Ball_Dis ** 2) / (2 * sigma ** 2))
    fuzzy_cardinality = np.sum(Sim, axis=1)
    fuzzy_ratio = fuzzy_cardinality / np.sum(fuzzy_cardinality)
    fuzzy_ratio = scaler.fit_transform(fuzzy_ratio.reshape(-1, 1)).reshape(-1)

    NNtool = NNSearch(Ball_Dis)
    dis_index, nn, rnn = NNSearch.get_dis_index(NNtool)
    t, nn, rnn = NNtool.natural_search(dis_index, nn, rnn)
    # # 通过自然搜索得到的邻居关系，计算每个粒球的邻居粒球
    nbGroup = NNtool.get_nb_group(nn, rnn)
    set1_sizes = np.array([len(nbGroup[i]) for i in range(len(nbGroup))])
    set1_sizes = scaler.fit_transform(set1_sizes.reshape(-1, 1)).reshape(-1)
    # set1_sizes = set1_sizes/np.max(set1_sizes)
    # AUC = 0.654
    # normalized_set_sizes = set1_sizes
    # AUC: 0.678
    # normalized_set_sizes = set1_sizes * density
    # AUC:0.647
    # normalized_set_sizes = density
    # AUC: 0.697
    # normalized_set_sizes = set1_sizes + density
    # normalized_set_sizes = 0.1 * set1_sizes + fuzzy_ratio
    # AUC:0.756
    normalized_set_sizes = np.power(set1_sizes * density, 1/2) * fuzzy_ratio
    # AUC:0.793
    # k[2,60] alpha[0,1] AUC:0.862
    # normalized_set_sizes = np.power(set1_sizes + density, 1/2) * fuzzy_ratio
    # k[2,60] alpha[0,1] AUC:0.860
    # normalized_set_sizes = np.power(set1_sizes + density, 1) * fuzzy_ratio
    normalized_set_sizes = scaler.fit_transform(normalized_set_sizes.reshape(-1, 1)).reshape(-1)
    normalized_set_sizes = 1 - normalized_set_sizes

    # normalized_set_sizes = 1 / set1_sizes
    # normalized_set_sizes = scaler.fit_transform(normalized_set_sizes.reshape(-1, 1)).reshape(-1)
    # fusion_nbG_propotion = normalized_set_sizes

    fuzzy_score = normalized_set_sizes
    force_scores = GBNAD.getDataPointAnomalyScore(centers, gb_list, dis_index, k, index, X,
                                                fuzzy_score, alpha)
    IsNan = np.isnan(force_scores).any()
    if IsNan:
        print('AnomalyScore has nan')

    score1 = scaler.fit_transform(force_scores.reshape(-1, 1)).reshape(-1)
    # score2 = scaler.fit_transform(force_scores.reshape(-1, 1)).reshape(-1)
    # Anomaly_score = score1 * alpha + score2 * (1-alpha)
    return score1

if __name__ == '__main__':
    algorithm_name = 'GBNAD'
    # load the data
    dir_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    dataname_path = os.path.join(dir_path, 'datasets\\all_datalists_outlier.mat')  # Categorical Mixed Numerical
    datalists = io.loadmat(dataname_path)['datalists']

    # no_data_ID = [8, 9, 26, 32, 34, 39, 47, 48, 51] + list(range(62, 68))
    no_data_ID = [8, 9, 25, 26, 32, 34, 38, 39, 47, 48, 51] + list(range(62, 68))
    # 生成27-83的所有数字
    full_range = range(27, 85)  # 注意：range右边界不包含，所以84才能包含83
    # 过滤掉no_data中的数字
    result = [num-1 for num in full_range if num not in no_data_ID]
    # data_ID = [29, 31, 33, 35, 36, 40, 42, 46, 49, 56, 57, 60, 68, 69, 71, 73, 74, 77, 82, 83]
    data_ID = [29, 31, 33, 35, 36, 40, 42, 46, 49, 56, 57, 60, 69, 71, 73, 74, 82, 83]
    # data_ID = [33]

    for data_i in data_ID:
        print('data_i=', data_i)
        data_name = datalists[data_i][0][0]
        print('old_dataset:', data_name)
        if data_i in no_data_ID:
            print('Dataset:' + data_name + ' 运行不出来！！！')
            continue
        add_folder = os.path.join(
            os.path.join(dir_path, 'Experiment_Results\\' + algorithm_name + '_results\\' + data_name))
        # if os.path.exists(add_folder):
        #     print(data_name + " 已经有实验结果！！！")
        #     continue
        # os.mkdir(add_folder)

        data_path = os.path.join(dir_path, 'datasets\\' + data_name + '.mat')
        trandata = io.loadmat(data_path)['trandata']

        oridata = trandata.copy()
        trandata = trandata.astype(float)
        # 标准化原始数据
        ID = (trandata >= 1).all(axis=0) & (trandata.max(axis=0) != trandata.min(axis=0))
        scaler = MinMaxScaler()
        if any(ID):
            trandata[:, ID] = scaler.fit_transform(trandata[:, ID])

        X = trandata[:, :-1]  # X是去除标签之后的数据
        labels = trandata[:, -1]
        k = 10
        alpha = 0

        opt_AUC = 0
        opt_out_scores = np.zeros(len(labels))
        opt_k = 0
        opt_T = 0
        opt_delta = 0
        opt_alpha = 0

        # for delta in np.arange(0.1, 1.1, 0.1):
        for k in range(2, 60):
        # for k in [30]:
        #     for alpha in np.arange(0, 1.05, 0.05):
            for alpha in [1.0]:
        # for delta in np.arange(5, 105, 5):
                delta = 5
                # out_scores = GBNAD_experiment(X, k, alpha, delta)
                out_scores = GBNAD_experiment(X, k, alpha)
                # results_name1 = data_name + '_' + algorithm_name + '_k-' + str(k) + '_alpha-' + str(alpha) + '.mat'
                results_name1 = data_name + '_' + algorithm_name + '_delta-' + str(delta) + '.mat'
                AUC = roc_auc_score(labels, out_scores)
                save_path = os.path.join(add_folder, results_name1)
                # io.savemat(save_path, {'out_scores': out_scores})
                print(f"k:{k}")
                print(f"alpha:{alpha}")
                print(f"AUC:{AUC}")
                # print(f"delta:{delta}")
                if AUC > opt_AUC:
                    opt_AUC = AUC
                    opt_out_scores = out_scores
                    opt_k = k
                    opt_alpha = alpha
                    # opt_delta = delta
        print('opt_AUC=', opt_AUC)
        # print('opt_delta=', opt_delta)
        # print('opt_k=', opt_k)
        print('opt_alpha=', opt_alpha)
        T_temp = np.zeros((len(opt_out_scores), 1))
        T_temp[0] = opt_delta
        T_temp[1] = opt_AUC
        T_temp[2] = opt_T
        # T_temp[3] = opt_alpha
        opt_out_scores = opt_out_scores.reshape(-1, 1)
        # 添加一列实验记录T_temp
        opt_out_scores = np.column_stack((opt_out_scores, T_temp))
        results_name2 = data_name + '_' + algorithm_name + '.mat'
        save_path = os.path.join(add_folder, results_name2)
        # io.savemat(save_path, {'opt_out_scores': opt_out_scores})




