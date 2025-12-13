from logging import raiseExceptions

import numpy as np
import argparse
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter, defaultdict
import sys
import random
from sklearn.metrics.pairwise import euclidean_distances

from torch.utils.hipify.hipify_python import meta_data

sys.path.append(".")
from models.gcxyuan.mcre import DeepRe, MoCo, load_model, MLPAE_for_DeepRe, DeepRe_Reload_Emb, reload_test_DeepRe
import models.gcxyuan.data_loader as data_loader


class KMeansSampleSelector:
    def __init__(self, n_clusters=10, shot_type = 'note'):
        """
        初始化KMeans样本选择器

        参数:
            n_clusters: 聚类的数量
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans_centers = None
        self.kmeans_label_map = {}

        self.cluster_centers = None
        self.cluster_labels = None

        self.train_samples = None
        self.train_gt_label = None
        self.clusters = None  # 存储每个聚类的样本索引

        self.shot_type = shot_type


    def fit(self, train_samples, train_gt_label = None):
        """
        使用训练集拟合KMeans模型

        参数:
            train_samples: 训练样本数组，形状为 (n_samples, n_features)
        """
        self.train_samples = train_samples
        self.train_cluster_labels = self.kmeans.fit_predict(train_samples)
        self.cluster_centers = self.kmeans.cluster_centers_

        if self.shot_type == 'annote' and train_gt_label == None:
            print(f'Error, shot_type with annote must have gt_label')
        self.train_gt_label = train_gt_label

        # 按聚类分组样本索引
        self.clusters = {}
        for i in range(self.n_clusters):
            self.clusters[i] = np.where(self.train_cluster_labels == i)[0]

    def select_samples_label(self, n_class, val_feas=None, val_labels=None, n_samples=150, shot_type = 'note'):
        """
        根据验证集在聚类中的分布选择样本

        参数:
            val_samples: 验证集样本数组，形状为 (n_val_samples, n_features)
            val_labels: 验证集标签数组，形状为 (n_val_samples,)
            n_samples: 每个类别或每个验证样本需要选择的样本数量

        返回:
            selected_samples: 筛选出的新数据集样本
            selected_indices: 筛选出的样本在原始训练集中的索引
        """
        if shot_type == 'note':
            self.shot_type = shot_type
            assert val_feas is not None and val_labels is not None
            # 预测验证集所属的聚类
            val_cluster_labels = self.kmeans.predict(val_feas)
            selected_indices = []
            selected_labels = []

            no_val_cluster = []


            # 遍历每个聚类
            for cluster_id in range(self.n_clusters):
                # 获取该聚类中的验证样本索引
                val_in_cluster = np.where(val_cluster_labels == cluster_id)[0]

                if not len(val_in_cluster):
                    print(f'no val in cluster {cluster_id}')
                    no_val_cluster.append(cluster_id)
                    continue  # 跳过没有验证样本的聚类

                # 获取这些验证样本的标签
                val_labels_in_cluster = val_labels[val_in_cluster]
                val_feas_in_cluster = val_feas[val_in_cluster]
                unique_labels = np.unique(val_labels_in_cluster)

                # 情况1：聚类中只有一种验证集类别
                if len(unique_labels) == 1:
                    self.kmeans_label_map[cluster_id] = int(unique_labels[0])
                    # 获取该聚类中的所有训练样本
                    train_cluster_sample_indices = self.clusters[cluster_id]
                    train_cluster_samples = self.train_samples[train_cluster_sample_indices]


                    # 计算到聚类中心的距离
                    train_distances = np.linalg.norm(
                        train_cluster_samples - self.cluster_centers[cluster_id],
                        axis=1
                    )
                    train_sorted_indices = np.argsort(train_distances)

                    train_closest_indices = train_cluster_sample_indices[train_sorted_indices[:int(n_samples * n_class /self.n_clusters)+1]]
                    selected_indices.extend(train_closest_indices)
                    selected_labels.extend([int(unique_labels.tolist()[0])]* len(train_closest_indices))

                # 情况2：聚类中有多种验证集类别
                else:
                    self.kmeans_label_map[cluster_id] = int(Counter(val_labels_in_cluster).most_common(1)[0][0])
                    # 获取该聚类中的所有训练样本
                    train_cluster_sample_indices = self.clusters[cluster_id]
                    train_cluster_samples = self.train_samples[train_cluster_sample_indices]

                    # 计算到聚类中心的距离
                    train_distances = np.linalg.norm(
                        train_cluster_samples - self.cluster_centers[cluster_id],
                        axis=1
                    )

                    most_val_in_cluster = np.where(val_labels_in_cluster == Counter(val_labels_in_cluster).most_common(1)[0][0])[0]
                    most_val_feas_in_cluster = val_feas_in_cluster[most_val_in_cluster]

                    val_distances = np.linalg.norm(most_val_feas_in_cluster - self.cluster_centers[cluster_id],
                                                   axis=1)
                    val_distances_mean = np.mean(val_distances)

                    # 按距离排序，取最近的n_samples个样本
                    train_mask = train_distances <= val_distances_mean
                    train_closest_indices = train_cluster_sample_indices[train_mask]
                    train_closest_indices = random.choices(train_closest_indices, k=int(n_samples * n_class /self.n_clusters)+1)
                    selected_indices.extend(train_closest_indices)
                    selected_labels.extend([self.kmeans_label_map[cluster_id]] * len(train_closest_indices))

                    # # 对每个验证样本，找到最近的n_samples个训练样本
                    # for val_idx in val_in_cluster:
                    #     val_sample = val_dataset.train_samples[val_idx]
                    #
                    #     # 获取该聚类中的所有训练样本
                    #     cluster_sample_indices = self.clusters[cluster_id]
                    #     cluster_samples = self.train_samples[cluster_sample_indices]
                    #
                    #     # 计算距离并找到最近的样本
                    #     distances = np.linalg.norm(cluster_samples - val_sample, axis=1)
                    #     sorted_indices = np.argsort(distances)
                    #
                    #     # 取最近的n_samples / len(unique_labels)个样本
                    #     closest_indices = cluster_sample_indices[sorted_indices[:int(n_samples/len(unique_labels))]]
                    #     selected_indices.extend(closest_indices)
                    #     selected_labels.extend(int([Counter(val_labels_in_cluster).most_common(1)[0][0]]) * len(closest_indices.shape[0]))

            selected_samples = self.train_samples[selected_indices]
            selected_gt_labels = self.train_gt_label[selected_indices]
            selected_labels = np.array(selected_labels)

            print(f'note purity is : {sum(selected_labels == selected_gt_labels) / len(selected_gt_labels)}')

            return selected_samples, selected_labels, selected_gt_labels, self.kmeans_label_map, no_val_cluster

        elif shot_type == 'annote':
            self.shot_type = shot_type
            selected_indices = []
            selected_labels = []
            # 遍历每个聚类
            for cluster_id in range(self.n_clusters):
                if cluster_id not in self.kmeans_label_map.keys():
                    # 获取该聚类中的所有训练样本
                    train_cluster_sample_indices = self.clusters[cluster_id]
                    train_cluster_samples = self.train_samples[train_cluster_sample_indices]

                    # 计算到聚类中心的距离
                    train_distances = np.linalg.norm(
                        train_cluster_samples - self.cluster_centers[cluster_id],
                        axis=1
                    )
                    train_sorted_indices = np.argsort(train_distances)

                    train_closest_indices = train_cluster_sample_indices[train_sorted_indices[:int(n_samples * n_class /self.n_clusters)+1]]
                    selected_indices.extend(train_closest_indices)
                    train_closest_indice = train_cluster_sample_indices[train_sorted_indices[0]]
                    selected_closest_gt_label = self.train_gt_label[train_closest_indice]
                    self.kmeans_label_map[cluster_id] = selected_closest_gt_label
                    selected_labels.extend([int(selected_closest_gt_label)] * len(train_closest_indices))

            selected_samples = self.train_samples[selected_indices]
            selected_gt_labels = self.train_gt_label[selected_indices]
            selected_labels = np.array(selected_labels)

            print(f'annote purity is : {sum(selected_labels == selected_gt_labels) / len(selected_gt_labels)}')

            return np.array(selected_samples), np.array(selected_labels), np.array(selected_gt_labels), self.kmeans_label_map

    def gen_softlabel(self, soft_label_k, num_classes):
        # 计算每个样本到所有聚类中心的距离
        # 初始化距离矩阵
        train_distances_all = np.zeros((len(self.train_samples), len(self.cluster_centers))) # 形状: (n_samples, n_clusters)
        # 计算每个样本到每个聚类中心的距离
        for i in range(len(self.train_samples)):
            for j in range(len(self.cluster_centers)):
                # 计算欧氏距离
                train_distances_all[i, j] = np.sqrt(np.sum((self.train_samples[i] - self.cluster_centers[j]) ** 2))

        # train_distances_all = np.sqrt(((self.train_samples[:, np.newaxis] - self.cluster_centers) ** 2).sum(axis=2))
        # train_distances_all = pairwise_distances(self.train_samples, self.cluster_centers)
        # 获取每个样本到最近k个聚类中心的距离和对应的聚类ID
        # 对距离排序，取最小的k个
        train_nearest_indices = np.argsort(train_distances_all, axis=1)[:, :soft_label_k]  # 最近k个聚类的索引
        train_nearest_distances = np.take_along_axis(train_distances_all, train_nearest_indices, axis=1)  # 对应的距离

        # 获取所有唯一的真实标签并创建映射
        # self.kmeans_label_map

        # 初始化软标签数组
        soft_labels = np.zeros((self.train_samples.shape[0], num_classes))

        # 为每个样本计算软标签
        for i in range(self.train_samples.shape[0]):
            # 最近k个聚类的ID和距离
            clusters = train_nearest_indices[i]
            dists = train_nearest_distances[i]
            true_labels = [self.kmeans_label_map[j] for j in clusters]

            # 分组 - 记录a中每个元素对应的所有b值和索引
            element_groups = defaultdict(lambda: {'values': [], 'indices': [], 'finalvals': 0})
            for idx, (a_val, b_val) in enumerate(zip(true_labels, dists)):
                element_groups[a_val]['values'].append(b_val)
                element_groups[a_val]['indices'].append(idx)
            for a_val, group in element_groups.items():
                values = group['values']
                # 如果是相同元素（组内元素数量大于1）
                if len(values) > 1:
                    min_val = min(values)
                    count = len(values)
                    # 计算最小值除以相同个数
                    processed_val = min_val / count
                    element_groups[a_val]['finalvals'] = processed_val
                else:
                    # 对于唯一元素，直接使用对应的b值
                    element_groups[a_val]['finalvals'] = values[0]

            true_labels = element_groups.keys()
            dists = [element_groups[a_val]['finalvals'] for a_val in true_labels]

            # 避免除零错误（距离为0时的处理）
            dists = np.maximum(dists, 1e-10)

            # 计算距离的反比作为权重
            weights = 1.0 / dists

            # 归一化权重，使其总和为1
            weights /= np.sum(weights)

            # 将权重分配到对应的真实标签
            for true_label, weight in zip(true_labels, weights):
                soft_labels[i, true_label] += weight

        return soft_labels

class huge_deal():
    def __init__(self):
        self.val_centers =None

    def calculate_class_centers(self, validation_data, validation_labels):
        """计算每个类别的中心点"""
        unique_classes = np.unique(validation_labels)
        centers = {}

        for cls in unique_classes:
            # 获取该类别的所有样本
            class_samples = validation_data[validation_labels == cls]
            # 计算中心点（均值）
            centers[cls] = np.mean(class_samples, axis=0)

        self.val_centers = centers

        return centers

    def gen_softlabel(self, train_samples, soft_label_k, num_classes):
        # 计算每个样本到所有聚类中心的距离
        # 提取类别和对应的中心点
        cluster_classes = list(self.val_centers.keys())  # same with label
        centers = np.array(list(self.val_centers.values()))
        # 初始化距离矩阵
        train_distances_all = np.zeros(
            (len(train_samples), len(centers)))  # 形状: (n_samples, n_clusters)
        # 计算每个样本到每个聚类中心的距离
        for i in range(len(train_samples)):
            for j in range(len(centers)):
                # 计算欧氏距离
                train_distances_all[i, j] = np.sqrt(np.sum((train_samples[i] - centers[j]) ** 2))
        # 初始化软标签数组
        soft_labels = np.zeros((train_samples.shape[0], num_classes))
        for i in range(train_samples.shape[0]):
            # 获取当前样本到所有类别的距离
            sample_distances = train_distances_all[i]

            # 找到最小的k个距离及其对应的类别索引
            k_indices = np.argsort(sample_distances)[:soft_label_k]

            # 获取这k个最小距离
            k_distances = sample_distances[k_indices]

            # 计算距离的反比（加一个小值避免除零错误）
            inv_distances = 1 / (k_distances + 1e-8)

            # 归一化，使权重之和为1
            weights = inv_distances / np.sum(inv_distances)

            # 分配软标签
            for idx, weight in zip(k_indices, weights):
                soft_labels[i, idx] = weight

        return soft_labels



    def generate_soft_labels(self, train_data, k=3):
        """
        生成训练集的软标签和预测标签

        参数:
        - train_data: 训练集数据
        - class_centers: 每个类别的中心点字典
        - k: 考虑的最近邻数量

        返回:
        - soft_labels: 软标签数组
        - predicted_labels: 预测标签数组
        """
        # 提取类别和对应的中心点
        classes = list(self.class_centers.keys())
        centers = np.array(list(self.class_centers.values()))

        # 计算每个训练样本到所有中心点的距离
        distances = euclidean_distances(train_data, centers)

        # 初始化软标签和预测标签
        num_samples = train_data.shape[0]
        num_classes = len(classes)
        soft_labels = np.zeros((num_samples, num_classes))
        predicted_labels = np.zeros(num_samples, dtype=int)


        return soft_labels, predicted_labels

def load_data(npy_path, labels_npz_path, val_npz_path):
    """加载数据文件"""
    # 读取npy文件（特征数据）
    features = np.load(npy_path)

    # 读取样本标签npz文件
    labels_data = np.load(labels_npz_path)
    # 假设npz文件中包含"labels"键，根据实际情况修改
    sample_labels = labels_data['labels']

    # 读取验证样本npz文件
    val_data = np.load(val_npz_path)
    # 假设npz文件中包含"val_features"和"val_labels"键，根据实际情况修改
    val_features = val_data['val_features']
    val_labels = val_data['val_labels']

    return features, sample_labels, val_features, val_labels






def expand_meta(config, train_dataset, val_dataset, logger, shot_type = 'note'):  #shot_type = 'note' 'annote'

    def determine_optimal_clusters(features, method='elbow', max_k=10, begin=1):
        """通过肘部法或轮廓系数确定最佳聚类数量"""
        if method == 'elbow':
            # 肘部法
            distortions = []
            K_range = range(begin, max_k + 1)

            for k in K_range:
                print(k, end=' ')
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(features)
                distortions.append(kmeans.inertia_)

            # 绘制肘部曲线
            plt.figure(figsize=(10, 6))
            plt.plot(K_range, distortions, 'bo-')
            plt.xlabel('聚类数量 k')
            plt.ylabel('畸变程度 (Distortion)')
            plt.title('肘部法确定最佳聚类数量')
            plt.grid(True)
            plt.show()

            # 简单逻辑：找到畸变程度下降变缓的点（实际应用中可能需要手动选择）
            # 这里使用二阶导数法寻找拐点
            if len(distortions) >= 3:
                second_derivatives = np.diff(np.diff(distortions))
                optimal_k = np.argmax(second_derivatives) + 2  # +2 是因为二阶导数索引偏移
                print(f"肘部法建议的最佳聚类数量: {begin + optimal_k}")
                return begin + optimal_k
            else:
                return begin + 2  # 默认值

        elif method == 'silhouette':
            # 轮廓系数法
            silhouette_scores = []
            K_range = range(2, max_k + 1)  # 轮廓系数不适用于k=1

            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(features)
                silhouette_avg = silhouette_score(features, labels)
                silhouette_scores.append(silhouette_avg)
                print(f"k = {k}, 平均轮廓系数 = {silhouette_avg:.4f}")

            # 绘制轮廓系数曲线
            plt.figure(figsize=(10, 6))
            plt.plot(K_range, silhouette_scores, 'bo-')
            plt.xlabel('聚类数量 k')
            plt.ylabel('平均轮廓系数')
            plt.title('轮廓系数法确定最佳聚类数量')
            plt.grid(True)
            plt.show()

            # 选择轮廓系数最大的k值
            optimal_k = K_range[np.argmax(silhouette_scores)]
            print(f"轮廓系数法建议的最佳聚类数量: {optimal_k}")
            return optimal_k

        else:
            raise ValueError("方法必须是 'elbow' 或 'silhouette'")

    def analyze_cluster_labels(features, true_labels, n_clusters, percentage=0.1):
        """分析每个聚类中心最近的样本的真实标签分布"""
        # 执行KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        # 计算每个样本到其聚类中心的距离
        distances = np.sqrt(((features - centers[cluster_labels]) ** 2).sum(axis=1))

        # 分析每个聚类
        cluster_label_distributions = {}

        for cluster_id in range(n_clusters):
            # 获取该聚类的所有样本索引
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            # 获取这些样本的距离和真实标签
            cluster_distances = distances[cluster_indices]
            cluster_true_labels = true_labels[cluster_indices]

            # 按距离排序，取最近的percentage比例的样本
            n_samples = len(cluster_indices)
            n_nearest = max(1, int(n_samples * percentage))  # 至少取1个样本
            nearest_indices = np.argsort(cluster_distances)[:n_nearest]
            nearest_true_labels = cluster_true_labels[nearest_indices]

            # 计算标签分布
            label_counts = Counter(nearest_true_labels)
            total = sum(label_counts.values())
            label_distribution = {label: count / total for label, count in label_counts.items()}

            cluster_label_distributions[cluster_id] = label_distribution
            print(f"\n聚类 {cluster_id} 中最近的 {percentage * 100}% 样本的标签分布:  totle:{len(cluster_indices)}")
            for label, ratio in label_distribution.items():
                print(f"  标签 {label}: {ratio:.2%}")

        return kmeans, cluster_label_distributions

    def map_cluster_to_true_labels(kmeans, val_features, val_labels):
        """根据验证样本推测聚类标签与真实标签的映射关系"""
        # 预测验证样本的聚类
        val_cluster_labels = kmeans.predict(val_features)

        # 建立聚类标签到真实标签的映射
        cluster_to_true = {}

        for cluster_id in np.unique(val_cluster_labels):
            # 获取该聚类中所有验证样本的真实标签
            true_labels_in_cluster = val_labels[val_cluster_labels == cluster_id]

            if len(true_labels_in_cluster) == 0:
                continue

            # 找到出现次数最多的真实标签作为映射
            most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
            cluster_to_true[cluster_id] = most_common_label

        print("\n聚类标签到真实标签的映射关系:")
        for cluster, true_label in cluster_to_true.items():
            print(f"  聚类 {cluster} -> 真实标签 {true_label}")

        return cluster_to_true

    def rebuild_dataset(config, train_dataset, logger):
        if config['cluster_fea_type'] == 'onlyfea':
            return train_dataset.train_data.detach().numpy(), train_dataset.train_gt_labels.detach().numpy(), train_dataset.train_labels.detach().numpy()
        else:
            train_emb, train_gt_label, train_label = DeepRe_Reload_Emb(config, train_dataset, logger)
            if config['cluster_fea_type'] == 'onlyemb':
                return train_emb, train_gt_label, train_label
            elif config['cluster_fea_type'] == 'feaemb':
                # 拼接数据
                train_fea = train_dataset.train_data.detach().numpy()
                if train_fea.shape[0] != train_emb.shape[0]:
                    raise ValueError("原始数据和嵌入层数据的样本数量不匹配")
                train_fea_emb = np.concatenate([train_fea, train_emb], axis=1)
                return train_fea_emb, train_gt_label, train_label

    train_fea, train_gt_label, train_label = rebuild_dataset(config, train_dataset, logger)

    logger.info(f"特征数据形状: {train_fea.shape}")
    logger.info(f"样本标签数量: {len(train_label)}")

    # # 确定最佳聚类数量
    # # can del =======================================
    # cluster_method = input("请选择确定聚类数量的方法 (elbow/silhouette): ").strip().lower()
    # if cluster_method not in ['elbow', 'silhouette']:
    #     print("无效的方法，使用默认的肘部法")
    #     cluster_method = 'elbow'
    #
    # max_k = int(input(f"请输入最大聚类数量尝试值 (默认 {2 * config['num_classes']}): ") or str(2 * config['num_classes']))
    # # optimal_k = determine_optimal_clusters(train_fea, method=cluster_method, max_k=max_k, begin=config['num_classes'])
    # optimal_k = determine_optimal_clusters(train_fea, method=cluster_method, max_k=40, begin=20)
    # # 允许用户手动调整聚类数量
    # user_k = input(f"是否使用建议的聚类数量 {optimal_k}? 如需修改请输入数字，否则按回车: ")
    # if user_k.strip().isdigit():
    #     optimal_k = int(user_k)
    # optimal_k = 80
    #
    # # 分析聚类与标签的关系
    # # percentage = float(input("请输入分析最近样本的百分比 (0-1，默认0.1): ") or "0.1")
    # percentage = 0.1
    # kmeans, cluster_distributions = analyze_cluster_labels(
    #     train_fea, train_gt_label, optimal_k, percentage)
    # kmeans, cluster_distributions = analyze_cluster_labels(
    #     train_fea[:,:-32], train_gt_label, optimal_k, 0.1)
    # # can del =======================================

    if config['dataset'] == 'tls23' and config['label_init_type'] == 'kmeans':
        optimal_k = config['optimal_k']
        selector = KMeansSampleSelector(n_clusters=optimal_k, shot_type='note')
        selector.fit(train_fea, train_gt_label)
        # # save kmeans model
        # kmeans_name = '_'.join(['model_kmeans',config['cluster_fea_type'], config['noise_pattern']+str(config['noise_ratio'])+'.pkl'])
        # with open(os.path.join(config['savedir'], kmeans_name), 'wb') as f:
        #     pickle.dump(selector.kmeans, f)

        val_fea, val_gt_label, val_label = rebuild_dataset(config, val_dataset, logger)
        meta_data_ex, meta_flabel_ex, meta_gtlabel_ex, kmeans_label_map,  no_val_cluster = selector.select_samples_label(config['num_classes'], val_fea, val_label, n_samples=150, shot_type ='note')
        if len(no_val_cluster) > 0:
            meta_data_ex2, meta_flabel_ex2, meta_gtlabel_ex2, kmeans_label_map = selector.select_samples_label(config['num_classes'], n_samples=150, shot_type ='annote')
            meta_data_ex = np.concatenate((meta_data_ex, meta_data_ex2), axis=0)
            meta_flabel_ex = np.concatenate((meta_flabel_ex, meta_flabel_ex2), axis=0)
            meta_gtlabel_ex = np.concatenate((meta_gtlabel_ex, meta_gtlabel_ex2), axis=0)

        print(f'extended meta data shape: {meta_data_ex.shape}')
        print(f'extended meta gt_label : {Counter(meta_gtlabel_ex).items()}')
        print(f'extended meta data purity : {sum(meta_flabel_ex == meta_gtlabel_ex) / len(meta_gtlabel_ex)}')

        meta_data_ex = meta_data_ex[:, :config['input_dim']]
        meta_data_ex = np.concatenate((meta_data_ex, val_dataset.train_data), axis=0)
        meta_flabel_ex = np.concatenate((meta_flabel_ex, val_dataset.train_labels), axis=0)
        meta_gtlabel_ex = np.concatenate((meta_gtlabel_ex, val_dataset.train_gt_labels), axis=0)

        meta_ex_name = '_'.join(['meta_data', config['cluster_fea_type'], config['noise_pattern'] + str(config['noise_ratio']) + '.npz'])
        np.savez(os.path.join(config['savedir'],meta_ex_name),  X_meta=meta_data_ex, y_meta=meta_flabel_ex, y_gt_meta=meta_gtlabel_ex)

        soft_label_kmeans = selector.gen_softlabel(config['meta_soft_label_k'], config['num_classes'])
        pre_label_kmeans = np.argmax(soft_label_kmeans, axis=1)

        print(f'train kmeans pre label purity : {sum(train_gt_label == pre_label_kmeans) / len(train_gt_label)}')
        print(f'train kmeans pre label counter : {Counter(pre_label_kmeans)}')

        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(pre_label_kmeans),
            y=pre_label_kmeans
        )

        # 生成完整的连续标签序列
        continuous_labels = np.arange(0, config['num_classes'])
        # 找到缺失的标签
        missing_labels = np.setdiff1d(continuous_labels, np.unique(pre_label_kmeans))
        counter = 0
        for ind in missing_labels:
            weights = np.insert(weights, ind+counter, max(weights))
            counter += 1

        init_weight = [weights[i] for i in pre_label_kmeans]

        kmeans_label_map_name =  '_'.join(['model_kmeans_labels_map', config['cluster_fea_type'], config['noise_pattern'] + str(config['noise_ratio']) + '.npz'])
        np.savez(os.path.join(config['savedir'], kmeans_label_map_name),
                 X_train_ex = train_fea, train_soft_label=soft_label_kmeans, train_pre_label = pre_label_kmeans,
                 label_map=kmeans_label_map, train_cluster_label=selector.train_cluster_labels, init_weight = init_weight)

    elif config['dataset'] == 'ids17c' and config['label_init_type'] == 'meta':

        val_fea, val_gt_label, val_label = rebuild_dataset(config, val_dataset, logger)

        kmeans_info2 = huge_deal()
        kmeans_info2.calculate_class_centers(val_fea, val_gt_label)
        soft_label_kmeans = kmeans_info2.gen_softlabel(train_fea, config['meta_soft_label_k'], config['num_classes'])
        pre_label_kmeans = np.argmax(soft_label_kmeans, axis=1)

        print(f'train kmeans pre label purity : {sum(train_gt_label == pre_label_kmeans) / len(train_gt_label)}')
        print(f'train kmeans pre label counter : {Counter(pre_label_kmeans)}')

        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(pre_label_kmeans),
            y=pre_label_kmeans
        )

        # 生成完整的连续标签序列
        continuous_labels = np.arange(0, config['num_classes'])
        # 找到缺失的标签
        missing_labels = np.setdiff1d(continuous_labels, np.unique(pre_label_kmeans))
        counter = 0
        for ind in missing_labels:
            weights = np.insert(weights, ind + counter, max(weights))
            counter += 1

        init_weight = [weights[i] for i in pre_label_kmeans]

        kmeans_label_map_name = '_'.join(['model_meta_labels_map', config['cluster_fea_type'],
                                          config['noise_pattern'] + str(config['noise_ratio']) + '.npz'])
        np.savez(os.path.join(config['savedir'], kmeans_label_map_name),
                 X_train_ex=train_fea, train_soft_label=soft_label_kmeans, train_pre_label=pre_label_kmeans,
                  train_cluster_label=pre_label_kmeans, init_weight=init_weight)


from config_meta2 import config_meta
from utils import setup_realtime_logging

# 初始化实时日志
logger = setup_realtime_logging("script_output_lcm.log")

# 用logger替代print输出
logger.info("脚本开始执行...")



# load config
for noise_pattern in ['asym', 'sym']:
    for noise_ratio in [0.2, 0.4, 0.6, 0.8]:
# for noise_pattern in ['sym']:
#     for noise_ratio in [0.4, 0.6, 0.8]:
#         args = {'dataset': 'tls23',
#                 'noise_pattern': noise_pattern,  ##asy or sy
#                 'noise_ratio': noise_ratio}
        args = {'dataset': 'ids17c',
                'noise_pattern': noise_pattern,  ##asy or sy
                'noise_ratio': noise_ratio}

        config = config_meta(args)

        logger.info(args)

        # ========================


        # load data
        train_data_path = os.path.join(config['datadir'], config['train_data_file'])
        train_label_path = os.path.join(config['datadir'], 'noise_' + str(config['noise_pattern']) + '_labels.npz')
        test_path = os.path.join(config['datadir'], config['test_data_file'])
        meta_data_path = os.path.join(config['datadir'], config['meta_data_file'])

        train_data, train_labels, train_gt_labels, val_data, val_labels, val_gt_labels = data_loader.load_and_norm_data(train_data_path, label_path=train_label_path, meta_path=meta_data_path, is_train=True, is_meta=True, ratio=config['noise_ratio'], dataset=config['dataset'])
        test_data, test_labels = data_loader.load_and_norm_data(test_path, is_train=False, dataset=config['dataset'])

        logger.info(f'Total number of data points in train_loader: {len(train_data)}')
        logger.info(f'Total number of data points in val_loader: {len(val_data)}')
        logger.info(f'Total number of data points in test_loader: {len(test_data)}')

        train_dataset = data_loader.Data_Process(data=train_data,
                                     label=train_labels,
                                     label_gt=train_gt_labels,
                                     train=True
                                     )
        val_dataset = data_loader.Data_Process(data=val_data,
                                     label=val_labels,
                                     label_gt=val_gt_labels,
                                     train=True
                                     )
        test_dataset = data_loader.Data_Process(data=test_data,
                                     label=None,
                                     label_gt=test_labels,
                                     train=False
                                     )

        # # obtain potent verctor
        DeepRe(config, train_dataset, test_dataset, logger)



