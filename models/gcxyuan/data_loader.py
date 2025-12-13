import numpy as np
import torch
from sklearn.preprocessing import normalize

# sydeal  dataset
# def load_and_norm_data(data_path, label_path = None, meta_path = None, is_train=True, is_meta=False, random_seed=42, ratio=0.2, dataset='tls23'):
#     """
#     加载数据并进行预处理，根据train参数决定是否划分验证集
#
#     参数:
#     data_path (str): 数据文件路径
#     train (bool): 是否为训练模式，默认为True
#     random_seed (int): 随机种子，确保结果可复现
#
#     返回:
#     tuple: 根据train参数返回不同的结果
#         - 如果train为True，返回(train_data, train_labels, val_data, val_labels)
#         - 如果train为False，返回(test_data, test_labels)
#     """
#     # 设置随机种子
#     np.random.seed(random_seed)
#
#     if is_train:
#         if dataset == 'ids17c':
#             data = np.load(data_path)
#             features = data['data']
#             normalized_features = features  # no need
#             labels = data['label']
#             random_indices = np.random.choice(features.shape[0],
#                                               int(0.1 * features.shape[0]), replace=False)
#             normalized_features = normalized_features[random_indices, :]
#             labels = labels[random_indices]
#
#             # 合并归一化特征和标签
#             processed_data = np.hstack((normalized_features, labels.reshape(-1, 1)))
#
#             noise_data = np.load(label_path)
#             noise_labels = {field: noise_data[field] for field in noise_data.dtype.names}
#             noi_label_name = 'y' + str(ratio) if ratio > 0 else 'y'
#             noise_labels = noise_labels[noi_label_name]
#             noise_labels = noise_labels[random_indices]
#
#             del data, noise_data
#
#         elif dataset == 'tls23':
#             data = np.load(data_path)
#             features = data[:, :-1]  # 除最后一列外的所有列
#             labels = data[:, -1].astype(np.int32)  # 最后一列作为标签
#             normalized_features = features  # no need
#             # 合并归一化特征和标签
#             processed_data = np.hstack((normalized_features, labels.reshape(-1, 1)))
#
#             noise_data = np.load(label_path)
#             noise_labels = {field: noise_data[field] for field in noise_data.dtype.names}
#             noi_label_name = 'y' + str(ratio) if ratio > 0 else 'y'
#             noise_labels = noise_labels[noi_label_name]
#
#             del data, noise_data
#
#         else:
#             # # 对特征进行归一化处理（按行）
#             # normalized_features = normalize(features, norm='l2', axis=1)
#             raise NotImplementedError
#
#         # 分离特征和标签
#         train_split = processed_data
#         train_noilabels_split = noise_labels
#         train_data, train_labels, train_noilabels = train_split[:, :-1], train_split[:, -1], train_noilabels_split
#
#
#         if is_meta:
#             if dataset == 'tls23':
#                 data = np.load(meta_path)
#                 features = data['data']
#                 normalized_features = features
#                 val_data = normalized_features
#                 val_labels = data['label']
#                 val_noilabels = data['label']
#
#             elif dataset == 'ids17c':
#                 # 训练模式：划分训练集和验证集
#                 indices = np.arange(len(processed_data))
#                 np.random.shuffle(indices)
#
#                 val_size = int(0.1 * len(processed_data))
#                 val_indices = indices[:val_size]
#
#                 val_split = processed_data[val_indices]
#                 val_noilabels_split = noise_labels[val_indices]
#
#                 val_data, val_labels, val_noilabels = val_split[:, :-1], val_split[:, -1], val_noilabels_split
#
#             return train_data, train_noilabels, train_labels, val_data, val_noilabels, val_labels
#
#         return train_data, train_noilabels, train_labels
#
#     else:
#         # 测试模式：直接返回处理后的数据
#         if dataset == 'ids17c':
#             data = np.load(data_path)
#             features = data['data']
#             normalized_features = features  # no need
#             labels = data['label']
#             random_indices = np.random.choice(features.shape[0],
#                                               int(0.1 * features.shape[0]), replace=False)
#             normalized_features = normalized_features[random_indices, :]
#             labels = labels[random_indices]
#
#         elif dataset == 'tls23':
#             data = np.load(data_path)
#             features = data[:, :-1]  # 除最后一列外的所有列
#             labels = data[:, -1].astype(np.int32)  # 最后一列作为标签
#             normalized_features = features  # no need
#
#         else:
#             raise NotImplementedError
#
#         return normalized_features, labels


def load_and_norm_data(data_path, label_path = None, meta_path = None, is_train=True, is_meta=False, random_seed=42, ratio=0.2, dataset='tls23'):
    """
    加载数据并进行预处理，根据train参数决定是否划分验证集

    参数:
    data_path (str): 数据文件路径
    train (bool): 是否为训练模式，默认为True
    random_seed (int): 随机种子，确保结果可复现

    返回:
    tuple: 根据train参数返回不同的结果
        - 如果train为True，返回(train_data, train_labels, val_data, val_labels)
        - 如果train为False，返回(test_data, test_labels)
    """
    # 设置随机种子
    np.random.seed(random_seed)

    if is_train:
        if dataset == 'ids17c':
            data = np.load(data_path)
            features = data['X_train']
            normalized_features = features  # no need
            labels = data['y_train']
            random_indices = np.random.choice(features.shape[0],
                                              int(0.1 * features.shape[0]), replace=False)
            normalized_features = normalized_features[random_indices, :]
            labels = labels[random_indices]

            # 合并归一化特征和标签
            processed_data = np.hstack((normalized_features, labels.reshape(-1, 1)))

            noise_data = np.load(label_path)
            noise_labels = noise_data['y' + str(int(ratio*10))]
            noise_labels = noise_labels[random_indices]

            del data, noise_data

        elif dataset == 'tls23':
            data = np.load(data_path)
            features = data['X_train']
            normalized_features = features  # no need
            labels = data['y_train']
            normalized_features = normalized_features
            labels = labels

            # 合并归一化特征和标签
            processed_data = np.hstack((normalized_features, labels.reshape(-1, 1)))

            noise_data = np.load(label_path)
            noise_labels = noise_data['y' + str(int(ratio*10))]

            del data, noise_data

        else:
            # # 对特征进行归一化处理（按行）
            # normalized_features = normalize(features, norm='l2', axis=1)
            raise NotImplementedError

        # 分离特征和标签
        train_split = processed_data
        train_noilabels_split = noise_labels
        train_data, train_labels, train_noilabels = train_split[:, :-1], train_split[:, -1], train_noilabels_split


        if is_meta:
            if dataset == 'tls23' or dataset == 'ids17c':
                data = np.load(meta_path)
                features = data['X_meta']
                normalized_features = features
                val_data = normalized_features
                val_labels = data['y_meta']
                val_noilabels = data['y_meta']

            return train_data, train_noilabels, train_labels, val_data, val_noilabels, val_labels

        return train_data, train_noilabels, train_labels

    else:
        # 测试模式：直接返回处理后的数据
        if dataset == 'ids17c':
            data = np.load(data_path)
            features = data['X_test']
            normalized_features = features  # no need
            labels = data['y_test']
            random_indices = np.random.choice(features.shape[0],
                                              int(0.1 * features.shape[0]), replace=False)
            normalized_features = normalized_features[random_indices, :]
            labels = labels[random_indices]

        elif dataset == 'tls23':
            data = np.load(data_path)
            features = data['X_test']
            normalized_features = features  # no need
            labels = data['y_test']
            normalized_features = normalized_features
            labels = labels

        else:
            raise NotImplementedError

        return normalized_features, labels

# class Data_Process():
#     # print(data)
#     def __init__(self, data, label, label_gt, train=True):
#         self.train = train  # training set or test set
#
#         if self.train:
#             self.train_data = torch.tensor(data, dtype=torch.float32)
#             self.train_labels = torch.tensor(label_gt, dtype=torch.int64)
#             self.train_noisy_labels = torch.tensor(label, dtype=torch.int64)
#             self.noise_or_not = torch.tensor([self.train_noisy_labels[i] == self.train_labels[i]
#                                               for i in range(self.train_noisy_labels.shape[0])])
#         else:
#             self.test_data = torch.tensor(data, dtype=torch.float32)
#             self.test_labels = torch.tensor(label_gt, dtype=torch.int64)
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         if self.train:
#             img, target = self.train_data[index], self.train_noisy_labels[index]
#         else:
#             img, target = self.test_data[index], self.test_labels[index]
#
#         return img, target, index
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data)
#         else:
#             return len(self.test_data)

class Data_Process():
    # print(data)
    def __init__(self, data, label, label_gt, train=True):
        self.train = train  # training set or test set

        if self.train:
            self.train_data = torch.tensor(data, dtype=torch.float32)
            self.train_labels = torch.tensor(label, dtype=torch.int64)
            self.train_gt_labels = torch.tensor(label_gt, dtype=torch.int64)
            self.noise_or_not = torch.tensor([self.train_gt_labels[i] == self.train_labels[i]
                                              for i in range(self.train_gt_labels.shape[0])])
        else:
            self.test_data = torch.tensor(data, dtype=torch.float32)
            self.test_labels = torch.tensor(label_gt, dtype=torch.int64)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)