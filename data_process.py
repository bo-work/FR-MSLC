'''
almost follow the code in DSDIR, but remain groundtrue labels and save file space
'''




import glob
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="根据命令行参数构造文件名。")

# 添加 corruption_ratio 参数
parser.add_argument('--dataset_name', type=str, default='BoAu')

# 添加 corruption_ratio 参数
parser.add_argument('--noise_type', type=str, default='asym')



# 解析命令行参数
args = parser.parse_args()

# preprocess tls23 dataset
args.dataset_name = 'tls23'
args.noise_type = 'asym'
args.data_dir ='./data/tls23'
NOISEDATA_SAVE_FOLDER = args.data_dir


# preprocess ids17c dataset
args.dataset_name = 'ids17c'
args.noise_type = 'asym'
args.data_dir ='./data/ids17c'
args.save_dir = f'./data/feat/{args.dataset_name}/'
NOISEDATA_SAVE_FOLDER = args.data_dir
# args.TRAFFIC_LABEL = {'BENIGN': 0,
#                  'FTP-BruteForce': 1, 'SSH-BruteForce': 1,
#                  'DoS GoldenEye': 2, 'DoS Slowloris': 2,
#                  'DoS Slowhttptest': 2, 'DoS Hulk': 2,
#                  'Infilteration': 3,
#                  'Infiltration - Dropbox Download':3, 'Infiltration - NMAP Portscan':3,
#                  'Infiltration - Communication Victim Attacker':3,
#                  'DDoS-LOIC-HTTP': 4, 'DDoS-LOIC-UDP': 4, 'DDoS-HOIC': 4,
#                  'Web Attack - Brute Force': 5,
#                  'Web Attack - XSS': 5,
#                  'Botnet Ares': 6,
#                  'Web Attack - SQL': 7
#                  }
# args.re_LABEL = {0: 0, 1: 6, 2:4, 3:2,  4:2, 5:2, 6:2, 7:1, 8:7, 9:3, 10:3, 11:3, 12:1, 13:5, 14:5, 15:5}
# args.name_re_LABEL = {'benign': 0,
#                       'Botnet': 6,
#                       'DDoS': 4,
#                       'DoS GoldenEye': 2, 'DoS Hulk': 2, 'DoS Slowhttptest': 2, 'DoS Slowloris': 2,
#                       'FTP-Patator': 1, 'SSH-Patator': 1,
#                       'Infiltration': 3, 'Infiltration - Portscan': 3, 'Portscan': 3,
#                       'Web Attack - Brute Force': 5, 'Web Attack - SQL Injection': 5, 'Web Attack - XSS': 5,
#                       'Heartbleed': 7,}
args.re_LABEL = {0: 0, 1: 5, 2:4, 3:2, 4:2, 5:2, 6:2, 7:1, 9:3, 10:3, 11:3, 12:1}
args.del_LABEL = [13, 14, 15, 8]
args.name_re_LABEL = {'benign': 0,
                      'Botnet': 5,
                      'DDoS': 4,
                      'DoS GoldenEye': 2, 'DoS Hulk': 2, 'DoS Slowhttptest': 2, 'DoS Slowloris': 2,
                      'FTP-Patator': 1, 'SSH-Patator': 1,
                      'Infiltration': 3, 'Infiltration - Portscan': 3, 'Portscan': 3}





print(args)


class CICIDS2017Preprocessor(object):
    def __init__(self, data_name, data_path, training_size, validation_size, testing_size):
        self.data_name = data_name
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size

        self.data = None
        self.features = None
        self.labels = None
        self.labels_encoded = []  # labels_encoded[i]表示数字i对应的label是什么

    def read_data(self):
        """"""
        filenames = []
        if not os.path.exists(self.data_path):
            print(f"错误：路径不存在 - {self.data_path}")
        else:
            print(f"路径存在：{self.data_path}")
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.lower().endswith('.csv'):
                        filenames.append(os.path.join(root, file))
        print(self.data_path)
        print(filenames)
        datasets = [pd.read_csv(filename, encoding='latin1', low_memory=True) for filename in filenames]

        # Remove white spaces and rename the columns
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # Concatenate the datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        # self.data.drop(labels=['fwd_header_length.1'], axis=1, inplace=True)

    def _clean_column_name(self, column):
        """"""
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column

    def remove_duplicate_values(self):
        """"""
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        """"""
        # Remove missing values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        """"""
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        self.data.dropna(axis=0, how='any', inplace=True)

    def find_non_numeric_columns(self):  # 删除非数值列
        label_columns = ['label', 'label_category']
        self.data = self.data.drop(columns=label_columns)
        # print(self.data.head(1))
        # non_numeric_columns = self.data.select_dtypes(exclude=['number']).columns  # 找到非数值列
        # self.non_numeric_data = self.data[non_numeric_columns]  # 存储非数值列
        # self.data.drop(columns=non_numeric_columns, inplace=True)  # 删除非数值列
        # return non_numeric_columns

    def remove_constant_features(self, threshold=0.01):
        """"""
        # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
        data_std = self.data.std(numeric_only=True)

        # Find Features that meet the threshold
        constant_features = [column for column, std in data_std.items() if std < threshold]

        # Drop the constant features
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features_pre(self, threshold=0.98):
        """"""
        # Correlation matrix
        data_corr = self.data.corr()

        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find Features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.9):
        """
        Remove highly correlated features from the data.
        """
        # Select only the numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])

        # Correlation matrix
        data_corr = numeric_data.corr()
        # print(data_corr)
        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        """"""
        # Proposed Groupings
        attack_group = None
        # if self.data_name == 'CIC-IDS-2017':
        #     attack_group = {
        #         'BENIGN': 'Benign',
        #         'PortScan': 'PortScan',
        #         'DDoS': 'DoS/DDoS',
        #         'DoS Hulk': 'DoS/DDoS',
        #         'DoS GoldenEye': 'DoS/DDoS',
        #         'DoS slowloris': 'DoS/DDoS',
        #         'DoS Slowhttptest': 'DoS/DDoS',
        #         'Heartbleed': 'DoS/DDoS',
        #         'FTP-Patator': 'Brute Force',
        #         'SSH-Patator': 'Brute Force',
        #         'Bot': 'Botnet ARES',
        #         'Web Attack =Brute Force': 'Web Attack',
        #         'Web Attack =Sql Injection': 'Web Attack',
        #         'Web Attack =XSS': 'Web Attack',
        #         'Infiltration': 'Infiltration'
        #     }





        # Create grouped label column
        if attack_group is not None:
            self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
        else:
            self.data['label_category'] = self.data['label']

        if self.data_name == 'ids17c':
            self.data.loc[self.data['attempted_category'] != -1, 'label_category'] = 'BENIGN'
            self.data = self.data.drop(labels=['attempted_category'], axis=1)

        self.labels = self.data['label_category']
        self.features = self.data.drop(labels=['label', 'label_category'], axis=1)


    def train_valid_test_split(self, seed=42):
        """"""

        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.labels,
            test_size=(self.validation_size + self.testing_size),
            random_state=seed,
            stratify=self.labels
        )
        if self.validation_size > 0:
            X_test, X_val, y_test, y_val = train_test_split(
                X_test,
                y_test,
                test_size=self.validation_size / (self.validation_size + self.testing_size),
                random_state=seed
            )
        else:
            X_val, y_val = None, None
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def scale(self, training_set, validation_set, testing_set):
        """"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set

        categorical_features = X_train.select_dtypes(exclude=["number"]).columns
        numeric_features = X_train.select_dtypes(exclude=[object]).columns

        preprocessor = ColumnTransformer(transformers=[
            # ('categoricals', OrdinalEncoder(), categorical_features),
            ('numericals', QuantileTransformer(), numeric_features)
            # ('numericals', 'passthrough', numeric_features)
        ])
        preprocessor2 = ColumnTransformer(transformers=[

            # ('numericals', 'passthrough', numeric_features)
        ])

        # Preprocess the features
        columns_numeric = numeric_features.tolist()
        columns_categorical = categorical_features.tolist()

        print('数值列:', len(columns_numeric), columns_numeric)
        print('非数值列:', len(columns_categorical), columns_categorical)
        columns = columns_numeric # 全部的列

        print()

        # print(X_train.shape)
        # tmp = preprocessor.fit_transform(X_train)
        # print(tmp.shape)
        # exit(0)
        print(X_train.shape)
        print(X_test.shape)

        # 不能执行数值列的归一化

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=columns)

        # X_train = pd.DataFrame(preprocessor2.fit_transform(X_train), columns=columns)
        # X_test = pd.DataFrame(preprocessor2.fit_transform(X_test), columns=columns)

        # 获取生成的数值列名
        columns_numeric = numeric_features
        # 获取生成的所有列名
        columns = preprocessor.get_feature_names_out()

        # X_train = pd.DataFrame(preprocessor2.fit_transform(X_train), columns=columns_categorical)
        # X_test = pd.DataFrame(preprocessor2.fit_transform(X_test), columns=columns_categorical)

        # X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        # if X_val is not None:
        #     X_val = pd.DataFrame(preprocessor.fit_transform(X_val), columns=columns)
        # X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=columns)

        # 定义标签映射字典
        # 手动列出 'benign' 的12种大小写变体
        # 输入字符串
        import itertools

        input_str = "benign"

        # 获取所有可能的组合
        combinations = list(itertools.product(*[(char.lower(), char.upper()) for char in input_str]))

        # 将组合列表转换为字符串列表
        result = [''.join(combination) for combination in combinations]

        # 创建映射字典
        label_mapping = {combination: 0 for combination in result}

        # 输出结果
        for key in label_mapping:
            print(f"'{key}': {label_mapping[key]}")


        # 获取所有唯一的标签，并按字典序排序
        all_labels = sorted(set(y_train) | set(y_test))

        # 初始化计数器，跳过已经映射的 'benign' 和 'BENIGN'
        self.labels_encoded.append('benign')
        counter = 1
        for label in all_labels:
            if label not in label_mapping:
                label_mapping[label] = counter
                counter += 1
                self.labels_encoded.append(label)

        print(label_mapping)

        # 应用标签映射规则
        y_train = pd.DataFrame([label_mapping.get(label, 0) for label in y_train], columns=["label"])
        y_test = pd.DataFrame([label_mapping.get(label, 0) for label in y_test], columns=["label"])


        # 输出结果查看
        # print("y_train:")
        # print(y_train)
        # print("\ny_test:")
        # print(y_test)

        # 查看每个数字对应的原始label
        # self.labels_encoded.append('benign')
        # self.labels_encoded.append('malicious')


        # 获取label编码，len就是label数量
        # print(self.labels_encoded)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_data(data_dir, data_name, train_size, val_size, test_size, batch_size, seed, sample_strategy,
              target_strategy):  # 路径名、数据集名称、训练集比例、验证集比例、测试集比例，0、1、2：欠采样、重采样、欠+重，0:中位数、1:平均数
    # cicids2017 = CICIDS2017Preprocessor(
    #     data_name=data_name,
    #     data_path=dir_name,
    #     training_size=train_size,
    #     validation_size=val_size,
    #     testing_size=test_size
    # )

    cicids2017 = CICIDS2017Preprocessor(
        data_name=data_name,
        data_path=data_dir,
        # data_path='/home/maybo/Documents/Project/Paper/etc-longtail-py/dataset/ids17/TrafficLabelling',
        # data_path='/home/maybo/Documents/Project/Paper/etc-longtail-py/val-DSDIR/data/dataset/tls23',
        training_size=train_size,
        validation_size=val_size,
        testing_size=test_size
    )

    # Read datasets
    cicids2017.read_data()  # 读数据
    # cicids2017.print_data(1, 0)

    # Remove NaN, -Inf, +Inf, Duplicates
    cicids2017.remove_duplicate_values()  # 删除重复列
    cicids2017.remove_missing_values()  # 删除缺失值行
    cicids2017.remove_infinite_values()  # 删除无穷值
    # cicids2017.print_data(1)
    # Drop constant & correlated features
    cicids2017.remove_constant_features()  # 删除常量
    cicids2017.remove_correlated_features()  # 删除强相关变量
    # cicids2017.print_data(1)
    # Create new label category
    cicids2017.group_labels()  # 重定义label, correct attempted_category
    # cicids2017.print_data(5, 0)
    # cicids2017.find_non_numeric_columns()  # 去掉非数值列，不然后边的报错

    # Split & Normalise data sets
    training_set, validation_set, testing_set = cicids2017.train_valid_test_split(seed=seed)  # 训练集、测试集划分

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cicids2017.scale(training_set, validation_set,
                                                                            testing_set)  # 归一化


    print(type(X_train), type(y_train))

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # 将 y_train 转换为一维数组，并找出唯一值
    unique_labels, counts = np.unique(y_train, return_counts=True)

    # 输出唯一值和每个唯一值的出现次数
    print("Unique labels:", unique_labels)
    print("Train Counts:", counts)
    print("Test Counts:", np.unique(y_test, return_counts=True)[1])

    if args.dataset_name == 'ids17c':
        train_set = np.concatenate((X_train, y_train), axis=1)
        test_set = np.concatenate((X_test, y_test), axis=1)

        print("Train data set:", X_train.shape)
        print("Test data set:", X_test.shape)

        # 构造完整的文件路径
        train_file_path = f'./data/feat/{args.dataset_name}/clean_train.npz'
        test_file_path = f'./data/feat/{args.dataset_name}/clean_test.npz'

        for file_path in [train_file_path, test_file_path]:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)

        # 保存数据到文件
        np.savez(train_file_path, data=X_train, label = y_train)
        np.savez(test_file_path, data=X_test, label = y_test)

    else:
        train_set = np.concatenate((X_train, y_train), axis=1)
        test_set = np.concatenate((X_test, y_test), axis=1)

        print("Train set:", train_set.shape)
        print("Test set:", test_set.shape)

        # 构造完整的文件路径
        train_file_path = f'./data/feat/{args.dataset_name}/clean_train.npy'
        test_file_path = f'./data/feat/{args.dataset_name}/clean_test.npy'

        for file_path in [train_file_path, test_file_path]:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)

        # 保存数据到文件
        np.save(train_file_path, train_set)
        np.save(test_file_path, test_set)



def set_noise(train_file_dir, seed, corruption_ratios, noise_type):

    if args.dataset_name == 'ids17c':
        train_file_path = os.path.join(train_file_dir, 'clean_train.npz')
        train_set = np.load(train_file_path)
        y_train = train_set['label']
    else:
        train_file_path = os.path.join(train_file_dir, 'clean_train.npy')
        train_set = np.load(train_file_path)
        y_train = train_set[:,-1]

    y_train_noises = {}

    from collections import Counter

    # 将 y_train 二维数组展平为一维数组
    y_train_flattened = y_train.reshape(-1).astype(np.int32)
    y_train_noises['y'] = y_train_flattened

    # 使用 Counter 统计每个类别的出现次数
    cnt_per_class_pre = Counter(y_train_flattened)
    print(cnt_per_class_pre)


    np.random.seed(seed)

    benign_idx = 0
    bef_be_num = sum(y_train == benign_idx)

    print('良性:', bef_be_num)
    print('恶意:', len(y_train) - bef_be_num)

    if noise_type == 'asym':
        # 良性数据是不变的，恶意的翻转过来。
        for ratio in corruption_ratios:
            y_train_noise = y_train_flattened.copy()
            for idx, family in enumerate(np.unique(y_train_flattened)):
                if family == benign_idx:
                    continue
                family_idx = np.where(y_train_noise == family)[0]
                family_size = len(family_idx)
                filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
                y_train_noise[filter_idx] = benign_idx
                print(
                    f'ratio: {ratio}\tfamily: {family}\tstart: {family_size}\tpot: {len(filter_idx)}\tend: {family_size - len(filter_idx)}')
                print(f'y_train_family: {Counter(y_train_noise)}\n\n')
            y_train_noises['y'+str(ratio)] = y_train_noise

            bef_be_num = sum(y_train_noise == benign_idx)
            print('良性(带恶意噪声):', bef_be_num)
            print('恶意(带良性噪声):', len(y_train_noise) - bef_be_num)

    elif noise_type == 'sym':
        family_list = np.unique(y_train_flattened)
        for ratio in corruption_ratios:
            y_train_ori = y_train_flattened.copy()
            y_train_noise = y_train_flattened.copy()
            for idx, family in enumerate(family_list):
                if family == benign_idx:
                    family_idx = np.where(y_train_ori == family)[0]
                    family_size = len(family_idx)
                    filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
                    np.random.shuffle(filter_idx)
                    filter_idx_list = np.array_split(filter_idx, len(family_list) - 1)
                    i = 0
                    for idx1, family1 in enumerate(family_list):
                        if family1 == 0:
                            continue
                        y_train_noise[filter_idx_list[i]] = family1
                        i += 1
                    print(
                        f'ratio: {ratio}\tfamily: {family}\tstart: {family_size}\tend: {family_size - len(filter_idx)}')
                else:
                    family_idx = np.where(y_train_ori == family)[0]
                    family_size = len(family_idx)
                    filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
                    y_train_noise[filter_idx] = benign_idx
                    print(
                        f'ratio: {ratio}\tfamily: {family}\tstart: {family_size}\tend: {family_size - len(filter_idx)}')
                print(f'y_train_family: {Counter(y_train_noise)}\n\n')
            y_train_noises['y'+str(ratio)] = y_train_noise

            bef_be_num = sum(y_train_noise == benign_idx)
            print('良性(带恶意噪声):', bef_be_num)
            print('恶意(带良性噪声):', len(y_train_noise) - bef_be_num)

    elif noise_type == 'ask':
        print('error noise type')

    # 构造完整的文件路径
    noilabel_file_path = f'./data/feat/{args.dataset_name}/train_noiselabels_' + str(args.noise_type) + '.npy'

    for file_path in [noilabel_file_path]:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

    # 保存数据到文件
    dt = np.dtype([(key, 'i8') for key in y_train_noises.keys()])
    structured_array = np.array(
        list(zip(*y_train_noises.values())),
        dtype=dt
    )
    np.save(noilabel_file_path, structured_array)

    print("Datasets saved successfully.")


def reset_label(data_dir, map_dict, map_del_dict= None):

    def relabel(data_set, map_dict, map_del_dict= None):
        # 提取最后一列
        other_cols = data_set['data']
        ori_label = data_set['label']
        del data_set
        ori_label = ori_label.reshape(-1).astype(np.int32)
        # 统计原始最后一列的分布
        original_dist = Counter(ori_label)
        print(original_dist)

        if map_del_dict is not None:
            mask = ~np.isin(ori_label, np.array(map_del_dict))
            other_cols = other_cols[mask]
            ori_label = ori_label[mask]
            # 统计原始最后一列的分布
            print('after del dist: ')
            original_dist = Counter(ori_label)
            print(original_dist)

        last_col = ori_label.copy()

        # 创建映射后的列
        mapped_col = np.array([map_dict[value] for value in last_col])

        # 统计映射后的分布
        mapped_dist = Counter(mapped_col)
        print(mapped_dist)

        return other_cols, mapped_col, ori_label

    train_file_path = os.path.join(data_dir, 'clean_train.npz')
    train_set = np.load(train_file_path)
    train_data, re_label, ori_label = relabel(train_set, map_dict, map_del_dict)
    np.savez(train_file_path, data=train_data, label=re_label, ori_label = ori_label)

    test_file_path = os.path.join(data_dir, 'clean_test.npz')
    test_set = np.load(test_file_path)
    test_data, re_label, ori_label = relabel(test_set, map_dict, map_del_dict)
    np.savez(test_file_path, data=test_data, label=re_label, ori_label = ori_label)

def reset_data_label_to_npz():
#     load data
    if args.dataset_name == 'tls23':
        args.read_dir = '/home/maybo/project/etc-longtail-py/task/sydeal/data/feat/tls23/'
        args.save_dir = './data/tls23/'

        data = np.load(os.path.join(args.read_dir, 'clean_train.npy'))
        features = data[:, :-1]  # 除最后一列外的所有列
        labels = data[:, -1].astype(np.int32)  # 最后一列作为标签
        np.savez(os.path.join(args.save_dir, 'clean_train.npz'), X_train=features, y_train=labels)

        data = np.load(os.path.join(args.read_dir, 'clean_test.npy'))
        features = data[:, :-1]  # 除最后一列外的所有列
        labels = data[:, -1].astype(np.int32)  # 最后一列作为标签
        np.savez(os.path.join(args.save_dir, 'clean_test.npz'), X_test=features, y_test=labels)

        noise_data = np.load(os.path.join(args.read_dir, 'train_noiselabels_asym.npy'))
        noise_labels = {field: noise_data[field] for field in noise_data.dtype.names}
        np.savez(os.path.join(args.save_dir, 'noise_asym_labels.npz'), y0=noise_labels['y'], y2=noise_labels['y0.2'], y4=noise_labels['y0.4'], y6=noise_labels['y0.6'], y8=noise_labels['y0.8'])

        noise_data = np.load(os.path.join(args.read_dir, 'train_noiselabels_sym.npy'))
        noise_labels = {field: noise_data[field] for field in noise_data.dtype.names}
        np.savez(os.path.join(args.save_dir, 'noise_sym_labels.npz'), y0=noise_labels['y'],
                 y2=noise_labels['y0.2'], y4=noise_labels['y0.4'], y6=noise_labels['y0.6'], y8=noise_labels['y0.8'])

    elif args.dataset_name == 'ids17c':
        args.read_dir = '/home/maybo/project/etc-longtail-py/task/sydeal/data/feat/ids17c/'
        args.save_dir = './data/ids17c/'

        data = np.load(os.path.join(args.read_dir, 'clean_train.npz'))
        features = data['data']  # 除最后一列外的所有列
        labels = data['label'].astype(np.int32)  # 最后一列作为标签
        np.savez(os.path.join(args.save_dir, 'clean_train.npz'), X_train=features, y_train=labels)

        data = np.load(os.path.join(args.read_dir, 'clean_test.npz'))
        features = data['data']  # 除最后一列外的所有列
        labels = data['label'].astype(np.int32)  # 最后一列作为标签
        np.savez(os.path.join(args.save_dir, 'clean_test.npz'), X_test=features, y_test=labels)

        noise_data = np.load(os.path.join(args.read_dir, 'train_noiselabels_asym.npy'))
        noise_labels = {field: noise_data[field] for field in noise_data.dtype.names}
        np.savez(os.path.join(args.save_dir, 'noise_asym_labels.npz'), y0=noise_labels['y'],
                 y2=noise_labels['y0.2'], y4=noise_labels['y0.4'], y6=noise_labels['y0.6'], y8=noise_labels['y0.8'])

        noise_data = np.load(os.path.join(args.read_dir, 'train_noiselabels_sym.npy'))
        noise_labels = {field: noise_data[field] for field in noise_data.dtype.names}
        np.savez(os.path.join(args.save_dir, 'noise_sym_labels.npz'), y0=noise_labels['y'],
                 y2=noise_labels['y0.2'], y4=noise_labels['y0.4'], y6=noise_labels['y0.6'], y8=noise_labels['y0.8'])



#  load label



def balance_getmeta(noisy_SAVE_FOLDER, save_name, size, save=True):
    data_file_path = os.path.join(noisy_SAVE_FOLDER, 'clean_train.npz')
    raw_data = np.load(data_file_path)
    X_train, y_train = raw_data['X_train'], raw_data['y_train']

    family_sample_num = size

    for idx, family in enumerate(np.unique(y_train)):
        family_idx = np.where(y_train == family)[0]
        filter_idx = np.random.choice(family_idx, size=family_sample_num, replace=False)
        X_train_family = X_train[filter_idx, :]
        y_train_family = y_train[filter_idx]
        print(f'idx: {idx}\tfamily: {family}')
        print(f'X_train_family: {X_train_family.shape}')
        print(f'y_train_family: {Counter(y_train_family)}\n\n')
        if idx == 0:
            X_train_sampling = X_train_family
            y_train_sampling = y_train_family
        else:
            X_train_sampling = np.concatenate((X_train_sampling, X_train_family), axis=0)
            y_train_sampling = np.concatenate((y_train_sampling, y_train_family), axis=0)

    X_meta, y_meta = X_train_sampling, y_train_sampling

    print(f'X_meta: {X_meta.shape}, y_meta: {y_meta.shape}')
    print(f'y_meta labels: {Counter(y_meta)}')
    if save:
        np.savez_compressed(os.path.join(noisy_SAVE_FOLDER, save_name),
                            X_meta=X_meta, y_meta=y_meta, y_gt_meta=y_meta)
        print('generated data file saved')
    else:
        return   X_meta, y_meta


def set_noise_label_ask(noisy_SAVE_FOLDER, ratio=1):
    # 读取数据
    #
    input('Attention!!   you must check the labels you want to change!!!!   u can enter without input')

    # save npz key!!!check
    # label_list = [2, 5]    # tls23
    label_list = [1, 5]    #ids17c

    saved_noise_path = os.path.join(noisy_SAVE_FOLDER, 'noise_sym_labels.npz')
    raw_data = np.load(saved_noise_path)
    y_train = raw_data['y0']
    print(f'before y_train Counter: {Counter(y_train)}')

    # ask标签
    y_noise_dict = {}
    for asklabel in label_list:
        y_train_ori = y_train.copy()
        y_train_noise = y_train.copy()

        orig_dist = np.bincount(y_train_ori, minlength=6)

        asklabel = int(asklabel)
        target_indices = np.where(y_train == asklabel)[0]
        # 如果没有目标标签，直接返回原数组
        if len(target_indices) == 0:
            print(f"警告: 目标标签 {asklabel} 不在数组中")
        # 生成随机数，为每个目标标签位置选择新标签
        # 其他标签列表（不包含目标标签）
        other_labels = [i for i in np.unique(y_train) if i != asklabel]
        # 为每个目标标签位置生成均匀分布的随机选择
        random_choices = np.random.choice(other_labels, size=len(target_indices))
        # 将目标标签位置的值替换为随机选择的新标签
        y_train_noise[target_indices] = random_choices

        # 计算最终标签分布
        final_dist = np.bincount(y_train_noise, minlength=6)

        print(f"原始标签分布: {orig_dist}")
        print(f"处理后标签分布: {final_dist}")
        print(f"修改后各标签的增量: {final_dist - orig_dist}")

        y_noise_dict[asklabel] = y_train_noise

    saved_noise_label_path = os.path.join(noisy_SAVE_FOLDER, 'noise_ask_labels_random.npz')
    np.savez_compressed(saved_noise_label_path, y=y_train, y1=y_noise_dict[1], y5=y_noise_dict[5])

def set_noise_label_ask_toclose(NOISEDATA_SAVE_FOLDER, save_name):
    # 读取数据
    #
    input('Attention!!   you must check the labels you want to change!!!!   u can enter without input')

    # save npz key!!!check
    label_list = [2, 5]    # tls23
    # label_list = [1, 5]    #ids17c

    from sklearn.metrics.pairwise import euclidean_distances
    # 读取数据
    saved_noise_path = os.path.join(NOISEDATA_SAVE_FOLDER, 'clean_train.npz')
    raw_data = np.load(saved_noise_path)
    X_train = raw_data['X_train']
    y_train = raw_data['y_train']
    print(f'before y_train Counter: {Counter(y_train)}')

    """计算每个类别的中心点"""
    class_labels = np.unique(y_train)
    class_centers = {}

    for label in class_labels:
        # 找到该类别的所有样本
        class_samples = X_train[y_train == label]
        # 计算中心点（均值）
        center = np.mean(class_samples, axis=0)
        class_centers[label] = center
        print(f"类别 {label} 的中心点计算完成，样本数: {len(class_samples)}")

    # ask标签
    y_noise_dict = {}
    ask_list = [5, 2]
    for asklabel in ask_list:

        y_train_ori = y_train.copy()
        y_train_noise = y_train.copy()
        orig_dist = np.bincount(y_train_ori, minlength=6)

        ask_mask = y_train_ori == asklabel
        ask_samples = X_train[ask_mask]
        num_ask_samples = len(ask_samples)

        print(f"找到 {num_ask_samples} 个类别为 {asklabel} 的样本，开始重新分配标签...")

        # 获取其他类别的中心
        other_centers = {label: center for label, center in class_centers.items()
                         if label != asklabel}
        other_labels = list(other_centers.keys())
        other_centers_array = np.array(list(other_centers.values()))

        # 计算每个目标样本到其他类别中心的距离
        distances = euclidean_distances(ask_samples, other_centers_array)

        # 找到最近的类别并更新标签
        nearest_indices = np.argmin(distances, axis=1)
        new_labels_for_target = [other_labels[i] for i in nearest_indices]

        # 更新新标签数组
        y_train_noise[ask_mask] = new_labels_for_target

        # 计算最终标签分布
        final_dist = np.bincount(y_train_noise, minlength=6)

        print(f"原始标签分布: {orig_dist}")
        print(f"处理后标签分布: {final_dist}")
        print(f"修改后各标签的增量: {final_dist - orig_dist}")

        y_noise_dict[asklabel] = y_train_noise

    saved_noise_label_path = os.path.join(NOISEDATA_SAVE_FOLDER, 'noise_ask_labels_close.npz')
    np.savez_compressed(saved_noise_label_path, y=y_train, y5=y_noise_dict[5], y2=y_noise_dict[2])


ratio_start = 0.2
ratio_distance = 0.2
ratio_end = 0.8
ratio_list = np.around(np.arange(ratio_start, ratio_end+ratio_distance, ratio_distance), decimals=1)
seed = 0
load_data(args.data_dir, args.dataset_name, 0.8, 0.0, 0.2, 128, seed, 'none', 'none')
if args.dataset_name == 'ids17c':
    reset_label(args.save_dir, args.re_LABEL, args.del_LABEL)
print('============set noise labels===================')
set_noise(f'./data/feat/{args.dataset_name}', seed, ratio_list, args.noise_type)

# unannotated 将要求的某类按照距离划分给其他类别
set_noise_label_ask_toclose(NOISEDATA_SAVE_FOLDER, 'noise_ask_labels_close.npz')
