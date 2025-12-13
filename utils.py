from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys


def class_weight():
    # # ids18
    # class_counts = np.array([83145, 12533, 40486, 12848, 99709, 22584])
    # # tls23
    # class_counts = np.array([3920, 3838, 2626, 2613, 2604, 2549])
    # # ano16
    # class_counts = np.array([1249996, 977623])
    # # ids18c1
    # class_counts = np.array([88028, 72000, 65960, 6865, 4523, 4308])
    # ids18c
    class_counts = np.array([73357, 60000, 54967, 11442, 7539, 7180])
    # classes = np.arange(2)  # 标签0-5

    classes = np.arange(6)  # 标签0-5

    # 创建完整的标签数组（每个标签重复出现对应次数）
    all_labels = []
    for i, count in enumerate(class_counts):
        all_labels.extend([i] * count)
    all_labels = np.array(all_labels)

    # 计算平衡权重
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=all_labels
    )
    # 输出结果
    for i, (count, weight) in enumerate(zip(class_counts, weights)):
        print(f"标签 {i}: 样本数 {count}, 权重 {weight:.4f}")


def heatmap_drag(df_cm):
    categories = ['Benign', 'Bot', 'BruteForce', 'DoS Hulk', 'HTTP DDoS', 'Infilteration']

    # for i, j in zip(y_gt, y_noi):
    #     y_gt_name.append(categories[i])
    #     y_noi_name.append(categories[j])
    # y_gt_name = np.array(y_gt_name)
    # y_noi_name = np.array(y_noi_name)
    # df_cm = pd.crosstab(y_gt_name, y_noi_name,
    #                     rownames=['Actual'], colnames=['Predicted'],
    #                     dropna=False, margins=False, normalize='index').round(4)
    # print(df_cm.to_dict())

    # ask1  'BruteForce': {'Benign': 0.0, 'Bot': 0, 'BruteForce': 0.0, 'DoS Hulk': 0., 'HTTP DDoS': 0.0, 'Infilteration': 0}
    # ==========
    # ask3  'Infilteration': {'Benign': 0.0, 'Bot': 0, 'BruteForce': 0.0, 'DoS Hulk': 0., 'HTTP DDoS': 0.0, 'Infilteration': 0}

    df_cm = pd.DataFrame()

    df_cm = df_cm[categories]
    df_cm = df_cm.reindex(categories)
    df_cm = df_cm.round(4)

    plt.figure(figsize=(16, 10))
    sns.set(font_scale=2)
    sns.heatmap(df_cm, cmap="viridis", annot=True, annot_kws={"size": 16})
    # sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    # 设置x、y轴标签为类别名称
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    # plt.xticks([])
    ####解决保存图时坐标文字显示不全#######
    plt.ylabel('')
    plt.xlabel('')
    plt.tight_layout()
    png_path = './results/imgs/ori.png'
    plt.show()



import logging


def setup_realtime_logging(log_file="output.log"):
    """配置实时日志，同时输出到控制台和文件"""
    # 创建日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 格式化日志（包含时间戳）
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（实时写入）
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    # 设置无缓冲模式
    file_handler.stream = open(log_file, 'a', 1)  # 1表示行缓冲
    logger.addHandler(file_handler)

    return logger



def analyze_confusion_matrix(confusion_matrix):
    # 样本数量
    sample_counts = confusion_matrix.sum(axis=1)

    # 准确率（Accuracy）
    accuracy_per_class = np.diag(confusion_matrix) / sample_counts * 100

    # 召回率（Recall）
    recall_per_class = np.diag(confusion_matrix) / sample_counts * 100

    # 将结果存储在一个字典中
    results = {
        "Class": np.arange(len(sample_counts)),
        "Sample Count": sample_counts,
        "Accuracy": accuracy_per_class,
        "Recall": recall_per_class
    }

    # 转换为结构化数组以便排序
    dtype = [('Class', int), ('Sample Count', int), ('Accuracy', float), ('Recall', float)]
    structured_results = np.array([tuple(results[key][i] for key in results) for i in range(len(sample_counts))],
                                  dtype=dtype)

    # 按样本数量排序
    sorted_results = np.sort(structured_results, order='Sample Count')

    # 按20%、30%、50%比例划分
    n = len(sample_counts)
    few_threshold = int(n * 0.5)
    medium_threshold = int(n * 0.8)

    categories = np.empty(n, dtype='<U6')  # 字符串数组用于存储分类结果
    categories[:few_threshold] = 'few'
    categories[few_threshold:medium_threshold] = 'medium'
    categories[medium_threshold:] = 'many'

    group_acc = [np.mean(sorted_results['Accuracy'][:few_threshold]), np.mean(sorted_results['Accuracy'][few_threshold:medium_threshold]), np.mean(sorted_results['Accuracy'][medium_threshold:])]
    # group_recall = [np.mean(sorted_results['Recall'][:few_threshold]),
    #              np.mean(sorted_results['Recall'][few_threshold:medium_threshold]),
    #              np.mean(sorted_results['Recall'][medium_threshold:])]

    # print(group_acc)
    return np.mean(group_acc), group_acc, sorted_results['Accuracy']

# confusion_matrix =
# analyze_confusion_matrix(confusion_matrix)

#
# class_weight()
# count_AAEC_wsith_K(2,20)
# count_gmodel()
# count_ptest('iid')
# count_ptest('pair')
# count_ptest('bais')
# count_detecter_model(2,20)