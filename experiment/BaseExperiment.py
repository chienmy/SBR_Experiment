import logging
import math
import os
import re

import pandas as pd

from encoder import Encoder
from model import Model


class BaseExperiment:
    """
    实验基类
    """
    # ====实验参数设置====
    # 初始化采样方法
    init_sample_method = "random"
    # 采样个数（初始化采样、确定性采样、不确定性采样）
    sample_number = 10
    # 开始训练的SBR阈值
    SBR_threshold = 1
    # presumptive non-relevant 最小采样数目
    pnr_sample = 100
    # aggressive undersampling 阈值
    aggressive_threshold = 15
    # 主动学习循环次数上限
    learning_cycle = 50
    # 召回率阈值
    recall_threshold = 0.65
    # 确定性采样和不确定性采样的分界值
    query_threshold = 10
    # 查找策略选择的样本数
    query_number = 10
    # 是否打印日志
    log_output = True

    def __init__(self, encoder: Encoder, model: Model):
        # 特征提取数据集
        self.data_dict = {}
        # 特征编码后数据集
        self.encoded_data_dict = {}
        # 数据对应的标签
        self.label_dict = {}
        # 已标记样本ID集合
        self.labeled_set = set()
        # 未标记样本ID集合
        self.unlabeled_set = set()
        # human oracle的顺序
        self.oracle_list = []
        # 使用的特征提取
        self.encoder = encoder
        # 使用的模型
        self.model = model

    def init_data_dict(self, report_file) -> None:
        """
        根据一个文件初始化实验数据集

        :param report_file: datasets/report/目录下文件名
        """
        # 读取报告csv
        df = pd.read_csv(os.path.join("datasets/report/", report_file))
        # 取后一半
        df = df.iloc[int(len(df)/2):, :]
        # 读取停用词列表
        stop_words = pd.read_csv("datasets/stopwords.csv", header=None).iloc[:, 0].tolist()
        for line in df.itertuples():
            words = []
            # 拼接summary和description
            s = line.summary + " " + line.description
            # 分词
            for w in re.split(r'\W+', s.lower()):
                # 去除停用词、去除空字符串、去除数字
                if w not in stop_words and len(w) > 0 and not w.isdigit():
                    words.append(w)
            self.data_dict[line.id] = words
            self.label_dict[line.id] = line.security
            self.unlabeled_set.add(line.id)
        self.log_info("Sentence Size: %d" % len(self.data_dict))

    def human_oracle(self, sample_list) -> bool:
        """
        模拟人类审核，假设审核结果一定正确

        :param sample_list: 样本列表（ID表示）
        :return: 是否满足开始训练的条件
        """
        # 将未标记样本设置为已标记
        for sample_id in sample_list:
            self.unlabeled_set.remove(sample_id)
            self.labeled_set.add(sample_id)
        self.oracle_list.extend(sample_list)
        # 计算SBR的总数，用于决定是否开始训练
        SBR_num = len(self.get_data_id_by_label(1, self.labeled_set))
        return SBR_num >= self.SBR_threshold

    def get_data_id_by_label(self, label: int, data_id_set) -> list:
        """
        获取带有指定标签的数据

        :param label: 标签值：0或1
        :param data_id_set: 数据Id集合
        :return: 数据列表
        """
        return list(filter(lambda i: self.label_dict[i] == label, data_id_set))

    def get_data_and_label(self, data_id_set) -> tuple:
        """
        根据数据ID得到数据和对应标签

        :param data_id_set: 数据ID集合
        :return: 训练集数据, 训练集标签
        """
        x_train = [self.encoded_data_dict[i] for i in data_id_set]
        y_train = [self.label_dict[i] for i in data_id_set]
        return x_train, y_train

    def log_info(self, log: str) -> None:
        """
        输出log

        :param log: 输出信息
        """
        if self.log_output:
            logging.info(log)

    def get_recall_target(self, recall: float) -> int:
        """
        计算为了达到召回率要找到的SBR数目

        :param recall: 召回率
        :return: SBR数目
        """
        real_pos_num = len(self.get_data_id_by_label(1, self.data_dict.keys()))
        return int(math.ceil(real_pos_num * recall))

    def get_cost(self, recall: float) -> int:
        """
        实验结束后，计算达到某一召回率所需要的cost

        :param recall: 召回率
        :return: 达到召回率所需的cost，如果达不到则返回-1
        """
        label_seq = [self.label_dict[i] for i in self.oracle_list]
        target = self.get_recall_target(recall)
        num = 0
        for i in range(len(label_seq)):
            num += label_seq[i]
            if num >= target:
                return i
        return -1

    def get_recall(self, cost: int) -> float:
        """
        实验结束后，计算某一cost下的召回率

        :param cost:
        :return: 返回cost对应的召回率，如果无法达到返回-1
        """
        label_seq = [self.label_dict[i] for i in self.oracle_list]
        if cost > len(label_seq):
            return -1
        return 1.0 * sum(label_seq[:cost]) / len(self.get_data_id_by_label(1, self.data_dict.keys()))

    def run(self) -> None:
        """
        提供具体的实验逻辑
        """
        pass
