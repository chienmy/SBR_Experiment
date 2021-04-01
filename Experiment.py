import logging
import os
import random
import re

import numpy as np
import pandas as pd

from encoder import Encoder
from model import Model


class Experiment:
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

    def init_data_dict(self, report_file):
        """
        根据一个文件初始化实验数据集

        :param report_file: datasets/report/目录下文件名
        """
        # 读取报告csv
        df = pd.read_csv(os.path.join("datasets/report/", report_file))
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
        logging.info("Sentence Size: %d" % len(self.data_dict))

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
        return SBR_num >= Experiment.SBR_threshold

    def get_data_id_by_label(self, label: int, data_id_set) -> list:
        """
        获取带有指定标签的数据

        :param label: 标签值：0或1
        :param data_id_set: 数据Id集合
        :return: 数据列表
        """
        return list(filter(lambda i: self.label_dict[i] == label, data_id_set))

    def get_data_and_label(self, data_id_set):
        """
        根据数据ID得到数据和对应标签

        :param data_id_set: 数据ID集合
        :return: 训练集数据, 训练集标签
        """
        x_train = [self.encoded_data_dict[i] for i in data_id_set]
        y_train = [self.label_dict[i] for i in data_id_set]
        return x_train, y_train

    def run(self):
        """
        主动学习实验逻辑
        """
        # 计算达到召回率所找到的SBR数目
        real_pos_num = len(self.get_data_id_by_label(1, self.data_dict.keys()))
        target = int(real_pos_num * Experiment.recall_threshold) + 1
        # 特征提取
        self.encoded_data_dict = self.encoder.handle(self.data_dict)
        # 初始化采样循环
        while True:
            # 随机初始化采样方法
            if Experiment.init_sample_method == "random":
                sample_list = self.encoder.random_sample(self.unlabeled_set, Experiment.sample_number)
            # 风险策略初始化采样方法
            elif Experiment.init_sample_method == "risk":
                sample_list = self.encoder.risk_sample(self.unlabeled_set, self.encoded_data_dict,
                                                       Experiment.sample_number)
            else:
                return
            # 人工审核后如果SBR数目达到阈值则结束初始化采样
            if self.human_oracle(sample_list):
                break
        # 开始训练
        for i in range(Experiment.learning_cycle):
            labeled_pos_data_id = self.get_data_id_by_label(1, self.labeled_set)
            labeled_neg_data_id = self.get_data_id_by_label(0, self.labeled_set)
            labeled_x_train, labeled_y_train = self.get_data_and_label(self.labeled_set)
            # 从未标记样本中随机采样一定数目的样本
            unlabeled_x_train, unlabeled_y_train = self.get_data_and_label(random.sample(
                self.unlabeled_set, max(len(labeled_pos_data_id), Experiment.pnr_sample)))
            # 将这些样本全部标记为0
            unlabeled_y_train = [0] * len(unlabeled_x_train)
            # log
            logging.info("pos: %d, neg: %d, unlabeled: %d" %
                         (len(labeled_pos_data_id), len(labeled_neg_data_id), len(self.unlabeled_set)))
            # 拼接并训练
            self.model.train(labeled_x_train + unlabeled_x_train, labeled_y_train + unlabeled_y_train)

            labeled_neg_x_predict, labeled_neg_y_predict = self.get_data_and_label(labeled_neg_data_id)
            unlabeled_x_predict, unlabeled_y_predict = self.get_data_and_label(self.unlabeled_set)
            # aggressive undersampling
            if len(labeled_pos_data_id) >= Experiment.aggressive_threshold:
                # 输出所有负样本（已标记中的负样本和所有未标记的样本）的预测概率
                x_predict = labeled_neg_x_predict + unlabeled_x_predict
                prob = self.model.predict(x_predict)
                # 取概率最小的前（和已标记中正样本数目相同）个
                prob_index = np.argsort(prob)[:len(labeled_pos_data_id)]
                labeled_pos_x_predict, labeled_pos_y_predict = self.get_data_and_label(labeled_pos_data_id)
                x_train = labeled_pos_x_predict + [x_predict[i] for i in prob_index]
                y_train = labeled_pos_y_predict + [0] * len(prob_index)
                self.model.train(x_train, y_train)
            else:
                # 输出所有未标记样本的预测概率
                prob = self.model.predict(unlabeled_x_predict)
                # 取其中概率靠前的一半
                prob_index = np.argsort(prob)[:int(len(unlabeled_x_predict) / 2)]
                x_train = labeled_x_train + [unlabeled_x_predict[i] for i in prob_index]
                y_train = labeled_y_train + [0] * len(prob_index)
                self.model.train(x_train, y_train)
            # 记录召回率
            recall = float(len(labeled_pos_data_id)) / real_pos_num
            logging.info("Epoch: %d, Recall: %f, Label Rate: %f" %
                         (i + 1, recall, len(self.labeled_set) / len(self.label_dict)))
            # 达到召回率则退出循环
            if len(labeled_pos_data_id) >= target:
                break
            # Query strategy
            unlabeled_id_list = list(self.unlabeled_set)
            unlabeled_x_predict = [self.encoded_data_dict[unlabeled_id_list[i]] for i in range(len(unlabeled_id_list))]
            prob = self.model.predict(unlabeled_x_predict)
            # 不确定采样
            if len(labeled_pos_data_id) < Experiment.query_threshold:
                order = np.argsort(np.abs(np.array(prob) - 0.5))[:Experiment.sample_number]
            # 确定性采样
            else:
                order = np.argsort(prob)[::-1][:Experiment.sample_number]
            sample_list = [unlabeled_id_list[i] for i in order]
            # 最后交给人审核
            self.human_oracle(sample_list)
