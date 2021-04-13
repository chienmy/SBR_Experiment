import logging
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from encoder import Encoder


class OneHotEncoder(Encoder):

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.counter = Counter()

    def handle(self, origin_data_dict: dict) -> dict:
        # 对单词计数
        for v in origin_data_dict.values():
            self.counter.update(v)
        # 使用LabelEncoder编码
        self.label_encoder.fit(list(self.counter.keys()))
        result_dict = {}
        for data_id, data in origin_data_dict.items():
            result_dict[data_id] = self._encode(data)
        return result_dict

    def risk_sample(self, label_set: set, encoded_dict: dict, sample_num: int) -> set:
        # 剩余样本不足则返回全部
        if len(label_set) <= sample_num:
            return set(label_set)
        # 取出现的前1/4单词计算风险
        key_words = list(map(lambda e: e[0], self.counter.most_common(int(len(self.counter) / 4))))
        a = np.array(self._encode(key_words))

        def risk_sum(one_hot_code):
            b = np.array(one_hot_code)
            return np.sum(a * b)
        # 风险计算
        risk_list = [(k, risk_sum(one_hot_code)) for k, one_hot_code in
                     filter(lambda item: (item[0] in label_set), encoded_dict.items())]
        # id按风险排序
        risk_list = sorted(risk_list, key=lambda e: e[1], reverse=True)
        # 返回风险最大的n个作为样本
        return set(map(lambda e: e[0], risk_list[:sample_num]))

    def handle_append(self, labeled_data_dict: dict, unlabeled_data_dict: dict) -> dict:
        """
        边训练边提取向量

        :param labeled_data_dict: 已标记的数据集合
        :param unlabeled_data_dict: 未标记的数据集合
        """
        # 对单词计数
        for v in labeled_data_dict.values():
            self.counter.update(v)
        # 使用LabelEncoder编码
        self.label_encoder.fit(list(self.counter.keys()))
        result_dict = {}
        for data_id, data in unlabeled_data_dict.items():
            result_dict[data_id] = self._encode(data)
        return result_dict

    def _encode(self, data):
        # 将单词序列转换为one-hot编码
        encoded_data = [0] * len(self.counter)
        data = list(filter(lambda s: s in self.label_encoder.classes_, data))
        for n in self.label_encoder.transform(data):
            encoded_data[n] = 1
        return encoded_data
