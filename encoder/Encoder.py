import abc
import random


class Encoder:

    @abc.abstractmethod
    def handle(self, origin_data_dict: dict) -> dict:
        """
        对原始数据进行特征提取

        :param origin_data_dict: {id: data}
        :return 特征字典 {id: features}
        """
        pass

    @abc.abstractmethod
    def risk_sample(self, label_set: set, encoded_dict: dict, sample_num: int) -> set:
        """
        与特征提取方法对应的风险策略采样计算

        :param label_set: 待采样的id集合
        :param encoded_dict: 特征提取后的数据集
        :param sample_num: 采样个数
        :return: 从未标记id中按风险排序抽取的一部分样本
        """
        pass

    def random_sample(self, label_set: set, sample_num: int) -> set:
        """
        随机采样策略

        :param label_set: 待采样的id集合
        :param sample_num: 采样个数
        :return: 从未标记id中随机抽取的一部分样本
        """
        # 剩余样本不足则返回全部
        if len(label_set) <= sample_num:
            sample_list = set(label_set)
        # 否则随机抽取m个样本的ID
        else:
            sample_list = set(random.sample(label_set, sample_num))
        return sample_list
