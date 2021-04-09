import random

import numpy as np

from experiment import BaseExperiment


class ExperimentTwo(BaseExperiment):
    """
    边训练边提取特征
    """

    def get_origin_data(self, data_id_set):
        """
        根据数据ID得到原始数据

        :param data_id_set: 数据ID集合
        :return: 原始数据
        """
        x_origin = {i: self.data_dict[i] for i in data_id_set}
        return x_origin
    
    def run(self) -> None:
        target = self.get_recall_target(self.recall_threshold)
        # 初始化采样循环
        while True:
            # 随机初始化采样方法
            if ExperimentTwo.init_sample_method == "random":
                sample_list = self.encoder.random_sample(self.unlabeled_set, ExperimentTwo.sample_number)
            # 风险策略初始化采样方法
            elif ExperimentTwo.init_sample_method == "risk":
                sample_list = self.encoder.risk_sample(self.unlabeled_set, self.encoded_data_dict,
                                                       ExperimentTwo.sample_number)
            else:
                return
            # 人工审核后如果SBR数目达到阈值则结束初始化采样
            if self.human_oracle(sample_list):
                break

        for i in range(ExperimentTwo.learning_cycle):
            # 已经标记过的原始数据
            labeled_train_data = self.get_origin_data(self.labeled_set)
            unlabeled_data = self.get_origin_data(self.unlabeled_set)

            # 特征提取 合并已标记和未标记的数据
            self.encoded_data_dict = self.encoder.handle(labeled_train_data)
            self.encoded_data_dict.update(self.encoder.handle_append(labeled_train_data, unlabeled_data))

            labeled_x_train, labeled_y_train = self.get_data_and_label(self.labeled_set)

            labeled_pos_data_id = self.get_data_id_by_label(1, self.labeled_set)
            labeled_neg_data_id = self.get_data_id_by_label(0, self.labeled_set)

            # 从未标记样本中随机采样一定数目的样本
            unlabeled_x_train, unlabeled_y_train = self.get_data_and_label(random.sample(
                self.unlabeled_set, max(len(labeled_pos_data_id), ExperimentTwo.pnr_sample)))

            # 将这些样本全部标记为0
            unlabeled_y_train = [0] * len(unlabeled_x_train)
            # log
            self.log_info("pos: %d, neg: %d, unlabeled: %d" %
                         (len(labeled_pos_data_id), len(labeled_neg_data_id), len(self.unlabeled_set)))
            # 拼接并训练
            self.model.train(labeled_x_train + unlabeled_x_train, labeled_y_train + unlabeled_y_train)

            labeled_neg_x_predict, labeled_neg_y_predict = self.get_data_and_label(labeled_neg_data_id)

            unlabeled_x_predict, unlabeled_y_predict = self.get_data_and_label(self.unlabeled_set)

            # aggressive undersampling
            if len(labeled_pos_data_id) >= ExperimentTwo.aggressive_threshold:
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
            # 达到召回率则退出循环
            if len(labeled_pos_data_id) >= target:
                break
            # Query strategy
            unlabeled_id_list = list(self.unlabeled_set)

            # 魔改unlabeled_x_predict
            unlabeled_x_predict = [self.encoded_data_dict[unlabeled_id_list[i]] for i in range(len(unlabeled_id_list))]

            prob = self.model.predict(unlabeled_x_predict)
            # 不确定采样
            if len(labeled_pos_data_id) < ExperimentTwo.query_threshold:
                order = np.argsort(np.abs(np.array(prob) - 0.5))[:ExperimentTwo.sample_number]
            # 确定性采样
            else:
                order = np.argsort(prob)[::-1][:ExperimentTwo.sample_number]
            sample_list = [unlabeled_id_list[i] for i in order]
            # 最后交给人审核
            self.human_oracle(sample_list)
