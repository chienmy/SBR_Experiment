import random

import numpy as np

from experiment import BaseExperiment


class ExperimentOne(BaseExperiment):

    def run(self, **kwargs) -> None:
        """
        主动学习实验逻辑
        """
        target = self.get_recall_target(self.recall_threshold)
        # 特征提取
        if len(self._encoded_data_dict) == 0:
            self._encoded_data_dict = self._encoder.handle(self._data_dict)
        # 初始化采样循环
        while True:
            # 随机初始化采样方法
            if self.init_sample_method == "random":
                sample_list = self._encoder.random_sample(self._unlabeled_set, self.sample_number)
            # 风险策略初始化采样方法
            elif self.init_sample_method == "risk":
                sample_list = self._encoder.risk_sample(self._unlabeled_set, self._encoded_data_dict,
                                                        self.sample_number)
            else:
                return
            # 人工审核后如果SBR数目达到阈值则结束初始化采样
            if self.human_oracle(sample_list):
                break
        # 开始训练
        pre_pos_num = 0
        for i in range(self.learning_cycle):
            labeled_pos_data_id = self.get_data_id_by_label(1, self._labeled_set)
            if "bar" in kwargs.keys():
                kwargs["bar"](incr=(len(labeled_pos_data_id) - pre_pos_num))
                pre_pos_num = len(labeled_pos_data_id)
            labeled_neg_data_id = self.get_data_id_by_label(0, self._labeled_set)
            labeled_x_train, labeled_y_train = self.get_data_and_label(self._labeled_set)
            # 从未标记样本中随机采样一定数目的样本
            if max(len(labeled_pos_data_id), self.pnr_sample) <= len(self._unlabeled_set):
                unlabeled_sample = random.sample(self._unlabeled_set, max(len(labeled_pos_data_id), self.pnr_sample))
            else:
                unlabeled_sample = self._unlabeled_set
            unlabeled_x_train, unlabeled_y_train = self.get_data_and_label(unlabeled_sample)
            # 将这些样本全部标记为0
            unlabeled_y_train = [0] * len(unlabeled_x_train)
            # log
            self.log_info("pos: %d, neg: %d, unlabeled: %d" %
                          (len(labeled_pos_data_id), len(labeled_neg_data_id), len(self._unlabeled_set)))
            # 拼接并训练
            self._model.train(labeled_x_train + unlabeled_x_train, labeled_y_train + unlabeled_y_train)

            labeled_neg_x_predict, labeled_neg_y_predict = self.get_data_and_label(labeled_neg_data_id)
            unlabeled_x_predict, unlabeled_y_predict = self.get_data_and_label(self._unlabeled_set)
            # aggressive undersampling
            if len(labeled_pos_data_id) >= self.aggressive_threshold:
                # 输出所有负样本（已标记中的负样本和所有未标记的样本）的预测概率
                x_predict = labeled_neg_x_predict + unlabeled_x_predict
                prob = self._model.predict(x_predict)
                # 取概率最小的前（和已标记中正样本数目相同）个
                prob_index = np.argsort(prob)[:len(labeled_pos_data_id)]
                labeled_pos_x_predict, labeled_pos_y_predict = self.get_data_and_label(labeled_pos_data_id)
                x_train = labeled_pos_x_predict + [x_predict[i] for i in prob_index]
                y_train = labeled_pos_y_predict + [0] * len(prob_index)
                self._model.train(x_train, y_train)
            else:
                # 输出所有未标记样本的预测概率
                prob = self._model.predict(unlabeled_x_predict)
                # 取其中概率靠前的一半
                prob_index = np.argsort(prob)[:int(len(unlabeled_x_predict) / 2)]
                x_train = labeled_x_train + [unlabeled_x_predict[i] for i in prob_index]
                y_train = labeled_y_train + [0] * len(prob_index)
                self._model.train(x_train, y_train)
            # 达到召回率则退出循环
            if len(labeled_pos_data_id) >= target:
                break
            # Query strategy
            unlabeled_id_list = list(self._unlabeled_set)

            # unlabeled 特征需要边run边vector
            unlabeled_x_predict = [self._encoded_data_dict[unlabeled_id_list[i]] for i in range(len(unlabeled_id_list))]
            prob = self._model.predict(unlabeled_x_predict)
            # 不确定采样
            if len(labeled_pos_data_id) < self.query_threshold:
                order = np.argsort(np.abs(np.array(prob) - 0.5))[:self.sample_number]
            # 确定性采样
            else:
                order = np.argsort(prob)[::-1][:self.sample_number]
            sample_list = [unlabeled_id_list[i] for i in order]
            # 最后交给人审核
            self.human_oracle(sample_list)
