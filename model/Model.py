import abc


class Model:

    @abc.abstractmethod
    def train(self, x_train, y_train):
        """
        模型训练

        :param x_train: 训练数据
        :param y_train: 训练标签
        """
        pass

    @abc.abstractmethod
    def predict(self, x_predict):
        """
        返回预测为正值的概率

        :param x_predict: 预测数据
        :return: 预测为正概率
        """
        pass
