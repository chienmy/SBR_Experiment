import abc


class Encoder:

    @abc.abstractmethod
    def handle(self, origin_data_dict: dict) -> dict:
        """
        对原始数据进行特征提取

        :param origin_data_dict: {id: data}
        :return 特征字典 {id: features}
        """
        pass
