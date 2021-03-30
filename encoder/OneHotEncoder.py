import logging
from functools import reduce

from sklearn.preprocessing import LabelEncoder

from encoder import Encoder


class OneHotEncoder(Encoder):

    def handle(self, origin_data_dict: dict) -> dict:
        word_set = set(reduce(lambda a, b: a + b, list(origin_data_dict.values())))
        logging.info("Word Dictionary Size: %d" % len(word_set))
        label_encoder = LabelEncoder()
        label_encoder.fit(list(word_set))
        result_dict = {}
        for data_id, data in origin_data_dict.items():
            encoded_data = [0] * len(word_set)
            for n in label_encoder.transform(data):
                encoded_data[n] = 1
            result_dict[data_id] = encoded_data
        return result_dict
