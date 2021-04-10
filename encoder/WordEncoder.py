from encoder import Encoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class WordEncoder(Encoder):

    def __init__(self, vectorizer_name: str):
        self.name = vectorizer_name
        self.vectorizer = TfidfVectorizer() if vectorizer_name == "tf" else CountVectorizer()

    def handle(self, origin_data_dict: dict) -> dict:
        result_dict = {}
        str_data_dict = {data_id: " ".join(data) for data_id, data in origin_data_dict.items()}
        v_fit = self.vectorizer.fit_transform(list(str_data_dict.values()))
        feature_arr = v_fit.toarray()
        index = 0
        for data_id, data in origin_data_dict.items():
            result_dict[data_id] = feature_arr[index:index + 1].flatten()
            index += 1
        return result_dict

    def risk_sample(self, label_set: set, encoded_dict: dict, sample_num: int) -> set:
        # 剩余样本不足则返回全部
        res = {}
        if len(label_set) <= sample_num:
            return set(label_set)
        for key, value in encoded_dict.items():
            target = 0.0
            for v in value:
                target += v
            res[key] = target
        sorted_key_list = sorted(list(label_set), key=lambda x: res[x])
        # 返回风险最大的n个作为样本
        return set(sorted_key_list[:sample_num])

    def handle_append(self, labeled_data_dict: dict, unlabeled_data_dict: dict) -> dict:
        result_dict = {}
        str_data_dict = {data_id: " ".join(data) for data_id, data in labeled_data_dict.items()}
        v_fit = self.vectorizer.fit_transform(list(str_data_dict.values()))
        vocabulary = self.vectorizer.vocabulary_
        for data_id, data in unlabeled_data_dict.items():
            encoded_data = [0] * len(vocabulary.keys())
            for word in data:
                if vocabulary.get(word) is not None:
                    if self.name == "tf":
                        encoded_data[vocabulary.get(word)] = vocabulary.values()[word]
                    else:
                        encoded_data[vocabulary.get(word)] += 1
            result_dict[data_id] = encoded_data
        return result_dict


