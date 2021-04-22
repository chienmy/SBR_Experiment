from encoder import Encoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class WordEncoder(Encoder):

    def __init__(self, vectorizer_name: str):
        self.name = vectorizer_name
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000) \
            if vectorizer_name == "tf" else CountVectorizer(stop_words="english", max_features=1000)

    def handle(self, origin_data_dict: dict) -> dict:
        keys, values = zip(*origin_data_dict.items())
        v_fit = self.vectorizer.fit_transform(list(values)).toarray()
        return {keys[i]: v_fit[i] for i in range(len(keys))}

    def risk_sample(self, label_set: set, encoded_dict: dict, sample_num: int) -> set:
        # 剩余样本不足则返回全部
        if len(label_set) <= sample_num:
            return set(label_set)
        sorted_key_list = sorted(list(label_set), key=lambda x: sum(encoded_dict[x]))
        # 返回风险最大的n个作为样本
        return set(sorted_key_list[:sample_num])

    def handle_append(self, labeled_data_dict: dict, unlabeled_data_dict: dict) -> dict:
        # 没有未标记数据时直接返回，否则下方zip会报错
        if len(unlabeled_data_dict) == 0:
            return {}
        self.vectorizer.fit(list(labeled_data_dict.values()))
        keys, values = zip(*unlabeled_data_dict.items())
        v_fit = self.vectorizer.transform(list(values)).toarray()
        return {keys[i]: v_fit[i] for i in range(len(keys))}


