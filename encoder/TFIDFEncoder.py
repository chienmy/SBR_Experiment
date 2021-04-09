from encoder import Encoder
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFEncoder(Encoder):

    def handle(self, origin_data_dict: dict) -> dict:
        result_dict = {}
        str_data_dict = self.listTostr(origin_data_dict)
        tf = TfidfVectorizer()
        tf_fit = tf.fit_transform(list(str_data_dict.values()))
        feature_arr = tf_fit.toarray()
        index = 0
        for data_id, data in origin_data_dict.items():
            result_dict[data_id] = feature_arr[index:index + 1].flatten()
            index += 1
            # print(result_dict[data_id])
        return result_dict

    def listTostr(self, origin_data_dict: dict) -> dict:
        res = {}
        for data_id, data in origin_data_dict.items():
            str = ''
            for word in data:
                str = str + " " + word
            res[data_id] = str
        return res

    def risk_sample(self, label_set: set, encoded_dict: dict, sample_num: int) -> set:
        # 剩余样本不足则返回全部
        res = {}
        if len(label_set) <= sample_num:
            return set(label_set)
        for key , value in encoded_dict.items():
            target = 0.0
            for v in value:
                target += v
            res[key] = target
        sorted_key_list = sorted(res, key=lambda x: res[x])
        # 返回风险最大的n个作为样本
        return sorted_key_list[:sample_num]

    # 边训练边提取向量
    def handle_append(self, labeled_data_dict: dict, unlabeled_data_dict: dict) -> dict:
        result_dict = {}
        str_data_dict = self.listTostr(labeled_data_dict)
        tf = TfidfVectorizer()
        tf_fit = tf.fit_transform(list(str_data_dict.values()))
        vocabulary = tf.vocabulary_
        for data_id, data in unlabeled_data_dict.items():
            encoded_data = [0] * len(vocabulary.keys())
            for word in data:
                # print(vocabulary.get(word))
                if vocabulary.get(word) != None:
                    encoded_data[vocabulary.get(word)] = vocabulary.values()[word]
            result_dict[data_id] = encoded_data
            # print(result_dict[data_id])
        return result_dict


