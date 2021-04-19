from .WordEncoder import WordEncoder


class OneHotEncoder(WordEncoder):

    def __init__(self):
        super().__init__("word")

    def handle(self, origin_data_dict: dict) -> dict:
        result_dict = super().handle(origin_data_dict)
        return {data_id: list(map(lambda n: 1 if n > 0 else 0, data)) for data_id, data in result_dict.items()}

    def handle_append(self, labeled_data_dict: dict, unlabeled_data_dict: dict) -> dict:
        result_dict = super().handle_append(labeled_data_dict, unlabeled_data_dict)
        return {data_id: list(map(lambda n: 1 if n > 0 else 0, data)) for data_id, data in result_dict.items()}
