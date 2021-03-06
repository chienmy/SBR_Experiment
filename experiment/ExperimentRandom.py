from encoder import Encoder
from experiment import BaseExperiment
from model import Model


class ExperimentRandom(BaseExperiment):

    def __init__(self, name: str, model: Model):
        super().__init__(name, Encoder(), model)

    def run(self, **kwargs) -> None:
        """
        随机采样实验
        """
        target = self.get_recall_target(self.recall_threshold)
        pre_pos_num = 0
        while True:
            self.human_oracle(self._encoder.random_sample(self._unlabeled_set, self.sample_number))
            pos_number = sum(map(lambda s: s[1], filter(
                lambda item: item[0] in self._labeled_set, self._label_dict.items())))
            self.log_info("pos: %d, neg: %d, unlabeled: %d" %
                          (pos_number, len(self._labeled_set) - pos_number, len(self._unlabeled_set)))
            if "bar" in kwargs.keys():
                kwargs["bar"](incr=(pos_number - pre_pos_num))
                pre_pos_num = pos_number
            if pos_number >= target:
                return
