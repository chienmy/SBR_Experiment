from encoder import Encoder
from experiment import BaseExperiment
from model import Model


class ExperimentRandom(BaseExperiment):

    def __init__(self, model: Model):
        super().__init__(Encoder(), model)

    def run(self) -> None:
        """
        随机采样实验
        """
        target = self.get_recall_target(self.recall_threshold)
        while True:
            self.human_oracle(self.encoder.random_sample(self.unlabeled_set, self.sample_number))
            pos_number = sum(map(lambda s: s[1], filter(lambda item: item[0] in self.labeled_set, self.label_dict.items())))
            self.log_info("pos: %d, neg: %d, unlabeled: %d" %
                          (pos_number, len(self.labeled_set) - pos_number, len(self.unlabeled_set)))
            if pos_number >= target:
                return
