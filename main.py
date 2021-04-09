import logging

from experiment import ExperimentOne, ExperimentRandom
from encoder import Encoder, OneHotEncoder, WordEncoder
from model import SvmModel

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # ExperimentOne.log_output = False
    # e = ExperimentOne(encoder=Encoder(), model=SvmModel())
    e = ExperimentRandom(model=SvmModel())
    e.recall_threshold = 1.0
    e.init_data_dict("Ambari.csv")
    e.run()
