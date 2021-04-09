import logging

from Experiment import Experiment
from Experiment2 import Experiment2
from encoder import OneHotEncoder,WordSequenceEncoder
from encoder import TFIDFEncoder
from model import SvmModel

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # e = Experiment(encoder=WordSequenceEncoder(), model=SvmModel())
    e = Experiment2(encoder=WordSequenceEncoder(), model=SvmModel())
    e.init_data_dict("Ambari.csv")
    e.run()
