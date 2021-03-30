import logging

from Experiment import Experiment
from encoder import OneHotEncoder
from model import SvmModel

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    e = Experiment(encoder=OneHotEncoder(), model=SvmModel())
    e.init_data_dict("Ambari.csv")
    e.run()
