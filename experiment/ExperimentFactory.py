from encoder import OneHotEncoder, WordEncoder
from model import SvmModel
from experiment import ExperimentOne, ExperimentTwo, ExperimentRandom


class ExperimentFactory:

    # 默认配置
    default_config = {
        "name": "Unknown",
        "type": "extract_random",
        "encoder": "onehot",
        "model": "svm"
    }
    # encoder和model的创建
    encoder_dict = {
        "onehot": OneHotEncoder(),
        "word": WordEncoder("word"),
        "tf": WordEncoder("tf")
    }
    model_dict = {
        "svm": SvmModel()
    }

    @staticmethod
    def build(config: dict):
        """
        根据配置字典自动生成实验

        :param config: 配置
        :return: 实验对象
        """
        default_config = ExperimentFactory.default_config.copy()
        default_config.update(config)
        e = None
        if default_config["type"] == "extract_first":
            e = ExperimentOne(
                default_config["name"],
                ExperimentFactory.encoder_dict[default_config["encoder"]],
                ExperimentFactory.model_dict[default_config["model"]])
        elif default_config["type"] == "extract_process":
            e = ExperimentTwo(
                default_config["name"],
                ExperimentFactory.encoder_dict[default_config["encoder"]],
                ExperimentFactory.model_dict[default_config["model"]])
        elif default_config["type"] == "extract_random":
            e = ExperimentRandom(
                default_config["name"],
                ExperimentFactory.model_dict[default_config["model"]])
        # 根据配置参数更新实验参数
        if e is not None:
            for k, v in default_config.items():
                if k in dir(e):
                    setattr(e, k, v)
        return e
