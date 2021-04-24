from encoder import OneHotEncoder, WordEncoder
from model import SvmModel, RFModel, NBModel, KNNModel, MLPModel
from experiment import ExperimentOne, ExperimentTwo, ExperimentRandom, ExperimentML


class ExperimentFactory:

    # 默认配置
    default_config = {
        "name": "Unknown",
        "type": "extract_random",
        "encoder": "onehot",
        "model": "svm"
    }
    # 取对应的encoder和model
    encoder_dict = {
        "onehot": OneHotEncoder(),
        "word": WordEncoder("word"),
        "tf": WordEncoder("tf")
    }
    model_dict = {
        "svm": SvmModel(),
        "rf": RFModel(),
        "nb": NBModel(),
        "knn": KNNModel(),
        "mlp": MLPModel()
    }

    @staticmethod
    def build(config: dict):
        """
        根据配置字典自动生成实验，配置说明如下：\n
        type: [extract_first | extract_process | extract_random] 先提取再训练 | 边训练边提取 | 随机提取 \n
        encoder: [onehot | word | tf] One-Hot编码 | Word序列 | TF-IDF编码 \n
        model: [svm] \n
        其他参数配置参见 BaseExperiment 中实验参数设置，使用同样的键名即可

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
        elif default_config["type"] == "ml":
            e = ExperimentML(
                default_config["name"],
                ExperimentFactory.encoder_dict[default_config["encoder"]],
                ExperimentFactory.model_dict[default_config["model"]])
        # 根据配置参数更新实验参数
        if e is not None:
            for k, v in default_config.items():
                if k in dir(e):
                    setattr(e, k, v)
        return e
