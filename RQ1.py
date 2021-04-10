import numpy as np
import pandas as pd
import logging
from alive_progress import alive_bar
from experiment import ExperimentFactory

recall_list = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
repeat_num = 10
config_list = [
    {
        "name": "One-hot+随机",
        "type": "extract_first",
        "encoder": "onehot"
    },
    {
        "name": "One-hot+风险策略",
        "type": "extract_first",
        "init_sample_method": "risk",
        "encoder": "onehot"
    },
    {
        "name": "Word sequence+随机",
        "type": "extract_first",
        "encoder": "word"
    },
    {
        "name": "Word sequence+风险策略",
        "type": "extract_first",
        "init_sample_method": "risk",
        "encoder": "word"
    },
    {
        "name": "Tf-idf+随机",
        "type": "extract_first",
        "encoder": "tf"
    },
    {
        "name": "Tf-idf+风险策略",
        "type": "extract_first",
        "init_sample_method": "risk",
        "encoder": "tf"
    },
    {
        "name": "完全随机",
        "type": "extract_random"
    }
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    result_dict = {}
    for config in config_list:
        e = ExperimentFactory.build(config)
        e.log_output = False
        e.recall_threshold = 1.0
        e.init_data_dict("./datasets", "Ambari.csv")
        result = []
        with alive_bar(e.get_recall_target(1.0) * repeat_num, title=config["name"]) as bar:
            for i in range(repeat_num):
                e.run(bar=bar)
                result.append([e.get_cost(recall) for recall in recall_list])
                e.clear()
            result_dict[config["name"]] = result
    average_data = []
    average_index = []
    for name, data in result_dict.items():
        average_data.append(np.mean(np.array(data), axis=0))
        average_index.append(name)
    # 保存结果至xlsx
    with pd.ExcelWriter('output.xlsx') as writer:
        df = pd.DataFrame(average_data, index=average_index, columns=recall_list)
        df.to_excel(writer, sheet_name="平均结果")
        for name, data in result_dict.items():
            df = pd.DataFrame(data, columns=recall_list)
            df.to_excel(writer, sheet_name=name)
