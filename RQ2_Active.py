import numpy as np
import pandas as pd
import logging
from alive_progress import alive_bar
from experiment import ExperimentFactory

recall_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8 , 0.9 , 1.0]
repeat_num = 10
config_list = [
    {
        "name": "One-hot+随机",
        "type": "extract_first",
        "encoder": "onehot"
    }
]

file_list = [
    "Ambari.csv" , "Camel.csv" , "Chromium.csv" , "Derby.csv" , "Wicket.csv"
]

def active_cost():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    result_dict = {}
    for config in config_list:
        for file_name in file_list:
            sheet_name = config["name"] + "+" + file_name
            print(sheet_name)
            e = ExperimentFactory.build(config)
            e.log_output = False
            e.recall_threshold = 1.0
            e.init_data_dict("./datasets", file_name)
            result = []
            with alive_bar(e.get_recall_target(1.0) * repeat_num, title=config["name"]) as bar:
                for i in range(repeat_num):
                    e.run(bar=bar)
                    result.append([e.get_cost(recall) for recall in recall_list])
                    e.clear()
                result_dict[sheet_name] = result
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

def active_recall():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    result_dict = {}
    for config in config_list:
        for file_name in file_list:
            sheet_name = config["name"] + "+" + file_name
            print(sheet_name)
            e = ExperimentFactory.build(config)
            e.log_output = False
            e.recall_threshold = 1.0
            e.init_data_dict("./datasets", file_name)
            result = []
            with alive_bar(e.get_recall_target(1.0) * repeat_num, title=config["name"]) as bar:
                for i in range(repeat_num):
                    e.run(bar=bar)
                    result.append([e.get_cost(recall) for recall in recall_list])
                    e.clear()
                result_dict[sheet_name] = result
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

if __name__ == "__main__":
    active_cost()
    active_recall()