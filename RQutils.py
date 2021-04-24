import numpy as np
import pandas as pd


def save_excel(file_path: str, result_dict: dict, column_list: list) -> None:
    """
    计算实验结果的平均值并保存到指定文件

    :param file_path: 文件存储位置
    :param result_dict: 结果数据字典
    :param column_list: 列名
    """
    average_data = []
    # 计算均值
    for name, data in result_dict.items():
        np_data = np.array(data)
        mean_data = np.mean(np_data[np.all(np_data != -1, axis=1)], axis=0).tolist()
        average_data.append(name.split("|") + mean_data)
    # 保存结果至xlsx
    with pd.ExcelWriter(file_path) as writer:
        df = pd.DataFrame(average_data, columns=column_list)
        df.to_excel(writer, sheet_name="平均结果")
