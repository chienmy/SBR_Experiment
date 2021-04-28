from RQ2 import RQ2
from RQutils import save_excel

if __name__ == "__main__":
    r = RQ2()
    for train_name in ["Ambari", "Camel", "Derby", "Wicket"]:
        for test_name in ["Ambari", "Camel", "Derby", "Wicket"]:
            if train_name == test_name:
                continue
            for model_name in ["svm", "rf", "nb", "knn", "mlp"]:
                r.run(train_name, test_name, model_name)
    save_excel("output-RQ3.xlsx", r.result_dict, ["train", "test", "model"] + r.recall_list)
