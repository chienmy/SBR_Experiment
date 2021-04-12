import os

import pandas as pd
import time
import warnings



from model import utilities
from sklearn.feature_extraction.text import CountVectorizer
from model import RFModel
from  model import dimension_reduce as dr

warnings.filterwarnings('ignore')


def read_data(path):
    data = pd.read_csv(path)
    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1)
    return data


def prediction(data_dir, report_file):
    
    #    all_dataset = read_data(dataset_path)

    # 读取报告csv
    df = read_data(os.path.join(data_dir, "report", report_file))
    # 取后一半
    num_n = len(df)
    train_dataset = df[:int(0.5*num_n)]
    test_dataset = df[int(0.5*num_n):]

#    print(dataset_path)
    learners = ["RF", "SVM"] # ,"MLP","LR","KNN"

    global train_x
    global test_x
    global train_y
    global test_y

    train_content = train_dataset.description
    train_label = train_dataset.security.tolist()
#    train_label = train_dataset.iloc[:, -2:-1] 

    test_content = test_dataset.description
    test_label = test_dataset.security.tolist()
    
    vectorizer = CountVectorizer(stop_words='english')
    train_content_matrix = vectorizer.fit_transform(train_content)
    test_content_matrix = vectorizer.transform(test_content)
    train_content_matrix_dr, test_content_matrix_dr  = dr.selectFromLinearSVC2(train_content_matrix, train_label, test_content_matrix)   

    
    train_x = train_content_matrix_dr.toarray()
    train_y = train_label
    test_x = test_content_matrix_dr.toarray()
    test_y = test_label
    utilities.data_transfer(train_x, train_y, test_x, test_y)
    print(train_x)

    for l in learners:
        if l == "RF":
#            print("===============RF without turning===============")
            rf_train_start_time = time.time()
            # clf = RandomForestClassifier(oob_score=True, n_estimators=30)
            # clf.fit(train_x, train_y)
            clf = RFModel()
            clf.train(train_x , train_y)
            predicted = clf.predict(test_x)
            # predicted = clf.predict(test_x)
            print(predicted)

def main():
    # output = "../output_noise/predict_with_text_5_lner_simple_output.csv"
    # csv_file = open(output, "w", newline='')
    # writer = csv.writer(csv_file, delimiter=',')
    # writer.writerow(['Dataname','Version', 'Approach', 'TN', 'FP', 'FN', 'TP', 'pd', 'pf', 'prec', 'fmeasure', 'Gmeasure','Cost'])
    #
    # datanames = ["chromium","ambari", "camel", "derby", "wicket"] #"ambari", "camel", "derby", "wicket", "chromium"
    # for dataname in datanames:
    #     print("Start data: ", dataname)
    #     DATA_PATH_NOISE = r"../input/noise/" +dataname + ".csv"
    #     DATA_PATH_CLEAN = r"../input/clean/" +dataname + ".csv"

    prediction("../datasets", "Ambari.csv")
    print("")
    # csv_file.close()
    # print(output + '**************** finished************************')

if __name__ == "__main__":
    main()
