import math
import os
import random
import time
import numpy as np
import pandas

from skmultiflow import drift_detection

import DHDA
from base_model.basemodels import build_incremental_model
from utils.general import process_training_data


env_list1 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data1\\sac_2.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data1\\sac_4.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data1\\sac_5.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data1\\sac_6.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data1\\sac_7.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data1\\sac_8.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data1\\sac_9.csv"]

env_list2 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_0.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_1.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_2.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_3.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_4.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_5.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_6.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_7.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_8.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_9.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_10.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_11.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_12.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_13.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_14.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_15.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_16.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_17.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_18.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_19.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data2\\x264_20.csv", ]

env_list3 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj1_feature6.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj1_feature7.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj1_feature8.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj2_feature9.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj2_feature6.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj1_feature7.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj2_feature8.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data3\\storm-obj2_feature9.csv"]

env_list4 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_0.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_1.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_2.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_3.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_4.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_5.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_6.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_7.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_8.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data4\\spear_9.csv"]

env_list5 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_10.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_11.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_16.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_17.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_18.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_19.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_44.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_45.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_52.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_59.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_73.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_79.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_94.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_96.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data5\\sqlite_97.csv"]

env_list6 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data6\\nginx1.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data6\\nginx2.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data6\\nginx3.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data6\\nginx4.csv"]

env_list7 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data7\\exastencils1.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data7\\exastencils2.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data7\\exastencils3.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data7\\exastencils4.csv"]

env_list8 = ["D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data8\\deeparch1.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data8\\deeparch2.csv",
             "D:\\CodePycharm\\dhda-artifact\\dhda-main\\drift_data\\data8\\deeparch3.csv"
             ]


def test_DaL(env_list, n_drifts, regressor='RF'):
    if regressor == 'LR':
        isstanderlize = True
    else:
        isstanderlize = False
    raw = process_training_data(env_list, n_drifts=n_drifts, init_train_count=100, min_occupation=0.8, max_count=4000,
                                isstandarlize=isstanderlize)
    true_change_point_x = list()
    detected_change_point_x = list()
    env_count = len(env_list)
    y1 = list()
    y2 = list()
    y3 = list()
    y4 = list()
    test_list = list()
    count = 0
    init_train_count = 0
    change_point = list()
    dataset_name = env_list[0].split('\\')[-1]
    print("the datasets are...")
    x_train = raw['x_train']
    y_train = raw['y_train']
    x_test = raw['x_test']
    y_test = raw['y_test']
    count_list = raw['count_list']
    print(count_list)
    test_size = len(y_test)
    err_list2 = []
    time1 = time.time()
    print("Online-DaL-RF-Retrain")

    err_list2 = []
    time1 = time.time()
    print("O-DAL")
    dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor)
    dal.fit_model(x_train, y_train)
    c = 0
    for i in range(test_size):
        c += 1
        y_pre = dal.predict(x_test[i][np.newaxis, :])
        # val = abs(y_pre - y_test[i])[0][0] change
        if y_test[i] != 0:
            val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
            err_list2.append(val)
            dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
        if c % 32 == 0:
            err2 = np.mean(err_list2)
            y2.append(err2)
            err_list2 = list()

    time2 = time.time()
    DAL_TIME = time2 - time1
    time1 = time.time()


    time1 = time.time()
    print("Online-model")
    err_list1 = []

    model = build_incremental_model(regressor)
    model.fit(x_train, y_train)
    x_accessible = x_train
    y_accessible = y_train
    drift_adwin = drift_detection.ADWIN(0.002)
    warning_adwin = drift_detection.ADWIN(0.01)
    warning_adwin.set_clock(16)
    c = 0
    mode = False
    new_count = 0
    for i in range(test_size):
        c += 1
        y_pre = model.predict(x_test[i][np.newaxis, :])
        # val = abs(y_pre - y_test[i])[0][0] change
        if y_test[i] != 0:
            val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
            err_list1.append(val)
            warning_adwin.add_element(val)
            drift_adwin.add_element(val)

        if warning_adwin.detected_change():
            new_count = 0
            mode = True
            warning_adwin.reset()
            print("warning")
        if mode:
            new_count += 1
        if drift_adwin.detected_change():
            if new_count < 8:
                new_count = 8
            x_accessible = x_accessible[-new_count:]
            y_accessible = y_accessible[-new_count:]
            drift_adwin.reset()
            warning_adwin.reset()
            new_count = 0
            print("drift")
            model.fit(x_accessible, y_accessible)

        x_accessible = np.vstack((x_accessible, x_test[i][np.newaxis, :]))
        y_accessible = np.vstack((y_accessible, y_test[i][np.newaxis, :]))

        if c % 32 == 0:
            """if len(y_accessible)>=32:
                model.update(x_accessible[-32:], y_accessible[-32:])
            else:
                model.update(x_accessible, y_accessible)"""
            model.update(x_accessible, y_accessible)
            err1 = np.mean(err_list1)
            y1.append(err1)
            err_list1 = list()

    time2 = time.time()
    model_time = time2 - time1



    mean_RF_retrain = np.mean(y2)
    mean_DeepPerf_Retrain = np.mean(y1)
    if type(mean_DeepPerf_Retrain) is np.ndarray:
        mean_DeepPerf_Retrain = mean_DeepPerf_Retrain[0]
    if type(mean_RF_retrain) is np.ndarray:
        mean_RF_retrain = mean_RF_retrain[0]

    performance_list = [round(mean_DeepPerf_Retrain, 4), round(mean_RF_retrain, 4)]

    return {'performance': performance_list}


baseline_models = ['LR', 'RF', 'KNN', 'HT']
system_list = [env_list1, env_list2, env_list3, env_list4, env_list5, env_list6, env_list7, env_list8]


def test_DaL_result(regressor='RF', test_times=30):
    for system in system_list:
        all_performance = list()
        i = 0
        while i < test_times:
            random.seed(i * 2 + 1)
            np.random.seed(i * 2 + 1)
            result = test_DaL(system, n_drifts=4, regressor=regressor)
            i += 1
            performance = result['performance']
            all_performance.append(performance)
        df_performance = pandas.DataFrame(all_performance, columns=['None-DaL', 'DaL-Model'])
        save_dir = r'D:\CodePycharm\dhda-artifact\dhda-main\RQ2'
        filename = 'RQ2_' + regressor + '_' + system[0].split('\\')[-1]
        data_path = os.path.join(save_dir, filename)
        df_performance.to_csv(data_path)


for regressor in baseline_models:
    test_DaL_result(regressor)
