import math
import os
import random
import time
import numpy as np
import pandas

from skmultiflow import drift_detection
import DHDA

from utils.general import process_training_data


from base_model.basemodels import build_incremental_model

env_list1 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_4.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_5.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_9.csv"]

env_list2 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_0.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_4.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_5.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_9.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_10.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_11.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_12.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_13.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_14.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_15.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_16.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_17.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_18.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_19.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_20.csv", ]

env_list3 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature9.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature9.csv"]

env_list4 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_0.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_4.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_5.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_9.csv"]

env_list5 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_10.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_11.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_16.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_17.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_18.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_19.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_44.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_45.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_52.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_59.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_73.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_79.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_94.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_96.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_97.csv"]

env_list6 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx4.csv"]

env_list7 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils4.csv"]

env_list8 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data8\\deeparch1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data8\\deeparch2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data8\\deeparch3.csv"
             ]


def test_timely_retrain(env_list, n_drifts):
    raw = process_training_data(env_list, n_drifts=n_drifts, init_train_count=100, min_occupation=0.8, max_count=4000)
    true_change_point_x = list()
    init_train_count = 100

    print("the datasets are...")
    x_train = raw['x_train']
    y_train = raw['y_train']
    x_test = raw['x_test']
    y_test = raw['y_test']
    count_list = raw['count_list']
    print(count_list)
    test_size = len(y_test)

    test_t_list = [1, 2, 3, 4, 5, 6, 7]
    error_list = []
    time_list = []
    for t in test_t_list:
        y = []
        time1 = time.time()
        print("RF_RETRAIN")
        err_list = []
        dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model='RF', Hybrid_Frequency=t)
        dal.fit_model(x_train, y_train)
        c = 0
        for i in range(test_size):
            c += 1
            y_pre = dal.predict(x_test[i][np.newaxis, :])
            # val = abs(y_pre - y_test[i])[0][0] change
            if y_test[i] != 0:
                val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
                err_list.append(val)
                dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
            if c % 32 == 0:
                err = np.mean(err_list)
                y.append(err)
                err_list = list()
        time2 = time.time()
        time_cost = time2 - time1
        error_list.append(np.mean(y))
        time_list.append(time_cost)
    print(error_list)
    print(time_list)
    return {'performance': error_list, 'time': time_list}


baseline_models = ['RF']
system_list = [env_list1, env_list2, env_list3, env_list4, env_list5, env_list6, env_list7, env_list8]

def test_DaL_result(regressor='RF', test_times=20):
    i = 10
    for system in system_list:
        all_performance = list()
        all_times = list()
        while i < test_times:
            random.seed(i * 2 + 1)
            np.random.seed(i * 2 + 1)
            result = test_timely_retrain(system, n_drifts=4)
            i += 1
            performance = result['performance']
            times = result['time']
            all_times.append(times)
            all_performance.append(performance)
        df_performance = pandas.DataFrame(all_performance, columns=['t=1', 't=2', 't=3', 't=4', 't=5', 't=6', 't=7'])
        df_time = pandas.DataFrame(all_times, columns=['t=1', 't=2', 't=3', 't=4', 't=5', 't=6', 't=7'])

        save_path = r'D:\CodePycharm\dhda-artifact\dhda-main\RQ3'
        mape_filename = 'RQ4_MAPE_' + system[0].split('\\')[-1]
        time_filename = 'RQ4_TIME_' + system[0].split('\\')[-1]
        data_path = os.path.join(save_path, mape_filename)
        data_path1 = os.path.join(save_path, time_filename)
        df_performance.to_csv(data_path)
        df_time.to_csv(data_path1)


for regressor in baseline_models:
    test_DaL_result(regressor)
