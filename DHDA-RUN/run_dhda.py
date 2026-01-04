import math
import os
import random
import time
import numpy as np
import pandas

import DHDA

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


def test_different_algorithm(env_list, n_drifts, regressor):
    raw = process_training_data(env_list, n_drifts=n_drifts, init_train_count=100, min_occupation=0.8, max_count=4000,
                                reuse_test=False)

    y1 = list()
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

    err_list1 = []
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
            err_list1.append(val)
            dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
        if c % 32 == 0:
            err1 = np.mean(err_list1)
            y1.append(err1)
            err_list1 = list()

    time2 = time.time()
    DAL_TIME = time2 - time1
    O_DAL_ERROR = np.mean(y1)
    if type(O_DAL_ERROR) is np.ndarray:
        O_DAL_ERROR = O_DAL_ERROR[0]


    performance_list = [round(O_DAL_ERROR, 4)]
    timelist = [round(DAL_TIME, 3)]
    return {'performance': performance_list, 'time_cost': timelist}


system_list = [env_list2, env_list3, env_list4, env_list5, env_list6, env_list7, env_list8]


def test_DaL_result(test_times=30):
    i = 0
    for system in system_list:
        i=0
        all_performance = list()
        all_time_cost = list()
        while i < test_times:
            random.seed(2*i +1)
            np.random.seed(2*i +1)
            result = test_different_algorithm(system, 4, regressor="RF")
            i += 1
            time_cost_list = result['time_cost']
            performance = result['performance']
            all_performance.append(performance)
            all_time_cost.append(time_cost_list)
            print(result)


        df_performance = pandas.DataFrame(all_performance)
        df_time_cost = pandas.DataFrame(all_time_cost)

        save_dir = r'D:\CodePycharm\dhda-artifact\dhda-main\DHDA-RUN'
        mape_filename = 'DHDA_MAPE_' + system[0].split('\\')[-1]
        time_filename = 'DHDA_TIME_' + system[0].split('\\')[-1]
        data_path = os.path.join(save_dir, mape_filename)
        data_path_time = os.path.join(save_dir, time_filename)
        df_performance.to_csv(data_path)
        df_time_cost.to_csv(data_path_time)


test_DaL_result()
