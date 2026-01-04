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


def test_CART_ablation(env_list, n_drifts, reuse_test, regressor='RF', seed=1):
    raw = process_training_data(env_list, n_drifts=n_drifts, init_train_count=50, min_occupation=0.8, max_count=4000,
                                reuse_test=reuse_test)
    true_change_point_x = list()
    detected_change_point_x = list()
    env_count = len(env_list)
    y1 = list()
    y2 = list()
    y3 = list()
    y4 = list()

    test_list = list()
    count = 0
    init_train_count = 100
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

    time1 = time.time()
    print("no upper adapt & update")
    err_list2 = []
    dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor, Upper_Adapt=False, Lower_Adapt=True, Hybrid_Update=True)
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
    no_cart_change_time = time2 - time1

    time1 = time.time()
    print("ban Lower adapt")
    err_list3 = []
    dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor, Upper_Adapt=True, Lower_Adapt=False, Hybrid_Update=True)
    dal.fit_model(x_train, y_train)
    c = 0
    for i in range(test_size):
        c += 1
        y_pre = dal.predict(x_test[i][np.newaxis, :])
        # val = abs(y_pre - y_test[i])[0][0] change
        if y_test[i] != 0:
            val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
            err_list3.append(val)
            dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
        if c % 32 == 0:
            err3 = np.mean(err_list3)
            y3.append(err3)
            err_list3 = list()
    time2 = time.time()
    cart_change_but_no_hoeffding_time = time2 - time1

    time1 = time.time()
    print("ban Lower Hybrid update")
    err_list4 = []
    dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor, Upper_Adapt=True, Lower_Adapt=True, Hybrid_Update=False)
    dal.fit_model(x_train, y_train)
    c = 0
    for i in range(test_size):
        c += 1
        y_pre = dal.predict(x_test[i][np.newaxis, :])
        # val = abs(y_pre - y_test[i])[0][0] change
        if y_test[i] != 0:
            val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
            err_list4.append(val)
            dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
        if c % 32 == 0:
            err4 = np.mean(err_list4)
            y4.append(err4)
            err_list4 = list()
    time2 = time.time()
    cart_change_but_no_hoeffding_time = time2 - time1

    m2 = np.mean(y2)
    m3 = np.mean(y3)
    m4 = np.mean(y4)
    mylist = [m2, m3, m4]
    for i, m in enumerate(mylist):
        if type(m) is np.ndarray:
            mylist[i] = m[0]
    performance_list = []

    for m in mylist:
        performance_list.append(round(m, 4))

    # baseline model & DaL-model
    return {'performance': performance_list}


baseline_models = ['RF']
system_list = [env_list1, env_list2, env_list3, env_list4, env_list5, env_list6, env_list7, env_list8]


def test_DaL_result(regressor='RF', test_times=30):
    i = 0
    for system in system_list:
        all_performance = list()
        i = 0
        while i < test_times:
            random.seed(2*i + 1)
            np.random.seed(2*i + 1)
            result = test_CART_ablation(system, reuse_test=False, n_drifts=4, seed=2*i + 1)
            i += 1
            performance = result['performance']
            all_performance.append(performance)

        df_performance = pandas.DataFrame(all_performance,
                                          columns=['NO-UPPER-ADAPT', "NO-LOWER-ADAPT", "NO-LOWER-HYBRID"])
        save_path = r'D:\CodePycharm\dhda-artifact\dhda-main\RQ3'
        filename = 'RQ3_' + system[0].split('\\')[-1]
        data_path = os.path.join(save_path, filename)
        df_performance.to_csv(data_path)



for regressor in baseline_models:
    test_DaL_result(regressor)
