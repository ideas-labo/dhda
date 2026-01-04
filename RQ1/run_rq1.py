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


def test_DaL(env_list, n_drifts, regressor='RF'):
    if regressor == 'LR':
        isstanderlize = True
    else:
        isstanderlize = False
    raw = process_training_data(env_list, n_drifts=n_drifts, init_train_count=50, min_occupation=0.8, max_count=4000,
                                isstandarlize=isstanderlize)
    true_change_point_x = list()
    detected_change_point_x = list()
    env_count = len(env_list)
    y1 = list()
    y2 = list()
    y3 = list()
    y4 = list()
    y5 = list()
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

    time1 = time.time()
    print("rf-alpha-retrain")
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
        x_accessible = np.vstack((x_accessible, x_test[i][np.newaxis, :]))
        y_accessible = np.vstack((y_accessible, y_test[i][np.newaxis, :]))
        if c % 96 == 0:
            model.fit(x_accessible, y_accessible)
            err1 = np.mean(err_list1)
            y1.append(err1)
            err_list1 = list()
    time2 = time.time()
    Re_RF_alpha_time = time2 - time1

    err_list2 = []
    time1 = time.time()
    print("dal-alpha-retrain")
    dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor)
    dal.fit_model(x_train, y_train)
    x_accessible = x_train
    y_accessible = y_train
    c = 0
    for i in range(test_size):
        c += 1
        y_pre = dal.predict(x_test[i][np.newaxis, :])
        x_accessible = np.vstack((x_accessible, x_test[i][np.newaxis, :]))
        y_accessible = np.vstack((y_accessible, y_test[i][np.newaxis, :]))
        # val = abs(y_pre - y_test[i])[0][0] change
        if y_test[i] != 0:
            val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
            err_list2.append(val)
            # dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
        if c % 32 == 0:
            err2 = np.mean(err_list2)
            y2.append(err2)
            err_list2 = list()
        if c % 96 == 0:
            dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor)
            dal.fit_model(x_accessible, y_accessible)

    time2 = time.time()
    Re_DaL_alpha_time = time2 - time1

    time1 = time.time()
    print("rf-per-retrain")
    err_list3 = []
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
            err_list3.append(val)
            warning_adwin.add_element(val)
            drift_adwin.add_element(val)
        x_accessible = np.vstack((x_accessible, x_test[i][np.newaxis, :]))
        y_accessible = np.vstack((y_accessible, y_test[i][np.newaxis, :]))
        if c % 32 == 0:
            model.fit(x_accessible, y_accessible)
            err3 = np.mean(err_list3)
            y3.append(err3)
            err_list3 = list()

    time2 = time.time()
    Re_RF_per_time = time2 - time1

    err_list4 = []
    time1 = time.time()
    print("dal-per-retrain")
    dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor)
    dal.fit_model(x_train, y_train)
    x_accessible = x_train
    y_accessible = y_train
    c = 0
    for i in range(test_size):
        c += 1
        y_pre = dal.predict(x_test[i][np.newaxis, :])
        x_accessible = np.vstack((x_accessible, x_test[i][np.newaxis, :]))
        y_accessible = np.vstack((y_accessible, y_test[i][np.newaxis, :]))
        # val = abs(y_pre - y_test[i])[0][0] change
        if y_test[i] != 0:
            val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
            err_list4.append(val)
            # dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
        if c % 32 == 0:
            err4 = np.mean(err_list4)
            y4.append(err4)
            err_list4 = list()
            dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor)
            dal.fit_model(x_accessible, y_accessible)

    time2 = time.time()
    Re_DaL_per_time = time2 - time1

    err_list5 = []
    time1 = time.time()
    print("fixed-dal")
    dal = DHDA.DHDA_Regressor(depth=1, min_clock=32, base_model=regressor)
    dal.fit_model(x_train, y_train)
    x_accessible = x_train
    y_accessible = y_train
    c = 0
    for i in range(test_size):
        c += 1
        y_pre = dal.predict(x_test[i][np.newaxis, :])
        x_accessible = np.vstack((x_accessible, x_test[i][np.newaxis, :]))
        y_accessible = np.vstack((y_accessible, y_test[i][np.newaxis, :]))
        # val = abs(y_pre - y_test[i])[0][0] change
        if y_test[i] != 0:
            val = ((abs(y_pre - y_test[i])[0]) / y_test[i])[0]
            err_list5.append(val)
            # dal.learn_data(x_test[i][np.newaxis, :], y_test[i][np.newaxis, :], val)
        if c % 32 == 0:
            err5 = np.mean(err_list5)
            y5.append(err5)
            err_list5 = list()

    time2 = time.time()
    fixed_DaL_time = time2 - time1

    Re_DaL_alpha_mape = np.mean(y2)
    Re_RF_alpha_mape = np.mean(y1)
    Re_DaL_per_mape = np.mean(y4)
    Re_RF_per_mape = np.mean(y3)
    fixed_DaL_mape = np.mean(y5)

    mean_RF_retrain = np.mean(y2)
    mean_DeepPerf_Retrain = np.mean(y1)
    if type(Re_RF_alpha_mape) is np.ndarray:
        Re_RF_alpha_mape = Re_RF_alpha_mape[0]
    if type(Re_DaL_alpha_mape) is np.ndarray:
        Re_DaL_alpha_mape = Re_DaL_alpha_mape[0]

    performance_list = [round(Re_RF_alpha_mape, 4), round(Re_DaL_alpha_mape, 4), round(Re_RF_per_mape, 4),
                        round(Re_DaL_per_mape, 4), round(fixed_DaL_mape, 4)]
    time_list = [Re_RF_alpha_time, Re_DaL_alpha_time, Re_RF_per_time, Re_DaL_per_time, fixed_DaL_time]

    return {'performance': performance_list, 'time': time_list}


baseline_models = ['RF']
system_list = [env_list1, env_list2, env_list3, env_list4, env_list5, env_list6, env_list7, env_list8]


def test_DaL_result(regressor='RF', test_times=30):
    for system in system_list:
        all_performance = list()
        all_time = list()
        i = 0
        while i < test_times:
            random.seed(i * 2 + 1)
            np.random.seed(i * 2 + 1)
            result = test_DaL(system, n_drifts=4, regressor=regressor)
            print(result)
            i += 1
            performance = result['performance']
            all_performance.append(performance)
            times = result['time']
            all_time.append(times)
        df_performance = pandas.DataFrame(all_performance,
                                          columns=['alpha-Re-RF', 'alpha-Re-DaL', 'per-Re-RF', 'per-Re-DaL', 'fixed-DaL'])
        df_time_cost = pandas.DataFrame(all_time, columns=['alpha-Re-RF', 'alpha-Re-DaL', 'per-Re-RF', 'per-Re-DaL', 'fixed-DaL'])

        save_dir = r'D:\CodePycharm\dhda-artifact\dhda-main\RQ1'
        mape_filename = 'RQ1_MAPE_' + system[0].split('\\')[-1]
        time_filename = 'RQ1_TIME_' + system[0].split('\\')[-1]
        data_path = os.path.join(save_dir, mape_filename)
        data_path_time = os.path.join(save_dir, time_filename)
        df_performance.to_csv(data_path)
        df_time_cost.to_csv(data_path_time)


for regressor in baseline_models:
    test_DaL_result(regressor)
