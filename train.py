'''
Confidential !!!
These are codes for HandyAPP.
Please do not share the codes with others. Thanks~
'''
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import gc
import csv
import time
import json
import random
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
warnings.simplefilter("ignore")
from preprocess import pre_process
from utils import cal_metrics, get_average_results
from models import create_ann, create_cnn, create_lstm, create_lstmcnn, create_tcn
from models import create_lr, create_lda, create_knn, create_svm, create_dt

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

'''
-model_name: ann, cnn, lstm, lstmcnn, tcn, ...
-task_name: slide-2, cradle-3, all-7
-device: cpu("-1"), gpu("0","1","2","3")
'''

def merge_data(Xdata): # merge 3 rows
    Xs = []
    for i in range(Xdata.shape[0]):
        if i == 0:
            mg_row = np.vstack([Xdata[i][:-1], Xdata[i][:-1], Xdata[i][:-1]])
        elif i == 1:
            if Xdata[i-1][-1] == Xdata[i][-1]: # the same user id
                mg_row = np.vstack([Xdata[i-1][:-1], Xdata[i][:-1], Xdata[i][:-1]])
            else:
                mg_row = np.vstack([Xdata[i][:-1], Xdata[i][:-1], Xdata[i][:-1]])  
        else:
            if (Xdata[i-2][-1] == Xdata[i-1][-1]) and (Xdata[i-2][-1] == Xdata[i][-1]):
                mg_row = np.vstack([Xdata[i-2][:-1], Xdata[i-1][:-1], Xdata[i][:-1]])
            else:
                if (Xdata[i-1][-1] == Xdata[i][-1]):
                    mg_row = np.vstack([Xdata[i-1][:-1], Xdata[i][:-1], Xdata[i][:-1]])
                else:
                    mg_row = np.vstack([Xdata[i][:-1], Xdata[i][:-1], Xdata[i][:-1]])
        Xs.append(mg_row)
    X_data = np.array(Xs)
    return X_data

def main(args):
    dnn_models = ["ann", "cnn", "lstm", "lstmcnn", "tcn"]
    ml_models = ["lr", "lda", "svm", "knn", "dt", "lgb"]
    print(args.gpu_id, args.task_name, args.model_name, args.merge)
    assert args.gpu_id in ["-1", "0", "1", "2", "3"]
    assert args.task_name in ["slide-2", "cradle-3", "all-7"]
    assert args.model_name in dnn_models + ml_models
    assert args.merge in [False, True]
    
    time_start = time.time()

    # params
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id 
    model_name = args.model_name
    task_name = args.task_name

    # check gpu env
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        print('use cpu')
    else:
        print('available gpu num:', len(gpus), ', use gpu id:', gpu_id)

    # set random seed
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # pre process
    data_dir = './data_sample/'
    data_list = []
    for file_id in range(1,56):
        file_path = data_dir + f'{file_id}.csv'
        df = pd.read_csv(file_path)
        data_list.append(df)
    df = pd.concat(data_list, ignore_index=True)

    del data_list
    tf.keras.backend.clear_session()
    gc.collect()

    df = pre_process(df)
    print('total:', df.shape)

    # features
    path_len = 24
    sensor_len = 16

    sensors = [ "gravity", "gyroscope", "linearacceleration", "ryp", "real_gravity", "real_gyroscope", "real_linearacceleration", "real_ryp", "live_gravity", "live_gyroscope", "live_linearacceleration", "live_ryp"]

    pathx = ["pathx_" + str(i) for i in range(1,int(path_len/2)+1)]
    pathy = ["pathy_" + str(i) for i in range(1,int(path_len/2)+1)] 
    paths = [item for pair in zip(pathx, pathy) for item in pair]

    radiusx = ['radiusx_'+str(i) for i in range(1,int(path_len/2)+1)]
    radiusy = ['radiusy_'+str(i) for i in range(1,int(path_len/2)+1)]
    radii = [item for pair in zip(radiusx, radiusy) for item in pair]

    angles = [f"angle_{i}" for i in range(1, int(path_len/2)+2)]

    distances = [f"distance_{i}" for i in range(1, int(path_len/2)+2)]

    # our features w/o motion sensors
    features = ["device_model", "duration", "speed", "xy_1", "xy_2", "angle", "angles", "screen", "slide_direction", "2_1", "real_distance"]

    time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    output_dir = f'./output/{time_now}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gpu_name = "cpu"
    if gpu_id != "-1":
        gpu_name = "gpu" + gpu_id
    result_path = output_dir + f"{model_name}_{task_name}_{gpu_name}.json"

    feas = []
    for fea in features:
        if fea in ["device_model", "duration", "speed", "angle", "slide_direction", "distance", "real_distance"]:
            feas.append(fea)
        elif fea == "xy_1":
            feas = feas + ["x1", "y1"]
        elif fea == "xy_2":
            feas = feas + ["x2", "y2"]
        elif fea == "pathxy":
            feas = feas + paths
        elif fea == "radius":
            feas = feas + radii
        elif fea == "angles":
            feas = feas + angles
        elif fea == "start_end_touch_position":
            feas = feas + ["start_touch_position", "end_touch_position"]
        elif fea == "screen":
            feas = feas + ["screenh", "screenw"]
        elif fea == "resolution":
            feas = feas + ["resolutionh", "resolutionw"]
        elif fea == "2_1":
            feas = feas + ["x2_x1", "y2_y1"]
        elif fea == "distances":
            feas = feas + distances
        else:
            print(f'{fea} not exist!')

    sensor_feas = [f'real_gyroscope_x_{tt}' for tt in range(1, sensor_len+1)] + [f'real_gyroscope_y_{tt}' for tt in range(1, sensor_len+1)] + [f'real_gyroscope_z_{tt}' for tt in range(1, sensor_len+1)]
    print("sensor_feas:", sensor_feas)

    feas = feas + sensor_feas

    if args.merge:
        feas = feas + ["device_id"] # for sample construction

    # Record the metrics of each fold
    final_results = {
        "time_now": time_now,
        "end": False,
        "task": task_name,
        "model": model_name,
        "gpu": gpu_name,
        "total_time_cost": -1,
        "features": features
    }

    split_file = 'split.json'
    with open(split_file, 'r') as file:
        splits = json.load(file)

    summary = True
    for split in splits.keys():
        time_begin = time.time()
        print(f'---------fold: {split}---------')
        train_ids, test_ids = splits[split]["train"], splits[split]["test"]
        train_df, test_df = df[df['device_id'].isin(train_ids)], df[df['device_id'].isin(test_ids)]
        print('train:', train_df.shape, 'test:', test_df.shape)
        gc.collect()

        train_df, test_df = train_df.sort_values(by=['device_id', 'time']), test_df.sort_values(by=['device_id', 'time'])
        if task_name == "all-7":
            X_train, X_test, y_train, y_test = train_df[feas].values, test_df[feas].values, train_df['label_7'].values, test_df['label_7'].values
            if args.merge:
                X_train, X_test = merge_data(X_train), merge_data(X_test)
            if model_name not in ["lr", "lda", "svm", "lgb"]:
                encoder = OneHotEncoder(sparse_output=False)
                y_train = encoder.fit_transform(y_train.reshape(-1, 1))
                y_test = encoder.transform(y_test.reshape(-1, 1))
        if task_name == "cradle-3":
            X_train, X_test, y_train, y_test = train_df[feas].values, test_df[feas].values, train_df['label_3'].values, test_df['label_3'].values
            if args.merge:
                X_train, X_test = merge_data(X_train), merge_data(X_test)
            if model_name not in ["lr", "lda", "svm", "lgb"]:
                encoder = OneHotEncoder(sparse_output=False)
                y_train = encoder.fit_transform(y_train.reshape(-1, 1))
                y_test = encoder.transform(y_test.reshape(-1, 1))
        if task_name == "slide-2":
            train_df, test_df = train_df[train_df['label_2'].isin([0, 1])], test_df[test_df['label_2'].isin([0, 1])]
            train_df, test_df = train_df.sort_values(by=['device_id', 'time']), test_df.sort_values(by=['device_id', 'time'])
            X_train, X_test, y_train, y_test = train_df[feas].values, test_df[feas].values, train_df['label_2'].values, test_df['label_2'].values
            if args.merge:
                X_train, X_test = merge_data(X_train), merge_data(X_test)

        if model_name == "lgb":
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        gc.collect()

        # model train
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        if model_name == "ann":
            input_shape = (X_train.shape[1],)
            if args.merge:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
                input_shape = (X_train.shape[1],)
            model = create_ann(input_shape, task_name)
        if model_name == "cnn":
            input_shape = (X_train.shape[1], 1)
            if args.merge:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
                input_shape = (X_train.shape[1], X_train.shape[2])
            model = create_cnn(input_shape, task_name)
        if model_name == "lstm":
            X_train = X_train.reshape((X_train.shape[0], 1, -1))
            X_test = X_test.reshape((X_test.shape[0], 1, -1))
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = create_lstm(input_shape, task_name)
        if model_name == "lstmcnn":
            X_train = X_train.reshape((X_train.shape[0], 1, -1))
            X_test = X_test.reshape((X_test.shape[0], 1, -1))
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = create_lstmcnnNorm(input_shape, task_name)
        if model_name == "tcn":
            if args.merge:
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[2], X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[2], X_test.shape[1]))
                input_shape = (X_train.shape[1], X_train.shape[2])
            else:
                X_train = X_train.reshape((X_train.shape[0], 1, -1))
                X_test = X_test.reshape((X_test.shape[0], 1, -1))
                input_shape = (X_train.shape[1], X_train.shape[2])
            model = create_tcn(input_shape, task_name)
        if model_name == "lr":
            model = create_lr(task_name)
        if model_name == "lda":
            model = create_lda(task_name)
        if model_name == "svm":
            model = create_svm(task_name)
        if model_name == "knn":
            model = create_knn(task_name)
        if model_name == "dt":
            model = create_dt(task_name)
        if model_name == "lgb":
            num_class = 2
            if task_name == "cradle-3":
                num_class = 3
            if task_name == "all-7":
                num_class = 7
            params = {
                'objective': 'multiclass',
                'num_class': num_class,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 12,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'max_depth': 4
                }
            num_rounds = 200
            early_stop = lgb.early_stopping(stopping_rounds=20)   
        
        if model_name in dnn_models:
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=0, callbacks=[early_stopping]) # verbose=2
            if summary:
                model.summary()
                summary = False
        if model_name in ["lr", "lda", "svm", "knn", "dt"]:
            model.fit(X_train, y_train)
        if model_name == "lgb":
            model = lgb.train(params, train_data, num_rounds, valid_sets=[test_data], callbacks=[early_stop])

        # evaluation
        # val_acc = model.evaluate(X_test, y_test, verbose=0)[1]
        if model_name == "lgb":
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        else:
            y_pred = model.predict(X_test)

        if task_name == "slide-2":
            if model_name == "lgb":
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_pred_classes = (y_pred >= 0.5).astype(int)
            y_true_classes = y_test
        else:
            if model_name == "lgb":
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = y_test
            elif model_name not in ["lr", "lda", "svm"]:
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test, axis=1)
            else:
                y_pred_classes = y_pred
                y_true_classes = y_test
        result = cal_metrics(y_true_classes, y_pred_classes, task_name)

        # save model
        if model_name in dnn_models:
            model_save_path = output_dir + f'model_{split}.keras'
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

        time_cost = time.time() - time_begin
        print('fold time cost:', time_cost)
        result['fold_time_cost'] = time_cost
        final_results[split] = result

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # final results
    avg_results = get_average_results(final_results)
    final_results['avg'] = avg_results
    accuracy_avg = avg_results['accuracy_avg']

    final_results["end"] = True
    total_time_cost = time.time() - time_start
    final_results["total_time_cost"] = total_time_cost
    with open(result_path, 'w') as file:
        json.dump(final_results, file, indent=4)
    print('finished...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, help='cpu -1 or gpu 0 1 2 3', required=True)
    parser.add_argument('--task_name', type=str, help='slide-2, cradle-3, all-7', required=True)
    parser.add_argument('--model_name', type=str, help='ann, cnn, tcn, ...', required=True)
    parser.add_argument('--merge', type=bool, help='True or False', required=True)
    args = parser.parse_args()
    
    main(args)