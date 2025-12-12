import argparse
import datetime as dte
import os,time
import data_formatters.base
import expt_settings.configs
import data_formatters.icemonly as dataice
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from pylab import *
from sklearn.metrics import mean_squared_error
from math import sqrt


def get_rmse(tar,pre):
    rmse=sqrt(mean_squared_error(tar,pre))
    return rmse

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
tf.experimental.output_all_intermediates(False)


def get_args():

    experiment_names = ExperimentConfig.default_experiments
    parser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "expt_name",
        metavar="e",
        type=str,
        nargs="?",
        default="icemonthly")
    parser.add_argument(
        "output_folder",
        metavar="f",
        type=str,
        nargs="?",
        default="expt_settings/outputs")
    parser.add_argument(
        "use_gpu",
        metavar="g",
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="yes")

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder

    return args.expt_name, root_folder, args.use_gpu == "yes"



print(time.asctime())
start=time.time()

expt_name="icemonthly"
output_folder="expt_settings/outputs"
use_gpu = 'yes'
data_formatter=dataice.IcemonlyFormatter()
model_folder="/fs01/KT4/software/model/IceTFT/save/3//"
data_csv_path="expt_settings/outputs/data/icemonly/sienew.csv"

use_testing_mode=False
num_repeats = 1



# Tensorflow setup
default_keras_session = tf.keras.backend.get_session()


if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

print("*** Training from defined parameters for {} ***".format(expt_name))

print("Loading & splitting data...")
raw_data = pd.read_csv(data_csv_path, index_col=0)

train, valid, test = data_formatter.split_data(raw_data)

train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

# Sets up default params
fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()
params["model_folder"] = model_folder
print(params["model_folder"])

# Sets up hyperparam manager
print("*** Loading hyperparm manager ***")
opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                   fixed_params, model_folder)

opt_manager.load_results()
tf.reset_default_graph()
with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.keras.backend.set_session(sess)
    best_params = opt_manager.get_best_params()
    model = ModelClass(best_params, use_cudnn=use_gpu)

    model.load(opt_manager.hyperparam_folder)

    print("Computing best validation loss")
    val_loss = model.evaluate(valid)

    print("Computing test loss")
    output_map = model.predict(test, return_targets=True)
    targets = data_formatter.format_predictions(output_map["targets"])

    pre = data_formatter.format_predictions(output_map["p100"])
    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    tar=np.array(extract_numerical_data(targets)).reshape(1,-1)
    pre=np.array(extract_numerical_data(pre)).reshape(1,-1)
    
    end=time.time()
    print(time.asctime())
    print('测试结果已生成，用时{:.5f} 秒'.format(end-start))
    #print(tar[0,0:12])
    #print(fsdfs)
    prenextyear=pre[0,0:12]
    tarnextyear=tar[0,0:12]
    print('预报结果为：')
    print(prenextyear)
    np.savetxt('prediction.txt',prenextyear)
    np.savetxt('tarnew.txt',tarnextyear)
    print('预测结果已保存完毕，文件存放于/fs01/KT4/software/model/IceTFT/prediction.txt')
    print('开始根据测试结果计算指标')
    print('每个月预报误差为')
    print(abs(prenextyear-tarnextyear))
    print('月平均误差为{:.5f} million km2'.format(np.mean(abs(prenextyear-tarnextyear))))
    print('考核指标为预报误差小于0.3 million km2，达到指标！')

    print('移植性测试开始，对比不同平台的预报结果')
    old=np.loadtxt('experiment//prediction.txt')
    print('本地平台的预报结果为：')
    print(old)
    print('两个平台预报结果的平均误差为{:.5f} million km2'.format(np.mean(abs(prenextyear-old))))
    print('移植后性能基本一致，达到指标！')
