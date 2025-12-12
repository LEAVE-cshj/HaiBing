# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

import argparse
import datetime as dte
import os
from sklearn.metrics import mean_squared_error
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

import tensorflow_model_optimization as tfmot
import zipfile
import tempfile

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
tf.experimental.output_all_intermediates(True)

#GPU ID
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'




def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        print("Apply pruning to Dense")
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

def get_gzipped_model_size(model):
    tf.keras.models.save_model(model, 'test.h5', include_optimizer=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("baseline_model.tflite", "wb").write(tflite_model)

    return os.path.getsize(zipped_file)

def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):
    """Trains tft based on defined model params.

    Args:
      expt_name: Name of experiment
      use_gpu: Whether to run tensorflow with GPU operations
      model_folder: Folder path where models are serialized
      data_csv_path: Path to csv file containing data
      data_formatter: Dataset-specific data fromatter (see
        expt_settings.dataformatter.GenericDataFormatter)
      use_testing_mode: Uses a smaller models and data sizes for testing purposes
        only -- switch to False to use original default settings
    """

    num_repeats = 1

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))

    # Tensorflow setup
    default_keras_session = tf.keras.backend.get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="/gpu:3", gpu_id=3)

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

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10
        

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_loss = np.Inf
    metrics_name=''
    clonemodel=None
    for _ in range(num_repeats):

        tf.reset_default_graph()

        with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

            tf.keras.backend.set_session(sess)

            params = opt_manager.get_next_parameters()
            tftmodel = ModelClass(params, use_cudnn=use_gpu)

            if not tftmodel.training_data_cached():
                tftmodel.cache_batched_data(train, "train", num_samples=train_samples)
                tftmodel.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.global_variables_initializer())
            tftmodel.fit()

            val_loss= tftmodel.evaluate()

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, tftmodel)
                best_loss = val_loss



            metrics_name=tftmodel.model.metrics_names
            #clonemodel=tftmodel.get_model()
            
            tf.keras.backend.set_session(default_keras_session)


            

    print("*** Running tests ***")
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
        

        #p50_forecast = data_formatter.format_predictions(output_map["p50"])
        #p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]
        print(targets)
        tar=np.array(extract_numerical_data(targets)).reshape(1,-1)
        pre=np.array(extract_numerical_data(pre)).reshape(1,-1)

        np.savetxt("prenew.txt",pre)
        np.savetxt("tarnew.txt",tar)
        #print(mean_squared_error(pre, tar))
       
        
        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    


if __name__ == "__main__":
    def get_args():
        """Gets settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="icemonly",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default="expt_settings/outputs/",
            help="Path to folder for data download")
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=bool,
            nargs="?",
            choices=[False, True],
            default=True,
            help="Whether to use gpu for training.")

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu


    name, output_folder, use_tensorflow_with_gpu = get_args()
    

    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    # Customise inputs to main() for new datasets.
    for num in range(0,20):
        main(
            expt_name=name,
            use_gpu=use_tensorflow_with_gpu,
            model_folder=os.path.join(config.model_folder, "fixed"),
            data_csv_path=config.data_csv_path,
            data_formatter=formatter,
            use_testing_mode=False)  # Change to false to use original default params
        os.system('mkdir save/'+str(num))
        os.system('cp prenew.txt save/'+str(num)+'/prenew.txt')
        os.system('cp tarnew.txt save/'+str(num)+'/tarnew.txt')        
        os.system('cp expt_settings/outputs/saved_models/icemonly/fixed/results.csv save/'+str(num)+'/results.csv')
        os.system('cp expt_settings/outputs/saved_models/icemonly/fixed/params.csv save/'+str(num)+'/params.csv')
        os.system('cp expt_settings/outputs/saved_models/icemonly/fixed/checkpoint save/'+str(num)+'/checkpoint')
        os.system('cp expt_settings/outputs/saved_models/icemonly/fixed/TemporalFusionTransformer.ckpt.data-00000-of-00001 save/'+str(num)+'/TemporalFusionTransformer.ckpt.data-00000-of-00001')
        os.system('cp expt_settings/outputs/saved_models/icemonly/fixed/TemporalFusionTransformer.ckpt.index save/'+str(num)+'/TemporalFusionTransformer.ckpt.index')
        os.system('cp expt_settings/outputs/saved_models/icemonly/fixed/TemporalFusionTransformer.ckpt.meta save/'+str(num)+'/TemporalFusionTransformer.ckpt.meta')
