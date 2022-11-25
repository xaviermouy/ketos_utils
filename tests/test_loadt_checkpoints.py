# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:33:29 2022

@author: xavier.mouy
"""
import tensorflow as tf

# from ketos.neural_networks.dev_utils.nn_interface import NNInterface
from ketos.neural_networks.resnet import ResNetInterface
from ketos.data_handling.data_feeding import BatchGenerator
import ketos.data_handling.database_interface as dbi
import os

db_file = r"D:\Detector\ketos\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700\database.h5"
recipe_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\FS-400ms-bs256-ep50-norm\recipe.json"
checkpoint_dir = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\FS-400ms-bs256-ep50-norm"
out_dir = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\test"
spec_config_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\FS-400ms-bs256-ep50-norm\spec_config.json"
batch_size = 16


db = dbi.open_file(db_file, "r")
train_data = dbi.open_table(db, "/train/data")
val_data = dbi.open_table(db, "/test/data")


train_generator = BatchGenerator(
    batch_size=batch_size,
    data_table=train_data,
    output_transform_func=ResNetInterface.transform_batch,
    shuffle=True,
    refresh_on_epoch_end=True,
)

val_generator = BatchGenerator(
    batch_size=batch_size,
    data_table=val_data,
    output_transform_func=ResNetInterface.transform_batch,
    shuffle=True,
    refresh_on_epoch_end=False,
)


latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
resnet = ResNetInterface.build_from_recipe_file(recipe_file)
resnet.train_generator = train_generator
resnet.val_generator = val_generator
resnet.checkpoint_dir = checkpoint_dir
resnet.log_dir = out_dir


resnet.model.load_weights(latest_checkpoint).expect_partial()

resnet.save_model(
    os.path.join(out_dir, "ketos_model.kt"), audio_repr_file=spec_config_file
)
resnet.save_model(
    os.path.join(out_dir, "ketos_model.ktpb"), audio_repr_file=spec_config_file
)
#


# latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
# resnet = ResNetInterface.build_from_recipe_file(recipe_file)


# resnet = tf.model.load_weights(latest_checkpoint).expect_partial()

# # resnet.train_loop(n_epochs=params['n_epochs'], log_csv=True, verbose=True)

# from ketos.neural_networks.resnet import ResNetInterface
# path_to_pre_trained_model = "my_model/killer_whale.kt"
# killer_whale_classifier = ResNetInterface.load_model_file(path_to_pre_trained_model)


# In the meantime, you can try to load the weights from the latest checkpoint and then proceed with your training.
# After instantiating your network, you can add the following lines before you call the train_loop() method

# import tensorflow as tf
# latest_checkpoint = tf.train.latest_checkpoint("path/to/checkpoints")
# my_network.model.load_weights(latest_checkpoint).expect_partial()
# my_network.train_loop(...)

# Let me know if this works for you.

# Cheers,
# Fabio
