import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score
import h5py
import cv2
import numpy as np
import os
import tifffile as tiff
import sys

from baseline import BaselineModel
from dataset import HDF5Dataset
from evaluate import predict, Score
from parameters import DATA_PATH, FILE_PATH, LOG_PATH, VAL_SPLIT_RATIO
from train import train
from dataloader import CombinedDataLoader
from preprocess_data import preprocess
from attention_unet import AttentionUNet
from dense_unet import DenseUNet

### ---------- Prepare device ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### ---------- Prepare models ----------------------------------------
model_1_name = 'baseline'
model_2_name = 'denseUnet'
model_3_name = 'attentionUnet'
model_4_name = ''
model_5_name = ''
all_model_name = [model_1_name, model_2_name, model_3_name, model_4_name, model_5_name]

#TODO: Add models
models = {}
models[model_1_name] = BaselineModel().to(device)
models[model_2_name] = DenseUNet().to(device)
models[model_3_name] = AttentionUNet().to(device)
#models[model_4_name] = 
#models[model_5_name] = 


### ---------- Get parameters ----------------------------------------
args = sys.argv[1:]

force_preprocess = False
skip_preprocess = False
run_preprocess = False

train_all = False
train_model = []
train_skip_model = []
skip_all_train = False
train_queue = False
train_skip_queue = False

evaluate_all = False
evaluate_model = []
evaluate_skip_model = []
skip_all_evaluation = False
evaluation_queue = False
evaluation_skip_queue = False

for arg in args:
    if arg in ('-pf'):
        force_preprocess = True
        preprocess_specified = True
    elif arg in ('-ps'):
        skip_preprocess = True
        preprocess_specified = True
    elif arg in ('-pr'):
        run_preprocess = True
        preprocess_specified = True
    elif arg in ('-ta'):
        train_all = True
    elif arg in ('-tm'):
        train_queue = True
    elif arg in ('-tsa'):
        skip_all_train = True
    elif arg in ('-tsm'):
        train_skip_queue = True
    elif train_queue:
        train_model.append(arg)
        train_queue = False
    elif train_skip_queue:
        train_skip_model.append(arg)
        train_skip_queue = False
    elif arg in ('-ea'):
        evaluate_all = True
    elif arg in ('-em'):
        evaluation_queue = True
    elif arg in ('-esa'):
        skip_all_evaluation = True
    elif arg in ('-esm'):
        evaluation_skip_queue = True
    elif evaluation_queue:
        evaluate_model.append(arg)
        evaluation_queue = False
    elif evaluation_skip_queue:
        evaluate_skip_model.append(arg)
        evaluation_skip_queue = False
    else:
        raise Exception("Unknown argument!")

#Arg validation
if (force_preprocess + skip_preprocess + run_preprocess) > 1 :
    raise Exception("Preprocess type flags are used more than once! Please remove until one remains! (-pf, -ps, -pr)")
if (train_all + any(train_model) + any(train_skip_model) + skip_all_train) > 1:
    raise Exception("Train type flag conflict! Use only one train type flag!")
if (evaluate_all + any(evaluate_model) + any(evaluate_skip_model) + skip_all_evaluation) > 1:
    raise Exception("Evaluation type flag conflict! Use only one evaluation type flag")


### ---------- Pipeline ----------------------------------------

### ---------- Preprocess ----------

#if force_preprocess:
elif run_preprocess or not skip_preprocess:
    preprocess()


### ---------- Training ----------

if train_skip_model:
    train_models = all_model_name - train_skip_model
elif train_model:
    train_models = train_model
elif train_all:
    train_models = all_model_name
elif skip_all_train:
    train_models = []
else:
    train_models = all_model_name


for model in train_models:
    if model not in models.keys():
        raise Exception("Unknown model name!")
    dataset = HDF5Dataset(DATA_PATH, subset_size=5)
    dataloader = CombinedDataLoader(dataset, VAL_SPLIT_RATIO, 4)
    train(models[model], dataloader, 1000, device, 5, 5, model)


### ---------- Evaluate ----------

if evaluate_skip_model:
    evaluation_models = all_model_name - evaluate_skip_model
elif evaluate_model:
    evaluation_models = evaluate_model
elif evaluate_all:
    evaluation_models = all_model_name
elif skip_all_evaluation:
    evaluation_models = []
else:
    evaluation_models = all_model_name

for model in evaluation_models:
    if model not in models.keys():
        raise Exception("Unknown model name!")
    mu, sd = dataloader.dataset.get_mean_std()
    predict(FILE_PATH, models[model], mu, sd, device)
    Score(FILE_PATH, LOG_PATH)
