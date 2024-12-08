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


from dataset import HDF5Dataset, compute_mean_and_std, debug_dataset
from evaluate import predict
from parameters import DATA_PATH, FILE_PATH, LOG_PATH, VAL_SPLIT_RATIO, MODEL_PATH
from train import train_model as train
from dataloader import get_hdf5_data_loaders
from preprocess_data import preprocess
from ensemble_model_outputs import ensemble_model_outputs
from model_list import models, all_model_name

NUM_EPOCHS = 2
BATCH_SIZE = 16

### ---------- Prepare device ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



### ---------- Get parameters ----------------------------------------
all_args = ('-pf', '-ps', '-pr', '-lm', '-ta', '-tm', '-tsa', '-tsm', '-ea', '-em', '-esa', '-esm', '-eea', '-eem', '-eesa', '-eesm',)
args = sys.argv[1:]

force_preprocess = False
skip_preprocess = False
run_preprocess = False

load_model = []
load_queue = False

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

e_evaluate_all = False
e_evaluate_model = []
e_evaluate_skip_model = []
e_skip_all_evaluation = False
e_evaluation_queue = False
e_evaluation_skip_queue = False

for arg in args:
    if arg in all_args:
        train_queue = False
        train_skip_queue = False
        evaluation_queue = False
        evaluation_skip_queue = False
        load_queue = False
        e_evaluation_queue = False
        e_evaluation_skip_queue = False
    if arg in ('-pf'):
        force_preprocess = True
        preprocess_specified = True
    elif arg in ('-ps'):
        skip_preprocess = True
        preprocess_specified = True
    elif arg in ('-pr'):
        run_preprocess = True
        preprocess_specified = True
    elif arg in ('-lm'):
        load_queue = True
    elif load_queue:
        load_model.append(arg)
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
    elif train_skip_queue:
        train_skip_model.append(arg)
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
    elif evaluation_skip_queue:
        evaluate_skip_model.append(arg)
    elif arg in ('-eea'):
        e_evaluate_all = True
    elif arg in ('-eem'):
        e_evaluation_queue = True
    elif arg in ('-eesa'):
        e_skip_all_evaluation = True
    elif arg in ('-eesm'):
        e_evaluation_skip_queue = True
    elif e_evaluation_queue:
        e_evaluate_model.append(arg)
    elif e_evaluation_skip_queue:
        e_evaluate_skip_model.append(arg)
    else:
        raise Exception("Unknown argument!")

#Arg validation
if (force_preprocess + skip_preprocess + run_preprocess) > 1 :
    raise Exception("Preprocess type flags are used more than once! Please remove until one remains! (-pf, -ps, -pr)")
if (train_all + any(train_model) + any(train_skip_model) + skip_all_train) > 1:
    raise Exception("Train type flag conflict! Use only one train type flag!")
if (evaluate_all + any(evaluate_model) + any(evaluate_skip_model) + skip_all_evaluation) > 1:
    raise Exception("Evaluation type flag conflict! Use only one evaluation type flag")
if (e_evaluate_all + any(e_evaluate_model) + any(e_evaluate_skip_model) + e_skip_all_evaluation) > 1:
    raise Exception("Ensemble evaluation type flag conflict! Use only one ensemble evaluation type flag")

### ---------- Pipeline ----------------------------------------

### ---------- Preprocess ----------

#if force_preprocess:
elif run_preprocess or not skip_preprocess:
    preprocess()


### ---------- Loading model ----------

for load_name in load_model:
    model_name = load_name.split('_')[0]
    models[model_name].load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{load_name}.pth")))


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


for model_name in train_models:
    if model_name not in models.keys():
        raise Exception("Unknown model name!")
    dataset = HDF5Dataset(DATA_PATH)
    mu, sd = compute_mean_and_std(dataset)
    print(f"Mean (mu): {mu}, Standard Deviation (sd): {sd}")
    debug_dataset(dataset)
    train_loader, val_loader = get_hdf5_data_loaders(DATA_PATH, BATCH_SIZE)
    models[model_name] = train(models[model_name], train_loader, val_loader, device, NUM_EPOCHS)





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

for model_name in evaluation_models:
    if model_name not in models.keys():
        raise Exception("Unknown model name!")
    predict(dir_path=FILE_PATH, model_name="unetR", CNN_model=models[model_name], mu=mu, sd=sd, device=device)


### ---------- Evaluate ensemble model ----------

if e_evaluate_skip_model:
    e_evaluation_models = all_model_name - e_evaluate_skip_model
elif e_evaluate_model:
    e_evaluation_models = e_evaluate_model
elif e_evaluate_all:
    e_evaluation_models = all_model_name
elif e_skip_all_evaluation:
    e_evaluation_models = []
else:
    e_evaluation_models = all_model_name

ensemble_model_outputs(base_dir=FILE_PATH, model_names=e_evaluation_models)
