import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd
import pickle
import numpy as np
from training import TrainLogger, TrainConfig

# load pickle file
def hyperopt_analysis(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    lowest_loss = np.inf
    for i, ((lr, batch_size), log) in enumerate(results):
        print(f"{i} -> acc val/train {log.get_best_val_acc()} / {log.get_best_train_acc()} / loss {log.get_best_loss()}")
        if log.get_best_loss() < lowest_loss:
            lowest_loss = log.get_best_loss()
            best_log = log
            print(f"new best loss: {lowest_loss}")
    print(f"best loss: {lowest_loss}")
    # print("debug")

# after 3 epochs
hyperopt_analysis("outputs/hyperopt_clf-pert-pretext-lr10-5_20230825_210356_all.pickle")

# after 6 epochs
# hyperopt_analysis("outputs/hyperopt_clf-pert-pretext-lr10-5_20230826_034528_all.pickle")