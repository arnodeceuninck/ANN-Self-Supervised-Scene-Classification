import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.optim import Adam, SGD

import os

import torch

## settings
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 15
BATCH_SIZE = 15
NUM_EPOCHS = 20
LEARNING_RATE = 0.00005
OPTIMIZER = "Adam"


class TrainLogger:
    def __init__(self, train_config, train_loader, val_loader, model, descr="", verbose=True):
        self.train_config = train_config

        self.train_loss_list_per_epoch = []
        self.train_loss_list_per_itr = []
        self.train_loss_list_all_itr = []
        self.train_accuracy_per_epoch = []

        self.val_loss_list = []
        self.val_accuracy_per_epoch = []

        self.time_s = time.time()

        time_str = time.strftime("%Y%m%d_%H%M%S")
        self.identifier = descr + "_" + time_str

        self.output_folder = os.path.join("outputs", self.identifier)
        os.makedirs(self.output_folder, exist_ok=True)

        self.time_e = None

        self.train_size = len(train_loader.dataset)
        self.val_size = len(val_loader.dataset)

        self.batch_size = train_loader.batch_size

        self.fc_layers = get_n_fc_layers(model)

        self.verbose = verbose

    def notify_itr_end(self, itr, train_loss):
        if itr % 10 == 0:
            self.train_loss_list_per_itr.append(train_loss.item())

    def notify_epoch_end(self, epoch, model, eval_loss, eval_acc, train_loss=None, train_acc=None):

        if train_loss is None:
            train_loss = np.mean(self.train_loss_list_per_itr)
        self.train_loss_list_per_epoch.append(train_loss)
        self.train_loss_list_all_itr.extend(self.train_loss_list_per_itr)
        self.train_loss_list_per_itr = []

        self.train_accuracy_per_epoch.append(train_acc)

        self.val_loss_list.append(eval_loss)
        self.val_accuracy_per_epoch.append(eval_acc)
        #
        # try:
        #     self.save_model(model, epoch)
        # except Exception as e:
        #     print(f"Ignoring exception occurred while saving model: {e}")

        self.save_log_info()

        if self.verbose:
            print(f'Epoch {epoch} finished. '
                  f'Train loss: {self.train_loss_list_per_epoch[-1]}, '
                  f'Val loss: {eval_loss}, '
                  f'Val accuracy: {eval_acc}')

    def getting_worse(self, early_stop_count):
        if len(self.val_loss_list) < early_stop_count + 1:
            return False

        epoch_before = self.val_loss_list[-early_stop_count - 1]

        past_epochs = self.val_loss_list[-early_stop_count:]
        best_recent = np.min(past_epochs)

        return epoch_before < best_recent

    def get_train_duration(self):
        """Returns the duration of the training in minutes"""
        return (self.time_e - self.time_s) / 60

    def notify_train_end(self, model):
        self.time_e = time.time()
        self.save_log_info()
        self.save_model(model)
        self.plot_all()

    def analysis(self):
        # print loss and accuracy for train and validation per epoch
        print("Epoch: \tTrain loss / \tValidation loss ; \tTrain accuracy / \tValidation accuracy")
        prev_train_loss = None
        prev_train_acc = None
        prev_val_loss = None
        prev_val_acc = None

        first_val_loss_increase = None
        for i in range(len(self.train_loss_list_per_epoch)):
            curr_train_loss = self.train_loss_list_per_epoch[i]
            curr_val_loss = self.val_loss_list[i]
            curr_train_acc = self.train_accuracy_per_epoch[i]
            curr_val_acc = self.val_accuracy_per_epoch[i]

            # add + or - to indicate if the loss or accuracy is better than the previous epoch
            train_loss_arrow = "+" if prev_train_loss is None or curr_train_loss < prev_train_loss else "-"
            train_acc_arrow = "+" if prev_train_acc is None or curr_train_acc > prev_train_acc else "-"
            val_loss_arrow = "+" if prev_val_loss is None or curr_val_loss < prev_val_loss else "-"
            val_acc_arrow = "+" if prev_val_acc is None or curr_val_acc > prev_val_acc else "-"

            if first_val_loss_increase is None and prev_val_loss is not None and curr_val_loss > prev_val_loss:
                first_val_loss_increase = i

            arrows = train_loss_arrow + val_loss_arrow + train_acc_arrow + val_acc_arrow

            print(
                f"{arrows} {i}: \t{curr_train_loss:.8f} / \t{curr_val_loss:.8f} ; \t{curr_train_acc:.8f} / \t{curr_val_acc:.8f}")

            prev_train_loss = curr_train_loss
            prev_train_acc = curr_train_acc
            prev_val_loss = curr_val_loss
            prev_val_acc = curr_val_acc

        print("\n")
        print(f"Training duration: {self.get_train_duration()} minutes")
        print(f"Best validation accuracy: {self.get_best_val_acc(epoch=True)}")
        print(f"Best validation loss: {self.get_best_loss(epoch=True)}")
        print(f"First validation loss increase: {first_val_loss_increase}")

        self.plot_all(save=True)

    def plot_all(self, save=True):
        plt.plot(np.arange(len(self.train_loss_list_per_epoch)), self.train_loss_list_per_epoch, color='blue',
                 label='Train')
        plt.plot(np.arange(len(self.val_loss_list)), self.val_loss_list, color='red', label='Validation')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Train and Validation Loss')
        plt.savefig(f'{self.output_folder}/train_val_loss.png') if save else None

        plt.show()

        plt.cla()

        plt.plot(np.arange(len(self.val_accuracy_per_epoch)), self.val_accuracy_per_epoch, color='red',
                 label='Validation')
        plt.plot(np.arange(len(self.train_accuracy_per_epoch)), self.train_accuracy_per_epoch, color='blue',
                 label='Train')
        plt.legend()

        plt.title('Train and validation accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(f'{self.output_folder}/validation_accuracy.png') if save else None

        plt.show()

        plt.cla()

    def get_info_table(self):
        table = ""

        table_values = {
            "Input image size": self.train_config.input_size,
            "Learning rate": self.train_config.lr,
            "Optimizer": self.train_config.optimizer,
            "Batch size": self.batch_size,
            "Number of epochs": self.train_config.num_epochs,
            "Number of fully-connected layers": self.fc_layers
        }

        for key, value in table_values.items():
            table += f"{key}\t{value}\n"

        return table

    def save_model(self, model, epoch=None):
        if epoch is not None:
            filename = f'{self.output_folder}/model_epoch_{epoch}.pt'
        else:
            filename = f'{self.output_folder}/model.pt'

        torch.save(model.state_dict(), filename)

    def save_log_info(self):
        filename = f'{self.output_folder}/log.pickle'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        print(f"Saved log info to {filename}") if self.verbose else None

    def get_best_val_acc(self, epoch=False):
        # epoch or not indicates whether the epoch number should also be returned
        if epoch:
            return np.max(self.val_accuracy_per_epoch), np.argmax(self.val_accuracy_per_epoch)
        else:
            return np.max(self.val_accuracy_per_epoch)

    def get_best_loss(self, epoch=False):
        # epoch or not indicates whether the epoch number should also be returned
        if epoch:
            return np.min(self.val_loss_list), np.argmin(self.val_loss_list)
        else:
            return np.min(self.val_loss_list)



class TrainConfig:
    def __init__(self, lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, criterion=None, optimizer=OPTIMIZER,
                 input_size=INPUT_SHAPE, batch_size=BATCH_SIZE):
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = criterion if criterion else torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.batch_size = batch_size # warning: be sure to use train_with_custom_batch_size

        if input_size != INPUT_SHAPE:
            print(
                f"Warning: different input size from config not supported (since this is the train config and not loader config)")

        self.input_size = input_size

    def get_optimizer(self, model):
        if self.optimizer == "Adam":
            return Adam(model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            return SGD(model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")


def get_n_fc_layers(model):
    """
    Returns the number of fully connected layers of a model.

    :param model: the model to get the number of fully connected layers from

    :return: the number of fully connected layers
    """
    n_fc_layers = 0
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            n_fc_layers += 1

    return n_fc_layers

def load_log_info(path):
    with open(path, 'rb') as f:
        log = pickle.load(f)
    return log