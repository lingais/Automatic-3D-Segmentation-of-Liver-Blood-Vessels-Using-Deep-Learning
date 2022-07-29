"""Class to do training of the network."""

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from dvn import losses as ls
from dvn import misc as ms


class Solver(object):
    default_optim_args = {"lr": 0.01,
                          "weight_decay": 0.}

    def __init__(self, optim=torch.optim.SGD, optim_args={},
                 loss_func=ls.dice_loss):

        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.best_train_dice = -1
        self.best_val_dice = -1
        self.best_train_model = None
        self.best_val_model = None

        self._reset_histories()
        self.writer = SummaryWriter()

    def _reset_histories(self):
        """Resets train and val histories for the accuracy and the loss. """

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        self.train_dice_coeff_history = []
        self.val_dice_coeff_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: object initialized from a torch.nn.Module
        - train_loader: train data (currently using nonsense data)
        - val_loader: val data (currently using nonsense data)
        - num_epochs: total number of epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        model.to(device)
        model.train()

        print("START TRAIN")

        for epoch in range(num_epochs):
            # Training
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
                optim.zero_grad()

                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                dice_coeff = ms.dice_coeff(outputs, targets).detach().cpu().numpy()
                self.train_loss_history.append(loss.detach().cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    dice_coeff = ms.dice_coeff(outputs, targets).detach().cpu().numpy()
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' %
                          (i + epoch * iter_per_epoch,
                           iter_per_epoch * num_epochs,
                           train_loss))
                    self.writer.add_scalar("Soft dice loss", train_loss, i + epoch * iter_per_epoch)
                    self.writer.add_scalar("Dice coefficient", dice_coeff, i + epoch * iter_per_epoch)

            _, preds = torch.max(outputs, 1)
            train_acc = np.mean((preds == targets).detach().cpu().numpy())
            dice_coeff = ms.dice_coeff(outputs, targets).detach().cpu().numpy()
            self.train_acc_history.append(train_acc)
            self.train_dice_coeff_history.append(dice_coeff)

            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss/dice: %.3f/%.3f/%.3f' %
                      (epoch + 1, num_epochs, train_acc, train_loss, dice_coeff))

            # Validation
            val_losses = []
            val_scores = []
            model.eval()
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                val_losses.append(loss.detach().cpu().numpy())

                _, preds = torch.max(outputs, 1)
                scores = np.mean((preds == targets).detach().cpu().numpy())
                val_scores.append(scores)

            self.val_dice_coeff_history.append(dice_coeff)
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss/dice: %.3f/%.3f/%.3f' % (epoch + 1,
                                                                             num_epochs,
                                                                             val_acc,
                                                                             val_loss, dice_coeff))

            model.train()

        #################################################################

        print("FINISH")
