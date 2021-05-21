# coding:utf-8

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
import models
import utils
import config as cfg


def eval(net, x_test, y_test, plot=False):
    x_valid = x_test
    y_valid = y_test
    print('\nStart evaluating...\n')
    net.eval()
    rmse_train_list = []
    rmse_valid_list = []
    mae_valid_list = []
    y_valid_pred_final = []
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    h_state = None
    y_valid_pred_final = []
    rmse_valid = 0.0
    cnt = 0
    for start in range(len(x_valid) - cfg.batch_size + 1):
        x_input_valid = torch.tensor(x_valid[start:start + cfg.batch_size], dtype=torch.float32)
        y_true_valid = torch.tensor(y_valid[start:start + cfg.batch_size], dtype=torch.float32)
        if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
            y_valid_pred, _h_state = net(x_input_valid, h_state)
        else:
            y_valid_pred = net(x_input_valid)
        y_valid_pred_final.extend(y_valid_pred.data.numpy())
        loss_valid = criterion(y_valid_pred, y_true_valid).data
        mse_valid_batch = loss_valid.numpy()
        rmse_valid_batch = np.sqrt(mse_valid_batch)
        rmse_valid += mse_valid_batch
        cnt += 1
    y_valid_pred_final = np.array(y_valid_pred_final).reshape((-1, 1))
    rmse_valid = np.sqrt(rmse_valid / cnt)
    mae_valid = metrics.mean_absolute_error(y_valid, y_valid_pred_final)
    r2_valid = metrics.r2_score(y_valid, y_valid_pred_final)

    print('\nRMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}\n'.format(rmse_valid, mae_valid, r2_valid))
    return rmse_valid, mae_valid, r2_valid


def main():
    # Hyper Parameters
    cfg.print_params()
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)

    # Load data
    print('\nLoading data...\n')
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y)

    # Generate model
    net = None
    if cfg.model_name == 'RNN':
        net = models.SimpleRNN(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'GRU':
        net = models.SimpleGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'LSTM':
        net = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'TCN':
        net = models.TCN(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN':
        net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                          num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))

    # Load model parameters
    net.load_state_dict(torch.load(cfg.model_save_pth))
    print(utils.get_param_number(net=net))

    # Evaluation
    eval(net, x_test, y_test)


if __name__ == '__main__':
    main()
