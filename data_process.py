# coding:utf-8

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time, datetime
import utils


def main():
    # extract station id list in Beijing
    df_airq = pd.read_csv('./data/microsoft_urban_air_data/airquality.csv')
    station_id_list = np.unique(df_airq['station_id'])[:36]     # first 36 stations are in Beijing
    print(station_id_list)
    
    # Calculate the influence degree (defined as the Pearson correlation coefficient) between the center station and other stations
    r_thred = 0.85
    center_station_id = 1013
    station_id_related_list = []
    df_one_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(center_station_id))
    v_list_1 = list(df_one_station['PM25_Concentration'])
    for station_id_other in station_id_list:
        df_one_station_other = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(station_id_other))
        v_list_2 = list(df_one_station_other['PM25_Concentration'])
        r, p = stats.pearsonr(v_list_1, v_list_2)
        if r > r_thred:
            station_id_related_list.append(station_id_other)
        print('{}  {}  {:.3f}'.format(center_station_id, station_id_other, r))
    print(len(station_id_related_list), station_id_related_list)
    
    # generate x and y
    # x_shape: [example_count, num_releated, seq_step, feat_size]
    # y_shape: [example_count,]
    print('Center station: {}\nRelated stations: {}'.format(center_station_id, station_id_related_list))
    feat_names = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration',
                  'weather', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
    x_length = 24
    y_length = 1
    y_step = 1
    x = []
    y = []
    for station_id in station_id_related_list:
        df_one_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(station_id))
        x_one = []
        for start_id in range(0, len(df_one_station)-x_length-y_length+1-y_step+1, y_length):
            x_data = np.array(df_one_station[feat_names].iloc[start_id: start_id+x_length])
            y_list = np.array(df_one_station['PM25_Concentration'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
            if np.isnan(x_data).any() or np.isnan(y_list).any():
                continue
            x_one.append(x_data)
            if station_id == center_station_id:
                y.append(np.mean(y_list))
        if len(x_one) <= 0:
            continue
        x_one = np.array(x_one)
        x.append(x_one)
        print('station_id: {}  x_shape: {}'.format(station_id, x_one.shape))

    x = np.array(x)
    x = x.transpose((1, 0, 2, 3))
    y = np.array(y)
    print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))
    
    # Save the four dimensional data as pickle file
    utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
    utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
    print('x_shape: {}\ny_shape: {}'.format(x.shape, y.shape))


if __name__ == '__main__':
    main()
