# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:29:30 2022

@author: xdata
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
import warnings
import pdb
warnings.filterwarnings('ignore')
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化


set_seed(42)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train = train.dropna()
train.reset_index(drop=True,inplace=True)

label = 'Main_steam_flow_rate'

# =============================================================================
# ['时间', '主蒸汽流量', 'CO含量', 'HCL含量', 'NOx含量', 'SO2含量', '一次风调门', '一次风量',
#        '主蒸汽流量设定值', '二次风调门', '二次风量', '引风机转速', '推料器启停', '推料器手动指令',
#        '推料器自动投退信号', '推料器自动指令', '氧量设定值', '汽包水位', '炉排启停', '炉排实际运行指令',
#        '炉排手动指令', '炉排自动投退信号', '给水流量']
# =============================================================================

train.columns = ['Time', 'Main_steam_flow_rate', 'CO_content', 'HCL_content', 'NOx_content', 'SO2_content', 'Primary_air_regulator', 'Primary_air_volume',
       'Main_steam_flow_rate_setting', 'Secondary_air_regulator', 'Secondary_air_volume', 'Inducer_fan_speed', 'Pusher_start_stop', 'Pusher_manual_command',
       'Pusher_automatic_throwback_signal', 'Pusher_automatic_command', 'Oxygen_setpoint', 'Ladle_water_level', 'Grate_start_stop', 'Grate_actual_operation_command',
       'Grate_manual_command', 'Grate_automatic_throw-out_signal', 'Feedwater_flow']

test.columns = ['Time', 'CO_content', 'HCL_content', 'NOx_content', 'SO2_content', 'Primary_air_regulator', 'Primary_air_volume',
       'Main_steam_flow_rate_setting', 'Secondary_air_regulator', 'Secondary_air_volume', 'Inducer_fan_speed', 'Pusher_start_stop', 'Pusher_manual_command',
       'Pusher_automatic_throwback_signal', 'Pusher_automatic_command', 'Oxygen_setpoint', 'Ladle_water_level', 'Grate_start_stop', 'Grate_actual_operation_command',
       'Grate_manual_command', 'Grate_automatic_throw-out_signal', 'Feedwater_flow']

data = pd.concat([train, test])



data['gas'] = data.CO_content + data.HCL_content + data.SO2_content + data.NOx_content
for f in ['Feedwater_flow', 'Oxygen_setpoint', 'Primary_air_volume', 'Main_steam_flow_rate_setting', 'Ladle_water_level']:
    shift_f = []
    shift_d = []
    for i in range(200):
        colname = f+'_shift_{}'.format(i+1)
        data[colname] = data[f].shift(i+1)
        shift_f.append(colname)

        if i>10:
            shift_d.append(colname)

        # .diff用于计算一列中某元素与该列中前？个元素的差值（默认前一个元素）
        colname = f+'_diff_{}'.format(i+1)
        data[colname] = data[f].diff(i+1)
# =============================================================================
#         if (i+1)%50==0:
#             data[f+'_fore_{}_steps_mean'.format(i)] = data[shift_f].mean(1)
#             data[f+'_fore_{}_steps_max'.format(i)] = data[shift_f].max(1)
#             data[f+'_fore_{}_steps_min'.format(i)] = data[shift_f].min(1)
#             data[f+'_fore_{}_steps_std'.format(i)] = data[shift_f].std(1)
# =============================================================================

    # data[f+'_diff'] = data[f].diff(1)
    # 对每一行：shift_列的值取平均，即将前n天的值取平均，取最大值，最最小值，最标准差
    #
    data[f+'_fore_7_steps_mean'] = data[shift_f].mean(1)
    data[f+'_fore_7_steps_max'] = data[shift_f].max(1)
    data[f+'_fore_7_steps_min'] = data[shift_f].min(1)
    data[f+'_fore_7_steps_std'] = data[shift_f].std(1)


    data.drop(shift_d, axis=1, inplace=True)
#
data['Grate_start_stop'] = data['Grate_start_stop'].map(int)
data['Pusher_start_stop'] = data['Pusher_start_stop'].map(int)
data['Pusher_automatic_throwback_signal'] = data['Pusher_automatic_throwback_signal'].map(int)
data['Grate_automatic_throw-out_signal'] = data['Grate_automatic_throw-out_signal'].map(int)



test = data.tail(1800)
train = data.iloc[:-1800, :]
train.reset_index(drop=True,inplace=True)

valid = train.iloc[-1800:, :]
train = train.iloc[:-1800, :]

valid.reset_index(drop=True, inplace=True)



features = train.columns.drop(['Time',label]).tolist()

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'l2_root',
    'learning_rate': 0.01,
    'reg_alpha': 0.7,
    'reg_lambda': 35,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'feature_fraction': 0.7,
    "random_seed": 1,
}

dtrain = lgb.Dataset(train[features],label=train[label])
dvalid = lgb.Dataset(valid[features],label=valid[label])

watchlist = [dtrain,dvalid]
model = lgb.train(params,train_set=dtrain,num_boost_round=10000,early_stopping_rounds=1000,valid_sets=watchlist,verbose_eval=200)

feature_importance = pd.DataFrame()
feature_importance['fea_name'] = features
feature_importance['fea_imp'] = model.feature_importance()
feature_importance = feature_importance.sort_values('fea_imp',ascending = False)




sub = test[['Time']]
sub['lgb'+label] = model.predict(test[features])
sub.reset_index(inplace=True)
sub.columns=['ID','Time','Steam_flow']

sub.to_csv('result.csv',index=False)


