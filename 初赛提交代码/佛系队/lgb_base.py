# %%
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
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold


# %%

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化


set_seed(42)

# %%

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train = train.dropna()
train.reset_index(drop=True, inplace=True)

label = 'Main_steam_flow_rate'

# =============================================================================
# ['时间', '主蒸汽流量', 'CO含量', 'HCL含量', 'NOx含量', 'SO2含量', '一次风调门', '一次风量',
#        '主蒸汽流量设定值', '二次风调门', '二次风量', '引风机转速', '推料器启停', '推料器手动指令',
#        '推料器自动投退信号', '推料器自动指令', '氧量设定值', '汽包水位', '炉排启停', '炉排实际运行指令',
#        '炉排手动指令', '炉排自动投退信号', '给水流量']
# =============================================================================

train.columns = ['Time', 'Main_steam_flow_rate', 'CO_content', 'HCL_content', 'NOx_content', 'SO2_content',
                 'Primary_air_regulator', 'Primary_air_volume',
                 'Main_steam_flow_rate_setting', 'Secondary_air_regulator', 'Secondary_air_volume', 'Inducer_fan_speed',
                 'Pusher_start_stop', 'Pusher_manual_command',
                 'Pusher_automatic_throwback_signal', 'Pusher_automatic_command', 'Oxygen_setpoint',
                 'Ladle_water_level', 'Grate_start_stop', 'Grate_actual_operation_command',
                 'Grate_manual_command', 'Grate_automatic_throw-out_signal', 'Feedwater_flow']

test.columns = ['Time', 'CO_content', 'HCL_content', 'NOx_content', 'SO2_content', 'Primary_air_regulator',
                'Primary_air_volume',
                'Main_steam_flow_rate_setting', 'Secondary_air_regulator', 'Secondary_air_volume', 'Inducer_fan_speed',
                'Pusher_start_stop', 'Pusher_manual_command',
                'Pusher_automatic_throwback_signal', 'Pusher_automatic_command', 'Oxygen_setpoint', 'Ladle_water_level',
                'Grate_start_stop', 'Grate_actual_operation_command',
                'Grate_manual_command', 'Grate_automatic_throw-out_signal', 'Feedwater_flow']

data = pd.concat([train, test])

# %%

data['gas'] = data.CO_content + data.HCL_content + data.SO2_content + data.NOx_content
for f in ['Feedwater_flow', 'Oxygen_setpoint', 'Primary_air_volume', 'Main_steam_flow_rate_setting', 'Ladle_water_level']:
    shift_f = []
    shift_d = []
    for i in range(200):
        colname = f + '_shift_{}'.format(i + 1)
        data[colname] = data[f].shift(i + 1)
        shift_f.append(colname)

        # # 与上n分钟的变化比例特征
        # if i%60==0:
        #     temp_name = f+'_ratio_{}'.format(i)
        #     data[temp_name] = data[colname]/data[colname].shift(i+1)

        # .diff用于计算一列中某元素与该列中前？个元素的差值（默认前一个元素）
        colname = f + '_diff_{}'.format(i + 1)
        data[colname] = data[f].diff(i + 1)

    # data[f+'_diff'] = data[f].diff(1)
    # 对每一行：shift_列的值取平均，即将前n天的值取平均，取最大值，最最小值，最标准差

    data[f + '_fore_steps_mean'] = data[shift_f].mean(1)
    data[f + '_fore_steps_max'] = data[shift_f].max(1)
    data[f + '_fore_steps_min'] = data[shift_f].min(1)
    data[f + '_fore_steps_std'] = data[shift_f].std(1)
    data[f + '_fore_steps_std'] = data[shift_f].skew(1)

    data.drop(shift_f, axis=1, inplace=True)
#
data['Grate_start_stop'] = data['Grate_start_stop'].map(int)
data['Pusher_start_stop'] = data['Pusher_start_stop'].map(int)
data['Pusher_automatic_throwback_signal'] = data['Pusher_automatic_throwback_signal'].map(int)
data['Grate_automatic_throw-out_signal'] = data['Grate_automatic_throw-out_signal'].map(int)


# %%

# groupby + shift(差值特征) + transform(count, mean, max, min, skew)

def get_shift_feats(data, gap_list=[1], gp_col='', target_col=''):
    # gp_col可以是个列表进行多次分组，e.g. gp_col = [id1, id2]
    for gap in gap_list:
        # 后面减前面
        data[f'{gp_col}_{target_col}_next_{gap}'] = data.groupby(gp_col)[target_col].shift(-gap)
        data[f'{gp_col}_{target_col}_next_{gap}'] = data[f'{gp_col}_{target_col}_next_{gap}'] - data[target_col]

        # 前面减后面
        data[f'{gp_col}_{target_col}_prev_{gap}'] = data.groupby(gp_col)[target_col].shift(+gap)
        data[f'{gp_col}_{target_col}_prev_{gap}'] = data[f'{gp_col}_{target_col}_next_{gap}'] - data[target_col]

        # 统计其不为nan的值
        data[f'{gp_col}_{target_col}_next_{gap}_count'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_next_{gap}'].transform('count')
        data[f'{gp_col}_{target_col}_prev_{gap}_count'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_prev_{gap}'].transform('count')

        # 统计其平均值
        data[f'{gp_col}_{target_col}_next_{gap}_mean'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_next_{gap}'].transform('mean')
        data[f'{gp_col}_{target_col}_prev_{gap}_mean'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_prev_{gap}'].transform('mean')

        # 统计其最大值
        data[f'{gp_col}_{target_col}_next_{gap}_max'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_next_{gap}'].transform('max')
        data[f'{gp_col}_{target_col}_prev_{gap}_max'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_prev_{gap}'].transform('max')

        # 统计其最小值
        data[f'{gp_col}_{target_col}_next_{gap}_min'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_next_{gap}'].transform('min')
        data[f'{gp_col}_{target_col}_prev_{gap}_min'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_prev_{gap}'].transform('min')

        # 统计其skew值
        data[f'{gp_col}_{target_col}_next_{gap}_skew'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_next_{gap}'].transform('skew')
        data[f'{gp_col}_{target_col}_prev_{gap}_skew'] = data.groupby(gp_col)[
            f'{gp_col}_{target_col}_prev_{gap}'].transform('skew')

    return data


# %%

# gp_tg_cols = [(['Pusher_start_stop'], 'Feedwater_flow'),
#               (['Pusher_automatic_throwback_signal'], 'Feedwater_flow'),
#               (['Grate_start_stop'], 'Feedwater_flow'),
#               (['Grate_automatic_throw-out_signal'], 'Feedwater_flow'),###########
#               (['Pusher_start_stop'], 'Ladle_water_level'),
#               (['Pusher_automatic_throwback_signal'], 'Ladle_water_level'),
#               (['Grate_start_stop'], 'Ladle_water_level'),
#               (['Grate_automatic_throw-out_signal'], 'Ladle_water_level'),##########
#                (['Pusher_start_stop'], 'Inducer_fan_speed'),
#               (['Pusher_automatic_throwback_signal'], 'Inducer_fan_speed'),
#               (['Grate_start_stop'], 'Inducer_fan_speed'),
#               (['Grate_automatic_throw-out_signal'], 'Inducer_fan_speed')]##########

gp_tg_cols = [(['Main_steam_flow_rate_setting'], 'Feedwater_flow')]

# %%

for col in tqdm(gp_tg_cols):
    data = get_shift_feats(data, gap_list=list(range(1, 10)), gp_col=col[0], target_col=col[1])

# %%

import re

data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

# %%

# for col in tqdm(gp_tg_cols):
#     data = get_shift_feats(data, gap_list=[1,2,3], gp_col=col[0], target_col=col[1])

# %%

# import re
# data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# %%

print('nums features: ', len(data.columns))

# %%

# 3、内存压缩
# 3、内存压缩
# def reduce_mem_usage(df, verbose=True):
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     start_mem = df.memory_usage().sum() / 1024 ** 2
#     for col in df.columns:
#         col_type = df[col].dtypes
#         if col_type in numerics:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#     end_mem = df.memory_usage().sum() / 1024 ** 2
#     if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
#                 start_mem - end_mem) / start_mem))
#     return df


# 压缩使用内存
# 由于数据比较大，所以合理的压缩内存节省空间尤为的重要
# 使用reduce_mem_usage函数可以压缩近70%的内存占有。
# data = reduce_mem_usage(data)
# Mem. usage decreased to 2351.47 Mb (69.3% reduction)

# %%


test_data = data.tail(1800)
train = data.iloc[:-1800, :]
train.reset_index(drop=True, inplace=True)
features = train.columns.drop(['Time', label]).tolist()
train_x, train_y = train[features], train[label]

# %%

# 交叉验证所使用的第三方库


kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_x = test_data[features]
test_lgb = np.zeros(test_x.shape[0])
for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
    print('************************************  {} fold************************************'.format(str(i + 1)))
    trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], \
                                 train_x.iloc[valid_index], train_y[valid_index]

    dtrain = lgb.Dataset(trn_x, label=trn_y)
    dvalid = lgb.Dataset(val_x, label=val_y)

    watchlist = [dtrain, dvalid]

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
    model = lgb.train(params, train_set=dtrain, num_boost_round=10000, early_stopping_rounds=1000, valid_sets=watchlist,
                      verbose_eval=200)

    test_pred = model.predict(test_x)
    test_lgb += test_pred / kf.n_splits

# %%


# %%

# %%

sub = test_data[['Time']]
sub['lgb' + label] = test_lgb
sub.reset_index(inplace=True)
sub.columns = ['ID', 'Time', 'Steam_flow']

sub.to_csv('result.csv', index=False)