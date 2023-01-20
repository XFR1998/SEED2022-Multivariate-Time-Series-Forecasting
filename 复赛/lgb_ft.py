#%%

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

#%%

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化

set_seed(42)


#%%

# label: '推料器自动指令'
# train_data_path = './data/train/'
# test_data_path = './data/test/'
#
# train_cols = ['推料器自动指令', 'CO含量', 'HCL含量', 'NOx含量', 'SO2含量', '二次风调门', '二次风量', '给水流量',
#         '炉排实际运行指令', '炉排自动投退信号', '汽包水位', '推料器启停', '推料器自动投退信号',
#         '氧量设定值', '一次风调门', '一次风量', '引风机转速', '主蒸汽流量']
#
# test_cols = ['CO含量', 'HCL含量', 'NOx含量', 'SO2含量', '二次风调门', '二次风量', '给水流量',
#         '炉排实际运行指令', '炉排自动投退信号', '汽包水位', '推料器启停', '推料器自动投退信号',
#         '氧量设定值', '一次风调门', '一次风量', '引风机转速', '主蒸汽流量']
#
# print(len(train_cols))
# print(len(test_cols))

#%%

# train_df = pd.DataFrame()
# train_df['时间'] = pd.read_csv(train_data_path+'CO含量.csv')['时间']
#
# for n in tqdm(train_cols):
#     train_df[n] = pd.read_csv(train_data_path+f'{n}.csv')[n]
#
#

#%%

# test_df = pd.DataFrame()
# test_df['时间'] = pd.read_csv(test_data_path+'CO含量.csv')['时间']
#
# for n in tqdm(test_cols):
#     test_df[n] = pd.read_csv(test_data_path+f'{n}.csv')[n]
#
#

#%%

# # 有重复的行，需要去重
# print('训练集中重复的行数：', train_df.duplicated().sum())
# train_df = train_df.drop_duplicates()
#
# print('测试集中重复的行数：', test_df.duplicated().sum())
# test_df = test_df.drop_duplicates()
#
#

#%%

# train_df.to_csv('./data/train.csv', index=False)
# test_df.to_csv('./data/test.csv', index=False)


#%%

train_df = pd.read_csv('./data/train.csv')
train_df = train_df.dropna()

test_df = pd.read_csv('./data/test.csv')
train_df.info()
test_df.info()


#%%



#%%

data = pd.concat([train_df, test_df])

data.info()

#%%

data['推料器启停'].value_counts()

#%%

data = data[data['推料器启停']==1]
# data.info()

#%%

data = data.drop(columns=['推料器启停'])
data.info()

#%%



#%%



#%%



#%%



#%%



#%%



#%%

data['gas'] = data['CO含量'] + data['HCL含量'] + data['NOx含量'] + data['SO2含量']
# feat_list = ['二次风调门', '二次风量', '给水流量', '炉排实际运行指令', '汽包水位', '氧量设定值', '一次风调门', '一次风量', '引风机转速', '主蒸汽流量']
# ['给水流量', '氧量设定值', '一次风量', '主蒸汽流量', '汽包水位']
feat_list = ['炉排实际运行指令','gas','CO含量','HCL含量','NOx含量','SO2含量','二次风调门', '二次风量', '给水流量', '汽包水位', '氧量设定值', '一次风调门', '一次风量', '引风机转速', '主蒸汽流量']
for f in tqdm(feat_list):
    shift_f = []
    shift_lf = []
    shift_d = []
    for i in range(200):
        colname = f + '_shift_{}'.format(i + 1)
        data[colname] = data[f].shift(i + 1)
        shift_f.append(colname)

        colname = f + '_lshift_{}'.format(-(i + 1))
        data[colname] = data[f].shift(-(i + 1))
        shift_lf.append(colname)
        # # 与上n分钟的变化比例特征
        # if i%60==0:
        #     temp_name = f+'_ratio_{}'.format(i)
        #     data[temp_name] = data[colname]/data[colname].shift(i+1)

        # .diff用于计算一列中某元素与该列中前？个元素的差值（默认前一个元素）
        # colname = f + '_diff_{}'.format(i + 1)
        # data[colname] = data[f].diff(i + 1)
        # colname = f + '_diff_{}'.format(-(i + 1))
        # data[colname] = data[f].diff(-(i + 1))

    # data[f+'_diff'] = data[f].diff(1)
    # 对每一行：shift_列的值取平均，即将前n天的值取平均，取最大值，最最小值，最标准差

    data[f + '_fore_steps_mean'] = data[shift_f].mean(1)
    data[f + '_fore_steps_max'] = data[shift_f].max(1)
    data[f + '_fore_steps_min'] = data[shift_f].min(1)
    data[f + '_fore_steps_std'] = data[shift_f].std(1)
    data[f + '_fore_steps_std'] = data[shift_f].skew(1)

    data[f + '_lfore_steps_mean'] = data[shift_lf].mean(1)
    data[f + '_lfore_steps_max'] = data[shift_lf].max(1)
    data[f + '_lfore_steps_min'] = data[shift_lf].min(1)
    data[f + '_lfore_steps_std'] = data[shift_lf].std(1)
    data[f + '_lfore_steps_std'] = data[shift_lf].skew(1)

    data.drop(shift_f, axis=1, inplace=True)
    data.drop(shift_lf, axis=1, inplace=True)

#%%

#
data['炉排自动投退信号'] = data['炉排自动投退信号'].map(int)
# data['推料器启停'] = data['推料器启停'].map(int)
data['推料器自动投退信号'] = data['推料器自动投退信号'].map(int)

#%%



#%%

import featuretools as ft

#%%



#%%



#%%

# from sklearn.datasets import load_iris
# import pandas as pd
# import featuretools as ft
#
# # Load data and put into dataframe
# iris = load_iris()
# df = pd.DataFrame(iris.data, columns = iris.feature_names)
# df['species'] = iris.target
# df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

#%%



#%%

# # Make an entityset and add the entity
# es = ft.EntitySet(id = 'iris')
# es.add_dataframe(dataframe_name = 'data', dataframe = df,
#                          make_index = True, index = 'index')
#
# # Run deep feature synthesis with transformation primitives
# feature_matrix, feature_defs = ft.dfs(entityset = es, target_dataframe_name = 'data',
#                                       trans_primitives = ['add_numeric', 'multiply_numeric'])
#
# feature_matrix.head()


#%%

label = data['推料器自动指令'].to_list()

data = data.drop(columns=['推料器自动指令'])

#%%

# Make an entityset and add the entity
es = ft.EntitySet(id = 'lj')
es.add_dataframe(dataframe_name = 'data', dataframe = data,
                         make_index = True, index = 'index')

# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(entityset = es, target_dataframe_name = 'data',
                                      trans_primitives = ['add_numeric', 'multiply_numeric'])

feature_matrix.head()


#%%

feature_defs

#%%

feature_matrix['推料器自动指令']  = label

#%%



#%%

# feature_matrix['推料器自动指令']
data = feature_matrix
data.head(5)

#%%


#%%


# gp_tg_cols = [(['推料器启停'], '给水流量'),
#               (['推料器启停'], '主蒸汽流量')]
#
#

#%%

# for col in tqdm(gp_tg_cols):
#     data = get_shift_feats(data, gap_list=list(range(1, 10)), gp_col=col[0], target_col=col[1])

#%%

# data.columns = [str(i) for i in data.columns]
# for i in data.columns:
#     print(i)

#%%

# # 内存压缩
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df

# 压缩使用内存
# 由于数据比较大，所以合理的压缩内存节省空间尤为的重要
# 使用reduce_mem_usage函数可以压缩近70%的内存占有。
data = reduce_mem_usage(data)

#%%



#%%

print('nums features: ', len(data.columns))

#%%

label = '推料器自动指令'
test_data = data.tail(1800)
train = data.iloc[:-1800, :]
train.reset_index(drop=True, inplace=True)
# features = train.columns.drop(['时间', label]).tolist()
features = train.columns.drop([label]).tolist()
train_x, train_y = train[features], train[label]

#%%

# train

#%%

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


    print("Features importance...")
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print('Top 50 features:\n', feat_imp.head(50))

#%%

pd.Series(test_lgb).describe()
# count    1800.000000
# mean       60.998414
# std         5.295495
# min        55.222541
# 25%        56.662194
# 50%        58.882301
# 75%        64.712533
# max        78.536112
# dtype: float64


#%%


# sub = test_data[['时间','推料器启停']]
# sub['lgb' + label] = test_lgb
# sub.reset_index(inplace=True)
# sub.columns = ['ID', 'Time', 'Aps', 'Ai']



sub = pd.DataFrame()
sub['ID'] = list(range(1,1801))
sub['Time'] = test_df['时间']
sub['Aps'] = test_df['推料器启停'].astype(bool)
sub['Ai'] = test_lgb

sub.to_csv('./lgb_ft.csv', index=False)