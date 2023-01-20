#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
import warnings
import pdb
warnings.filterwarnings('ignore')
import random
from tqdm import tqdm
import paddlets
from paddlets import TSDataset
from paddlets import TimeSeries
from paddlets.models.forecasting import MLPRegressor, LSTNetRegressor, InformerModel, NHiTSModel
from paddlets.transform import Fill, StandardScaler
from paddlets.metrics import MSE, MAE

#%%

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化

set_seed(42)

#%%

train_df = pd.read_csv('./data/train.csv')
train_df = train_df.dropna()

test_df = pd.read_csv('./data/test.csv')
train_df.info()
test_df.info()

#%%

# df = pd.concat([train_df, test_df])
#
# df.info()

#%%

train_val_dataset = TSDataset.load_from_dataframe(
    train_df,
    time_col='时间',
    target_cols='推料器自动指令',
    observed_cov_cols=['CO含量', 'HCL含量', 'NOx含量', 'SO2含量','二次风调门', '二次风量'],
    freq='1s',
    fill_missing_dates=True,
    fillna_method='pre'
)
# train_dev_dataset.plot(['CO含量', 'HCL含量', 'NOx含量', 'SO2含量','二次风调门', '二次风量'])

test_dataset = TSDataset.load_from_dataframe(
    test_df,
    time_col='时间',
    observed_cov_cols=['CO含量', 'HCL含量', 'NOx含量', 'SO2含量','二次风调门', '二次风量'],
    freq='1s',
    fill_missing_dates=True,
    fillna_method='pre'
)
# train_dev_dataset.plot(['CO含量', 'HCL含量', 'NOx含量', 'SO2含量','二次风调门', '二次风量'])

#%%

train_dataset, val_dataset = train_val_dataset.split(0.7)
train_dataset.plot(add_data=[val_dataset], labels=['val'])

#%%

scaler = StandardScaler()
scaler.fit(train_dataset)
train_dataset_scaled = scaler.transform(train_dataset)
val_dataset_scaled = scaler.transform(val_dataset)
test_dataset_scaled = scaler.transform(test_dataset)

#%%

# 查看PaddleTS库内置的评估指标
# ??paddlets.metrics

#%%

model = InformerModel(in_chunk_len=1800,
                       out_chunk_len=1800,
                       eval_metrics=['mse'],
                       batch_size=32,
                       max_epochs=20)

#%%

model.fit(train_dataset_scaled, val_dataset_scaled)


#%%

# save the model for multiple times.
model.save("model")
from paddlets.models.model_loader import load
# loaded_lstm = load("lstm")

#%%

# pred_test = lstm.predict(test_dataset_scaled)
# pred_test = lstm.recursive_predict(test_dataset_scaled, predict_length=1800)
# scaler.inverse_transform(val_dataset_scaled)['推料器自动指令']
# 利用训练集往后预测1800条
pred_test = model.recursive_predict(val_dataset_scaled, predict_length=1800)

#%%

pred_test = scaler.inverse_transform(pred_test)
pred_test

#%%

pred_test = pred_test.to_dataframe()
pred_test

#%%

sub = pd.DataFrame()
sub['ID'] = list(range(1,1801))
sub['Time'] = test_df['时间']
sub['Aps'] = test_df['推料器启停'].astype(bool)
sub['Ai'] = pred_test['推料器自动指令'].values

sub.to_csv('./informer_result.csv', index=False)
