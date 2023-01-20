import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#%%

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:29:30 2022

@author: xdata
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
import warnings
import pdb
warnings.filterwarnings('ignore')
import random
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset,SequentialSampler,RandomSampler,DataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#%%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.manual_seed(seed)


set_seed(2022)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('训练集的大小为： ', len(train))
print('测试集的大小为： ', len(test))


train = train.dropna()
train.reset_index(drop=True,inplace=True)

label = 'Main_steam_flow_rate'

# =============================================================================
# ['时间', '主蒸汽流量', 'CO含量', 'HCL含量', 'NOx含量', 'SO2含量', '一次风调门', '一次风量',
#        '主蒸汽流量设定值', '二次风调门', '二次风量', '引风机转速', '推料器启停', '推料器手动指令',
#        '推料器自动投退信号', '推料器自动指令', '氧量设定值', '汽包水位', '炉排启停', '炉排实际运行指令',
#        '炉排手动指令', '炉排自动投退信号', '给水流量']
# =============================================================================


#%%

train.columns = ['Time', 'Main_steam_flow_rate', 'CO_content', 'HCL_content', 'NOx_content', 'SO2_content', 'Primary_air_regulator', 'Primary_air_volume',
       'Main_steam_flow_rate_setting', 'Secondary_air_regulator', 'Secondary_air_volume', 'Inducer_fan_speed', 'Pusher_start_stop', 'Pusher_manual_command',
       'Pusher_automatic_throwback_signal', 'Pusher_automatic_command', 'Oxygen_setpoint', 'Ladle_water_level', 'Grate_start_stop', 'Grate_actual_operation_command',
       'Grate_manual_command', 'Grate_automatic_throw-out_signal', 'Feedwater_flow']

test.columns = ['Time', 'CO_content', 'HCL_content', 'NOx_content', 'SO2_content', 'Primary_air_regulator', 'Primary_air_volume',
       'Main_steam_flow_rate_setting', 'Secondary_air_regulator', 'Secondary_air_volume', 'Inducer_fan_speed', 'Pusher_start_stop', 'Pusher_manual_command',
       'Pusher_automatic_throwback_signal', 'Pusher_automatic_command', 'Oxygen_setpoint', 'Ladle_water_level', 'Grate_start_stop', 'Grate_actual_operation_command',
       'Grate_manual_command', 'Grate_automatic_throw-out_signal', 'Feedwater_flow']


#%%

len(train.columns), len(test.columns)

#%%

df = pd.concat([train, test])

df.info()

#%%

df.head(5)


#%%

print('df.shape: ', df.shape)

df.drop(columns=['Time'], inplace=True)

#%%
# df['gas'] = df.CO_content + df.HCL_content + df.SO2_content + df.NOx_content
# for f in ['Feedwater_flow', 'Oxygen_setpoint', 'Primary_air_volume', 'Main_steam_flow_rate_setting', 'Ladle_water_level']:
#     shift_f = []
#     shift_d = []
#     for i in range(200):
#         colname = f+'_shift_{}'.format(i+1)
#         df[colname] = df[f].shift(i+1)
#         shift_f.append(colname)
#
#         if i>10:
#             shift_d.append(colname)
#         colname = f+'_diff_{}'.format(i+1)
#         df[colname] = df[f].diff(i+1)
# # =============================================================================
# #         if (i+1)%50==0:
# #             data[f+'_fore_{}_steps_mean'.format(i)] = data[shift_f].mean(1)
# #             data[f+'_fore_{}_steps_max'.format(i)] = data[shift_f].max(1)
# #             data[f+'_fore_{}_steps_min'.format(i)] = data[shift_f].min(1)
# #             data[f+'_fore_{}_steps_std'.format(i)] = data[shift_f].std(1)
# # =============================================================================
#
#     # data[f+'_diff'] = data[f].diff(1)
#     df[f+'_fore_7_steps_mean'] = df[shift_f].mean(1)
#     df[f+'_fore_7_steps_max'] = df[shift_f].max(1)
#     df[f+'_fore_7_steps_min'] = df[shift_f].min(1)
#     df[f+'_fore_7_steps_std'] = df[shift_f].std(1)
#     df.drop(shift_d, axis=1, inplace=True)
#
# df['Grate_start_stop'] = df['Grate_start_stop'].map(int)
# df['Pusher_start_stop'] = df['Pusher_start_stop'].map(int)
# df['Pusher_automatic_throwback_signal'] = df['Pusher_automatic_throwback_signal'].map(int)
# df['Grate_automatic_throw-out_signal'] = df['Grate_automatic_throw-out_signal'].map(int)


#%%

df_test = df.tail(1800)
df_train = df.iloc[:-1800, :]
df_train.reset_index(drop=True,inplace=True)

df_valid = df_train.iloc[-1800:, :]
df_train = df_train.iloc[:-1800, :]

df_valid.reset_index(drop=True, inplace=True)

#%%

print('df_train.shape: ', df_train.shape)
print('df_valid.shape: ', df_valid.shape)
print('df_test.shape: ', df_test.shape)

#%%

scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_train)
df_for_validing_scaled = scaler.transform(df_valid)
# df_for_testing_scaled=scaler.transform(df_test)

#%%

df_for_training_scaled

#%%

def createXY(dataset,n_past):
  dataX = []
  dataY = []
  for i in range(n_past, len(dataset)):
      dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
      dataY.append(dataset[i,0])
  return np.array(dataX),np.array(dataY)


SEQ_LEN = 512 # 即时间窗口


trainX, trainY=createXY(df_for_training_scaled,SEQ_LEN)
validX, validY=createXY(df_for_validing_scaled,SEQ_LEN)
# testX, testY=createXY(df_for_testing_scaled,SEQ_LEN)

#%%

print('构建时间序列特征')
print('trainX.shape, trainY.shape: ', trainX.shape, trainY.shape)
print('validX.shape, validY.shape: ', validX.shape, validY.shape)
# print('testX.shape, testY.shape: ', testX.shape, testY.shape)

#%%

class My_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = dict(input=torch.FloatTensor(self.data[idx]),
                      label=torch.FloatTensor(np.array(self.labels[idx])))

        return sample


#%%

def create_data_loader(data=None, labels=None, batch_size=32):
    ds=My_Dataset(
        data = data,
        labels = labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
    # if test_mode:
    #     return DataLoader(ds, batch_size=batch_size, shuffle=False)
    # else:
    #     return DataLoader(ds, batch_size=batch_size, shuffle=True)

#%%

train_data_loader = create_data_loader(data=trainX,
                                       labels=trainY)

val_data_loader = create_data_loader(data=validX,
                                       labels=validY)

#%%

len(train_data_loader)

#%%

data = next(iter(train_data_loader))
data.keys()
print(data['input'].shape)
print(data['label'].shape)

#%%


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('use device: ', device)
#%%
class Time_Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = 256
        self.num_layers = 1
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size=22,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             bidirectional=self.bidirectional,
                             batch_first=True)

        self.fc = nn.Linear(self.hidden_size*2, 1)


    def forward(self, x):
        device = x.device
        batch_size, seq_len, emb_dim = x.shape
        # 初始化：双向就乘2
        h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(device)

        x, (_, _) = self.lstm(x, (h0, c0))
        x = x[:,-1,:] # 取最后一个时间步
        x = self.fc(x)

        return x

#%%
model = Time_Model()
model = model.to(device)
#%%

# MSE开方就是RMSE了，所以损失可以直接用来作为评估指标哦
loss_fn = torch.nn.MSELoss(reduction='mean').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 32
#%%

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []

    pred_list = []
    target_list = []

    for inputs in tqdm(data_loader):
        targets = inputs["label"].to(device)
        x = inputs['input'].to(device)
        outputs = model(x)
        preds = outputs


        pred_list.extend(preds.cpu().detach().numpy().tolist())
        target_list.extend(targets.cpu().detach().numpy().tolist())


        loss = loss_fn(outputs, targets)
        # MSE开方就是RMSE了
        losses.append(np.sqrt(loss.item()))
        loss.backward()



        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

    mean_loss = np.mean(losses)


    return mean_loss

#%%

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval() # 验证预测模式
    losses = []
    pred_list = []
    target_list = []

    with torch.no_grad():
        for inputs in tqdm(data_loader):
            targets = inputs["label"].to(device)
            x = inputs['input'].to(device)
            outputs = model(x)
            preds = outputs


            pred_list.extend(preds.cpu().detach().numpy().tolist())
            target_list.extend(targets.cpu().detach().numpy().tolist())


            loss = loss_fn(outputs, targets)


            # MSE开方就是RMSE了
            losses.append(np.sqrt(loss.item()))



    mean_loss = np.mean(losses)

    return mean_loss

#%%


from collections import defaultdict
history = defaultdict(list) # 记录10轮loss和acc
best_rmse = float('inf')


# -------------------控制早停--------------
early_stop_epochs = 5
no_improve_epochs = 0


for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_rmse = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
    )

    print(f'train_rmse : {train_rmse } \n ')

    val_rmse = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device
    )

    print(f'val_rmse : {val_rmse } \n ')
    print()

    history['train_rmse '].append(train_rmse)
    history['val_rmse '].append(val_rmse)


    if val_rmse  < best_rmse :
        print('best model saved!!!!!!!!!!!!!')
        torch.save(model.state_dict(), f'./save model/best_model.pt')
        best_rmse  = val_rmse

        no_improve_epochs = 0

    else:
        no_improve_epochs += 1



    if no_improve_epochs == early_stop_epochs:
        print('no improve score !!! stop train !!!')
        break

#%%



#%%
# 改变时间窗口后，这里要改动
df_days_past = df_valid[-SEQ_LEN:]
#df_days_past = df_valid
df_days_past.reset_index(drop=True,inplace=True)
df_days_past.info()
df_days_past.head(5)

#%%

df_days_future = df_test
df_days_future.info()
df_days_future['Main_steam_flow_rate'] = 0

df_days_future.head(5)
#%%

# print('df_train.shape: ', df_train.shape)
# print('df_valid.shape: ', df_valid.shape)
# print('df_test.shape: ', df_test.shape)

#%%
old_scaled_array=scaler.transform(df_days_past)
new_scaled_array=scaler.transform(df_days_future)
new_scaled_df=pd.DataFrame(new_scaled_array)
print('new_scaled_array.shape: ', new_scaled_array.shape)
new_scaled_df.iloc[:,0]=np.nan
full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)
full_df.info()
#%%



#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('use device: ', device)
model = Time_Model()


path = './save model/best_model.pt'
model.load_state_dict(torch.load(path))
model = model.to(device)


model.eval()

#%%

full_df_scaled_array=full_df.values
all_data=[]
time_step=SEQ_LEN
for i in range(time_step,len(full_df_scaled_array)):
    data_x=[]
    data_x.append(
    full_df_scaled_array[i-time_step :i , 0:full_df_scaled_array.shape[1]])
    data_x=np.array(data_x)

    data_x = torch.FloatTensor(data_x).to(device)
    with torch.no_grad():
        prediction=model(data_x)
    # print(prediction.shape)
    # break
    prediction = prediction.squeeze(-1)
    prediction = prediction.cpu().detach().numpy()

    all_data.append(prediction)
    full_df.iloc[i,0]=prediction

#%%

full_df.info()

#%%
new_array=np.array(all_data)
new_array=new_array.reshape(-1,1)
prediction_copies_array = np.repeat(new_array,22, axis=-1)
y_pred_future_days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),22)))[:,0]
print(y_pred_future_days)
#%%


len(all_data)
#%%



#%%

test = pd.read_csv('data/test.csv')

#%%


sub = pd.DataFrame({'ID': list(range(1,1801)),
                    'Time': test['时间'],
                    'Steam_flow': y_pred_future_days})
sub.to_csv('result.csv',index=False)