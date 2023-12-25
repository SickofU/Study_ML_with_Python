#import 할 부분들
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Attention,Concatenate
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.layers import Input



#데이터 가져오기
df = pd.read_csv('TSLA.csv')
#어디를 예측할지 정하기
data_to_use = df['Open'].values
#scaler 초기화
scaler = MinMaxScaler(feature_range=(0,1))
sequence_length = 50
#크기 나눠서 학습시킬 부분 나눠주고
train_data_len = int(len(data_to_use) * 0.8)
train_data = data_to_use[:train_data_len]

train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

X_train = []
y_train = []
for i in range(sequence_length, len(train_data_scaled)):
    X_train.append(train_data_scaled[i-sequence_length:i, 0])
    y_train.append(train_data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

test_data = data_to_use[train_data_len - sequence_length:]
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

X_test = []
y_test = []
for i in range(sequence_length, len(test_data_scaled)):
    X_test.append(test_data_scaled[i-sequence_length:i, 0])
    y_test.append(test_data_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# reshape 하기
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#모델 생성 LSTM
model_LSTM = Sequential()
model_LSTM.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_LSTM.add(Dropout(0.2))
model_LSTM.add(LSTM(units=50, return_sequences=False))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(units=1))

#optimizer은 adam으로 손실함수는 MSE로 설정함.
model_LSTM.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#모델 생성 RNN

model_RNN = Sequential()
model_RNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_RNN.add(Dropout(0.2))
model_RNN.add(SimpleRNN(units=50, return_sequences=False))
model_RNN.add(Dropout(0.2))

model_RNN.add(Dense(units=1))

# optimizer은 adam으로 손실함수는 MSE로 설정함.
model_RNN.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

#모델 생성 GRU 모델 생성

model_GRU = Sequential()
model_GRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_GRU.add(Dropout(0.2))
model_GRU.add(GRU(units=50, return_sequences=False))
model_GRU.add(Dropout(0.2))

model_GRU.add(Dense(units=1))

# optimizer은 adam으로 손실함수는 MSE로 설정함.
model_GRU.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')



#학습 
model_LSTM.fit(X_train, y_train, epochs=30, batch_size=50)
model_RNN.fit(X_train,y_train,epochs=15,batch_size=30)

model_GRU.fit(X_train,y_train,epochs=15,batch_size=30)

predicted_stock_price_LSTM = model_LSTM.predict(X_test)
predicted_stock_price_LSTM = scaler.inverse_transform(predicted_stock_price_LSTM)
real_stock_price_LSTM = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))


predicted_stock_price_RNN = model_RNN.predict(X_test)
predicted_stock_price_RNN = scaler.inverse_transform(predicted_stock_price_RNN)
real_stock_price_RNN = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

predicted_stock_price_GRU= model_GRU.predict(X_test)
predicted_stock_price_GRU = scaler.inverse_transform(predicted_stock_price_GRU)
real_stock_price_GRU = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

plt.figure(figsize=(14,5))
plt.plot(real_stock_price_LSTM, color='green', label='실제값')
plt.plot(predicted_stock_price_LSTM, color='blue', label='예측')
plt.title('LSTM')
plt.xlabel('시간')
plt.ylabel('주가')
plt.legend()
plt.show()


plt.figure(figsize=(14,5))
plt.plot(real_stock_price_RNN, color='green', label='실제값')
plt.plot(predicted_stock_price_RNN, color='blue', label='예측')
plt.title('RNN')
plt.xlabel('시간')
plt.ylabel('주가')
plt.legend()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(real_stock_price_GRU, color='green', label='실제값')
plt.plot(predicted_stock_price_GRU, color='blue', label='예측')
plt.title('GRU')
plt.xlabel('시간')
plt.ylabel('주가')
plt.legend()
plt.show()



#구글
#데이터 가져오기
df = pd.read_csv('GOOG.csv')
#어디를 예측할지 정하기
data_to_use = df['Open'].values
#scaler 초기화
scaler = MinMaxScaler(feature_range=(0,1))
sequence_length = 50
#크기 나눠서 학습시킬 부분 나눠주고
train_data_len = int(len(data_to_use) * 0.8)
train_data = data_to_use[:train_data_len]

train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

X_train = []
y_train = []
for i in range(sequence_length, len(train_data_scaled)):
    X_train.append(train_data_scaled[i-sequence_length:i, 0])
    y_train.append(train_data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

test_data = data_to_use[train_data_len - sequence_length:]
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

X_test = []
y_test = []
for i in range(sequence_length, len(test_data_scaled)):
    X_test.append(test_data_scaled[i-sequence_length:i, 0])
    y_test.append(test_data_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

#LSTM reshape 하기
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#모델 생성 LSTM
Gmodel_LSTM = Sequential()
Gmodel_LSTM.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
Gmodel_LSTM.add(Dropout(0.2))
Gmodel_LSTM.add(LSTM(units=50, return_sequences=False))
Gmodel_LSTM.add(Dropout(0.2))

Gmodel_LSTM.add(Dense(units=1))

#optimizer은 adam으로 손실함수는 MSE로 설정함.
Gmodel_LSTM.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#모델 생성 RNN

Gmodel_RNN = Sequential()
Gmodel_RNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
Gmodel_RNN.add(Dropout(0.2))
Gmodel_RNN.add(SimpleRNN(units=50, return_sequences=False))
Gmodel_RNN.add(Dropout(0.2))

Gmodel_RNN.add(Dense(units=1))

# optimizer은 adam으로 손실함수는 MSE로 설정함.
Gmodel_RNN.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#모델 생성 GRU 모델 생성

Gmodel_GRU = Sequential()
Gmodel_GRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
Gmodel_GRU.add(Dropout(0.2))
Gmodel_GRU.add(GRU(units=50, return_sequences=False))
Gmodel_GRU.add(Dropout(0.2))

Gmodel_GRU.add(Dense(units=1))

# optimizer은 adam으로 손실함수는 MSE로 설정함.
Gmodel_GRU.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#학습 
Gmodel_LSTM.fit(X_train, y_train, epochs=2000, batch_size=100)
Gmodel_RNN.fit(X_train,y_train,epochs=50,batch_size=50)

Gmodel_GRU.fit(X_train,y_train,epochs=2000,batch_size=50)


Gpredicted_stock_price_LSTM = Gmodel_LSTM.predict(X_test)

Gpredicted_stock_price_LSTM = scaler.inverse_transform(Gpredicted_stock_price_LSTM)
Greal_stock_price_LSTM = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))



Gpredicted_stock_price_RNN = Gmodel_RNN.predict(X_test)

Gpredicted_stock_price_RNN = scaler.inverse_transform(Gpredicted_stock_price_RNN)
Greal_stock_price_RNN = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))


Gpredicted_stock_price_GRU= Gmodel_GRU.predict(X_test)

Gpredicted_stock_price_GRU = scaler.inverse_transform(Gpredicted_stock_price_GRU)
Greal_stock_price_GRU = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

plt.figure(figsize=(20,5))
plt.plot(Greal_stock_price_LSTM, color='green', label='실제값')
plt.plot(Gpredicted_stock_price_LSTM, color='blue', label='예측')
plt.title('LSTM')
plt.xlabel('시간')
plt.ylabel('주가')
plt.legend()
plt.show()


plt.figure(figsize=(20,5))
plt.plot(Greal_stock_price_RNN, color='green', label='실제값')
plt.plot(Gpredicted_stock_price_RNN, color='blue', label='예측')
plt.title('RNN')
plt.xlabel('시간')
plt.ylabel('주가')
plt.legend()
plt.show()

plt.figure(figsize=(20,5))
plt.plot(Greal_stock_price_GRU, color='green', label='실제값')
plt.plot(Gpredicted_stock_price_GRU, color='blue', label='예측')
plt.title('GRU')
plt.xlabel('시간')
plt.ylabel('주가')
plt.legend()
plt.show()
