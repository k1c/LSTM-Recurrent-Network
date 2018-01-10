'''
Akhil Dalal - 100855466
Carolyne Pelletier - 101054962

Project

LSTM to predict ethereum prices
'''

import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
import math
from sklearn.metrics import mean_squared_error

# --- Dataset ---
# Prediction of closing price is affected by the following market conditions:
#  - Opening and Closing prices
#  - High & Low
#  - Volume traded
# ---------------

# Fetch data from Jan 1 2016 till latest
print("Loading data...")
market_start = "20160101"
ethereum_market = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=" + market_start + "&end=" +time.strftime("%Y%m%d"))[0]
ethereum_market = ethereum_market.assign(Date=pd.to_datetime(ethereum_market['Date']))
ethereum_market = ethereum_market.drop('Market Cap', 1)
print("Data loaded!")

# Split into train and test sets
split_date = "2017-07-01"
ethereum_market = ethereum_market.sort_values(by='Date')
training_set = ethereum_market[ethereum_market['Date'] < split_date]
testing_set = ethereum_market[ethereum_market['Date'] >= split_date]



# Display data
fig, ax1 = plt.subplots(1,1)
ax1.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
ax1.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
ax1.patch.set_facecolor('0.85')
ax1.set_ylabel('Closing Price ($)',fontsize=12)
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
ax1.plot(training_set['Date'].astype(datetime.datetime).values,training_set['Close'], label='Training')
ax1.plot(testing_set['Date'].astype(datetime.datetime).values,testing_set['Close'], label='Testing')
ax1.legend()
fig.tight_layout()
plt.savefig('test-train-split.png')

# No need for dates anymore.
training_set = training_set.drop('Date', 1)
testing_set = testing_set.drop('Date', 1)

# Prepare inputs (using sliding windows) and normalize columns, use numpy arrays - easier to work with vs pandas
# for normalized: -ve means it's lower compared with base value.

window_size = 15
pred_range = 5

# training
training_inputs = []
for i in range(len(training_set) - window_size):
    temp = training_set[i:(i+window_size)].copy()
    for col in list(training_set):
        temp.loc[:, col] = temp[col]/temp[col].iloc[0] - 1
    training_inputs.append(temp)

training_outputs = []
for i in range(window_size, len(training_set['Close'])-pred_range):
    training_outputs.append((training_set['Close'][i:i+pred_range].values/
                                  training_set['Close'].values[i-window_size])-1)


training_inputs = np.array([np.array(training_input) for training_input in training_inputs])
training_outputs = np.array(training_outputs)


# testing
test_inputs = []
for i in range(len(testing_set) - window_size):
    temp = testing_set[i:(i+window_size)].copy()
    for col in list(testing_set):
        temp.loc[:, col] = temp[col]/temp[col].iloc[0] - 1
    test_inputs.append(temp)

test_outputs = []
for i in range(window_size, len(testing_set['Close'])-pred_range):
    test_outputs.append((testing_set['Close'][i:i+pred_range].values/
                                  testing_set['Close'].values[i-window_size])-1)

test_inputs = np.array([np.array(test_inputs) for test_inputs in test_inputs])
test_outputs = np.array(test_outputs)



# --- Build model ---
# LSTM is a single LSTM layer. It "unrolls" into the individual cells.
# -------------------

inputs_shape = training_inputs.shape
dropout = 0.25
num_neurons = 100

model = Sequential()
model.add(LSTM(num_neurons, batch_input_shape=(1, inputs_shape[1], inputs_shape[2]), stateful=True, return_sequences=True))
model.add(LSTM(num_neurons, batch_input_shape=(1, inputs_shape[1], inputs_shape[2]), stateful=True))
model.add(Dropout(dropout))
model.add(Dense(units=pred_range))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="RMSProp")
model.summary()


# --- Train model ---
# -------------------

epochs = 50
for i in range(epochs):
    model.fit(training_inputs[:-pred_range], training_outputs, epochs=1, batch_size=1, verbose=2, shuffle=False)
    model.reset_states()

# --- Predict model ---
# ---------------------
predictions = model.predict(test_inputs[:-pred_range], batch_size=1)

# denormalize peredictions
eth_pred_prices = ((predictions[::pred_range]+1)*\
                   testing_set['Close'].values[:-(window_size + pred_range)][::pred_range].reshape(int(np.ceil((len(test_inputs) - pred_range)/float(pred_range))),1))

# plot predictions
pred_colors = ["#FF69B4", "#5D6D7E", "#F4D03F","#A569BD","#45B39D"]
fig, ax2 = plt.subplots(1,1)
ax2.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
ax2.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
ax2.plot(ethereum_market[ethereum_market['Date']>= split_date]['Date'][window_size:].astype(datetime.datetime),
         testing_set['Close'][window_size:], label='Actual')
for i, eth_pred in enumerate(eth_pred_prices):
    if i<5:
        ax2.plot(ethereum_market[ethereum_market['Date']>= split_date]['Date'][window_size:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
             eth_pred, color=pred_colors[i%5], label='Predicted')
    else:
        ax2.plot(ethereum_market[ethereum_market['Date']>= split_date]['Date'][window_size:].astype(datetime.datetime)[i*pred_range:i*pred_range+pred_range],
             eth_pred, color=pred_colors[i%5])
ax2.set_title('Test Set: ' + str(pred_range) + ' Timepoint Predictions',fontsize=13)
ax2.set_ylabel('Ethereum Price ($)',fontsize=12)
ax2.legend()
fig.tight_layout()
plt.savefig('predictions-with-'+str(window_size)+'-'+str(pred_range)+'-'+str(num_neurons)+ '.png')

# calculate RMSE between predictions and actual
# these are normalized
testScore = math.sqrt(mean_squared_error(test_outputs, predictions))
print('Test Score: %.2f RMSE' % (testScore))
