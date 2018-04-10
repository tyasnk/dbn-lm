import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dbn import SupervisedDBNRegression
np.random.seed(2018)

dataset = pd.read_csv("data/idraud.csv")

df = dataset["IDR/AUD"].values
df = df.astype('float32')

len_train = int(len(df)*0.70)
len_val = int(len(df)*0.10)

train = df[0:len_train]
validation = df[len_train:len_train+len_val]
test = df[len_train+len_val:]

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train)
train = scaler.transform(train)
validation = scaler.transform(validation)
test = scaler.transform(test)


# create time series from dataset
def create_sliding_windows(data, lag=1):
    X = []
    y = []
    for i in range(len(data)-lag-1):
        cek = data[i:(i+lag)]
        X.append(cek)
        y.append(data[i+lag])
    return np.array(X), np.array(y)


lag = 3
X_train, y_train = create_sliding_windows(train, lag)
y_train = np.reshape(y_train, (len(y_train), 1))
X_val, y_val = create_sliding_windows(validation, lag)
X_test, y_test = create_sliding_windows(test, lag)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[30, 40],
                                    learning_rate_rbm=0.001,
                                    learning_rate=0.001,
                                    n_epochs_rbm=30,
                                    n_iter_backprop=100,
                                    batch_size=32,
                                    activation_function='relu',
                                    dropout_p=0.2)
regressor.fit(X_train, y_train)

# Save the model
regressor.save('model.pkl')

# prediction
y_pred = regressor.predict(X_test)

# Restore it
regressor = SupervisedDBNRegression.load('model.pkl')

y_test_transform = scaler.inverse_transform(y_test)
y_pred_transform = scaler.inverse_transform(y_pred)


def dstat_measure(targets, predictions):
    n = len(targets)
    alpha = 0
    for i in range(n-1):
        if ((predictions[i + 1] - targets[i]) * (targets[i + 1] - targets[i]))>0:
            alpha += 1
    dstat = (1/n)*alpha*100
    return dstat


def mean_absolute_error(targets, predictions):
    return np.mean(np.abs(targets-predictions))


def mean_absolute_percentage_error(targets, predictions):
    return np.mean(np.abs((targets - predictions) / targets)) * 100


def root_mean_square_error(targets, predictions):
    return np.sqrt(np.mean((targets-predictions) ** 2))


# Test
print('Done.\nMAE: %f' % mean_absolute_error(y_test_transform,
                                             y_pred_transform))
print("MAPE: %f" % mean_absolute_percentage_error(y_test_transform,
                                                  y_pred_transform))
print("RMSE: %f" % root_mean_square_error(y_test_transform, y_pred_transform))
print("Dstat: %f" % dstat_measure(y_test, y_pred_transform))
