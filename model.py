import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.regression import mean_absolute_error

# from dbn.tensorflow import SupervisedDBNClassification
from dbn import SupervisedDBNRegression

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

# Restore it
regressor = SupervisedDBNRegression.load('model.pkl')

# Test
y_pred = regressor.predict(X_test)
print('Done.\nMAE: %f' % mean_absolute_error(y_test, y_pred))
