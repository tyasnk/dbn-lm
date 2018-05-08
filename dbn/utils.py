import numpy as np
import csv
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error

import math


def batch_generator(batch_size, data, labels=None):
    """
    Generates batches of samples
    :param data: array-like, shape = (n_samples, n_features)
    :param labels: array-like, shape = (n_samples, )
    :return:
    """
    n_batches = int(np.ceil(len(data) / float(batch_size)))
    idx = np.random.permutation(len(data))
    data_shuffled = data[idx]
    if labels is not None:
        labels_shuffled = labels[idx]
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if labels is not None:
            yield data_shuffled[start:end, :], labels_shuffled[start:end]
        else:
            yield data_shuffled[start:end, :]


def to_categorical(labels, num_classes):
    """
    Converts labels as single integer to row vectors. For instance, given a three class problem, labels would be
    mapped as label_1: [1 0 0], label_2: [0 1 0], label_3: [0, 0, 1] where labels can be either int or string.
    :param labels: array-like, shape = (n_samples, )
    :return:
    """
    new_labels = np.zeros([len(labels), num_classes])
    label_to_idx_map, idx_to_label_map = dict(), dict()
    idx = 0
    for i, label in enumerate(labels):
        if label not in label_to_idx_map:
            label_to_idx_map[label] = idx
            idx_to_label_map[idx] = label
            idx += 1
        new_labels[i][label_to_idx_map[label]] = 1
    return new_labels, label_to_idx_map, idx_to_label_map


def open_file(file_path, string_tag):
    with open(file_path) as csvfile:
        ratereader = csv.DictReader(csvfile)
        rate = []
        for row in ratereader:
            header = row[string_tag]
            rate.append(header)
        return rate


def normalize(x, y=None, params=None):
    xnormalize = []
    ynormalize = []
    returnMax = -100000
    returnMin = 100000

    thisparams = []

    if params is None:
        for i in range(0, len(x)):
            for j in range(0, len(x[i])):
                returnI = float(x[i][j])
                if returnI > returnMax:
                    returnMax = returnI
                if returnI < returnMin:
                    returnMin = returnI

            returnI = float(y[i][0])
            if returnI > returnMax:
                returnMax = returnI
            if returnI < returnMin:
                returnMin = returnI
        thisparams.append(returnMin)
        thisparams.append(returnMax)

    else:
        returnMin = params[0]
        returnMax = params[1]

        thisparams.append(returnMin)
        thisparams.append(returnMax)

    if params is None:
        for i in range(0, len(x)):
            itemxnormalize = []
            for j in range(0, len(x[i])):
                floatIx = float(x[i][j])
                normalizex = (floatIx - returnMin) / (returnMax - returnMin)
                itemxnormalize.append(normalizex)
            xnormalize.append(itemxnormalize)

            floatIy = float(y[i][0])
            normalizey = (floatIy - returnMin) / (returnMax - returnMin);
            ynormalize.append([normalizey])
        return xnormalize, ynormalize, thisparams
    else:
        for i in range(0, len(x)):
            itemxnormalize = []
            for j in range(0, len(x[i])):
                floatIx = float(x[i][j])
                normalizex = (floatIx - returnMin) / (returnMax - returnMin);
                itemxnormalize.append(normalizex)
            xnormalize.append(itemxnormalize)
            floatIy = float(y[i][0])
            normalizey = (floatIy - returnMin) / (returnMax - returnMin);
            ynormalize.append(normalizey)
        return xnormalize, ynormalize, thisparams


def renormalize(y, params_norm):
    yrenormalize = []
    returnMin = params_norm[0]
    returnMax = params_norm[1]

    for i in range(len(y)):
        renormalize = y[i] * (returnMax - returnMin) + returnMin
        yrenormalize.append(renormalize)
    return yrenormalize


def split_data(rate, sw, size_split):
    dataSize = len(rate)
    trainStart = 0
    trainEnd = dataSize * size_split[0] / 100 - sw - 1
    validStart = dataSize * size_split[0] / 100 + sw - 1
    validEnd = dataSize * size_split[1] / 100
    testStart = dataSize * size_split[1] / 100 + sw
    testEnd = dataSize

    print (trainStart)
    print (trainEnd)
    print (validStart)
    print (validEnd)
    print (testStart)
    print (testEnd)

    sliding_window = sw

    trainRealX = []
    trainRealY = []
    validRealX = []
    validRealY = []
    testRealX = []
    testRealY = []


    for i in range(int(trainStart), int(trainEnd)):
        if i > sw:
            dataNX = []
            for j in range(i - sliding_window, i):
                dataNX.append(float(rate[j]))
            trainRealX.append(dataNX)
            dataNY = []
            dataNY.append(float(rate[i]))
            trainRealY.append(dataNY)

    for i in range(int(validStart), int(validEnd)):
        datai = []
        for j in range(i - sliding_window, i):
            datai.append(float(rate[j]))
        validRealX.append(datai)
        dataiy = []
        dataiy.append(float(rate[i]))
        validRealY.append(dataiy)

    for i in range(int(testStart), int(testEnd)):
        datai = []
        for j in range(i - sliding_window, i):
            datai.append(float(rate[j]))
        testRealX.append(datai)
        dataiy = []
        dataiy.append(float(rate[i]))
        testRealY.append(dataiy)

    return trainRealX, trainRealY, validRealX, validRealY, testRealX, testRealY


def rmse(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def mape(y_true, y_pred, n):
    tmp = 0.0
    for i in range(0, n):
        tmp += math.fabs(y_true[i] - y_pred[i]) / y_true[i]
    mape = (tmp / n) * 100
    return mape


def da(x_true, y_true, y_pred):
    N = len(y_true)
    sigmaAlphaT = 0
    for i in range(0, N):
        if ((y_pred[i] - x_true[i][-1]) * (y_true[i][0] - x_true[i][-1])) > 0:
            sigmaAlphaT += 1
    da = (1 / float(N)) * float(sigmaAlphaT) * 100
    return da


class result:
    def __init__(self, input_node, hidden_node, iteration_rbm, lr, directional_accuracy):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.iteration_rbm = iteration_rbm
        self.lr = lr
        self.directional_accuracy = directional_accuracy


def find_best_input(hype_parameter_result, input_size_varian):
    array_average = []
    for i in range(len(input_size_varian)):
        array_da_result_input = []
        for j in range(len(hype_parameter_result)):
            if (hype_parameter_result[j].input_node == input_size_varian[i]):
                array_da_result_input.append(hype_parameter_result[j].directional_accuracy)
        average = np.average(array_da_result_input)
        print (average)
        print (len(array_da_result_input))
        array_average.append(average)

    best_input_accuracy = np.max(array_average)
    best_input = input_size_varian[array_average.index(best_input_accuracy)]
    da_input_average = array_average
    input_size_varian = input_size_varian

    return (best_input, best_input_accuracy, da_input_average, input_size_varian)


def find_best_hidden(hype_parameter_result, best_input, rbm_node_size_varian):
    best_input_parameter_result = []
    for i in range(len(hype_parameter_result)):
        if hype_parameter_result[i].input_node == best_input:
            best_input_parameter_result.append(hype_parameter_result[i])

    array_da_hidden = []
    for i in range(len(best_input_parameter_result)):
        array_da_hidden.append(best_input_parameter_result[i].directional_accuracy)

    best_node_accuracy = np.max(array_da_hidden)
    best_hidden = rbm_node_size_varian[array_da_hidden.index(best_node_accuracy)]
    rbm_node_size_varian = rbm_node_size_varian

    return best_hidden, best_node_accuracy, array_da_hidden, rbm_node_size_varian


def find_best_lr(hype_parameter_result, best_input, best_hidden, lr_varians):
    average_lr_da_accuracy = []
    for i in lr_varians:
        array_da_lr = []
        for j in range(len(hype_parameter_result)):
            if hype_parameter_result[j].lr == i:
                array_da_lr.append(hype_parameter_result[j].directional_accuracy)
        average = np.average(array_da_lr)
        average_lr_da_accuracy.append(average)

    best_lr_accuracy = np.max(average_lr_da_accuracy)
    best_lr = lr_varians[average_lr_da_accuracy.index(best_lr_accuracy)]
    lr_varians = lr_varians

    return best_input, best_hidden, best_lr, best_lr_accuracy, average_lr_da_accuracy, lr_varians
