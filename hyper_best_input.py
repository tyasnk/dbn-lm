import numpy as np
#import matplotlib.pyplot as plt

from dbn import SupervisedDBNRegression, split_data, open_file, normalize, renormalize, rmse, mae, mape, da, \
    find_best_input, find_best_hidden

from sklearn.metrics.regression import r2_score, mean_squared_error
import matplotlib.pyplot as plotInput
#import matplotlib.pyplot as plotlHidden
#from .utils import batch_generator
#from activations import SigmoidActivationFunction, ReLUActivationFunction, TanhActivationFunction

#import csv
#import math

import time

start_time = time.time()

# Your statements here

rate = []
name = "idraud"
fileName = "dataset/" + name + ".csv"
kurs = "IDR/AUD"

rate = open_file(fileName, kurs)

rbm_node_size_varian = []

for i in [14,8,10,12,16,20]:
    #for j in [4,8,10,12,16,20]:
     #   for k in [4,8,10,12,16,20]:
     #     for l in[8,16,24,32,40,48]:
            rbm_node_size_varian.append([i])

input_size_varian = [3,4,5,6,7]
iteration_rbm = [100]
string_output_varian = ''
lr = [0.001]
split_size = [80, 85, 100]

hype_parameter_result = []


class result:
    def __init__(self, input_node, hidden_node, iteration_rbm, lr, directional_accuracy):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.iteration_rbm = iteration_rbm
        self.lr = lr
        self.directional_accuracy = directional_accuracy

myfile = open('best_input_'+name+'.txt', 'w')
for input_size in input_size_varian:
     for node_size in rbm_node_size_varian:
         for rbm_iter in iteration_rbm:
             np.random.seed(0)

             trainRealX, trainRealY, validRealX, validRealY, testRealX, testRealY = split_data(rate, input_size,split_size)

             logXNormalizeTrain, logYNormalizeTrain, paramsTrain = normalize(trainRealX, trainRealY, None)
             logXNormalizeValid, logYNormalizeValid, params = normalize(validRealX, validRealY, paramsTrain)
             #logXNormalizeValid, logYNormalizeValid, paramsValid = normalize(validRealX, validRealY, paramsTrain)

             strX_train = ''
#
#             regressor = SupervisedDBNRegression(hidden_layers_structure=node_size,
#                                                 optimization_algorithm='sgd' ,
#                                                 learning_rate_rbm=lr[0],
#                                                 learning_rate=lr[0],9[8]
#                                                 n_epochs_rbm=rbm_iter,
#                                                 l2_regularization=0.0,
#                                                 batch_size=10,
#                                                 n_iter_backprop=rbm_iter,
#                                                 activation_function='relu')

             regressor = SupervisedDBNRegression(hidden_layers_structure=node_size,
                                                optimization_algorithm='adam',
                                                learning_rate_rbm=0.001,
                                                learning_rate=0.001,
                                                n_epochs_rbm=30,
                                                n_iter_backprop=100,
                                                l2_regularization=0.0,
                                                batch_size=10,
                                                activation_function='relu',
                                                dropout_p=0,
                                                train_optimization_algorithm='adam')
             
           #  regressor.fit(np.array(logXNormalizeTrain), np.array(logYNormalizeTrain))
             regressor.fit_and_validate(np.array(logXNormalizeTrain), np.array(logYNormalizeTrain), np.array(logXNormalizeValid), np.array(logYNormalizeValid))
    
             Y_predUnormalize = regressor.predict(np.array(logXNormalizeValid))

             Y_pred = renormalize(Y_predUnormalize, params)

             random_x = np.linspace(0, len(validRealX), len(validRealY))

             string_output = ''

             rmseFunc = rmse(validRealY, Y_pred)
             maeFunc = mae(validRealY, Y_pred)
             mapeFunc = mape(np.array(validRealY), np.array(Y_pred), len(Y_pred))
             daFunc = da(validRealX, validRealY, Y_pred)

             hype_parameter_result.append(result(input_size, node_size, iteration_rbm, lr[0], daFunc))
             print (input_size)
             myfile.write(str(input_size))
             print (node_size)
             myfile.write(str(node_size))
             print ('RMSE : %f ' % rmseFunc)
             myfile.write('RMSE : %f ' % rmseFunc)
             print ("MAE : %f" % maeFunc)
             myfile.write('MAE : %f ' % maeFunc)
             print ("MAPE : %f" % mapeFunc)
             myfile.write('MAPE : %f ' % mapeFunc)
             print ("DA : %f" % daFunc)
             myfile.write('DA : %f ' % daFunc)
myfile.close()


arrayInput = input_size_varian
arrayAccInput = []

for i in range(len(input_size_varian)):
     array_da_result_input = []
     for j in range(len(hype_parameter_result)):
         if (hype_parameter_result[j].input_node == input_size_varian[i]):
             array_da_result_input.append(hype_parameter_result[j].directional_accuracy)
     average = np.average(array_da_result_input)
     print (average)
     print (len(array_da_result_input))

print (find_best_input(hype_parameter_result, input_size_varian))
best_input = find_best_input(hype_parameter_result, input_size_varian)[0]
print (find_best_hidden(hype_parameter_result, best_input, rbm_node_size_varian))
best_hidden = find_best_hidden(hype_parameter_result, best_input, rbm_node_size_varian)[0]
best_hidden_da = find_best_hidden(hype_parameter_result, best_input, rbm_node_size_varian)[1]



N = len(rbm_node_size_varian)
x = range(N)

my_ticks = []
for ticks in rbm_node_size_varian:
     my_ticks.append("" + str(ticks) + "")
print (my_ticks)

plotInput.bar(x, find_best_hidden(hype_parameter_result, best_input, rbm_node_size_varian)[2], 0.9, color="blue",
               align="center")
 # my_ticks = ["[12, 12, 4]  ", "[12, 12, 8]  ", "[12, 12, 12]  ", "[12, 12, 16]  ", "[12, 12, 20]  "]
plotInput.ylim([np.min(find_best_hidden(hype_parameter_result, best_input, rbm_node_size_varian)[2]) * 0.7,
                 np.max(find_best_hidden(hype_parameter_result, best_input, rbm_node_size_varian)[2]) * 1.05])
plotInput.ylabel("Directional Accurary")
plotInput.xlabel("Node Size")
plotInput.xticks(x, my_ticks)
plotInput.show

print ("\n")
print (("\n"))
print ("Kurs                = " + kurs)
print ("Input Varians       = " + str(input_size_varian))
print ("Hidden Node Varians = " + str(rbm_node_size_varian))
print ("Input Optimal       = " + str(best_input))
print ("Hidden Node Optimal = " + str(best_hidden))
print ("DA                  = " + str(best_hidden_da))



 


#plt.plot(regressor.train_loss)
#plt.plot(regressor.validation_loss)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
#
#


			