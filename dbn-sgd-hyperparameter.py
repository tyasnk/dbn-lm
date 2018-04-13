import numpy as np

from dbn import SupervisedDBNRegression, split_data, open_file, normalize, renormalize, rmse, mae, mape, da, \
    find_best_input, find_best_hidden, find_best_lr

from sklearn.metrics.regression import r2_score, mean_squared_error
import matplotlib.pyplot as plotInput
import matplotlib.pyplot as plotlHidden

import csv
import math

import time

start_time = time.time()

# Your statements here

rate = []
name = "idraud"
fileName = "dataset/" + name + ".csv"
kurs = "IDR/AUD"

rate = open_file(fileName, kurs)

rbm_node_size_varian = []

for i in [ 4,8,12,16,20 ]:
# for j in [20 ]:
 # for k in [4, 8, 12,16,20 ]:
            rbm_node_size_varian.append([i])

input_size_varian = [3,4,5,6,7]
iteration_rbm = [30]
string_output_varian = ''
lr = [0,003]
split_size = [80, 85, 100]

hype_parameter_result = []


class result:
    def __init__(self, input_node, hidden_node, iteration_rbm, lr, directional_accuracy):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.iteration_rbm = iteration_rbm
        self.lr = lr
        self.directional_accuracy = directional_accuracy

myfile = open('best_input'+name+'.txt', 'w')
for input_size in input_size_varian:
     for node_size in rbm_node_size_varian:
         for rbm_iter in iteration_rbm:
             np.random.seed(0)

             trainRealX, trainRealY, validRealX, validRealY, testRealX, testRealY = split_data(rate, input_size,
                                                                                               split_size)

             logXNormalizeTrain, logYNormalizeTrain, paramsTrain = normalize(trainRealX, trainRealY, None)
             logXNormalizeValid, logYNormalizeValid, params = normalize(validRealX, validRealY, paramsTrain)

             strX_train = ''

             regressor = SupervisedDBNRegression(hidden_layers_structure=node_size,
                                                 learning_rate_rbm=lr[1],
                                                 learning_rate=lr[1],
                                                 n_epochs_rbm=rbm_iter,
                                                 l2_regularization=0.0,
                                                 batch_size=10,
                                                 n_iter_backprop=rbm_iter,
                                                 activation_function='relu')

             regressor.fit(np.array(logXNormalizeTrain), np.array(logYNormalizeTrain))

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



# lr_varians = [0.1, 0.0825, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001]
lr_varians = []
max_lr = 0.1
min_lr = 0.008
partition = 10

for i in xrange(0, partition):
    lr_varians.append(min_lr + (max_lr * i / partition))

lr_varians = [0.008, 0.025, 0.005, 0.003,0.01, 0.02, 0.03, 0.04,  0.09, 0.1]#
#lr_varians = [0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.01, 0.02, 0.03, 0.04, 0.05]
#lr_varians = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
print lr_varians

rbm_iter = 100

hype_parameter_result_lr = []

myfile = open('learning_rate'+name+'.txt', 'w')
for learning_rate in lr_varians:
    for i in range(0, 1):
        np.random.seed(0)
        trainRealX, trainRealY, validRealX, validRealY, testRealX, testRealY = split_data(rate, best_input, split_size)

        logXNormalizeTrain, logYNormalizeTrain, paramsTrain = normalize(trainRealX, trainRealY, None)
        logXNormalizeValid, logYNormalizeValid, params = normalize(testRealX, testRealY, paramsTrain)

        strX_train = ''

        regressor = SupervisedDBNRegression(hidden_layers_structure=best_hidden,
                                            learning_rate_rbm=learning_rate,
                                            learning_rate=learning_rate,
                                            n_epochs_rbm=rbm_iter,
                                            l2_regularization=0.0,
                                            batch_size=10,
                                            n_iter_backprop=rbm_iter,
                                            activation_function='relu')

        regressor.fit(np.array(logXNormalizeTrain), np.array(logYNormalizeTrain))

        Y_predUnormalize = regressor.predict(np.array(logXNormalizeValid))

        Y_pred = renormalize(Y_predUnormalize, params)

        random_x = np.linspace(0, len(testRealX), len(testRealY))

        string_output = ''
        
       

        rmseFunc = rmse(testRealY, Y_pred)
        maeFunc = mae(testRealY, Y_pred)
        mapeFunc = mape(np.array(testRealY), np.array(Y_pred), len(Y_pred))
        daFunc = da(testRealX, testRealY, Y_pred)

        hype_parameter_result_lr.append(result([best_input], best_hidden, rbm_iter, learning_rate, daFunc))

        print ("lr = " + str(learning_rate))
        print ('RMSE : %f ' % rmseFunc)
        print ("MAE : %f" % maeFunc)
        print ("MAPE : %f" % mapeFunc)
        print ("DA : %f" % daFunc)
        
        myfile.write("lr =" +str(learning_rate))
        myfile.write('RMSE : %f ' % rmseFunc)
        myfile.write('MAE : %f ' % maeFunc)
        myfile.write('MAPE : %f ' % mapeFunc)
        myfile.write('DA : %f ' % daFunc)
       
myfile.close()

best_input, best_hidden, best_lr, best_lr_accuracy, average_lr_da_accuracy, lr_varians = find_best_lr(
    hype_parameter_result_lr, [best_input], best_hidden, lr_varians)

myfile = open('best_parameter'+name+'.txt', 'w')
for i in hype_parameter_result_lr:
    print ("input node = " + str(i.input_node))
    print ("hidden_node = " + str(i.hidden_node))
    print ("iteration_rbm = " + str(i.iteration_rbm))
    print ("lr = " + str(i.lr))
    print ("directional_accuracy = " + str(i.directional_accuracy))
    print ("\n")
   
    myfile.write("input node="  +str(i.input_node))
    myfile.write("hidden node ="  +str(i.hidden_node))
    myfile.write("iteration rbm ="  +str(i.iteration_rbm))
    myfile.write("lr =" +str(i.lr))
    myfile.write("da ="  +str(i.directional_accuracy))
    myfile.write("\n")
    

print ("Input Optimal       = " + str(best_input))
print ("Hidden Node Optimal = " + str(best_hidden))
print ("LR Optimal          = " + str(best_lr))
print ("LR Varians          = " + str(lr_varians))
print ("LR Optimal Acc      = " + str(best_lr_accuracy))
print ("LR Optimal Acc Array= " + str(average_lr_da_accuracy))
print("--- %s seconds ---" % (time.time() - start_time))


myfile.write("input optimal ="  +str(best_input))
myfile.write("Hidden optimal="  +str(best_hidden))
myfile.write("lr optimal =" + str(best_lr))
myfile.write("LR Varians          = " + str(lr_varians))
myfile.write("LR Optimal Acc      = " + str(best_lr_accuracy))
myfile.write("LR Optimal Acc Array= " + str(average_lr_da_accuracy))
myfile.write("--- %s seconds ---" % (time.time() - start_time))
myfile.close()
    




			
