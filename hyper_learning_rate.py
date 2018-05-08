import numpy as np
import matplotlib.pyplot as plt

from dbn import SupervisedDBNRegression, split_data, open_file, normalize, renormalize, rmse, mae, mape, da, \
    find_best_input, find_best_hidden, find_best_lr
from sklearn.metrics.regression import r2_score, mean_squared_error

#from .utils import batch_generator
#from activations import SigmoidActivationFunction, ReLUActivationFunction, TanhActivationFunction

import time

start_time = time.time()

# Your statements here

rate = []
name = "idrusd"
fileName = "dataset/" + name + ".csv"
kurs = "IDR/USD"

rate = open_file(fileName, kurs)

#rbm_node_size_varian = []
#
#for i in [4]:
#   for j in [16]:
#      for k in [4,8,10,12,16,20]:
#     #     for l in[8,16,24,32,40,48]:
#            rbm_node_size_varian.append([i,j,k])
#
#input_size_varian = [4]
#iteration_rbm = [100]
#string_output_varian = ''
#lr = [0.001]
split_size = [80, 85, 100]
#
#hype_parameter_result = []


class result:
    def __init__(self, input_node, hidden_node, iteration_rbm, lr, directional_accuracy):
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.iteration_rbm = iteration_rbm
        self.lr = lr
        self.directional_accuracy = directional_accuracy

lr_varians = []
max_lr = 0.1
min_lr = 0.008
partition = 10

for i in range(0, partition):
    lr_varians.append(min_lr + (max_lr * i / partition))

lr_varians = [0.008,0.005, 0.003,0.01, 0.02, 0.03,0.1]
lr_varians = []

print (lr_varians)

rbm_iter = 100

hype_parameter_result_lr = []

myfile = open('learning_rate'+name+'.txt', 'w')
for learning_rate in lr_varians:
    for i in range(0, 1):
        np.random.seed(0)
        trainRealX, trainRealY, validRealX, validRealY, testRealX, testRealY = split_data(rate, 4, split_size)
        logXNormalizeTrain, logYNormalizeTrain, paramsTrain = normalize(trainRealX, trainRealY, None)
        logXNormalizeValid, logYNormalizeValid, params = normalize(testRealX, testRealY, paramsTrain)
        
            
        strX_train = ''

        regressor = SupervisedDBNRegression(hidden_layers_structure=[4,16,8],
                                            learning_rate_rbm=learning_rate,
                                            learning_rate=learning_rate,
                                            n_epochs_rbm=rbm_iter,
                                            l2_regularization=0.0,
                                            batch_size=5,
                                            n_iter_backprop=rbm_iter,
                                            activation_function='relu',
                                            train_optimization_algorithm='adam')

        #regressor.fit(np.array(logXNormalizeTrain), np.array(logYNormalizeTrain))
        regressor.fit_and_validate(np.array(logXNormalizeTrain), np.array(logYNormalizeTrain), np.array(logXNormalizeValid), np.array(logYNormalizeValid))
    
        Y_predUnormalize = regressor.predict(np.array(logXNormalizeValid))

        Y_pred = renormalize(Y_predUnormalize, params)

        random_x = np.linspace(0, len(testRealX), len(testRealY))

        string_output = ''
        
       

        rmseFunc = rmse(testRealY, Y_pred)
        maeFunc = mae(testRealY, Y_pred)
        mapeFunc = mape(np.array(testRealY), np.array(Y_pred), len(Y_pred))
        daFunc = da(testRealX, testRealY, Y_pred)

        hype_parameter_result_lr.append(result([4], [4,16,8], rbm_iter, learning_rate, daFunc))

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
#best_input, best_hidden
best_input, best_hidden, best_lr, best_lr_accuracy, average_lr_da_accuracy, lr_varians = find_best_lr(
    hype_parameter_result_lr, [4], [4,16,8], lr_varians)

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
    


plt.plot(regressor.train_loss)
plt.plot(regressor.validation_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()




			
