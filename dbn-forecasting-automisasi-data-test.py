import numpy as np

from dbn import SupervisedDBNRegression, split_data, open_file, normalize, renormalize, rmse, mae, mape, da
import matplotlib.pyplot as plt

import time

start_time = time.time()


def print_result(file_name, kurs, input_size, node_size, rmse_func, mae_func, mape_func, da_func):
    myfile = open('result1'+name+'.txt', 'w')
    print( "Dataset      = " + file_name)
    print ("Kurs         = " + kurs)
    print ("Input Size   = " + str(input_size))
    print (node_size)
    print ("Output Size  = " + "1")
    print ("")
    print ('Root Mean Square Error           : %f ' % rmse_func)
    print ("Mean Absolute Error              : %f" % mae_func)
    print ("Mean Absolute Percentage Error   : %f" % mape_func)
    print ("Directional Accuracy             : %f" % da_func)

    myfile.write("Dataset      = " + file_name)
    myfile.write("Kurs         = " + kurs)
    myfile.write("Input Size   = " + str(input_size))
    myfile.write(str(node_size))
    
    myfile.write('RMSE : %f ' % rmse_func)
    myfile.write('MAE : %f ' % mae_func)
    myfile.write('MAPE : %f ' % mape_func)
    myfile.write('DA : %f ' % da_func)
       
    myfile.close()
if __name__ == "__main__":
    print ('a')

    rate = []
    name = "idraud"
    file_name = "dataset/" + name + ".csv"
    kurs = "IDR/AUD"

    rate = open_file(file_name, kurs)
    input_size = 4
    node_size = [4,16 ,10]
    rbm_iter = 100
    lr = 0.003
    split_size = [80, 85, 100]

    np.random.seed(0)

    trainRealX, trainRealY, validRealX, validRealY, testRealX, testRealY = split_data(rate, input_size, split_size)

    logXNormalizeTrain, logYNormalizeTrain, paramsTrain = normalize(trainRealX, trainRealY, None)
    logXNormalizeTest, logYNormalizeTest, params = normalize(testRealX, testRealY, paramsTrain)

    print (logXNormalizeTest)

    strX_train = ''

#    regressor = SupervisedDBNRegression(hidden_layers_structure=node_size,
#                                        learning_rate_rbm=lr,
#                                        learning_rate=lr,
#                                        n_epochs_rbm=rbm_iter,
#                                       l2_regularization=0.0,
#                                        batch_size=10,
#                                        n_iter_backprop=rbm_iter,
#                                        activation_function='relu')
    
    regressor = SupervisedDBNRegression(hidden_layers_structure=node_size,
                                    optimization_algorithm='adam',
                                    learning_rate_rbm=0.03,
                                    learning_rate=lr,
                                    n_epochs_rbm=30,
                                
                                    n_iter_backprop=100,
                                    batch_size=5,
                                    activation_function='relu',
                                    dropout_p=0,
                                    train_optimization_algorithm='adam')
   # regressor.fit_and_validate(trainRealX, trainRealY, validRealX, validRealY)
    regressor.fit(np.array(logXNormalizeTrain), np.array(logYNormalizeTrain))

    Y_pred_unormalize = regressor.predict(np.array(logXNormalizeTest))

    Y_pred = renormalize(Y_pred_unormalize, params)

    random_x = np.linspace(0, len(testRealX), len(testRealY))

    rmse_func = rmse(testRealY, Y_pred)
    mae_func = mae(testRealY, Y_pred)
    mape_func = mape(np.array(testRealY), np.array(Y_pred), len(Y_pred))
    da_func = da(testRealX, testRealY, Y_pred)

    for i in range(0, len(Y_pred)):
        for j in range(0, len(testRealX[i])):
            print ("input day " + str(j + 1) + "     = " + str(testRealX[i][j]).replace('.',','))    
            # + ' ' + str(testRealX[i][j]).replace('.', ',')
        print ("actual day 4    = " + str(testRealY[i][0]).replace('.', ',') + ' \n' + "predicted day 4 = " + str(
            Y_pred[i][0]).replace('.', ',') + "\n")

    for i in range(0, len(Y_pred)):
        print (str(testRealY[i][0]).replace('.', ',') + " " + str(Y_pred[i][0]).replace('.', ','))

    print_result(file_name, kurs, input_size, node_size, rmse_func, mae_func, mape_func, da_func)

    N = len(testRealY)
    random_x = np.linspace(0, N, N)

    actual, = plt.plot(random_x, testRealY, 'g-', label='Actual')
    predicted, = plt.plot(random_x, Y_pred, "r-", label='Predicted')
    plt.legend([actual, predicted], ['Actual', 'predicted'])
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
    
    plt.plot(regressor.train_loss)
    plt.plot(regressor.validation_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
