from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
#np.random.seed(1337)
from activations import SigmoidActivationFunction, ReLUActivationFunction, TanhActivationFunction
from utils import batch_generator

#91 data
#Y_UnormalizeTest = [0.3499813641446142, 0.349608647036899, 0.3563175549757734, 0.34737234439060755, 0.3514722325754752, 0.3976891539321655, 0.3592992918374953, 0.39657100260901973, 0.37942601565411854, 0.4055162131941856, 0.38240775251584047, 0.46925083861349237, 0.476332463660082, 0.5065225493850167, 0.511367871785315, 0.4949683190458442, 0.5143496086470369, 0.5169586284010436, 0.5266492732016399, 0.5836749906820723, 0.5430488259411107, 0.551248602310846, 0.5642937010808796, 0.5609392471114424, 0.5762206485277674, 0.5635482668654491, 0.5027953783078644, 0.5061498322773015, 0.49534103615355946]
#X_UnormalizeTest = [[0.3507267983600447, 0.33134550875885205, 0.34066343645173314], [0.33134550875885205, 0.34066343645173314, 0.3499813641446142], [0.34066343645173314, 0.3499813641446142, 0.349608647036899], [0.3499813641446142, 0.349608647036899, 0.3563175549757734], [0.349608647036899, 0.3563175549757734, 0.34737234439060755], [0.3563175549757734, 0.34737234439060755, 0.3514722325754752], [0.34737234439060755, 0.3514722325754752, 0.3976891539321655], [0.3514722325754752, 0.3976891539321655, 0.3592992918374953], [0.3976891539321655, 0.3592992918374953, 0.39657100260901973], [0.3592992918374953, 0.39657100260901973, 0.37942601565411854], [0.39657100260901973, 0.37942601565411854, 0.4055162131941856], [0.37942601565411854, 0.4055162131941856, 0.38240775251584047], [0.4055162131941856, 0.38240775251584047, 0.46925083861349237], [0.38240775251584047, 0.46925083861349237, 0.476332463660082], [0.46925083861349237, 0.476332463660082, 0.5065225493850167], [0.476332463660082, 0.5065225493850167, 0.511367871785315], [0.5065225493850167, 0.511367871785315, 0.4949683190458442], [0.511367871785315, 0.4949683190458442, 0.5143496086470369], [0.4949683190458442, 0.5143496086470369, 0.5169586284010436], [0.5143496086470369, 0.5169586284010436, 0.5266492732016399], [0.5169586284010436, 0.5266492732016399, 0.5836749906820723], [0.5266492732016399, 0.5836749906820723, 0.5430488259411107], [0.5836749906820723, 0.5430488259411107, 0.551248602310846], [0.5430488259411107, 0.551248602310846, 0.5642937010808796], [0.551248602310846, 0.5642937010808796, 0.5609392471114424], [0.5642937010808796, 0.5609392471114424, 0.5762206485277674], [0.5609392471114424, 0.5762206485277674, 0.5635482668654491], [0.5762206485277674, 0.5635482668654491, 0.5027953783078644], [0.5635482668654491, 0.5027953783078644, 0.5061498322773015]]

#lebih banyak data
#Y_UnormalizeTest = [0.47931420052180396, 0.44726052925829296, 0.44055162131941855, 0.44986954901229964, 0.39060752888557587, 0.3607901602683563, 0.3660081997763697, 0.4129705553484905, 0.38017144986954904, 0.41371598956392097, 0.4219157659336564, 0.4058889303019009, 0.4308609765188222, 0.4189340290719344, 0.3876257920238539, 0.42079761461051063, 0.3660081997763697, 0.36451733134550873, 0.3760715616846813, 0.3469996272828923, 0.3447633246366008, 0.34066343645173314, 0.34662691017517705, 0.3540812523294819, 0.366380916884085, 0.36936265374580696, 0.33320909429742823, 0.3674990682072307, 0.34364517331345507]
#X_UnormalizeTest = [[0.4263883712262393, 0.4301155423033917, 0.4576966082743198], [0.4301155423033917, 0.4576966082743198, 0.47931420052180396], [0.4576966082743198, 0.47931420052180396, 0.44726052925829296], [0.47931420052180396, 0.44726052925829296, 0.44055162131941855], [0.44726052925829296, 0.44055162131941855, 0.44986954901229964], [0.44055162131941855, 0.44986954901229964, 0.39060752888557587], [0.44986954901229964, 0.39060752888557587, 0.3607901602683563], [0.39060752888557587, 0.3607901602683563, 0.3660081997763697], [0.3607901602683563, 0.3660081997763697, 0.4129705553484905], [0.3660081997763697, 0.4129705553484905, 0.38017144986954904], [0.4129705553484905, 0.38017144986954904, 0.41371598956392097], [0.38017144986954904, 0.41371598956392097, 0.4219157659336564], [0.41371598956392097, 0.4219157659336564, 0.4058889303019009], [0.4219157659336564, 0.4058889303019009, 0.4308609765188222], [0.4058889303019009, 0.4308609765188222, 0.4189340290719344], [0.4308609765188222, 0.4189340290719344, 0.3876257920238539], [0.4189340290719344, 0.3876257920238539, 0.42079761461051063], [0.3876257920238539, 0.42079761461051063, 0.3660081997763697], [0.42079761461051063, 0.3660081997763697, 0.36451733134550873], [0.3660081997763697, 0.36451733134550873, 0.3760715616846813], [0.36451733134550873, 0.3760715616846813, 0.3469996272828923], [0.3760715616846813, 0.3469996272828923, 0.3447633246366008], [0.3469996272828923, 0.3447633246366008, 0.34066343645173314], [0.3447633246366008, 0.34066343645173314, 0.34662691017517705], [0.34066343645173314, 0.34662691017517705, 0.3540812523294819], [0.34662691017517705, 0.3540812523294819, 0.366380916884085], [0.3540812523294819, 0.366380916884085, 0.36936265374580696], [0.366380916884085, 0.36936265374580696, 0.33320909429742823], [0.36936265374580696, 0.33320909429742823, 0.3674990682072307]]



class BaseModel(object):
    def save(self, save_path):
        import pickle

        with open(save_path, 'w') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, load_path):
        import pickle

        with open(load_path, 'r') as fp:
            return pickle.load(fp)


class BinaryRBM(BaseEstimator, TransformerMixin, BaseModel):
    """
    This class implements a Binary Restricted Boltzmann machine.
    """

    def __init__(self,
                 n_hidden_units=100,
                 activation_function='sigmoid',
                 optimization_algorithm='sgd',
                 learning_rate=1e-3,
                 n_epochs=10,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 verbose=True):
        self.n_hidden_units = n_hidden_units
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X):
        """
        Fit a model given data.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        # Initialize RBM parameters
        self.n_visible_units = X.shape[1]
        if self.activation_function == 'sigmoid':
            self.W = np.random.randn(self.n_hidden_units, self.n_visible_units) / np.sqrt(self.n_visible_units)
            self.c = np.random.randn(self.n_hidden_units) / np.sqrt(self.n_visible_units)
            self.b = np.random.randn(self.n_visible_units) / np.sqrt(self.n_visible_units)
            self._activation_function_class = SigmoidActivationFunction
        elif self.activation_function == 'relu':
            self.W = truncnorm.rvs(-0.2, 0.2, size=[self.n_hidden_units, self.n_visible_units]) / np.sqrt(
                self.n_visible_units)
            self.c = np.full(self.n_hidden_units, 0.1) / np.sqrt(self.n_visible_units)
            self.b = np.full(self.n_visible_units, 0.1) / np.sqrt(self.n_visible_units)
            self._activation_function_class = ReLUActivationFunction
        elif self.activation_function == 'tanh':
            self.W = np.random.randn(self.n_hidden_units, self.n_visible_units) / np.sqrt(self.n_visible_units)
            self.c = np.random.randn(self.n_hidden_units) / np.sqrt(self.n_visible_units)
            self.b = np.random.randn(self.n_visible_units) / np.sqrt(self.n_visible_units)
            self._activation_function_class = TanhActivationFunction
        else:
            raise ValueError("Invalid activation function.")

        if self.optimization_algorithm == 'sgd':
            self._stochastic_gradient_descent(X)
        else:
            raise ValueError("Invalid optimization algorithm.")
        return self

    def transform(self, X):
        """
        Transforms data using the fitted model.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
#        print "cus"
        if len(X.shape) == 1:  # It is a single sample
            return self._compute_hidden_units(X)
        transformed_data = self._compute_hidden_units_matrix(X)
#        print transformed_data
        return transformed_data

    def _reconstruct(self, transformed_data):
        """
        Reconstruct visible units given the hidden layer output.
        :param transformed_data: array-like, shape = (n_samples, n_features)
        :return:
        """
        return self._compute_visible_units_matrix(transformed_data)

    def _stochastic_gradient_descent(self, _data):
        """
        Performs stochastic gradient descend optimization algorithm.
        :param _data: array-like, shape = (n_samples, n_features)
        :return:
        """
        errorValueRBM = []
        string_output = ''
        print ("RBM Train")        
        sum_delta_W = np.zeros(self.W.shape)
        sum_delta_b = np.zeros(self.b.shape)
        sum_delta_c = np.zeros(self.c.shape)
        errorPrev = 10000
        
        for iteration in range(1, self.n_epochs + 1):
            idx = np.random.permutation(len(_data))
            data = _data[idx]
       #     print idx
       #     print data
            #print self._compute_free_energy(data[0])
            for batch in batch_generator(self.batch_size, data):
                sum_delta_W[:] = .0
                sum_delta_b[:] = .0
                sum_delta_c[:] = .0
                for sample in batch:
                    delta_W, delta_b, delta_c = self._contrastive_divergence(sample)
                    sum_delta_W += delta_W
                    sum_delta_b += delta_b
                    sum_delta_c += delta_c
                self.W += self.learning_rate * (sum_delta_W / self.batch_size)
                self.b += self.learning_rate * (sum_delta_b / self.batch_size)
                self.c += self.learning_rate * (sum_delta_c / self.batch_size)
            if self.verbose:
                #np.set_printoptions(threshold=np.inf)
                error = self._compute_reconstruction_error(data, iteration)
                print (">> Epoch %d finished \tRBM Reconstruction error %f" % (iteration, error))
                #print self.learning_rate   
                string_output += str(error).replace('.',',') + '\n'
#                if  errorPrev > error:
#                    self.learning_rate = self.learning_rate * (1. / (1. + 0.001 * iteration))                
#                    errorPrev = error
                
                self.learning_rate = self.learning_rate * (1. / (1. + self.learning_rate/self.n_epochs))                

#                print error
                
                errorValueRBM.append(error)
#            print self.learning_rate
            

#        import matplotlib.pyplot as plt
##   
#        numpyError = np.array(errorValueRBM)
#        N = self.n_epochs
#        random_x = np.linspace(0, N, N)
#        plt.plot(random_x, numpyError, 'b-')
#        plt.show
#        
        #import numpy as np

#        file = open('skripsi-'+'error-lr-rbm'+'.txt', 'w')
#        file.write(string_output)
#        file.close()

    def _contrastive_divergence(self, vector_visible_units):
        """
        Computes gradients using Contrastive Divergence method.
        :param vector_visible_units: array-like, shape = (n_features, )
        :return:
        """
        v_0 = vector_visible_units
        v_t = np.array(v_0)
        
        # Sampling
        for t in range(self.contrastive_divergence_iter):
#            print "yess"
            h_t = self._sample_hidden_units(v_t)
            v_t = self._compute_visible_units(h_t)

        # Computing deltas
        v_k = v_t
        h_0 = self._compute_hidden_units(v_0)
        h_k = self._compute_hidden_units(v_k)
        
#        print v_k
        delta_W = np.outer(h_0, v_0) - np.outer(h_k, v_k)
        delta_b = v_0 - v_k
        delta_c = h_0 - h_k

        return delta_W, delta_b, delta_c

#    def _contrastive_divergence(self, vector_visible_units):
#        """
#        Computes gradients using Contrastive Divergence method.
#        :param vector_visible_units: array-like, shape = (n_features, )
#        :return:
#        """
#        v_0 = vector_visible_units
#        v_t = np.array(v_0)
#        print "v_0"
#        print v_0
#        print "v_t"
#        print v_t
#        
#        
#        # Sampling
#        for t in range(self.contrastive_divergence_iter):
##            print "yess"
#            h_t = self._sample_hidden_units(v_t)
#            print "h_t"
#            print h_t
#            v_t = self._compute_visible_units(h_t)
#            print "v_t"            
#            print v_t
#            print " "
#        # Computing deltas
#        v_k = v_t
#        h_0 = self._compute_hidden_units(v_0)
#        print "h_0"
#        print h_0
#        h_k = self._compute_hidden_units(v_k)
#        print "h_k"        
#        print h_k
#        print " "
#        
##        print v_k
#        delta_W = np.outer(h_0, v_0) - np.outer(h_k, v_k)
#        print "delta_W"
#        print delta_W
#        delta_b = v_0 - v_k
#        delta_c = h_0 - h_k
#
#        return delta_W, delta_b, delta_c

    def _sample_hidden_units(self, vector_visible_units):
        """
        Computes hidden unit activations by sampling from a binomial distribution.
        :param vector_visible_units: array-like, shape = (n_features, )
        :return:
        """
        hidden_units = self._compute_hidden_units(vector_visible_units)
#        print (np.random.random_sample(len(hidden_units)) < hidden_units).astype(np.int64)
        return (np.random.random_sample(len(hidden_units)) < hidden_units).astype(np.int64)

    def _sample_visible_units(self, vector_hidden_units):
        """
        Computes visible unit activations by sampling from a binomial distribution.
        :param vector_hidden_units: array-like, shape = (n_features, )
        :return:
        """
        visible_units = self._compute_visible_units(vector_hidden_units)
        return (np.random.random_sample(len(visible_units)) < visible_units).astype(np.int64)

    def _compute_hidden_units(self, vector_visible_units):
        """
        Computes hidden unit outputs.
        :param vector_visible_units: array-like, shape = (n_features, )
        :return:
        """
#        print "vector_visible_units"
#        print vector_visible_units
        v = np.expand_dims(vector_visible_units, 0)
#        print "v"
#        print v
#        print "squeeze"
#        print np.squeeze(self._compute_hidden_units_matrix(v))        
        return np.squeeze(self._compute_hidden_units_matrix(v))

    def _compute_hidden_units_matrix(self, matrix_visible_units):
        """
        Computes hidden unit outputs.
        :param matrix_visible_units: array-like, shape = (n_samples, n_features)
        :return:
        """
        return np.transpose(self._activation_function_class.function(
            np.dot(self.W, np.transpose(matrix_visible_units)) + self.c[:, np.newaxis]))

    def _compute_visible_units(self, vector_hidden_units):
        """
        Computes visible (or input) unit outputs.
        :param vector_hidden_units: array-like, shape = (n_features, )
        :return:
        """
        h = np.expand_dims(vector_hidden_units, 0)
#        print np.squeeze(self._compute_visible_units_matrix(h))
        return np.squeeze(self._compute_visible_units_matrix(h))

    def _compute_visible_units_matrix(self, matrix_hidden_units):
        """
        Computes visible (or input) unit outputs.
        :param matrix_hidden_units: array-like, shape = (n_samples, n_features)
        :return:
        """
        return self._activation_function_class.function(np.dot(matrix_hidden_units, self.W) + self.b[np.newaxis, :])

    def _compute_free_energy(self, vector_visible_units):
        """
        Computes the RBM free energy.
        :param vector_visible_units: array-like, shape = (n_features, )
        :return:
        """
        v = vector_visible_units
        return - np.dot(self.b, v) - np.sum(np.log(1 + np.exp(np.dot(self.W, v) + self.c)))

    def _compute_reconstruction_error(self, data, iteration):
        np.set_printoptions(threshold=np.Infinity)

        """
        Computes the reconstruction error of the data.
        :param data: array-like, shape = (n_samples, n_features)
        :return:
        """
            
        data_transformed = self.transform(data)
        data_reconstructed = self._reconstruct(data_transformed)
        #if iteration == 300 :
#            print "data"
#            print data
#            print "reconstructed"
#            print data_reconstructed
        return np.mean(np.sum((data_reconstructed - data) ** 2, 1))


class UnsupervisedDBN(BaseEstimator, TransformerMixin, BaseModel):
    """
    This class implements a unsupervised Deep Belief Network.
    """

    def __init__(self,
                 hidden_layers_structure=[100, 100],
                 activation_function='sigmoid',
                 optimization_algorithm='sgd',
                 learning_rate_rbm=1e-3,
                 n_epochs_rbm=10,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 verbose=True):
        self.hidden_layers_structure = hidden_layers_structure
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate_rbm = learning_rate_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.rbm_layers = None
        self.verbose = verbose
        self.rbm_class = BinaryRBM

    def fit(self, X, y=None):
        """
        Fits a model given data.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        # Initialize rbm layers
        self.rbm_layers = list()
        for n_hidden_units in self.hidden_layers_structure:
            rbm = self.rbm_class(n_hidden_units=n_hidden_units,
                                 activation_function=self.activation_function,
                                 optimization_algorithm=self.optimization_algorithm,
                                 learning_rate=self.learning_rate_rbm,
                                 n_epochs=self.n_epochs_rbm,
                                 contrastive_divergence_iter=self.contrastive_divergence_iter,
                                 batch_size=self.batch_size,
                                 verbose=self.verbose)
            self.rbm_layers.append(rbm)

        # Fit RBM
        if self.verbose:
            print ("[START] Pre-training step:")
        input_data = X
        for rbm in self.rbm_layers:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)
        if self.verbose:
            print ("[END] Pre-training step")
        return self

    def transform(self, X):
        """
        Transforms data using the fitted model.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        input_data = X
        for rbm in self.rbm_layers:
            input_data = rbm.transform(input_data)
        return input_data


class AbstractSupervisedDBN(BaseEstimator, BaseModel):
    """
    Abstract class for supervised Deep Belief Network.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 unsupervised_dbn_class,
                 hidden_layers_structure=[100, 100],
                 activation_function='sigmoid',
                 optimization_algorithm='sgd',
                 learning_rate=1e-3,
                 learning_rate_rbm=1e-3,
                 n_iter_backprop=100,
                 l2_regularization=1.0,
                 n_epochs_rbm=10,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 dropout_p=0,  # float between 0 and 1. Fraction of the input units to drop
                 verbose=True):
        self.unsupervised_dbn = unsupervised_dbn_class(hidden_layers_structure=hidden_layers_structure,
                                                       activation_function=activation_function,
                                                       optimization_algorithm=optimization_algorithm,
                                                       learning_rate_rbm=learning_rate_rbm,
                                                       n_epochs_rbm=n_epochs_rbm,
                                                       contrastive_divergence_iter=contrastive_divergence_iter,
                                                       batch_size=batch_size,
                                                       verbose=verbose)
        self.unsupervised_dbn_class = unsupervised_dbn_class
        self.n_iter_backprop = n_iter_backprop
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.p = 1 - self.dropout_p
        self.verbose = verbose

    def fit(self, X, y=None, pre_train=True):
        """
        Fits a model given data.
        :param X: array-like, shape = (n_samples, n_features)
        :param y : array-like, shape = (n_samples, )
        :param pre_train: bool
        :return:
        """
        if pre_train:
            self.pre_train(X)
        self._fine_tuning(X, y)
        return self

    def predict(self, X):
        """
        Predicts the target given data.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        if len(X.shape) == 1:  # It is a single sample
            X = np.expand_dims(X, 0)
        transformed_data = self.transform(X)
#        print transformed_data
        predicted_data = self._compute_output_units_matrix(transformed_data)
#        print predicted_data
        return predicted_data

    def pre_train(self, X):
        """
        Apply unsupervised network pre-training.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        print ("masuk pre train")
        self.unsupervised_dbn.fit(X)
        return self

    def transform(self, *args):
        return self.unsupervised_dbn.transform(*args)

    @abstractmethod
    def _transform_labels_to_network_format(self, labels):
        return

    @abstractmethod
    def _compute_output_units_matrix(self, matrix_visible_units):
        return

    @abstractmethod
    def _determine_num_output_neurons(self, labels):
        return

    @abstractmethod
    def _stochastic_gradient_descent(self, data, labels):
        return

    @abstractmethod
    def _fine_tuning(self, data, _labels):
        return


class NumPyAbstractSupervisedDBN(AbstractSupervisedDBN):
    """
    Abstract class for supervised Deep Belief Network in NumPy
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(NumPyAbstractSupervisedDBN, self).__init__(UnsupervisedDBN, **kwargs)

    def _compute_activations(self, sample):
        """
        Compute output values of all layers.
        :param sample: array-like, shape = (n_features, )
        :return:
        """
        input_data = sample
        if self.dropout_p > 0:
            r = np.random.binomial(1, self.p, len(input_data))
            input_data *= r
        layers_activation = list()

        for rbm in self.unsupervised_dbn.rbm_layers:
            input_data = rbm.transform(input_data)
            if self.dropout_p > 0:
                r = np.random.binomial(1, self.p, len(input_data))
                input_data *= r
            layers_activation.append(input_data)

        # Computing activation of output layer
        input_data = self._compute_output_units(input_data)
        layers_activation.append(input_data)

        return layers_activation

    def _stochastic_gradient_descent(self, _data, _labels):
        """
        Performs stochastic gradient descend optimization algorithm.
        :param _data: array-like, shape = (n_samples, n_features)
        :param _labels: array-like, shape = (n_samples, targets)
        :return:
        """
        errorValue = []  
        errorValueValid = []
        string_output_BP = ''
        
        if self.verbose:
            matrix_error = np.zeros([len(_data), self.num_classes])
        num_samples = len(_data)
        sum_delta_W = [np.zeros(rbm.W.shape) for rbm in self.unsupervised_dbn.rbm_layers]
        sum_delta_W.append(np.zeros(self.W.shape))
        sum_delta_bias = [np.zeros(rbm.c.shape) for rbm in self.unsupervised_dbn.rbm_layers]
        sum_delta_bias.append(np.zeros(self.b.shape))
        errorPrev = 10000
        
        for iteration in range(1, self.n_iter_backprop + 1):
            idx = np.random.permutation(len(_data))
            data = _data[idx]
            labels = _labels[idx]
            i = 0
            for batch_data, batch_labels in batch_generator(self.batch_size, data, labels):
                # Clear arrays
                for arr1, arr2 in zip(sum_delta_W, sum_delta_bias):
                    arr1[:], arr2[:] = .0, .0
                for sample, label in zip(batch_data, batch_labels):
                    delta_W, delta_bias, predicted = self._backpropagation(sample, label)
                    for layer in range(len(self.unsupervised_dbn.rbm_layers) + 1):
                        sum_delta_W[layer] += delta_W[layer]
                        sum_delta_bias[layer] += delta_bias[layer]
                    if self.verbose:
                        loss = self._compute_loss(predicted, label)
                        matrix_error[i, :] = loss
                        i += 1

                layer = 0
                for rbm in self.unsupervised_dbn.rbm_layers:
                    # Updating parameters of hidden layers
                    rbm.W = (1 - (
                        self.learning_rate * self.l2_regularization) / num_samples) * rbm.W - self.learning_rate * (
                        sum_delta_W[layer] / self.batch_size)
                    rbm.c -= self.learning_rate * (sum_delta_bias[layer] / self.batch_size)
                    layer += 1
                # Updating parameters of output layer
                self.W = (1 - (
                    self.learning_rate * self.l2_regularization) / num_samples) * self.W - self.learning_rate * (
                    sum_delta_W[layer] / self.batch_size)
                self.b -= self.learning_rate * (sum_delta_bias[layer] / self.batch_size)

            if self.verbose:
                error = np.mean(np.sum(matrix_error, 1))/2
#
#                matrix_errorValid = np.zeros([len(Y_UnormalizeTest), 1])
#                l=0
#                for j in xrange(0, len(X_UnormalizeTest)):
#                    a, b, predicted = self._backpropagation(np.array(X_UnormalizeTest[j]), Y_UnormalizeTest[j])
#                    if self.verbose:
#                        loss = self._compute_loss(predicted, Y_UnormalizeTest[j])
#                        matrix_errorValid[j, :] = loss
#                        l += 1
#                
#                errorValid = np.mean(np.sum(matrix_errorValid, 1))                
                
#                print error
#                print errorValid
                
#                print"__"
#                print matrix_error
#                print matrix_errorValid
                
                print (">> Epoch %d finished \tANN training loss %f" % (iteration, error))
#                print self.learning_rate                
#                if  errorPrev > error:
                self.learning_rate = self.learning_rate * (1. / (1. + self.learning_rate/self.n_iter_backprop))                
#                    errorPrev = error
#                print errorValid
                
                errorValue.append(error)
#                errorValueValid.append(errorValid)
#                string_output_BP += str(errorValid).replace('.',',') + '\n'
#             
#        
#        import matplotlib.pyplot as plt
####
#        numpyError = np.array(errorValue)
#        numpyErrorValid = np.array(errorValueValid)
##        
#        N = self.n_iter_backprop
#        random_x = np.linspace(0, N, N)
#        plt.plot(random_x, numpyError, 'g-')
#        plt.show
#        
#        from datetime import datetime
#        print datetime.microsecond
        
        import datetime
        basename = "lr-with-time"
#        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
#        filename = "_".join([basename, suffix]) # e.g. 'mylogfile_120508_171442'

        
        file = open(basename +'.txt', 'w')
        file.write(string_output_BP)
        file.close()

    def _backpropagation(self, input_vector, label):
        """
        Performs Backpropagation algorithm for computing gradients.
        :param input_vector: array-like, shape = (n_features, )
        :param label: array-like, shape = (n_targets, )
        :return:
        """
        x, y = input_vector, label
        deltas = list()
        list_layer_weights = list()
        for rbm in self.unsupervised_dbn.rbm_layers:
            list_layer_weights.append(rbm.W)
        list_layer_weights.append(self.W)

        # Forward pass
        layers_activation = self._compute_activations(input_vector)

        # Backward pass: computing deltas
        activation_output_layer = layers_activation[-1]
        delta_output_layer = self._compute_output_layer_delta(y, activation_output_layer)
        deltas.append(delta_output_layer)
        layer_idx = range(len(self.unsupervised_dbn.rbm_layers))
        layer_idx.reverse()
        delta_previous_layer = delta_output_layer
        for layer in layer_idx:
            neuron_activations = layers_activation[layer]
            W = list_layer_weights[layer + 1]
            delta = np.dot(delta_previous_layer, W) * self.unsupervised_dbn.rbm_layers[
                layer]._activation_function_class.prime(neuron_activations)
            deltas.append(delta)
            delta_previous_layer = delta
        deltas.reverse()

        # Computing gradients
        layers_activation.pop()
        layers_activation.insert(0, input_vector)
        layer_gradient_weights, layer_gradient_bias = list(), list()
        for layer in range(len(list_layer_weights)):
            neuron_activations = layers_activation[layer]
            delta = deltas[layer]
            gradient_W = np.outer(delta, neuron_activations)
            layer_gradient_weights.append(gradient_W)
            layer_gradient_bias.append(delta)

        return layer_gradient_weights, layer_gradient_bias, activation_output_layer

    def _fine_tuning(self, data, _labels):
        """
        Entry point of the fine tuning procedure.
        :param data: array-like, shape = (n_samples, n_features)
        :param _labels: array-like, shape = (n_samples, targets)
        :return:
        """
        self.num_classes = self._determine_num_output_neurons(_labels)
        n_hidden_units_previous_layer = self.unsupervised_dbn.rbm_layers[-1].n_hidden_units
        self.W = np.random.randn(self.num_classes, n_hidden_units_previous_layer) / np.sqrt(
            n_hidden_units_previous_layer)
        self.b = np.random.randn(self.num_classes) / np.sqrt(n_hidden_units_previous_layer)

        labels = self._transform_labels_to_network_format(_labels)

        # Scaling up weights obtained from pretraining
        for rbm in self.unsupervised_dbn.rbm_layers:
            rbm.W /= self.p
            rbm.c /= self.p

        if self.verbose:
            print ("[START] Fine tuning step:")

        if self.unsupervised_dbn.optimization_algorithm == 'sgd':
            self._stochastic_gradient_descent(data, labels)
        else:
            raise ValueError("Invalid optimization algorithm.")

        # Scaling down weights obtained from pretraining
        for rbm in self.unsupervised_dbn.rbm_layers:
            rbm.W *= self.p
            rbm.c *= self.p

        if self.verbose:
            print ("[END] Fine tuning step")

    @abstractmethod
    def _compute_loss(self, predicted, label):
        return

    @abstractmethod
    def _compute_output_layer_delta(self, label, predicted):
        return


class SupervisedDBNRegression(NumPyAbstractSupervisedDBN, RegressorMixin):
    """
    This class implements a Deep Belief Network for regression problems.
    """

    def _transform_labels_to_network_format(self, labels):
        """
        Returns the same labels since regression case does not need to convert anything.
        :param labels: array-like, shape = (n_samples, targets)
        :return:
        """
        return labels

    def _compute_output_units(self, vector_visible_units):
        """
        Compute activations of output units.
        :param vector_visible_units: array-like, shape = (n_features, )
        :return:
        """
        v = vector_visible_units
        return np.dot(self.W, v) + self.b

    def _compute_output_units_matrix(self, matrix_visible_units):
        """
        Compute activations of output units.
        :param matrix_visible_units: shape = (n_samples, n_features)
        :return:
        """
#        print "oute yess"
#        print np.transpose(np.dot(self.W, np.transpose(matrix_visible_units)) + self.b[:, np.newaxis])

        return np.transpose(np.dot(self.W, np.transpose(matrix_visible_units)) + self.b[:, np.newaxis])

    def _compute_output_layer_delta(self, label, predicted):
        """
        Compute deltas of the output layer for the regression case, using common (one-half) squared-error cost function.
        :param label: array-like, shape = (n_features, )
        :param predicted: array-like, shape = (n_features, )
        :return:
        """
        return -(label - predicted)

    def _determine_num_output_neurons(self, labels):
        """
        Given labels, compute the needed number of output units.
        :param labels: shape = (n_samples, n_targets)
        :return:
        """
        if len(labels.shape) == 1:
            return 1
        else:
            return labels.shape[1]

    def _compute_loss(self, predicted, label):
        """
        Computes Mean squared error loss.
        :param predicted:
        :param label:
        :return:
        """
        error = predicted - label
        return error * error
