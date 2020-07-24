import numpy as np
class Netwerk(object):
    def __init__(self, groottes):
        self.aantal_lagen = len(groottes)
        self.groottes = groottes
        """
        Groottes is een lijst met de groottes van de respectievelijke neuronen bv (2,3,1) geeft 2 als input,
        3 als hidden en 1 als output
        """
        self.biases = [np.random.randn(y, 1) for y in groottes[1:]]
        """
        maakt een lijst van arrays (hoeveelheid afhankelijk van hoeveel hidden lagen) waarbij
        de arrays elk het aantal neuronen per laag bevatten aan items die elk gaussisch gegenereerde getallen
        zijn. Het vorig voorbeeld geeft dan bijvoorbeeld [array([[ 0.67097482],[-0.07362273],[-1.05704179]]),
        array([[-1.14542345]])]
        """
        self.gewichten = [np.random.randn(y, x) for x, y in zip(groottes[:-1], groottes[1:])]
        """
        de zip geeft een lijst terug van tuples die de overgangen voorstellen, met dit voorbeeld dus [(2,3),(3,1)].
        np.random.randn(y,x) geeft een array van y vectoren terug van elk x elementen, bij dit voorbeeld dus
        [array([[-0.5024744 , -0.26511926],
                [ 0.6797002 , -0.79437011],
                [ 0.00753035,  0.15947405]]),
         array([[ 0.98995451, -1.17216928, -2.05475119]])]
        """

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def feedforward(self, a):
        """geeft de output van het netwerk terug als 'a' de input is
        (b en w in die zip zijn arrays van de hele laag), ze loopt dus over elke laag"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Dit traint het neural network met mini-batch stochastic
        gradient descent. "training_data" Is een lijst van tuples
        "(x, y)" die de training inputs en de gewenste
        outputs voorstellen. als "test_data" wordt gegeven dan zal het
        netwerk geëvalueerd worden tegen de test data na elke
        epoch, en wordt de geleidelijke vooruitgang geprint. Dit is bruikbaar om te zien hoe het vordert
        maar vertraagd het process heel veel."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
