import numpy as np

class CNN():

    def init_parameter_conv(self, size):

        #initializes the parameter of size as specified by user with mean = 0 and standard deviation inversely
        #proportional to square root of number of units

        scale = 1.0 / np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = scale, size = size) * 0.01

    def init_parameter_fc(self, size):

        #initializes weights for fully connected layer
        (a, b) = size

        return np.random.rand(a, b) * 0.01

    def single_convolution(self, A_prev_slice, W, b):

        #performs single convolution operation using single filter and bias to give one scalar as output

        temp = A_prev_slice * W
        Z = np.sum(temp)
        Z = Z + b
        return Z

    def zero_pad(self, X, pad):

        #pads the input with zeros according to required pad amount

        return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

    def conv_forward(self, A_prev, W, b, stride, padding):
        
        #performs forward propagation on one layer of CNN

        (m, n_Hprev, n_Wprev, _) = A_prev.shape

        (f, f, _, n_C) = W.shape

        if padding == "same":
            pad = int(((n_Hprev - 1) * stride + f - n_Hprev) / 2)
        else:
            pad = 0

        n_H = int((n_Hprev + 2 * pad - f) / stride) + 1
        n_W = int((n_Wprev + 2 * pad - f) / stride) + 1

        Z = np.zeros((m, n_H, n_W, n_C))

        A_prev_pad = self.zero_pad(A_prev, pad)

        for i in range(m):
            for h in range(0, n_H):
                for w in range(0, n_W):
                    for c in range(0, n_C):

                        vert_start = h * stride
                        vert_end = h * stride + f
                        horiz_start = w * stride
                        horiz_end = w * stride + f
                        
                        a_prev_slice = A_prev_pad[i, vert_start : vert_end, horiz_start : horiz_end, :]                        

                        Z[i, h, w, c] = self.single_convolution(a_prev_slice, W[:, :, :, c], b[:, :, :, c])

        cache = (A_prev, W, b, stride, pad)
        return Z, cache

    def pool_forward(self, A_prev, stride, ksize, mode):

        (m, n_Hprev, n_Wprev, n_Cprev) = A_prev.shape

        n_H = (n_Hprev - ksize) // stride + 1
        n_W = (n_Wprev - ksize) // stride + 1
        n_C = n_Cprev

        Z = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):

                        vert_start = h * stride
                        vert_end = h * stride + ksize
                        horiz_start = w * stride
                        horiz_end = w * stride + ksize

                        a_slice_prev = A_prev[i, vert_start : vert_end, horiz_start : horiz_end, c]

                        if mode == "max":
                            Z[i, h, w, c] = np.max(a_slice_prev)
                        else:
                            Z[i, h, w, c] = np.mean(a_slice_prev)

        cache = (A_prev, stride, ksize, mode)

        return Z, cache

    def conv_backward(self, dZ, cache):

        #Back propagate the convolution operation

        (A_prev, W, b, stride, pad) = cache

        (f, f, _, n_C) = W.shape

        (m, n_H, n_W, n_C) = dZ.shape

        #initialize gradient matrices
        dA_prev = np.zeros(A_prev.shape)
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)

        #pad dA
        dA_prev_pad = self.zero_pad(dA_prev, pad)
        A_prev_pad = self.zero_pad(A_prev, pad)

        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):

                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        a_slice = A_prev_pad[i, vert_start : vert_end, horiz_start : horiz_end, :]

                        dA_prev_pad[i, vert_start : vert_end, horiz_start : horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += np.sum(dZ[i, h, w, c])

        dA_prev = dA_prev_pad[:, pad : -pad, pad : -pad, :]

        return dA_prev, dW, db

    def create_mask(self, X):

        #creates mask of boolean values which have value "True" for max element and "False" for all others
        #this is done because only the max value among the filtered values contribute to the calculation of Z

        mask = (X == np.max(X))

        return mask

    def distribute_value(self, value, shape):

        #creates matrix formed by equally distributing 'the average value after average pooling' among all elements

        (n_H, n_W) = shape

        average = value / (n_H * n_W)

        return np.ones(shape) * average

    def pool_backward(self, dA, cache):

        #creates backpropagatino of pooling layer using above defined functions

        A_prev, stride, f, mode = cache

        (m, n_H, n_W, n_C) = dA.shape

        dA_prev = np.zeros(A_prev.shape)

        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):

                        #define corners

                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        A_prev_slice = A_prev[i, vert_start : vert_end, horiz_start : horiz_end, c]

                        if mode == "max":
                            mask = self.create_mask(A_prev_slice)
                            dA_prev[i, vert_start : vert_end, horiz_start : horiz_end, c] += mask * dA[i, h, w, c]

                        else:
                            value = dA[i, h, w, c]
                            shape = (f, f)
                            dA_prev[i, vert_start : vert_end, horiz_start : horiz_end, c] += self.distribute_value(value, shape)

        return dA_prev

    def relu(self, X):
        return X * (X > 0), X

    def relu_back(self, X):
        return np.greater(X, 0).astype(int)

    def sigmoid(self, Z):
        np.clip(Z, -100, 100)
        den = 1.0 + np.exp(-Z)
        sig = 1.0 / den
        return sig, Z
    
    def sigmoid_back(self, Z):
        sig, _ = self.sigmoid(Z)
        return sig * (1 - sig)

    def softmax(self, X):
        temp = np.exp(X)
        return temp / np.sum(temp), X

    def linear_forward(self, A_prev, W, b):

        #Calculates and returns Z

        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def fc_forward(self, A_prev, W, b, activation):

        #Calculates activation of layer

        Z, cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, Z = self.sigmoid(Z)
        elif activation == "relu":
            A, Z = self.relu(Z)
        else:
            A, Z = self.softmax(Z)
        caches = (cache, Z)
        return A, caches

    def linear_back(self, dZ, cache):

        '''
        Description : Calculates derivatives of cost function
        Input : dZ, dA, dW, db (dZ given because activation function may vary depending on layer position)
        Variables:
            dA_prev = derivative of cost wrt activation of previous layer
            dW = derivative of cost wrt weights of current layer
            db = derivative of cost wrt biases of current layer'''

        A_prev, W, _ = cache
        n = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / n + (0.1 / n) * W
        db = np.sum(dZ, axis = 1, keepdims = True) / n
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def fc_back(self, dA, cache, activation):

        #Returns derivatives of cost function wrt previous layer activation, weights of current layer, 
        #biases of current layer

        linear_cache, Z = cache
        if activation == "relu":
            dZ = dA * self.relu_back(Z)
        elif activation == "sigmoid":
            dZ = dA * self.sigmoid_back(Z)
        dA_prev, dW, db = self.linear_back(dZ, linear_cache)
        return dA_prev, dW, db

    def adam_initializer(self, parameters):
        L = len(parameters) // 2
        v = {}
        s = {}
        for i in range(1, L + 1):
            v["dW" + str(i)] = np.zeros(parameters["W" + str(i)].shape)
            v["db" + str(i)] = np.zeros(parameters["b" + str(i)].shape)
            s["dW" + str(i)] = np.zeros(parameters["W" + str(i)].shape)
            s["db" + str(i)] = np.zeros(parameters["b" + str(i)].shape)
        return v, s

    def adam_optimizer_update(self, v, s, grads, learning_rate, parameters, t = 2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        L = len(parameters) // 2
        s_corrected = {}
        v_corrected = {}

        for i in range(1, L + 1):
            v["dW" + str(i)] = beta1 * v["dW" + str(i)] + (1 - beta1) * grads["dW" + str(i)]
            v["db" + str(i)] = beta1 * v["db" + str(i)] + (1 - beta1) * grads["db" + str(i)]

            v_corrected["dW" + str(i)] = v["dW" + str(i)] / (1 - beta1 ** 2)
            v_corrected["db" + str(i)] = v["db" + str(i)] / (1 - beta1 ** 2)

            s["dW" + str(i)] = beta2 * s["dW" + str(i)] + (1 - beta2) * (grads["dW" + str(i)] ** 2)
            s["db" + str(i)] = beta2 * s["db" + str(i)] + (1 - beta2) * (grads["db" + str(i)] ** 2)

            s_corrected["dW" + str(i)] = s["dW" + str(i)] / (1 - beta2 ** 2)
            s_corrected["db" + str(i)] = s["db" + str(i)] / (1 - beta2 ** 2)

            parameters["W" + str(i)] -= learning_rate * (v_corrected["dW" + str(i)]) / (np.sqrt(s_corrected["dW" + str(i)]) + epsilon)
            parameters["b" + str(i)] -= learning_rate * (v_corrected["db" + str(i)]) / (np.sqrt(s_corrected["db" + str(i)]) + epsilon)

        return parameters

    def gradient_descent_update(self, grads, parameters, learning_rate):
        
        #Updates parameters according to corresponding gradients and learning rate

        L = len(parameters) // 2
        for i in range(1, L):
            parameters["W" + str(i)] -= learning_rate * grads["dW" + str(i)]
            parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]
        return parameters

    def cost(self, AL, Y, parameters):

        #Calculates cost using cross entropy loss

        no_of_layers = len(parameters) // 2
        n = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / n
        L2_sum = 0
        for i in range(1, no_of_layers):
            L2_sum += np.sum(np.square(parameters["W" + str(i)]))
        L2_cost = (0.1 / n) * L2_sum
        cost += L2_cost
        cost = np.squeeze(cost)
        return cost

    def random_minibatches(self, X, Y, mini_batch_size):
        mini_batches = []
        n = X.shape[1]
        permutations = list(np.random.permutation(n))
        no_of_complete_batches = n // mini_batch_size
        X_shuffle = X[:, permutations]
        Y_shuffle = Y[:, permutations]
        for i in range(no_of_complete_batches):
            mini_batch_X = X_shuffle[:, mini_batch_size * i : mini_batch_size * (i + 1)]
            mini_batch_Y = Y_shuffle[:, mini_batch_size * i : mini_batch_size * (i + 1)]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        if n % mini_batch_size != 0:
            mini_batch_X = X_shuffle[:, mini_batch_size * no_of_complete_batches : n]
            mini_batch_Y = Y_shuffle[:, mini_batch_size * no_of_complete_batches : n]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        return mini_batches

