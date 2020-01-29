from builtins import range
from builtins import object
import numpy as np
import copy

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mu, sigma = 0, weight_scale
        W1 = np.random.normal(mu, sigma, (input_dim,hidden_dim))
        b1 = np.zeros((1,hidden_dim))
        W2 = np.random.normal(mu, sigma, (hidden_dim,num_classes))
        b2 = np.zeros((1,num_classes))
        self.params['W1']=W1
        self.params['W2']=W2
        self.params['b1']=b1
        self.params['b2']=b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        hidden, hidden_cache=affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores,scores_cache=affine_forward(hidden,self.params['W2'], self.params['b2'])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout=softmax_loss(scores,y)
        #add L2 regularisation to the loss
        loss+=0.5*self.reg*np.sum(self.params['W1']*self.params['W1']) + 0.5*self.reg*np.sum(self.params['W2']*self.params['W2']) 
        dhidden,grads['W2'],grads['b2']=affine_backward(dout,scores_cache)
        #add L2 regularisation to the gradient
        grads['W2']+=self.reg*self.params['W2']
        dx,grads['W1'],grads['b1']=affine_relu_backward(dhidden, hidden_cache)
        grads['W1']+=self.reg*self.params['W1']


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #found the bug here where copied list only points back to the original list
        dim_list=copy.deepcopy(hidden_dims)
        dim_list.insert(0, input_dim)
        dim_list.append(num_classes)

        mu, sigma = 0, weight_scale

        for i in range(self.num_layers):
            self.params[f'W{i+1}']=np.random.normal(mu, sigma, (dim_list[i],dim_list[i+1]))
            self.params[f'b{i+1}']=np.zeros((1,dim_list[i+1]))
            if i<self.num_layers-1 and (self.normalization=='batchnorm' or self.normalization=='layernorm'):
                #only the layers up to the penultimate layer has these two parameters
                self.params[f'gamma{i+1}']=np.ones((1,dim_list[i+1]))
                self.params[f'beta{i+1}']=np.zeros((1,dim_list[i+1]))

        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        input_matrix=X
        cache_dict={}
        #loop through all the hidden layers with affine_relu_forward       
        for i in range(self.num_layers-1):
            if self.normalization=='batchnorm':
                hidden, hidden_cache=affine_forward(input_matrix, self.params[f'W{i+1}'], self.params[f'b{i+1}'])
                hidden_norm,cache_norm=batchnorm_forward(hidden,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'],self.bn_params[i])
                hidden_final,relu_cache=relu_forward(hidden_norm)
                cache_dict[str(i+1)]=(hidden_cache,cache_norm,relu_cache)
                input_matrix=hidden_final
            elif self.normalization=='layernorm':
                hidden, hidden_cache=affine_forward(input_matrix, self.params[f'W{i+1}'], self.params[f'b{i+1}'])
                hidden_norm,cache_norm=layernorm_forward(hidden,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'],self.bn_params[i])
                hidden_final,relu_cache=relu_forward(hidden_norm)
                cache_dict[str(i+1)]=(hidden_cache,cache_norm,relu_cache)
                input_matrix=hidden_final
            else:
                hidden_final, hidden_cache=affine_relu_forward(input_matrix, self.params[f'W{i+1}'], self.params[f'b{i+1}'])
                cache_dict[str(i+1)]=hidden_cache
                input_matrix=hidden_final

            if self.use_dropout:
                hidden_final,drop_cache=dropout_forward(input_matrix,self.dropout_param)
                input_matrix=hidden_final
                cache_dict['drop'+str(i+1)]=drop_cache

        #last layer
        scores,scores_cache=affine_forward(hidden_final,self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout=softmax_loss(scores,y)

        #backprop the gradients through the penultimate layer:
        dhidden,grads[f'W{self.num_layers}'],grads[f'b{self.num_layers}']=affine_backward(dout,scores_cache)
        loss+=0.5*self.reg*np.sum(self.params[f'W{self.num_layers}']*self.params[f'W{self.num_layers}'])
        grads[f'W{self.num_layers}']+=self.reg*self.params[f'W{self.num_layers}']

        #backprop the gradients through all hidden layers with ReLU non-linearities
        #reverse the lists of storage for backprop
        for i in reversed(range(self.num_layers-1)):

            if self.use_dropout:
                drop_cache=cache_dict['drop'+str(i+1)]
                ddrop=dropout_backward(dhidden,drop_cache)
                dhidden=ddrop

            if self.normalization=='batchnorm':
                hidden_cache,cache_norm,relu_cache=cache_dict[str(i+1)]
                dx_relu=relu_backward(dhidden, relu_cache)
                dx_norm, grads[f'gamma{i+1}'], grads[f'beta{i+1}']=batchnorm_backward_alt(dx_relu, cache_norm)
                dx,grads[f'W{i+1}'],grads[f'b{i+1}']=affine_backward(dx_norm, hidden_cache)
                dhidden=dx
            elif self.normalization=='layernorm':
                hidden_cache,cache_norm,relu_cache=cache_dict[str(i+1)]
                dx_relu=relu_backward(dhidden, relu_cache)
                dx_norm, grads[f'gamma{i+1}'], grads[f'beta{i+1}']=layernorm_backward(dx_relu, cache_norm)
                dx,grads[f'W{i+1}'],grads[f'b{i+1}']=affine_backward(dx_norm, hidden_cache)
                dhidden=dx
            else:
                hidden_cache=cache_dict[str(i+1)]
                dx,grads[f'W{i+1}'],grads[f'b{i+1}']=affine_relu_backward(dhidden, hidden_cache)
                dhidden=dx



            #add regularisation
            loss+=0.5*self.reg*np.sum(self.params[f'W{i+1}']*self.params[f'W{i+1}'])
            grads[f'W{i+1}']+=self.reg*self.params[f'W{i+1}']


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

# class FullyConnectedNet(object):
#     """
#     A fully-connected neural network with an arbitrary number of hidden layers,
#     ReLU nonlinearities, and a softmax loss function. This will also implement
#     dropout and batch/layer normalization as options. For a network with L layers,
#     the architecture will be
#     {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
#     where batch/layer normalization and dropout are optional, and the {...} block is
#     repeated L - 1 times.
#     Similar to the TwoLayerNet above, learnable parameters are stored in the
#     self.params dictionary and will be learned using the Solver class.
#     """

#     def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
#                  dropout=1, normalization=None, reg=0.0,
#                  weight_scale=1e-2, dtype=np.float32, seed=None):
#         """
#         Initialize a new FullyConnectedNet.
#         Inputs:
#         - hidden_dims: A list of integers giving the size of each hidden layer.
#         - input_dim: An integer giving the size of the input.
#         - num_classes: An integer giving the number of classes to classify.
#         - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
#           the network should not use dropout at all.
#         - normalization: What type of normalization the network should use. Valid values
#           are "batchnorm", "layernorm", or None for no normalization (the default).
#         - reg: Scalar giving L2 regularization strength.
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - dtype: A numpy datatype object; all computations will be performed using
#           this datatype. float32 is faster but less accurate, so you should use
#           float64 for numeric gradient checking.
#         - seed: If not None, then pass this random seed to the dropout layers. This
#           will make the dropout layers deteriminstic so we can gradient check the
#           model.
#         """
#         self.normalization = normalization
#         self.use_dropout = dropout != 1
#         self.reg = reg
#         self.num_layers = 1 + len(hidden_dims)
#         self.dtype = dtype
#         self.params = {}

#         ############################################################################
#         # TODO: Initialize the parameters of the network, storing all values in    #
#         # the self.params dictionary. Store weights and biases for the first layer #
#         # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
#         # initialized from a normal distribution centered at 0 with standard       #
#         # deviation equal to weight_scale. Biases should be initialized to zero.   #
#         #                                                                          #
#         # When using batch normalization, store scale and shift parameters for the #
#         # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
#         # beta2, etc. Scale parameters should be initialized to ones and shift     #
#         # parameters should be initialized to zeros.                               #
#         ############################################################################
        
#         for i in range(self.num_layers):
#             W_i = 'W' + str(i+1)
#             b_i = 'b' + str(i+1)

#             # Output layer (without bath normalization)
#             if i == self.num_layers - 1:
#                 self.params[W_i] = np.random.randn(hidden_dims[len(hidden_dims)-1],
#                     num_classes) * weight_scale
#                 self.params[b_i] = np.zeros(num_classes)
#             # With batch normalization
#             else:
#                 # First hidden layer
#                 if i == 0:
#                     self.params[W_i] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
#                     self.params[b_i] = np.zeros(hidden_dims[0])
#                 # Intermediate hidden layer
#                 else:
#                     self.params[W_i] = np.random.randn(hidden_dims[i-1], hidden_dims[i]) * weight_scale
#                     self.params[b_i] = np.zeros(hidden_dims[i])

#                 # Batch or layer normalization layer
#                 if self.normalization in ['batchnorm', 'layernorm']:
#                     self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
#                     self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])
        
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################

#         # When using dropout we need to pass a dropout_param dictionary to each
#         # dropout layer so that the layer knows the dropout probability and the mode
#         # (train / test). You can pass the same dropout_param to each dropout layer.
#         self.dropout_param = {}
#         if self.use_dropout:
#             self.dropout_param = {'mode': 'train', 'p': dropout}
#             if seed is not None:
#                 self.dropout_param['seed'] = seed

#         # With batch normalization we need to keep track of running means and
#         # variances, so we need to pass a special bn_param object to each batch
#         # normalization layer. You should pass self.bn_params[0] to the forward pass
#         # of the first batch normalization layer, self.bn_params[1] to the forward
#         # pass of the second batch normalization layer, etc.
#         self.bn_params = []
#         if self.normalization=='batchnorm':
#             self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
#         if self.normalization=='layernorm':
#             # Modify `self.bn_params` to `self.lb_params`
#             self.ln_params = [{} for i in range(self.num_layers - 1)]

#         # Cast all parameters to the correct datatype
#         for k, v in self.params.items():
#             self.params[k] = v.astype(dtype)


#     def loss(self, X, y=None):
#         """
#         Compute loss and gradient for the fully-connected net.
#         Input / output: Same as TwoLayerNet above.
#         """
#         X = X.astype(self.dtype)
#         mode = 'test' if y is None else 'train'

#         # Set train/test mode for batchnorm params and dropout param since they
#         # behave differently during training and testing.
#         if self.use_dropout:
#             self.dropout_param['mode'] = mode
#         if self.normalization=='batchnorm':
#             for bn_param in self.bn_params:
#                 bn_param['mode'] = mode
#         scores = None
#         ############################################################################
#         # TODO: Implement the forward pass for the fully-connected net, computing  #
#         # the class scores for X and storing them in the scores variable.          #
#         #                                                                          #
#         # When using dropout, you'll need to pass self.dropout_param to each       #
#         # dropout forward pass.                                                    #
#         #                                                                          #
#         # When using batch normalization, you'll need to pass self.bn_params[0] to #
#         # the forward pass for the first batch normalization layer, pass           #
#         # self.bn_params[1] to the forward pass for the second batch normalization #
#         # layer, etc.                                                              #
#         ############################################################################
        
#         caches = {}

#         # Don't include the last layer (no relu, no batch normalization, just affine)
#         for i in range(self.num_layers-1):
#             W_i = 'W' + str(i+1)
#             b_i = 'b' + str(i+1)

#             # First hidden layer
#             if i == 0:
#                 # `out` denotes the output from last layer
#                 out = X

#             # With batch normalization
#             if self.normalization == 'batchnorm':
#                 # affine -> batch norm -> relu
#                 fc_out, fc_cache = affine_forward(out, self.params[W_i], self.params[b_i])
#                 bn_out, bn_cache = batchnorm_forward(fc_out, self.params['gamma'+str(i+1)],
#                     self.params['beta'+str(i+1)], self.bn_params[i])
#                 out, relu_cache = relu_forward(bn_out)
#                 caches[i+1] = (fc_cache, bn_cache, relu_cache)

#             # With layer normalization
#             elif self.normalization == 'layernorm':
#                 # affine -> layer norm -> relu
#                 fc_out, fc_cache = affine_forward(out, self.params[W_i], self.params[b_i])
#                 ln_out, ln_cache = layernorm_forward(fc_out, self.params['gamma'+str(i+1)],
#                     self.params['beta'+str(i+1)], self.ln_params[i])
#                 out, relu_cache = relu_forward(ln_out)
#                 caches[i+1] = (fc_cache, ln_cache, relu_cache)

#             # Without batch normalization and layer normalization
#             else:
#                 out, caches[i+1] = affine_relu_forward(out, self.params[W_i],
#                     self.params[b_i])

#             # Dropout layer after relu layer
#             if self.use_dropout:
#                 out, caches['dropout'+str(i+1)] = dropout_forward(out, self.dropout_param)

#         # The last layer
#         scores, caches[self.num_layers] = affine_forward(out,
#             self.params['W'+str(self.num_layers)],
#             self.params['b'+str(self.num_layers)])

#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################

#         # If test mode return early
#         if mode == 'test':
#             return scores

#         loss, grads = 0.0, {}
#         ############################################################################
#         # TODO: Implement the backward pass for the fully-connected net. Store the #
#         # loss in the loss variable and gradients in the grads dictionary. Compute #
#         # data loss using softmax, and make sure that grads[k] holds the gradients #
#         # for self.params[k]. Don't forget to add L2 regularization!               #
#         #                                                                          #
#         # When using batch/layer normalization, you don't need to regularize the scale   #
#         # and shift parameters.                                                    #
#         #                                                                          #
#         # NOTE: To ensure that your implementation matches ours and you pass the   #
#         # automated tests, make sure that your L2 regularization includes a factor #
#         # of 0.5 to simplify the expression for the gradient.                      #
#         ############################################################################
        
#         loss, dscores = softmax_loss(scores, y)

#         for i in range(self.num_layers, 0, -1):
#             # Loss with regularization
#             loss += 0.5 * self.reg * np.sum(np.square(self.params['W'+str(i)]))

#             # Last hidden layer (no batch normalization, no relu, no dropout)
#             if i == self.num_layers:
#                 # `dout` denotes gradient of output of the last layer
#                 dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dscores,
#                     caches[i])
#             else:
#                 # Dropout backward except the last layer (before relu backward)
#                 if self.use_dropout:
#                     dout = dropout_backward(dout, caches['dropout'+str(i)])

#                 if self.normalization == 'batchnorm':
#                     # drelu -> dbatchnorm -> daffine
#                     fc_cache, bn_cache, relu_cache = caches[i]
#                     dbn_out = relu_backward(dout, relu_cache)
#                     dfc_out, grads['gamma'+str(i)], grads['beta'+str(i)] = batchnorm_backward(dbn_out, bn_cache)
#                     dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dfc_out, fc_cache)
                
#                 elif self.normalization == 'layernorm':
#                     # drelu -> dlayernorm -> daffine
#                     fc_cache, ln_cache, relu_cache = caches[i]
#                     dln_out = relu_backward(dout, relu_cache)
#                     dfc_out, grads['gamma'+str(i)], grads['beta'+str(i)] = layernorm_backward(dln_out, ln_cache)
#                     dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dfc_out, fc_cache)

#                 else:
#                     dout, grads['W'+str(i)], grads['b'+str(i)] = affine_relu_backward(
#                         dout, caches[i])
            
#             # Regularization term
#             grads['W'+str(i)] += self.reg * self.params['W'+str(i)]

#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################

#         return loss, grads
