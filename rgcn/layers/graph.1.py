'''
Michael Cochcez 31 August 2018

An implementation of a Graph convolution layer able to deal with batches
The adjecancy matrix is inserted during onstruction time and is a simple 2D matrix

This implementation is made to work with Keras 2.0
'''

from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer

from keras.layers import Flatten, Reshape

import keras.backend as K


class GraphConvolution(Layer):
    def __init__(self, output_dim, adjecancies,
                 init='glorot_uniform', 
                 weights=None, W_regularizer=None, 
                 b_regularizer=None, bias=False,  **kwargs):
        """
        The original implementation had a num_bases=-1 argument. As I am not sure what it is used for, I left it out.

        Args:
            output_dim: The dimension to be used for the output vectors
            adjecancies: The adjecencies in the graph. A list of two dimensional arrays.
            Each list represents one relation. Iterating over the first dimension yields pairs (A,B) meaning there  is an arrow from A to B in the graph

        Returns:
            The return value. True for success, False otherwise.

        """
        
        self.init = initializers.get(init)
        self.output_dim = output_dim  # number of features per node
        self.adjecancies = adjecancies

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights

        # these will be defined during build()
        self.input_dim = None
        self.W = None
        self.W_comp = None
        self.b = None
        self.num_nodes = None

        super(GraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        print ("computing output shapes for %s" % str(input_shape))
        #TODO check whether the input_chape includes the batch size
        assert len(input_shape) == 3
        # input_shape = input_shapes[0] 
        batch_size = input_shape[0]
        number_of_nodes_in_graph = input_shape[1]
        output_shape = (batch_size, number_of_nodes_in_graph, self.output_dim)
        return output_shape  # (batch_size, nodes, output_dim)

    def build(self, input_shape):
        #input shape = (None - batch size, nodes, input_dim )
        print("building for ", input_shape)
        
        assert len(input_shape) == 3
        self.num_nodes = input_shape[1]
        self.input_dim = input_shape[2]
        # there was code for bases supprt here. Removed it till functionality is clear
       
        #for each relation type there is an own weight matrix
        self.W = [self.add_weight((self.input_dim, self.output_dim),
                                                     initializer=self.init,
                                                     name='{}_W_{}'.format(self.name, i),
                                                     regularizer=self.W_regularizer) for (i, _) in enumerate(self.adjecancies)]

        if self.bias:
             self.b = self.add_weight((self.output_dim,),
                                      initializer='zero',
                                      name='{}_b'.format(self.name),
                                      regularizer=self.b_regularizer)

        if self.initial_weights is not None:
             self.set_weights(self.initial_weights)
             del self.initial_weights

    def call(self, inputs, mask=None):
        print("Call called with input ", inputs)
        inputs = K.print_tensor(inputs)
        
        #input_shape = (None - batch size, nodes, input_dim )
        #output shape=(None - batch size, nodes, output_dim)
        
        # Placeholders to get dimensions correct
        # does not work, slice not implemented Theano backend result = K.slice(K.zeros_like(inputs), (0, 0, 0), (-1, -1, self.output_dim) )
        # randomizer = K.random_uniform_variable(shape=(self.input_dim, self.output_dim), low=0, high=1)
        # result = K.dot(inputs, randomizer)
        # return result
        

        #list with an item for each node. Each item is a list of tensors which need to be summed to get the output for that node. The final output is the concatenation of these sums.
        out_parts = [[]] * self.num_nodes

        #apply weights on links
        for (relationIndex, relAdj) in enumerate(self.adjecancies):
            relationWeight = self.W[relationIndex]
            for (source, dest) in relAdj:
                part = K.dot(inputs[:,source], relationWeight)
                out_parts[dest].append(part)
        #TODO apply weights for self loops
        #TODO apply bias



        #out_summed = [sum(nodePart) for nodePart in out_parts]

        out_summed = []
        for nodePart in out_parts:
            # TODO fix what happens when nothing is there.
            theSum = nodePart[0]
            # TODO re-enable
            #for nodePartPart in nodePart[1:]:
            #    theSum = theSum + nodePartPart
            out_summed.append(theSum)

        out = K.stack(out_summed)
        K.print_tensor(out)
        return out


    # Part of old code:    
        # features = inputs[0]
        # A = inputs[1:]  # list of basis functions

        # # convolve
        # supports = list()
        # for i in range(len(self.adjecancies)):
        #     supports.append(K.dot(A[i], features))
        # supports = K.concatenate(supports, axis=1)

        # output = K.dot(supports, self.W)

        # if self.bias:
        #     output += self.b
        # return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'num_bases': self.num_bases,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers import Reshape, Dense


    number_of_nodes_in_graph = 5
    #adjecancies = [[(1,2)], [], [(2,3), (3,4)]]
    #adjecancies = [[(1,2)], [(1, 2)], [(2,3), (3,4)]]
    adjecancies = [[(1,2)]]

    input_feature_dim = 11
    output_feature_dim = 7

    gc = GraphConvolution(output_dim = output_feature_dim, adjecancies = adjecancies)

    
    model = Sequential([
        gc,
        Reshape((number_of_nodes_in_graph*output_feature_dim, ))
       # Reshape((55,1)),
       # Dense(20)
    ])

    
    model.compile(optimizer='adagrad',
              loss='mean_squared_error',
              metrics=['accuracy'])


    #feed random input features
    import numpy as np
    samples = 13
    X = np.random.random((samples, number_of_nodes_in_graph, input_feature_dim))
    Y = np.random.randint(2, size=(samples, number_of_nodes_in_graph * output_feature_dim))

    # Train the model, iterating on the data in batches of 3 samples
    model.fit(X, Y, epochs=1, batch_size=3)
    

    model.summary()
    


