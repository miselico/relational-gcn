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

from keras.callbacks import TensorBoard

from keras.layers import Flatten, Reshape

import keras.backend as K
from itertools import accumulate

class GraphConvolution(Layer):
    def __init__(self, output_dim, adjecancies,
                 init='glorot_uniform',
                 weights=None, W_regularizer=None,
                 b_regularizer=None, bias=False, **kwargs):
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

        allDst = set()
        allSrc = set()
        for rel in adjecancies:
            for (src, dest) in rel:
                allSrc.add(src)
                allDst.add(dest)
        allIndices = allSrc.union(allDst)

        self.allSrc = allSrc
        self.allDst = allDst

        if len(allIndices) > 0 and min(allIndices) < 0:
            raise Exception("Index lower than 0 in adjecancies")
        self.maxIndexInAdjecencies = - \
            1 if len(allIndices) == 0 else max(allIndices)

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
        # TODO check whether the input_chape includes the batch size
        assert len(input_shape) == 3
        # input_shape = input_shapes[0]
        batch_size = input_shape[0]
        number_of_nodes_in_graph = input_shape[1]
        output_shape = (batch_size, number_of_nodes_in_graph, self.output_dim)
        return output_shape  # (batch_size, nodes, output_dim)

    def build(self, input_shape):
        # input shape = (None - batch size, nodes, input_dim )
        print("building for ", input_shape)

        assert len(input_shape) == 3
        self.num_nodes = input_shape[1]
        assert self.maxIndexInAdjecencies < self.num_nodes

        nodesWithNonZeroInDeg = len(self.allDst)
        assert nodesWithNonZeroInDeg <= self.num_nodes
        self.hasNodesWithZeroInDeg = not (
            nodesWithNonZeroInDeg == self.num_nodes)

        self.input_dim = input_shape[2]

        # there was code for bases supprt here. Removed it till functionality is clear

        # for each relation type there is an own weight matrix
        self.W = [self.add_weight((self.input_dim, self.output_dim),
                                  initializer=self.init,
                                  name='{}_W_{}'.format(self.name, i),
                                  regularizer=self.W_regularizer) for (i, _) in enumerate(self.adjecancies)]

        self.W_self = self.add_weight((self.input_dim, self.output_dim),
                                      initializer=self.init,
                                      name=self.name + '_selfweight',
                                      regularizer=self.W_regularizer)

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
        # inputs = K.print_tensor(inputs)

        # input_shape = (None - batch size, nodes, input_dim )
        # output shape=(None - batch size, nodes, output_dim)

        # Placeholders to get dimensions correct
        # does not work, slice not implemented Theano backend result = K.slice(K.zeros_like(inputs), (0, 0, 0), (-1, -1, self.output_dim) )
        # randomizer = K.random_uniform_variable(shape=(self.input_dim, self.output_dim), low=0, high=1)
        # result = K.dot(inputs, randomizer)
        # return result

        # list with an item for each node. Each item is a list of tensors which need to be summed to get the output for that node. The final output is the concatenation of these sums.

        out_parts = [list() for _ in range(self.num_nodes)]

        # make a collection of all input sourse slices so they get reused
        # this list contains None for nodes which have no outoging edges.
        # These would likely be pruned from the computation graph, though.
        inSlices = [
            inputs[:, i] if i in self.allSrc else None for i in range(self.num_nodes)]
        # inSlices = [inputs[:, i] for i in range(self.num_nodes)]

        print ("Made slices")

        # TODO investigate whether it is faster to slice and append more at the start to then in the end have less, but larger dot products and use backend.sum in a clever way to combine
        # apply weights on links
        for (relationIndex, relAdj) in enumerate(self.adjecancies):
            relationWeight = self.W[relationIndex]
            for (source, dest) in relAdj:
                part = K.dot(inSlices[source], relationWeight)
                out_parts[dest].append(part)

        print ("Created adjecancy network")
        # TODO apply bias

        if self.hasNodesWithZeroInDeg:
            # there are nodes with no in edge
            # TODO there is likely a better way to do achieve this, but can't figure it out
            zeroW = K.zeros((self.input_dim, self.output_dim))
            existingSlice = next(
                slice for slice in inSlices if slice is not None)
            zero_part = [K.dot(existingSlice, zeroW)]
            # TODO there might be some way to save here: many will be zero, hence adding larger blocks of zeroes might speed things up.
            out_parts = [zero_part if len(
                partList) == 0 else partList for partList in out_parts]

        print ("added zero parts for zero in degree nodes")

        def sumorsingle(aList):
            if len(aList) > 1:
                return sum(aList)
            else:
                return aList[0]
        out_summed = [sumorsingle(nodePart) for nodePart in out_parts]

        print ("Summed parts together")

        out_trough_adjecencies = self.stackOutTroughAjecency(out_summed)

        print ("Stacked parts")

        # apply weights for self loops

        out_self_loop = K.dot(inputs, self.W_self)
        print ("Computed self loop output")

        out = out_trough_adjecencies + out_self_loop

        # out_summed = []
        # for nodePart in out_parts:
        #     # TODO fix what happens when nothing is there.
        #     theSum = nodePart[0]
        #     for nodePartPart in nodePart:
        #         theSum = theSum + nodePartPart
        #     #theSum = K.print_tensor(theSum, message='thesum')
        #     out_summed.append(theSum)

        # TODO try whether performing the update add operations directly in the adjecancy look results in a more efficient or compact graph..

        # out = K.print_tensor(out, message='OUTPUT')
        print ("return from call()")
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

    def stackOutTroughAjecency(self, out_summed):
        # What this function tries to achieve:
        #out_trough_adjecencies = K.stack(out_summed, axis=1)

        #experiment with concatente: too slow
        #out_summed_reshaped = [K.reshape(part, (-1, 1, self.output_dim)) for part in out_summed]
        #out_trough_adjecencies = K.concatenate(out_summed_reshaped, axis=1)

        # stack in binary fashion -> works only for 2**n pieces, but reasonbly fast
        #out_trough_adjecencies = self._stackInPairs(out_summed, self.num_nodes)

        # the following did not work. the idea was to do the stacking manually. Unfortunately keras does not allow assignment to tensors
        # outshape = K.shape(inputs)
        # res = K.zeros(outshape)
        # # assign each of the results to res[:,i] = res_i
        # for i in range(self.num_nodes):
        #     res[:, i] = out_summed[i]

        # TODO idea: do first paiwaise stacking of 2^n ranges, then concatenate pieces
        partitionIndices = GraphConvolution._partitionDims(self.num_nodes)
        partitions = [ out_summed[start:end] for (start, end) in partitionIndices]
        stackedPartitions = [self._stackInPairsPow2( partition,  len(partition)) for partition in partitions]
        print ([K.int_shape(op) for op in stackedPartitions])
        stacked = K.concatenate(stackedPartitions, axis=1)
        return stacked


    def _stackInPairsPow2(self, out_summed, num_elements):
        assert num_elements != 0 and ((num_elements & (num_elements - 1)) == 0) # num_elements is a power of 2
        dims = [1]*num_elements
        result = self._stackInPairsPow2Rec(out_summed, dims)
        return result[0]

    #TODO this method can get rid of the dims.
    def _stackInPairsPow2Rec(self, out_summed, dims):
        assert len(dims) % 2 == 0
        print ("sinPairPow2 %d" % len(dims))
        
        stackedPairs = [K.stack([out_summed[i], out_summed[i+1]], axis=1)
                        for i in range(0, len(dims), 2)]

        dims = [dims[i] + dims[i+1] for i in range(0, len(dims), 2)]

        reshaped = [K.reshape(t,  (-1, dims[i], self.output_dim))
                    for (i, t) in enumerate(stackedPairs)]

        if len(dims) == 1:
            return reshaped[0]
        else:
            return self._stackInPairsPow2Rec(reshaped, dims)


    @staticmethod
    def _partitionDims(num_elements):
        '''returns 2**n sized partitions of the elements'''
        partitions = []
        num_elements_rest = num_elements
        while num_elements_rest != 0:
            partitions.append(num_elements_rest%2)
            num_elements_rest = num_elements_rest // 2
        counters = [2**index for (index, val) in enumerate(partitions) if val != 0]
        counters.reverse()
        assert num_elements == sum(counters)
        partition_ends = list(accumulate(counters))
        starts = [0]
        starts.extend(partition_ends[:-1])
        partitions = [(start, nextStart) for (start, nextStart) in  zip(starts,  partition_ends)]
        return partitions

    def _stackInPairs(self, out_summed, num_elements):
        dims = [1]*num_elements
        result = self._stackInPairsRec(out_summed, dims)
        return result[0]

    def _stackInPairsRec(self, out_summed, dims):
        print ("sinPair %d" % len(dims))
       # print ([K.int_shape(op) for op in out_summed])
        assert len(dims) == len(out_summed)
        if len(dims) == 1:
            return out_summed[0]
        if len(dims) % 2 == 0:
            return self._stackInPairsEven(out_summed, dims)
        else:
            return self._stackInPairsUnEven(out_summed, dims)

    def _stackInPairsEven(self, out_summed, dims):
        assert len(dims) % 2 == 0
        print ("sinPairE %d" % len(dims))
        #print ([K.int_shape(op) for op in out_summed])

        stackedPairs = [K.stack([out_summed[i], out_summed[i+1]], axis=1)
                        for i in range(0, len(dims), 2)]

        dims = [dims[i] + dims[i+1] for i in range(0, len(dims), 2)]

        reshaped = [K.reshape(t,  (-1, dims[i], self.output_dim))
                    for (i, t) in enumerate(stackedPairs)]

        return self._stackInPairsRec(reshaped, dims)

    def _stackInPairsUnEven(self, out_summed, dims):
        assert len(dims) % 2 == 1
        print ("sinPairUE %d" % len(dims))

        savedD = dims[-1]
        savedT = out_summed[-1]
        # making sure the dimension is compatible
        savedT = K.reshape(savedT, (-1, savedD, self.output_dim))

        restT = out_summed[:-1]
        restD = dims[:-1]
        restStacked = self._stackInPairsEven(restT, restD)
        restDim = sum(restD)

        new_out_summed = [restStacked, savedT]
        new_dims = [restDim, savedD]

        return self._stackInPairsEven(new_out_summed, new_dims)

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


def _createAdj(number_of_nodes_in_graph):
    #    numberOfRelations = 10000
    numberOfRelations = 2 * number_of_nodes_in_graph
    numberOfRelationTypes = 100

    adjecancies = []
    for _ in range(numberOfRelationTypes):
        rels = [((103079 * relationNumer) % number_of_nodes_in_graph, (101863 * relationNumer) %
                 number_of_nodes_in_graph) for relationNumer in range(numberOfRelations // numberOfRelationTypes)]
        adjecancies.append(rels)
    return adjecancies


if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers import Reshape, Dense

    #number_of_nodes_in_graph = 65536
    #number_of_nodes_in_graph = 1048576
    number_of_nodes_in_graph = 10000

    #adjecancies = []
    # adjecancies = [[(1,2)], [], [(2,3), (3,4)]]
    #adjecancies = [[(1, 2)], [(1, 2)], [(2, 3), (3, 4)], [(2, 3), (3, 4)]] * 50
    # adjecancies = [[(1, 2), (0, 0)]]
    # adjecancies = [[(1,2), (2, 3)]]
    adjecancies = [[(0, 1)]]
    #adjecancies = _createAdj(number_of_nodes_in_graph)

    input_feature_dim = 2
    internal_feature_dim = 5
    final_output_feature_dim = 3

    gc = GraphConvolution(output_dim=final_output_feature_dim,
                          adjecancies=adjecancies)

#    gcrepeat = GraphConvolution(
#        output_dim=internal_feature_dim, adjecancies=adjecancies)

#    gcfinal = GraphConvolution(output_dim=final_output_feature_dim, adjecancies=adjecancies)
    model = Sequential([
        gc,
        #       gcrepeat,
        #       gcrepeat,
        #       gcrepeat,
        #        gcfinal
    ])

    model.compile(optimizer='adagrad',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # feed random input features
    import numpy as np
    np.random.seed(0)
    samples = 100
    X = np.random.random(
        (samples, number_of_nodes_in_graph, input_feature_dim))
    Y = np.random.randint(
        2, size=(samples, number_of_nodes_in_graph, final_output_feature_dim))

    print("Number of nodes %d" % number_of_nodes_in_graph)
    print("Number of Relation types %d" % len(adjecancies))
    print("Number of Relations %d" % sum([len(rels) for rels in adjecancies]))
    print("Number of input features %d" % input_feature_dim)
    print("Number of output features %d" % final_output_feature_dim)
    print("Number of samples %d" % samples)

    tbcb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
    print ("Saving model to tensorboard")

    # Train the model, iterating on the data in batches of 3 samples
    model.fit(X, Y, epochs=20, batch_size=100, callbacks=[tbcb])

    model.summary()
