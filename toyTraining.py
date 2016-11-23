import numpy as np
import math
#logistic Sigmoidal NonLinear function
def nonlin(x, derivative=False):
    if(derivative==True):
        y = 1/(1 + np.exp(-x))
        return y*(1-y)
        # return 1.0 - x**2
    return 1/(1 + np.exp(-x))
    # return math.tanh(x)

def main():
    print " In the main program "
    inputNumber = 2
    hiddenNumber = 4
    outpuNumber = 1
    learningRate = 1
    numberOfIterations = 10000
    inputData = np.array(  [  [1.0,0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0],[0.5,0.5],[-0.5,0.5],[0.5,-0.5],[-0.5,-0.5] ] )
    outputData = np.array( [ [1,1,1,1,0,0,0,0] ] ).T
    np.random.seed(1)

    #intial Value of weights a 2*1 matrix
    # weight = 2*np.random.random((2,4)) -1
    #
    # weight2 = 2*np.random.random((4,1)) - 1
    # weight = np.random.randn(2,4)
    # weight2 = np.random.randn(4,1)
    input_range = 1.0 / inputNumber** (1 / 2)
    output_range = 1.0 / hiddenNumber ** (1 / 2)

    #weights for layer 1(hidden Layer) and layer 2(output layer)
    weight = np.random.normal(loc=0, scale=input_range, size=(inputNumber, hiddenNumber))
    weight2 = np.random.normal(loc=0, scale=input_range, size=(hiddenNumber, outpuNumber))

    #Baise is necessary
    baise = np.random.normal(loc=0, scale=input_range, size=(8, hiddenNumber))
    baise2= np.random.normal(loc=0, scale=input_range, size=(8,outpuNumber))

    for iterationNumber in xrange(numberOfIterations):

        layer0 = inputData
        netLayer1 =  np.add(np.dot(layer0, weight ), baise)
        layer1 = nonlin(netLayer1)

        netLayer2 = np.add( np.dot(layer1, weight2), baise2)
        layer2 = nonlin(netLayer2)

        layer2_error = outputData - layer2

        #Propogating error backword with gradient
        layer2_delta = layer2_error * nonlin(layer2, derivative=True)

        layer1_error = layer2_delta.dot(weight2.T)

        layer1_delta = layer1_error * nonlin( layer1 , derivative=True)

        weight += (learningRate)*np.dot(layer0.T,layer1_delta)
        weight2 += (learningRate)*np.dot(layer1.T, layer2_delta)

        baise +=  (learningRate)*layer1_delta
        baise2 += (learningRate)*layer2_delta

    # print "Output After Training"
    # print layer1
    # print weight
    print layer2

if __name__ == '__main__':
    main()

