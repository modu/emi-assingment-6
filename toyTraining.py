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
    inputData = np.array(  [  [1.0,0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0],[0.5,0.5],[-0.5,0.5],[0.5,-0.5],[-0.5,-0.5] ] )
    outputData = np.array( [ [1,1,1,1,0,0,0,0] ] ).T
    # outputData = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])
    np.random.seed(1)

    #intial Value of weights a 2*1 matrix
    synisodalFunction = np.random.random((2,8))
    synisodalFunction2 = np.random.random((8,1))

    for iterationNumber in xrange(10000):
        layer0 = inputData
        layer1 = nonlin(np.dot(layer0,synisodalFunction))
        layer2 = nonlin(np.dot(layer1, synisodalFunction2))

        layer2_error = outputData - layer2

        # if (iterationNumber%100 == 0):
        #     print "Error: " + str(np.mean(np.abs(layer2_error)))
        #     print "synisodalFunction: " + str(synisodalFunction)
        #     print "synisodalFunction2:" + str(synisodalFunction2)


        layer2_delta = layer2_error * nonlin(layer2, derivative=True)


        if iterationNumber == 1:
            print layer1.shape
            print layer2.shape
            print layer2_error.shape
            print layer2_delta.shape

        layer1_error = layer2_delta.dot(synisodalFunction2.T)

        layer1_delta = layer1_error * nonlin( layer1 , True)

        synisodalFunction += np.dot(layer0.T,layer1_delta)

        synisodalFunction2 += np.dot(layer1.T, layer2_delta)


    # print "Output After Training"
    print layer1
    # print synisodalFunction
    print layer2
if __name__ == '__main__':
    main()