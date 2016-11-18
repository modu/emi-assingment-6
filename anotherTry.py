import numpy as np
from sklearn.neural_network import MLPClassifier
# X = [[0., 0.], [1., 1.]]
# inputData = np.array(
#     [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
# X = inputData
# X = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]
X = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]

# outputData = np.array([[1, 1, 1, 1, 0, 0, 0, 0]]).T
# y = outputData
y = [1, 1, 1, 1, 0, 0, 0, 0]

# y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

print(clf.fit(X, y))

print(clf.predict([[-0.6, -0.6], [-1., -2.]]))