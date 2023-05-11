import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib 
import seaborn as sns

# Load the MNIST dataset
X, Y = fetch_openml('mnist_784',parser='auto', version=1, return_X_y=True, as_frame=False)

# TODO: display some input images
# Reshape one image from the dataset and display it
# img_idx = 20  # choose the first image in the dataset
# img = X[img_idx].reshape(28, 28)
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()

# Save the image as a PNG file
# filename = "mnist_image_{}.png".format(img_idx)
# plt.imsave(filename, img, cmap='gray')
# Scale the data and split it into train/test sets
X = X / 255.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = Y[:60000], Y[60000:]

# Create an MLP classifier with 2 hidden layers of 100 neurons each
mlp_mnist = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=70, alpha=1e-4,
                          solver='sgd', verbose=10, tol=1e-6, random_state=1,
                          learning_rate_init=.1)

# Train the classifier on the training data
mlp_mnist.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = mlp_mnist.score(X_test, y_test)
print("Test set accuracy: {:.5f}".format(accuracy))

# Save the trained model to disk
filename = "NN_model.sav"
joblib.dump(mlp_mnist, filename)

y_train_pred = mlp_mnist.predict(X_train)
y_test_pred = mlp_mnist.predict(X_test)

# Compute the confusion matrices for the training and test datasets
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

print("Training Set Confusion Matrix: \n")
print(train_cm)
print("Test Set Confusion Matrix: \n")
print(test_cm)
# sns.heatmap(train_cm, annot=True, cmap='Blues', ax=ax1)
# sns.heatmap(test_cm, annot=True, cmap='Blues', ax=ax2)
# ax1.set_title('')
# ax2.set_title('Test Set Confusion Matrix')
# plt.show()