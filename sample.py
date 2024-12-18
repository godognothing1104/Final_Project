import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot and save a sample image
plt.imshow(X_train[0], cmap='gray')  # Show the first training image
plt.title(f"Digit: {y_train[0]}")
plt.axis('off')
plt.savefig("mnist_sample.png", dpi=300)  # Save as 'mnist_sample.png'
plt.show()
