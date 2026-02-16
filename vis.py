import matplotlib.pyplot as plt
import tensorflow as tf

# Load the dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # Display the image in binary (black and white)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(f"Label: {y_train[i]}")

plt.show()