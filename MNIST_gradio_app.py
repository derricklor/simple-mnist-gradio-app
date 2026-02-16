import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import gradio as gr
import cv2

# Load and normalize the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Scale pixels to 0-1

MODEL_PATH = 'mnist_model.keras'

if os.path.exists(MODEL_PATH):
    # Load the pre-trained model
    model = load_model(MODEL_PATH)
    print("Model loaded successfully from disk.")
else:
    # Insert your training code here from the previous step
    # Build the model
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),    # Turn 2D image into 1D vector
        layers.Dense(128, activation='relu'),    # Hidden layer with 128 neurons
        layers.Dropout(0.2),                     # Prevent overfitting
        layers.Dense(10, activation='softmax')   # Output layer (10 digits)
    ])

    # Compile and train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.save(MODEL_PATH)
    print("Model trained and saved.")

def about_img(img):
    #img is dictionary of background, composite, and layers
    print("Background shape:", img['background'].shape)
    print("Composite shape:", img['composite'].shape)
    print("Layers:", img['layers'])
    print("Layers 0:", img['layers'][0])
    print("Layers 0 shape:", img['layers'][0].shape)


def predict_digit(data):
    # 1. Access the layers list
    # layers[0] gives us the first layer, which is the sketchpad drawing.
    layer_data = data['layers'][0]
    
    # 2. Extract the Alpha channel (index 3 of RGBA)
    # This ensures we only get the 'ink' from the sketchpad
    # layers[0] has 3 dims: rows, columns, and RGBA.(28x28x4)
    # keep only the rows and columns with the alpha channel, which is the 4th channel (index 3)
    alpha_channel = layer_data[:, :, 3]
    
    # 3. Normalize to 0.0 - 1.0 range
    normalized_img = alpha_channel.astype('float32') / 255.0
    
    # 5. Reshape to (1, 28, 28) for the model
    final_input = normalized_img.reshape(1, 28, 28)
    
    # 6. Predict
    prediction = model.predict(final_input)
    
    # Return a dictionary of labels and confidence scores
    return {str(i): float(prediction[0][i]) for i in range(10)}


# Define the interface
interface = gr.Interface(
    fn=predict_digit, 
    inputs=gr.Sketchpad(canvas_size=(28, 28),
                        brush = gr.Brush(default_size=1, colors=["#000000"]), 
                        type="numpy"), 
    outputs=gr.Label(num_top_classes=10),
    title="Digit Recognizer",
    description="Draw a number (0-9) and see the model predict it in real-time!"
)

interface.launch()