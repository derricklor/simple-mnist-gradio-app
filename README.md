# MNIST Digit Recognizer with Gradio

This project implements a simple handwritten digit recognition application using a Keras deep learning model and a Gradio web interface. Users can draw a digit (0-9) on a canvas, and the application will predict the digit in real-time.

## Description

The core of this application is a Sequential Neural Network built with TensorFlow/Keras, trained on the MNIST dataset. The model identifies handwritten digits from images. A user-friendly web interface is provided via Gradio, allowing real-time interaction where users can draw digits directly in their web browser. If a pre-trained model (`mnist_model.keras`) is not found, the application will train a new model upon first run.

## Installation

To set up and run this project, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/derricklor/simple-mnist-gradio-app.git
    cd simple-mnist-gradio-app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install tensorflow numpy gradio opencv-python matplotlib
    ```
    
    *(Note: `opencv-python` is for `cv2`, `matplotlib` for `vis.py`)*

## How to Use

1.  **Run the application:**
    After installing the dependencies and activating your virtual environment, run the main application script:
    ```bash
    python MNIST_gradio_app.py
    ```

2.  **Access the web interface:**
    Once the application starts, it will provide a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser.

3.  **Draw a digit:**
    Use the sketchpad in the Gradio interface to draw a digit (0-9). As you draw, the model will provide real-time predictions of what digit it thinks you have drawn.

4.  **Visualize MNIST dataset (optional):**
    You can also run the `vis.py` script to see examples from the MNIST dataset:
    ```bash
    python vis.py
    ```
    This will display a grid of 25 sample images from the MNIST training set along with their labels.
