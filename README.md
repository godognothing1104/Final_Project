
```markdown
# Handwritten Digit Recognition with CNN and GUI

This project implements a **handwritten digit recognition system** using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset. The project also includes a user-friendly **GUI (Graphical User Interface)** for users to draw digits and get predictions in real-time.

## Features
- Trains a CNN model on the MNIST dataset to classify digits (0-9).
- GUI for drawing digits and predicting the digit with confidence.
- Displays a grayscale version of the drawn digit and a confidence table for each class.
- Visualizes training and validation accuracy/loss.

## Technologies Used
- **Python**: Programming language.
- **TensorFlow/Keras**: For building and training the CNN model.
- **Tkinter**: For GUI implementation.
- **Pillow**: For image manipulation and preprocessing.
- **Matplotlib**: For plotting graphs and grayscale images.
- **NumPy**: For numerical computations.

## Installation and Setup

### Prerequisites
Ensure you have Python installed along with the following libraries:
- `tensorflow`
- `keras`
- `numpy`
- `Pillow`
- `matplotlib`

Install them using pip:
```bash
pip install tensorflow keras numpy pillow matplotlib
```

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. Train the model (or use the pre-trained model provided):
   ```bash
   jupyter notebook main.ipynb
   ```
   This will train the CNN model and save it as `cnn_mnist_model.keras`.

3. Launch the GUI application:
   ```bash
   jupyter notebook main.ipynb
   ```
   The GUI window will launch for digit recognition.

## How It Works
1. **Model Training**:
    - Loads the MNIST dataset.
    - Normalizes images and reshapes them for CNN input.
    - Builds and trains a CNN with dropout layers for regularization.
    - Saves the trained model for future use.

2. **GUI Features**:
    - Users draw digits in a canvas area.
    - The drawn digit is resized, preprocessed, and passed through the trained model for prediction.
    - Displays the predicted digit, confidence score, and a grayscale representation of the input.

3. **Visualization**:
    - Plots training/validation accuracy and loss graphs after training.

## File Structure
```
|-- main.ipynb                             # Jupyter Notebook for training and GUI implementation
|-- cnn_mnist_model.keras                  # Saved pre-trained model
|-- requirements.txt                       # Python dependencies
|-- Real_Time_Handwritten_Digit_Recognition_Report  # Detailed project report
|-- sample_video.mp4                       # Sample result of the project
|-- Loss,Accuracy_of_Epochs.png            # Loss and Accuracy Plot of training epochs
|-- README.md                              # Project overview and setup instructions
```
## Result
* **Accuracy and Loss Training Plot:**
   ![Loss,Accuracy_of_Epochs.png](Loss%2CAccuracy_of_Epochs.png)
* **Sample use of the project:**
   [sample_video.mp4](sample_video.mp4)
## Report
For a detailed explanation of the implementation, methodologies, and results, refer to the [Real_Time_Handwritten_Digit_Recognition_Report.pdf](Real_Time_Handwritten_Digit_Recognition_Report.pdf) in the project directory.

## Future Improvements
- Add more datasets for training.
- Extend GUI to allow uploading images instead of drawing.
- Implement additional model architectures for improved accuracy.

## Acknowledgments
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)


