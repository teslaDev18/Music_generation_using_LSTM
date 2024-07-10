# Music Generation with LSTM

## Project Overview
This project explores the fascinating intersection of music and machine learning by generating music using a Long Short-Term Memory (LSTM) network, a specific architecture of Recurrent Neural Networks (RNNs). By analyzing sequences of musical notes, the model learns to predict and generate subsequent notes, creating new musical pieces.

## Key Features
- **Music Prediction and Generation**: Generates music by predicting the sequence of notes that follow a given set of notes.
- **LSTM Architecture**: Utilizes the LSTM architecture to capture the temporal dependencies of musical notes in sequences.
- **Mozart Piano MIDI Dataset**: Trained on a comprehensive dataset of Mozart's piano compositions, sourced from [Piano MIDI](http://www.piano-midi.de/mozart.htm).

## Getting Started
To explore this project visit the deployed model at https://lstmmusic.streamlit.app/.

## Technology used
- **pretty_midi**: A Python library essential for processing MIDI files. It allows for easy manipulation and analysis of musical data, facilitating the model's training on the Mozart Piano MIDI dataset.
- **Pytorch and Keras**: These libraries form the backbone of the model, with TensorFlow providing a comprehensive ecosystem for machine learning and Keras offering a high-level API for neural network construction.
- **Streamlit**: Powers the web application that showcases the LSTM model, enabling users to interact with the model and generate music in a user-friendly environment.

## Data Collection and Preprocessing

### Dataset
- **Source**: The data is collected from Mozart's piano compositions, available at [Piano MIDI](http://www.piano-midi.de/mozart.htm).
- **Content**: The dataset focuses exclusively on piano compositions to ensure the model trains on relevant data. It includes notes from each major composition.

### Preprocessing
The preprocessing stage involves parsing MIDI files to extract musical notes and their properties using the `pretty_midi` library. This step is crucial for understanding the musical data and preparing it for the model.

#### Notable Properties of Notes
The MIDI files provide detailed information about each note, including:
- **Pitch**: Frequency of the note, represented as a number (0-128).
- **Velocity**: How hard the note is played.
- **Start Time**: When the note begins.
- **End Time**: When the note ends.
- **Channel**: MIDI channel used.
- **Instrument**: Instrument sound for the note.
- **Key Pressure**: Pressure sensitivity of the note.

#### Feature Extraction
For the model, the focus is on extracting three key properties:
- **Pitch**: Converted to a one-hot encoded vector for model input.
- **Step**: Time difference between consecutive notes.
- **Duration**: Length of time the note is played.

#### Note Construction
- Each note is represented by a combination of its **Pitch** (one-hot encoded), **Step**, and **Duration**. This composite representation forms the basis for our training data.

#### Sequence Generation
- Sequences of 50 consecutive notes is created as input to the model, with the immediate next note serving as the output target. This approach helps the model learn the structure and progression of musical compositions.

## Model Architecture

The `MusicGenerator` model is designed to generate music by predicting the next note in a sequence, given a series of notes. It is built using PyTorch and consists of several key components:

### LSTM Layers
- The model utilizes Long Short-Term Memory (LSTM) layers to process sequences of notes. This allows it to capture temporal dependencies and patterns in music.
- It is configured with a single LSTM layer (`lstm1`), which takes the input size, hidden layer size, and number of layers as parameters. The LSTM layer is designed to be unidirectional and supports dropout to prevent overfitting.

### Fully Connected Layers
- Following the LSTM layer, the model includes three fully connected (linear) layers (`fcP`, `fcS`, `fcD`) that map the LSTM output to the desired output sizes. These layers are responsible for predicting the pitch, step, and duration of the next note, respectively.
- Each of these layers is followed by a Rectified Linear Unit (ReLU) activation function to introduce non-linearity.

### Output
- The pitch predictions are passed through a softmax layer (`softmax_pitch`) to obtain a probability distribution over all possible pitches.
- The model outputs a concatenated tensor consisting of the pitch probabilities, step, and duration predictions for the next note.

### Device Compatibility
- The model is designed to be compatible with both CUDA-enabled GPUs and CPUs, allowing for flexible deployment based on the available hardware.

### Optimizer
- The model uses the Adam optimizer for adjusting the weights during training. Adam is chosen for its adaptive learning rate properties, which help in handling the sparse gradients and varying data scales in music generation tasks.

### Loss Function
- For pitch prediction, the model employs CrossEntropyLoss, which is suitable for classification tasks with multiple classes.
- For step and duration predictions, Mean Squared Error (MSE) Loss is used, catering to the regression nature of these outputs.
<p align="center">
  <img src="https://github.com/teslaDev18/Music_generation_using_LSTM/assets/134082150/9048a512-877e-481c-a380-9fc2b71c0d8f" alt="image">
</p>

## Results

The pitch accuracy obtained was upto 90%.

### Original music

https://github.com/teslaDev18/Music_generation_using_LSTM/assets/134082150/040b257f-a65c-4c71-a669-ed4136a23451

### Generated Music

https://github.com/teslaDev18/Music_generation_using_LSTM/assets/134082150/556d2e03-7b54-45f9-954a-a0b7749116b8

## Team:
This project was made by:
- [D.Hayagrivan](https://github.com/hikey-dj)
- [Kushal Gupta](https://github.com/teslaDev18)



