# LSTM-Based Next Word Prediction Using PyTorch

## 📌 Project Overview

This project implements a next-word prediction model using Long Short-Term Memory (LSTM), a type of Recurrent Neural Network (RNN), with PyTorch.

The model learns word sequences from a sample English text called *The Last Train Journey*. After training, it predicts the next word based on the input text.

This project demonstrates the fundamental concepts of Natural Language Processing (NLP), text preprocessing, sequence preparation, and deep learning with LSTM.

## 🎯 Objectives

- Understand text preprocessing for NLP tasks.
- Tokenize text into individual words.
- Build a vocabulary and convert words into numerical indices.
- Generate training sequences for next-word prediction.
- Apply padding to make sequences equal in length.
- Create a custom PyTorch Dataset and DataLoader.
- Build and train an LSTM-based neural network.
- Predict the next word from a given text input.
- Calculate model accuracy.

## 🛠️ Technologies and Libraries

- Python
- PyTorch
- NumPy
- NLTK
- Jupyter Notebook / JupyterLab
- Anaconda

## 📂 Project Structure

```text
LSTM-Next-Word-Prediction/
│
├── LSTM.ipynb
└── README.md
```

## ⚙️ Project Workflow

### 1. Text Dataset

A sample English story, *The Last Train Journey*, is used as the training text.

### 2. Text Tokenization

NLTK's `word_tokenize()` function splits the text into individual tokens. The text is converted to lowercase before tokenization.

### 3. Vocabulary Creation

A vocabulary is constructed from the tokens. Each word is assigned a unique numerical index, and the `<unk>` token is reserved for unknown words.

### 4. Numerical Encoding

The tokenized words are converted into numerical sequences using the vocabulary.

### 5. Training Sequence Preparation

Input sequences are generated from the text so that the model can learn to predict the next word from the preceding words.

### 6. Padding and Tensor Conversion

Sequences are padded with zeros to obtain a consistent length. The resulting sequences are converted into PyTorch tensors.

### 7. Dataset and DataLoader

A custom PyTorch `Dataset` class is implemented to manage the input sequences and their target words. A `DataLoader` creates batches with a batch size of 32 and shuffles the training samples.

### 8. LSTM Model Architecture

The model consists of three main layers:

- **Embedding Layer:** Converts vocabulary indices into dense vectors of size 100.
- **LSTM Layer:** Processes the embedded sequences with a hidden size of 150.
- **Fully Connected Layer:** Maps the final hidden representation to vocabulary-sized output scores.

The model uses `batch_first=True` in the LSTM layer.

### 9. Model Training

The model is trained using the following configuration:

| Parameter | Value |
|---|---|
| Embedding Dimension | 100 |
| LSTM Hidden Size | 150 |
| Batch Size | 32 |
| Number of Epochs | 50 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |

During training, the model performs forward propagation, calculates the loss, computes gradients using backpropagation, and updates the parameters using the Adam optimizer.

### 10. Next-Word Prediction

A prediction function processes an input text, converts its tokens into vocabulary indices, pads the sequence, and passes it through the trained model.

The word with the highest output score is selected as the predicted next word.

### 11. Accuracy Calculation

A custom accuracy function compares the model's predicted word indices with the target word indices and calculates the percentage of correct predictions on the DataLoader samples.

## 🚀 How to Run the Project

### Requirements

Make sure Python, JupyterLab, PyTorch, NumPy, and NLTK are installed.

Install the required libraries if necessary:

```bash
pip install torch numpy nltk jupyterlab
```

### Run Using JupyterLab

1. Open Anaconda Navigator.
2. Launch JupyterLab.
3. Navigate to the project folder.
4. Open `LSTM.ipynb`.
5. Run the notebook cells in order, starting from the first cell.
6. Observe the training loss, next-word prediction, and calculated accuracy.

The notebook also downloads the NLTK tokenizer resources required by the code.

## 📊 Results

The notebook displays the training loss for each epoch, generates a next-word prediction for an example input, and calculates accuracy using the implemented accuracy function.

The actual training loss, predicted word, and accuracy depend on running the notebook. No fixed performance score is claimed here.

## 📚 Learning Outcomes

Through this project, I practiced:

- Text tokenization and vocabulary construction.
- Converting text into numerical sequences.
- Preparing sequential training data.
- Implementing custom Dataset and DataLoader classes.
- Building an LSTM model using PyTorch.
- Understanding embeddings, hidden states, and output layers.
- Training a neural network with backpropagation and the Adam optimizer.
- Implementing next-word prediction and accuracy calculation.

## 🔮 Future Improvements

- Train the model on a larger text dataset.
- Improve text preprocessing and unknown-word handling.
- Generate multiple consecutive words instead of predicting only one next word.
- Evaluate performance on a separate validation or test dataset.
- Experiment with different embedding dimensions and LSTM hidden sizes.
- Save and load the trained model for future predictions.

## 👨‍💻 Author

Developed as a hands-on learning project to practice NLP and sequence modeling using PyTorch and LSTM.
