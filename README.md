# Implementation of LSTM in C++ using CUDA

This project contains an implementation of **LSTM (Long Short-Term Memory)** neural network in **C++** using **CUDA** to speed up calculations on **GPU**.

## Main files:

- *data.h/cpp* - class for scaling input data
- *model.h/cpp* - LSTM class of the model
- *main.cpp * - creating objects and launching training

## The *DataScaler* class performs input data preprocessing:

- Scaling to a specific range.

## The *LSTMModel* class contains an implementation of the LSTM network:

- The **CUDA** and **cuDNN** libraries are used for calculations on **GPU**:
1. Creating an LSTM descriptor with the necessary parameters
2. Memory allocation and initialization of weights
3. Methods **forwardPass** and **backwardPass** implement forward and reverse network traversal using **CUDA kernel** for calculations.

## In *main()* occurs:

- Data loading
- Creating class objects
- Model training and prediction

Thus, this code allows you to flexibly configure and train LSTM networks in **C++**, with acceleration on **GPU** using **CUDA** technology.
