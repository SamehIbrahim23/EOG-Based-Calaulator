<hr>EOG-Based Calculator Navigation</hr>
This project presents a novel approach to human-computer interaction, allowing users to control a graphical calculator interface using only their eye movements and blinks. The system uses Electrooculography (EOG) signals to detect and interpret user commands, translating them into navigation and input actions on the calculator.

Project Overview
The core of this project is a machine learning model that analyzes EOG data to classify five distinct eye commands:

Blink: Acts as a selection or "enter" command.

Up: Moves the selection highlight up.

Down: Moves the selection highlight down.

Left: Moves the selection highlight left.

Right: Moves the selection highlight right.

These commands are used to navigate a custom-built calculator interface designed for a cross-shaped layout. The system first trains a Support Vector Machine (SVM) model on a dataset of EOG signals, then uses this trained model to predict user commands in real time, enabling hands-free control of the calculator.

How It Works
The system workflow can be broken down into the following steps:

Data Loading: EOG signals, recorded from horizontal (h) and vertical (v) channels, are loaded from a structured directory.

Signal Preprocessing: Each EOG signal is preprocessed to remove noise and normalize the data. This includes applying a bandpass filter and scaling the values.

Feature Extraction: The preprocessed signals are transformed into numerical features that the machine learning model can understand. The project compares three different feature extraction methods:

Raw Features: Simple statistical properties of the signal.

Wavelet Features: Coefficients from a discrete wavelet transform, which are effective at capturing signal characteristics across different frequency scales.

Auto-Regression (AR): Coefficients from an AR model, which describe the relationship between a signal's current value and its past values.

Model Training: A Support Vector Machine (SVM) classifier is trained on the extracted features and their corresponding labels (e.g., 'blink', 'up', 'down'). The model that achieves the highest accuracy is saved for later use.

GUI Interface: A graphical calculator is built using the tkinter library. It features a unique cross-shaped button layout optimized for directional navigation.

Real-Time Prediction and Control: The system simulates the prediction of a sequence of commands (e.g., a mathematical calculation). It loads the best-performing model and uses it to predict the correct command from a set of new EOG features. These predictions are then used to control the calculator's selection and input, simulating a genuine hands-free user experience.
