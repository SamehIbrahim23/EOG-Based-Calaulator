# <h1 align="center">EOG-Based Calculator Navigation</h1>

<br>

## Project Overview

The core of this project is a machine learning model that analyzes EOG data to classify five distinct eye commands:

-   **Blink**: Acts as a selection or "enter" command.
-   **Up**: Moves the selection highlight up.
-   **Down**: Moves the selection highlight down.
-   **Left**: Moves the selection highlight left.
-   **Right**: Moves the selection highlight right.

---

## How It Works

The system workflow can be broken down into the following steps:

-   **Data Loading**: EOG signals, recorded from horizontal (h) and vertical (v) channels, are loaded from a structured directory.
-   **Signal Preprocessing**: Each EOG signal is preprocessed to remove noise and normalize the data. This includes applying a bandpass filter and scaling the values.
-   **Feature Extraction**: The preprocessed signals are transformed into numerical features that the machine learning model can understand...
