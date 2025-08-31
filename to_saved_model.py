import os
import numpy as np
import pandas as pd
import pywt
import joblib
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data(directory):
    data = []
    class_folders = {
        'Blink': 'blink',
        'Down': 'down',
        'Left': 'left',
        'Right': 'right',
        'Up': 'up'
    }
    for class_folder, label in class_folders.items():
        class_path = os.path.join(directory, class_folder)
        if not os.path.exists(class_path):
            continue
        file_pairs = {}
        for filename in os.listdir(class_path):
            if filename.endswith(('h.txt', 'v.txt')):
                try:
                    file_num = filename.split('.')[0][:-1]
                    suffix = filename[-5]
                    if file_num not in file_pairs:
                        file_pairs[file_num] = {}
                    file_pairs[file_num][suffix] = os.path.join(class_path, filename)
                except:
                    continue
        for file_num, files in file_pairs.items():
            if 'h' in files and 'v' in files:
                try:
                    h_signal = np.loadtxt(files['h'])
                    v_signal = np.loadtxt(files['v'])
                    data.append((h_signal, v_signal, label))
                except:
                    continue
    return data

def preprocess_signal(signal, fs=176):
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 20 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
    return normalized

def extract_raw_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.sum(signal**2),  # Energy
    ]

def extract_wavelet_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=2)
    return coeffs[0].tolist()

def preprocess_dataset(dataset):
    preprocessed = []
    for h, v, label in dataset:
        h_processed = preprocess_signal(h)
        v_processed = preprocess_signal(v)
        preprocessed.append((h_processed, v_processed, label))
    return preprocessed

def create_features(preprocessed_data, feature_type='wavelet'):
    X, y = [], []
    for h_signal, v_signal, label in preprocessed_data:
        if feature_type == 'raw':
            h_features = extract_raw_features(h_signal)
            v_features = extract_raw_features(v_signal)
        elif feature_type == 'wavelet':
            h_features = extract_wavelet_features(h_signal)
            v_features = extract_wavelet_features(v_signal)
        features = h_features + v_features
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='rbf', C=5, gamma=0.1)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return svm, acc

# ============================
# Main Script
# ============================

print("Loading data...")
train_data = load_data("data/train")
test_data = load_data("data/test")

print("Preprocessing signals...")
train_data_processed = preprocess_dataset(train_data)
test_data_processed = preprocess_dataset(test_data)

feature_types = ['raw', 'wavelet']
results = []
last_acc = 0.0

for feature_type in feature_types:
    print(f"\nProcessing with {feature_type} features...")

    X_train, y_train = create_features(train_data_processed, feature_type)
    X_test, y_test = create_features(test_data_processed, feature_type)

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train and evaluate
    svm, accuracy = train_and_evaluate(X_train, X_test, y_train_encoded, y_test_encoded)
    results.append({'Feature Type': feature_type, 'Accuracy': accuracy})

    print(f"Accuracy with {feature_type} features: {accuracy:.4f}")

    if accuracy > last_acc:
        last_acc = accuracy
        model_path = f"best_svm_model_{feature_type}.joblib"
        encoder_path = f"label_encoder_{feature_type}.joblib"
        joblib.dump(svm, model_path)
        joblib.dump(le, encoder_path)
        print(f"Best model so far saved to: {model_path}")
        print(f"Label encoder saved to: {encoder_path}")

# Display results
results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df.to_string(index=False))

# Report best
best_result = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"\nBest feature type: {best_result['Feature Type']} with accuracy {best_result['Accuracy']:.4f}")
