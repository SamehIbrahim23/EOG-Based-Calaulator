import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.tsa.ar_model import AutoReg

def load_data(directory):
    """Load and pair EOG signals from directory with class folders containing h/v files"""
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

def preprocess_signal(signal, fs=176,order=4):
    # Bandpass filter (0.5-20 Hz)
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 20 / nyq
    b, a = butter(order, [low, high], btype='band' , output='ba', analog=False)
    filtered = filtfilt(b, a, signal)
    
    # Normalize
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
    features = []
    coeffs = pywt.wavedec(signal, 'db4', level=2)
    features.extend(coeffs[0][:])  
    return features


def extract_auto_regression(signal, lags=5):
    try:
        model = AutoReg(signal, lags=lags, old_names=False).fit()
        return model.params.tolist()
    except Exception as e:
        # Return zero coefficients if AR model fails (e.g. short signal)
        return [0.0] * (lags + 1)

def preprocess_dataset(dataset):
    preprocessed = []
    for h, v, label in dataset:
        h_processed = preprocess_signal(h)
        v_processed = preprocess_signal(v)
        preprocessed.append((h_processed, v_processed, label))
    return preprocessed


def create_features(preprocessed_data, feature_type='wavelet'):
    """
    Create feature matrix from preprocessed signals.
    'raw' - only statistical features
    'wavelet' - only wavelet features
    """
    X = []
    y = []
    
    for h_signal, v_signal, label in preprocessed_data:
        if feature_type == 'raw':
            h_features = h_signal
            v_features = v_signal
        elif feature_type == 'wavelet':
            h_features = extract_wavelet_features(h_signal)
            v_features = extract_wavelet_features(v_signal)
        elif feature_type == 'ar':
            h_features = extract_auto_regression(h_signal)
            v_features = extract_auto_regression(v_signal)
        features = h_features + v_features  
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train SVM and evaluate on test set"""
    svm = SVC(kernel='rbf', C=1, gamma=0.05)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred)

# ============================
# Main
# ============================

train_data = load_data("data/train")
test_data = load_data("data/test")

train_data_processed = preprocess_dataset(train_data)
test_data_processed = preprocess_dataset(test_data)

feature_types = ['raw', 'wavelet' , 'ar']
results = []
best_accuracy = 0
best_model = None
best_feature_type = None

for feature_type in feature_types:
    X_train, y_train = create_features(train_data_processed, feature_type)
    X_test, y_test = create_features(test_data_processed, feature_type)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Train and evaluate
    svm = SVC(kernel='rbf', C=1, gamma=0.05)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({'Feature Type': feature_type, 'Accuracy': accuracy})

    print(f"Accuracy with {feature_type} features: {accuracy:.2f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = svm
        best_feature_type = feature_type
        best_le = le

if best_model is not None:
    model_package = {
        'model': best_model,
        'label_encoder': best_le,
        'feature_type': best_feature_type
    }
    joblib.dump(model_package, 'best_eog_model.pkl')

# Display results
results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df.to_string(index=False))

# Determine best feature type
best_result = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"\nBest feature type: {best_result['Feature Type']} with accuracy {best_result['Accuracy']:.4f}")