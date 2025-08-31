import os
import shutil
import numpy as np
from pathlib import Path

# Configuration
input_folder = 'class'
output_folder = 'data'
classes = ['Blink', 'Up', 'Down', 'Left', 'Right']
samples_per_file = 250
test_files_per_class = 5  # Number of file pairs (5 vertical + 5 horizontal = 10 files total per class)

# Create output directories
Path(os.path.join(output_folder, 'train')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, 'test')).mkdir(parents=True, exist_ok=True)

for class_name in classes:
    # Create class directories in train and test
    train_class_dir = os.path.join(output_folder, 'train', class_name)
    test_class_dir = os.path.join(output_folder, 'test', class_name)
    Path(train_class_dir).mkdir(parents=True, exist_ok=True)
    Path(test_class_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all files for this class and group them by their base name (without v/h)
    file_pairs = {}
    for root, _, files in os.walk(os.path.join(input_folder, class_name)):
        for file in files:
            base_name = file[:-5]  # Remove 'v.txt' or 'h.txt' (assuming format like kirp1v.txt)
            if base_name not in file_pairs:
                file_pairs[base_name] = {'v': None, 'h': None}
            if file.endswith('v.txt'):
                file_pairs[base_name]['v'] = os.path.join(root, file)
            elif file.endswith('h.txt'):
                file_pairs[base_name]['h'] = os.path.join(root, file)
    
    # Convert to list of pairs
    pair_list = []
    for base_name, files in file_pairs.items():
        if files['v'] and files['h']:  # Only include complete pairs
            pair_list.append((base_name, files['v'], files['h']))
    
    # Randomly select test pairs (consistent across runs)
    np.random.seed(42)  # for reproducibility
    test_pairs = np.random.choice(len(pair_list), size=test_files_per_class, replace=False)
    
    for i, (base_name, v_file, h_file) in enumerate(pair_list):
        # Determine if this pair is for testing
        if i in test_pairs:
            dest_dir = test_class_dir
        else:
            dest_dir = train_class_dir
        
        # Process vertical file
        try:
            with open(v_file, 'r') as f:
                lines = [next(f) for _ in range(samples_per_file)]
            output_file = os.path.join(dest_dir, os.path.basename(v_file))
            with open(output_file, 'w') as f:
                f.writelines(lines)
        except (StopIteration, ValueError):
            print(f"Warning: File {v_file} has fewer than {samples_per_file} samples")
        
        # Process horizontal file
        try:
            with open(h_file, 'r') as f:
                lines = [next(f) for _ in range(samples_per_file)]
            output_file = os.path.join(dest_dir, os.path.basename(h_file))
            with open(output_file, 'w') as f:
                f.writelines(lines)
        except (StopIteration, ValueError):
            print(f"Warning: File {h_file} has fewer than {samples_per_file} samples")

print("Data organization complete!")