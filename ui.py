import tkinter as tk
import numpy as np
from tkinter import messagebox
import csv
import joblib
import random

# Initialize calculator state
current_input = ""
operation = None
first_number = None
selected_button_text = 'Center'
buttons = {}

# Create main window
root = tk.Tk()
root.title("Simple Calculator")
root.geometry("650x600")
root.configure(bg="#9dcb91")
root.resizable(False, False)

# Display
display = tk.Entry(root, font=('Arial', 18), justify='right', bd=3, relief='sunken', bg='white', fg='black')
display.place(x=30, y=20, width=590, height=40)

# Widget configuration
default_bg_button = '#e0e0e0'
default_fg_button = 'black'
default_bg_label = 'black'
default_fg_label = 'white'

widget_width = 60
widget_height = 40
widget_pad_x = 5
widget_pad_y = 10

center_area_x = 650 / 2
center_area_y = 600 / 2 + 30
# Widget layout
widget_layout = [
    # Center Label
    ('Center', center_area_x - widget_width/2, center_area_y - widget_height/2, widget_width, widget_height, 'label'),

    # Top Cross
    ('Up', center_area_x - widget_width/2, center_area_y - widget_height/2 - 3.5*widget_height - widget_pad_y, widget_width, widget_height, 'label'),
    ('9', center_area_x - widget_width/2, center_area_y - widget_height/2 - 4*(widget_height + widget_pad_y), widget_width, widget_height, 'button'),
    ('C', center_area_x - widget_width/2 + 2*(widget_width + widget_pad_x) - (widget_width + widget_pad_x), center_area_y - widget_height/2 - 3.5*widget_height - widget_pad_y, widget_width, widget_height, 'button'),
    ('E', center_area_x - widget_width/2, center_area_y - widget_height/2 - widget_pad_y- 2.3*widget_height, widget_width, widget_height, 'button'),
    ('8', center_area_x - widget_width/2 - widget_width - widget_pad_x, center_area_y - widget_height/2 - 3.5*widget_height - widget_pad_y, widget_width, widget_height, 'button'),

    # Left Cross
    ('Left', center_area_x - widget_width/2 - 2.3*widget_width - widget_pad_x, center_area_y - widget_height/2, widget_width, widget_height, 'label'),
    ('3', center_area_x - widget_width/2 - 2.3*widget_width - widget_pad_x, center_area_y - widget_height/2 - widget_height - widget_pad_y, widget_width, widget_height, 'button'),
    ('1', center_area_x - widget_width/2 - 2.3*widget_width - widget_pad_x + widget_width + widget_pad_x, center_area_y - widget_height/2, widget_width, widget_height, 'button'),
    ('2', center_area_x - widget_width/2 - 2.3*widget_width - widget_pad_x, center_area_y - widget_height/2 + widget_height + widget_pad_y, widget_width, widget_height, 'button'),
    ('0', center_area_x - widget_width/2 - (3.5*widget_width + widget_pad_x), center_area_y - widget_height/2, widget_width, widget_height, 'button'),
    
    # Right Cross
    ('Right', center_area_x - widget_width/2 + 2.5*widget_width + widget_pad_x, center_area_y - widget_height/2, widget_width, widget_height, 'label'),
    ('4', center_area_x - widget_width/2 + 2.5*widget_width + widget_pad_x, center_area_y - widget_height/2 - widget_height - widget_pad_y, widget_width, widget_height, 'button'),
    ('7', center_area_x - widget_width/2 + 3.3*(widget_width + widget_pad_x), center_area_y - widget_height/2, widget_width, widget_height, 'button'),
    ('6', center_area_x - widget_width/2 + 2.5*widget_width + widget_pad_x, center_area_y - widget_height/2 + widget_height + widget_pad_y, widget_width, widget_height, 'button'),
    ('5', center_area_x - widget_width/2 + 2.3*(widget_width + widget_pad_x) - (widget_width + widget_pad_x), center_area_y - widget_height/2 - (widget_height + widget_pad_y) + (widget_height + widget_pad_y), widget_width, widget_height, 'button'),

    # Bottom Cross
    ('Down', center_area_x - widget_width/2, center_area_y - widget_height/2 + 3.5*widget_height + widget_pad_y, widget_width, widget_height, 'label'),
    ('/', center_area_x - widget_width/2, center_area_y - widget_height/2 + 2.3*widget_height + widget_pad_y, widget_width, widget_height, 'button'),
    ('-', center_area_x - widget_width/2 + widget_width + widget_pad_x, center_area_y - widget_height/2 + 3.5*widget_height + widget_pad_y, widget_width, widget_height, 'button'),
    ('+', center_area_x - widget_width/2, center_area_y - widget_height/2 + 4*(widget_height + widget_pad_y), widget_width, widget_height, 'button'),
    ('*', center_area_x - widget_width/2 - widget_width - widget_pad_x, center_area_y - widget_height/2 + 3.5*widget_height + widget_pad_y, widget_width, widget_height, 'button'),
]

# Navigation transitions
transitions = {
    'Center': {'up': 'Up', 'down': 'Down', 'left': 'Left', 'right': 'Right'},
    'Up': {'up': '9', 'down': 'E', 'left': '8', 'right': 'C'},
    'Left': {'up': '3', 'down': '2', 'left': '0', 'right': 'Center'},
    'Right': {'up': '4', 'down': '6', 'left': '5', 'right': '7'},
    'Down': {'up': '/', 'down': '+', 'left': '*', 'right': '-'},
    # 'Center': {'up': 'Up', 'down': 'Down', 'left': 'Left', 'right': 'Right'},
    # 'Up': {'up': '4', 'down': '6', 'left': '5', 'right': '7'},
    # '4': {'down': 'Up'},
    # '7': {'left': 'Up'},
    # '6': {'up': 'Up', 'down': 'Center'},
    # '5': {'right': 'Up'},
    # 'Left': {'up': '8', 'down': 'E', 'left': '9', 'right': 'C'},
    # '8': {'down': 'Left'},
    # '9': {'right': 'Left'},
    # 'E': {'up': 'Left'},
    # 'C': {'right': 'Center', 'left': 'Left'},
    # 'Right': {'up': '/', 'down': '+', 'left': '*', 'right': '-'},
    # '/': {'down': 'Right'},
    # '+': {'up': 'Right'},
    # '-': {'left': 'Right'},
    # '*': {'left': 'Center', 'right': 'Right'},
    # 'Down': {'up': '0', 'down': '2', 'left': '1', 'right': '3'},
    # '0': {'up': 'Center', 'down': 'Down'},
    # '2': {'up': 'Down'},
    # '3': {'up': 'Down'},
    # '1': {'right': 'Down'},
}

# Helper functions
def set_selected(widget_text):
    global selected_button_text
    
    # Reset previous selection
    if selected_button_text in buttons:
        prev_widget = buttons[selected_button_text]
        if isinstance(prev_widget, tk.Button):
            prev_widget.config(bg='#e0e0e0', fg='black', relief='raised')
        elif isinstance(prev_widget, tk.Label):
            prev_widget.config(bg='black', fg='white', relief='raised')
    
    # Set new selection
    if widget_text in buttons:
        selected_button_text = widget_text
        current_widget = buttons[selected_button_text]
        current_widget.config(bg='#fb7070', fg='white', relief='sunken')
    else:
        print(f"Warning: Widget '{widget_text}' not found.")
        selected_button_text = None

def move_selection(direction):
    if selected_button_text is None:
        set_selected('Center')
        return
    
    current_text = selected_button_text
    next_text = transitions.get(current_text, {}).get(direction)
    
    if next_text and next_text in buttons:
        set_selected(next_text)

def handle_number_input(digit):
    global current_input, first_number, operation
    
    current_input += digit
    display.delete(0, tk.END)
    display.insert(0, current_input)
    
    if first_number is not None and operation:
        calculate_result()

def handle_operator_input(op):
    global current_input, first_number, operation
    
    if current_input:
        try:
            first_number = float(current_input)
            operation = op
            current_input = ""
            display.delete(0, tk.END)
            display.insert(0, f"{first_number} {operation}")
        except ValueError:
            messagebox.showerror("Error", "Invalid number input")
            clear_all()
    elif first_number is not None:
        operation = op
        display.delete(0, tk.END)
        display.insert(0, f"{first_number} {operation}")

def calculate_result():
    global current_input, first_number, operation
    
    if first_number is not None and operation and current_input:
        try:
            b = float(current_input)
            if operation == '+':
                result = first_number + b
            elif operation == '-':
                result = first_number - b
            elif operation == '*':
                result = first_number * b
            elif operation == '/':
                if b == 0:
                    raise ZeroDivisionError
                result = first_number / b
            
            display.delete(0, tk.END)
            if result == int(result):
                display.insert(0, str(int(result)))
            else:
                display.insert(0, f"{result:.10f}".rstrip('0').rstrip('.'))
            
            current_input = str(result)
            operation = None
            first_number = None
            
        except ZeroDivisionError:
            messagebox.showerror("Error", "Cannot divide by zero")
            clear_all()
        except ValueError:
            messagebox.showerror("Error", "Invalid input after operator")
            clear_all()
        except Exception as e:
            messagebox.showerror("Error", f"Calculation error: {e}")
            clear_all()

def clear_all():
    global current_input, operation, first_number
    
    current_input = ""
    operation = None
    first_number = None
    display.delete(0, tk.END)
    set_selected('Center')

def on_button_click(button_text):
    print(f"Button clicked: {button_text}")
    
    if button_text in '0123456789':
        handle_number_input(button_text)
    elif button_text in '+-*/':
        handle_operator_input(button_text)
    elif button_text == 'C':
        clear_all()
    elif button_text == 'E':
        root.quit()

def on_blink(event=None):
    if selected_button_text is None:
        return
    
    widget_text = selected_button_text
    print(f"Blink on: {widget_text}")
    
    selected_widget = buttons.get(widget_text)
    
    if selected_widget is None:
        print(f"Error: Selected widget '{widget_text}' not found.")
        return
    
    if isinstance(selected_widget, tk.Label):
        direction = widget_text.lower()
        if direction in ['up', 'down', 'left', 'right']:
            move_selection(direction)
        elif widget_text == 'Center':
            if selected_button_text != 'Center':
                set_selected('Center')
    elif isinstance(selected_widget, tk.Button):
        on_button_click(widget_text)
    
    # Always return to center after blink
    root.after(500, lambda: set_selected('Center'))

# Create widgets
for (text, x, y, w, h, widget_type) in widget_layout:
    if widget_type == 'button':
        widget = tk.Button(root, text=text, 
                         bg=default_bg_button, fg=default_fg_button, relief='raised', bd=4,
                         font=('Arial', 14, 'bold'),
                         command=lambda t=text: on_button_click(t))
    elif widget_type == 'label':
        widget = tk.Label(root, text=text, anchor='center',
                         bg=default_bg_label, fg=default_fg_label, relief='raised', bd=4,
                         font=('Arial', 14, 'bold'))
    
    widget.place(x=x, y=y, width=w, height=h)
    buttons[text] = widget

# Initial selection
set_selected('Center')

# Key bindings
root.bind('<Up>', lambda e: move_selection('up'))
root.bind('<Down>', lambda e: move_selection('down'))
root.bind('<Left>', lambda e: move_selection('left'))
root.bind('<Right>', lambda e: move_selection('right'))
root.bind('<Return>', on_blink)

#=====================================================


def get_features_for_class(file_path, target_class):
    feature_list = []
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if row['Class'] == target_class:
                features = [float(row[f'f_{i}']) for i in range(134)]
                feature_list.append(features)
    
    return feature_list

file_path = 'saved_csv_data/test/test_features_wavelet.csv'

blink_features = get_features_for_class(file_path, 'blink')
up_features = get_features_for_class(file_path, 'up')
down_features = get_features_for_class(file_path, 'down')
right_features = get_features_for_class(file_path, 'right')
left_features = get_features_for_class(file_path, 'left')

def predict_sequence(scenario_sequence):
    model_package = joblib.load('best_eog_model.pkl')
    model = model_package['model']
    label_encoder = model_package['label_encoder']
    feature_type = model_package['feature_type']

    feature_map = {
        'blink': blink_features,
        'up': up_features,
        'down': down_features,
        'right': right_features,
        'left': left_features
    }
    
    predictions = []
    
    for command in scenario_sequence:
        features_list = feature_map.get(command.lower()) 
        idx =random.randint(0, 3)
        sample_features = features_list[0]
        sample_features = np.array(sample_features).reshape(1, -1)
        prediction = model.predict(sample_features)
        prediction_label = label_encoder.inverse_transform(prediction)[0]
        predictions.append(prediction_label)
    
    return predictions

# Example scenarios
scenario1 = ['left', 'up', 'blink', 'down', 'left', 'blink', 'right', 'left', 'blink']  # 3×5
scenario2 = ['left', 'down', 'blink', 'down', 'down', 'blink', 'left', 'up', 'blink']   # 2+3

def execute_sequence(sequence):
    """Execute a sequence of commands automatically"""
    predictions = predict_sequence(sequence)
    print(f"Executing sequence: {predictions}")
    
    # Clear current input and reset calculator
    clear_all()
    
    # Execute each predicted command with a delay
    for i, command in enumerate(predictions):
        # Schedule each command with increasing delay
        root.after(1500 * (i + 1), lambda cmd=command: execute_single_command(cmd))
    
    # Schedule final calculation and display after all commands
    root.after(1500 * (len(predictions) + 500, calculate_and_display))

def execute_single_command(command):
    """Execute a single command from the sequence"""
    print(f"Executing: {command}")
    
    if command in ['left', 'right', 'up', 'down']:
        # Move selection
        move_selection(command)
    elif command == 'blink':
        # Simulate blink (Enter key)
        on_blink()
    else:
        print(f"Unknown command: {command}")

def calculate_and_display():
    """Calculate and display the final result, then return to center"""
    if operation and current_input:
        calculate_result()
    # Return to center after calculation
    set_selected('Center')

# Create scenario buttons
scenario1_btn = tk.Button(root, text="Scenario 1 (3×5)", 
                         bg='#4CAF50', fg='white', relief='raised', bd=4,
                         font=('Arial', 12, 'bold'),
                         command=lambda: execute_sequence(scenario1))
scenario1_btn.place(x=50, y=550, width=200, height=30)

scenario2_btn = tk.Button(root, text="Scenario 2 (2+3)", 
                         bg='#2196F3', fg='white', relief='raised', bd=4,
                         font=('Arial', 12, 'bold'),
                         command=lambda: execute_sequence(scenario2))
scenario2_btn.place(x=400, y=550, width=200, height=30)

# Run the application
root.mainloop()