import os
import csv
import re
import sys

'''
Never ask why this file is necessary, there is a story 
'''

def extract_numbers(file_content):
    # Define regex patterns for each parameter
    patterns = {
        "hidden_size": r"'hidden_size':\s*(\d+)",
        "batch_size": r"'batch_size':\s*(\d+)",
        "optimizer": r"'...._optimizer':\s*<class '.*\..*\..*\.(.*)'>",
        "learning_rate": r"'lr':\s*([\d.e-]+)",
        "weight_decay": r"'weight_decay':\s*([\d.]+)",
        "density": r"'density':\s*([\d.]+)",
        "leakage_rate": r"'leakage_rate':\s*([\d.]+)",
        "spectral_radius": r"'spectral_radius':\s*([-\d.]+)",
        "input_scaling": r"'input_scaling':\s*([\d.]+)",
        "init_range_start": r"'init_range':\s*\(([-\d.]+),",
        "init_range_end": r"'init_range':\s*\([-+\d.]+,\s*([-\d.]+)\)",
        "testing_loss_prob": r"'Testing' loss: prob=([\d.]+)",
        "testing_loss_durr": r"'Testing' loss: prob=[\d.]+,\sdurr=([\d.]+)",
        "testing_accuracy": r"'Testing' Accuracy\s*([\d.]+)"
    
    }

    extracted_data = {}
    
    # Extract numbers using regex patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, file_content)
        if match:
            try:
                extracted_data[key] = float(match.group(1)) if '.' in match.group(1) or 'e' in match.group(1) else int(match.group(1))
            except ValueError:
                extracted_data[key] = match.group(1)
    
    return extracted_data

def main(name: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_path = dir_path
    print(os.path.dirname(dir_path))
    dir_path = os.path.join(os.path.dirname(dir_path),'gridsearch_raw', name)
    
    data: list[dict] = []
    for model_number in os.listdir(dir_path):
        model_folder = os.path.join(dir_path, model_number)
        if not os.path.isdir(model_folder):
            continue
        with open(os.path.join(model_folder, "summary.txt"), 'r') as file:
            file_content = file.read()
            extracted_data = extract_numbers(file_content)
            extracted_data['number'] = int(model_number) 
        data.append(extracted_data)
    with open(os.path.join(save_path, f'{name}_data.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, data[0].keys())
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        raise ValueError('Invalid amount of commandline arguments, run: python data_collector.py *name*')
    main(args[1])
