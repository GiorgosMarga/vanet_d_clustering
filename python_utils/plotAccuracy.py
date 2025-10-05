import os
import re
import matplotlib.pyplot as plt

IDS = [6 ,7 ,11 ,22 ,27 ,30 ,33 ,37, 41, 48, 52, 59]
def get_id(filepath):
    return filepath.split("/")[-1].split(".")[0].split("_")[1]
def extract_accuracies_from_file(filepath):
    node_id = get_id(filepath)
    if int(node_id) not in IDS :
        return
    with open(filepath, 'r') as f:
        content = f.read()
    print(filepath)

    # Extract accuracy values using regex
    accuracy_match = re.search(r'Accuracies:\s*\[([^\]]+)\]', content)
    if accuracy_match:
        accuracy_str = accuracy_match.group(1)
        accuracy_values = [float(x) for x in accuracy_str.split()]
        return accuracy_values
    return None
def extract_errors_from_file(filepath):
    node_id = get_id(filepath)
    if int(node_id) not in IDS:
        return
    with open(filepath, 'r') as f:
        content = f.read()
    print(filepath)

    # Extract accuracy values using regex
    errors_match = re.search(r'Errors:\s*\[([^\]]+)\]', content)
    if errors_match:
        error_str = errors_match.group(1)
        error_values = [float(x) for x in error_str.split()]
        return error_values
    return None

def plot_accuracies_from_folder(folder_path, save_path="accuracies_plot.png"):
    plt.figure(figsize=(12, 6))
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            accuracies = extract_accuracies_from_file(filepath)
            if accuracies:
                plt.plot(accuracies, label=f"Clusterhead ({get_id(filepath)})")

    plt.title("Cluster Heads' Accuracy")
    plt.xlabel("Rounds")
    plt.xticks(range(51))
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    # plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_errors_from_folder(folder_path, save_path="errors_plot.png"):
    plt.figure(figsize=(12, 6))
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            accuracies = extract_errors_from_file(filepath)
            if accuracies:
                plt.plot(accuracies, label=f"Clusterhead ({get_id(filepath)})")

    plt.title("Cluster Heads' Loss")
    plt.xlabel("Rounds")
    plt.xticks(range(51))
    plt.ylabel("Loss (MSE)")
    plt.legend(loc="lower right")
    # plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# Replace with your folder path
folder_path = "simple_lr_0.01/cars_info"
plot_accuracies_from_folder(folder_path)
plot_errors_from_folder(folder_path)

