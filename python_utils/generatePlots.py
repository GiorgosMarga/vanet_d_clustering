import os
import re
import numpy as np
import matplotlib.pyplot as plt
import shutil

def clear_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"{folder_path} is not a valid directory.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory and contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
def plot_arrays_from_text_files(folder_path, output_folder="plots"):
    array_pattern = re.compile(r"\[([^\[\]]+)\](?!:)")  # matches content inside square brackets

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            matches = array_pattern.findall(content)

            arrays = []
            for match in matches:
                # Split by whitespace and convert to float
                arr = np.array([float(x) for x in match.strip().split()])
                arrays.append(arr)

            title = "Node_" + filename.split(".")[0].split("_")[1]

            if len(arrays) >= 2:
                plt.figure(figsize=(10, 4))
                plt.plot(arrays[0], label='Predicted')
                plt.plot(arrays[1], label='Actual')
                plt.title(f"{title} - Predictions vs Actual")
                plt.ylabel("Temperature")
                plt.xlabel("Sample Index")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f"{title}_pred_vs_actual.png"))
                plt.close()

            if len(arrays) >= 3:
                plt.figure(figsize=(10, 3))
                plt.plot(arrays[2], color='red', label='Errors')
                plt.title(f"{title} - Errors")
                plt.tight_layout()
                plt.ylabel("Loss")
                plt.xlabel("Sample Index")
                plt.savefig(os.path.join(output_folder, f"{title}_errors.png"))
                plt.close()


            if len(arrays) >= 4:
                plt.figure(figsize=(10, 3))
                plt.plot(arrays[3], color='orange', label='Accuracy')
                plt.title(f"{title} - Accuracy")
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f"{title}_accuracy.png"))
                plt.close()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# clear plots folder
clear_folder("plots")
plot_arrays_from_text_files("./simple_lr_0.01/cars_info")