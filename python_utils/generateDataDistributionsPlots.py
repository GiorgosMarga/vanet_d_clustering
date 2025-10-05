import os
import re
import matplotlib.pyplot as plt

def read_floats_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # Extract all float-like numbers using regex
        float_strings = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', content)
        return [float(num) for num in float_strings]

def collect_data_from_folder(folder_path):
    all_data = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                floats = read_floats_from_file(file_path)
                all_data[filename] = floats
                print(f"Read {len(floats)} floats from {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return all_data

def plot_distribution(data, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, color='skyblue', edgecolor='black')
    plt.title("Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./data_distributions/"+output_path)
    print(f"Distribution saved as {output_path}")

if __name__ == "__main__":
    folder = "data"  # Replace with your folder path
    data = collect_data_from_folder(folder)
    for d in data:
        plot_distribution(data[d],d.split(".")[0]+".png")
    else:
        print("No valid data found.")