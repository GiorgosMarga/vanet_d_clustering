import re
import os
import matplotlib.pyplot as plt
import sys
def get_stats(filename):
    with open(filename, "r") as f:
        input_text = str(f.readlines())
        errors_match = re.search(r'Errors:\s*\[([^\]]+)\]', input_text)
        errors = [float(num) for num in errors_match.group(1).split()] if errors_match else []

        accuracies_match = re.search(r'Accuracies:\s*\[([^\]]+)\]', input_text)
        accuracies = [float(num) for num in accuracies_match.group(1).split()] if accuracies_match else []
        f.close()
        return accuracies,errors

def get_average_from_folder(foldername):
    avg_accuracies, avg_errors = [0 for i in range(51)],[0 for i in range(51)]
    for f in os.listdir(foldername):
        accuracies,errors = get_stats(f"{foldername}/{f}")
        for (idx,accuracy) in enumerate(accuracies):
            avg_accuracies[idx] += accuracy
        for (idx,err) in enumerate(errors):
            avg_errors[idx] += err
    for i in range(51):
        avg_errors[i] /= 60
        avg_accuracies[i] /= 60
    return avg_accuracies, avg_errors
def get_average_accuracies(foldername, keywords):
    avgs = {}
    for item in os.listdir(foldername):
        full_path = os.path.join(foldername, item)
        if os.path.isdir(full_path):
            cont = False
            for keyword in keywords:
                if keyword not in item:
                    cont = True
            if cont:
                continue
            acu,err = get_average_from_folder(f"{foldername}/{item}/cars_info")
            avgs[f"{item}"] = (acu,err)

    return avgs


def plot_avgs(results, png_name):
    plt.figure(figsize=(10, 5))
    for name, (accuracies, _) in results.items():
        x = list(range(1, len(accuracies) + 1))
        plt.plot(x, accuracies, marker='o', label=name)
    plt.title('All Accuracies Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{png_name}_accuracies")

# Error Plot (All lines in one figure)
    plt.figure(figsize=(10, 5))
    for name, (_, errors) in results.items():
        x = list(range(1, len(errors) + 1))
        plt.plot(x, errors, marker='x', label=name)
        plt.title('All Errors Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{png_name}_errors.png")


folder_name = sys.argv[1]
keywords = sys.argv[2:]
os.makedirs(f'{folder_name}/graphs', exist_ok=True)

plot_avgs(get_average_accuracies(folder_name, keywords), f'{folder_name}/graphs/{"_".join(keywords)}')
