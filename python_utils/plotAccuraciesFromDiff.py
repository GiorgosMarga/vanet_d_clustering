import os
import sys
import re
import matplotlib.pyplot as plt

def extract_accuracies_from_file(filepath):
    """Extracts the list of accuracies from a file with line: Accuracies: [..]"""
    with open(filepath, 'r') as f:
        content = f.read()

    match = re.search(r'Accuracies:\s*\[([^\]]+)\]', content)
    if not match:
        raise ValueError(f"No 'Accuracies:' line found in {filepath}")

    accuracy_strs = match.group(1).strip().split()
    accuracies = [float(val) for val in accuracy_strs]
    return accuracies

# t = {
#     "p_5_lr_0.01": "5 Coefficients",
#     "p_10_lr_0.01": "10 Coefficients",
#     "p_20_lr_0.01": "20 Coefficients",
#     "simple_lr_0.01": "ANP",
# }

t = {
    "rnp_0_50_lr_0.01": "RNP 50%",
    "rnp_0_75_lr_0.01": "RNP 75%",
    "simple_lr_0.01": "ANP",
}

# t = {
#     "lap_2_lr_0.01": "LAP",
#     "gap_2_lr_0.01": "GAP",
#     "simple_lr_0.01": "ANP",
# }

def plot_accuracies(accuracy_lists, labels,id):
    plt.figure(figsize=(10, 6))
    for accs, label in zip(accuracy_lists, labels):
        plt.plot(accs, label=t[label])
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Comparison CH ({id})")
    plt.xticks(range(51))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"accuracy_comparison_{id}.png")

def main():
    
    for number in [ 6,22,30,48,52]:
        # folder_names = ["gap_2_lr_0.01", "lap_2_lr_0.01", "simple_lr_0.01"]
        # folder_names = ["p_5_lr_0.01", "p_10_lr_0.01", "p_20_lr_0.01", "simple_lr_0.01"]
        folder_names = ["rnp_0_50_lr_0.01", "rnp_0_75_lr_0.01", "simple_lr_0.01"]
        file_pattern = f"cars_info/car_{number}.info"

        accuracy_lists = []
        labels = []

        for folder in folder_names:
            filepath = os.path.join(folder, file_pattern)
            print(folder+file_pattern)
            if os.path.exists(filepath):
                print(f"✅ Found: {filepath}")
                try:
                    accs = extract_accuracies_from_file(filepath)
                    accuracy_lists.append(accs)
                    labels.append(folder)
                except Exception as e:
                    print(f"⚠️ Failed to read accuracies from {filepath}: {e}")
            else:
                print(f"❌ File not found: {filepath}")

        if accuracy_lists:
            plot_accuracies(accuracy_lists, labels, number)
            for accuracy in accuracy_lists:
                print(accuracy[-1])
        else:
            print("❌ No accuracy data found in any folder.")

if __name__ == "__main__":
    main()
