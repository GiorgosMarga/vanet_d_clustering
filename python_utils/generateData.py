import os
import numpy as np


def generate_distinct_data_files(num_files=60, num_numbers=5000, output_dir="data_files"):
    """
    Generate files with numbers having distinct means and standard deviations.
    
    Parameters:
    num_files (int): Number of files to generate (default: 60)
    num_numbers (int): Number of numbers per file (default: 5000)
    output_dir (str): Directory to save the files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define ranges for means and standard deviations
    means = np.arange(0, num_files, 1)  # 0, 1, 2, ..., 59 for 60 files
    stds = np.linspace(1, 6, num_files // 2)  # 30 values from 1 to 6
    stds = np.concatenate([stds, stds + 0.5])  # Extend to 60 by adding offset
    
    # Generate and save files
    for i in range(num_files):
        # Generate 5000 numbers with unique mean and std
        data = np.random.normal(loc=means[i], scale=stds[i], size=num_numbers)
        
        # Save to file
        file_path = os.path.join(output_dir, f"car_{i}.txt")
        np.savetxt(file_path, data, fmt="%.6f")
    
    # Verify statistics
    print("Generated files with the following statistics:")
    stats_dict = {}
    for i in range(num_files):
        file_path = os.path.join(output_dir, f"car_{i}.txt")
        data = np.loadtxt(file_path)
        stats_dict[i] = {'avg': np.mean(data), 'std': np.std(data)}
        print(f"data_{i}.txt: mean={stats_dict[i]['avg']:.4f}, std={stats_dict[i]['std']:.4f}")
    
    return stats_dict

if __name__ == "__main__":
    # Generate the files
    stats = generate_distinct_data_files(output_dir="./data/my_data_3")