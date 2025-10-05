import os
import numpy as np

def load_temperature_series(filepath):
    """Load temperature data from a file with one value per line."""
    return np.loadtxt(filepath)

def add_noise(series, noise_level=0.5):
    """Add Gaussian noise to a temperature series."""
    noise = np.random.normal(loc=0.0, scale=noise_level, size=series.shape)
    return series + noise

def save_series(series, filepath):
    """Save a temperature series to a file, one value per line."""
    np.savetxt(filepath, series, fmt="%.2f")

def process_folder(input_dir, output_dir, noise_level=0.5):
    """Read all .id files from input_dir, add noise, and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing: {filename}")
            series = load_temperature_series(input_path)
            noise_level = np.random.uniform(0.5, 1)
            augmented = add_noise(series, noise_level=noise_level)
            save_series(augmented, output_path)
            print(series[:5])
            print(augmented[:5])

    print("✅ All files processed.")

# Example usage
input_folder = "data"       # Folder containing original .id files
output_folder = "augmented_data"  # Folder to save synthetic .id files
process_folder(input_folder, output_folder, noise_level=0.8)
