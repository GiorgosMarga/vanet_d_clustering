import os
import numpy as np
import sys
def calculate_file_stats(folder_path):
    stats = {}
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Process only files (not directories)
        if os.path.isfile(file_path):
            try:
                # Read all lines from the file
                with open(file_path, 'r') as file:
                    # Parse lines to float, ignoring invalid lines
                    numbers = []
                    for line in file:
                        try:
                            num = float(line.strip())
                            numbers.append(num)
                        except ValueError:
                            print(f"Warning: Skipping invalid line in {filename}: {line.strip()}")
                    
                    # Calculate statistics if we have valid numbers
                    if numbers:

                        avg = np.mean(numbers)
                        std = np.std(numbers)
                        # print(f"\nFile: {filename}")
                        # print(f"Average: {avg:.4f}")
                        # print(f"Standard Deviat""ion: {std:.4f}")
                        id = int(file_path.split("/")[-1].split(".")[0].split("_")[1])
                        print(f"Read {id}")
                        stats[id] = {
                            "avg": float("{:.4f}".format(avg)),
                            "std": float("{:.4f}".format(std))
                        }
                    else:
                        print(f"\nFile: {filename}")
                        print("No valid numbers found in file")
                        
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    return stats
def find_similar_nodes(stats_dict, avg_threshold=1, std_threshold=1):
    # Convert dict values to list for easier comparison
    nodes = [(node_id, val['avg'] if isinstance(val, dict) else val[0], 
              val['std'] if isinstance(val, dict) else val[1]) 
             for node_id, val in stats_dict.items()]
    
    # Find similar nodes
    groups = []
    used = set()
    
    for i, (node1, avg1, std1) in enumerate(nodes):
        if node1 in used:
            continue
        current_group = [node1]
        used.add(node1)
        
        for j, (node2, avg2, std2) in enumerate(nodes):
            if node2 not in used and node1 != node2:
                if (abs(avg1 - avg2) <= avg_threshold and 
                    abs(std1 - std2) <= std_threshold):
                    current_group.append(node2)
                    used.add(node2)
        
        if len(current_group) > 1:  # Only include groups with multiple nodes
            groups.append(current_group)
    
    # Print results
    if not groups:
        print("No nodes with similar averages and standard deviations found.")
    else:
        print("Nodes with similar averages and standard deviations:")
        for i, group in enumerate(groups, 1):
            print(f"\nGroup {i}:")
            for node in group:
                avg = stats_dict[node]['avg'] if isinstance(stats_dict[node], dict) else stats_dict[node][0]
                std = stats_dict[node]['std'] if isinstance(stats_dict[node], dict) else stats_dict[node][1]
                print(f"  {node}: avg={avg:.4f}, std={std:.4f}")


if __name__ == "__main__":
    # Specify the folder path here
    if len(sys.argv) <= 1:
        folder_path = "augmented_data" 
    else:
        folder_path = sys.argv[1]     
    stats = calculate_file_stats(folder_path)
    find_similar_nodes(stats,3,3)
    while(True):
        id = int(input("Id: "))
        if id > 59:
            id = id % 60

        if id not in stats:
            print(f"{id} is not in stats.")
            continue
        print(stats[id])
