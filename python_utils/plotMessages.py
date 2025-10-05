import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
# File path
file_path = 'messages.txt'

# Mapping protocols to index in array
protocol_indices = {'5 Coefficients': 0, '10 Coefficients': 1,'20 Coefficients':2, 'ANP': 3}

# Dictionary to hold message counts per node
node_messages = defaultdict(lambda: [None, None, None, None])

# Current protocol
current_protocol = None

# Regular expression to match node data
node_pattern = re.compile(r'Node \[(\d+)] .*?messages ([\d.]+)')
# Read and parse the file
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line in protocol_indices:
            current_protocol = line
        elif current_protocol and (match := node_pattern.search(line)):
            node_id = int(match.group(1))
            messages = float(match.group(2))/50
            index = protocol_indices[current_protocol]
            node_messages[node_id][index] = messages

# Example: print the result
for node_id, messages in sorted(node_messages.items()):
    print(f"Node {node_id}: {messages}")


while False:
    cluster = input("Cluster members: ")
    splitted = cluster.split(" ")
    avg = [0,0,0,0]

    for id in splitted:
        avg[0] += node_messages[int(id)][0] 
        avg[1] += node_messages[int(id)][1] 
        avg[2] += node_messages[int(id)][2] 
        avg[3] += node_messages[int(id)][3] 
    avg[0] /= len(splitted)
    avg[1] /= len(splitted)
    avg[2] /= len(splitted)     
    avg[3] /= len(splitted)     
    print(avg)


t = {
    6: [14865.386666666667, 10198.346666666666, 6395.573333333334, 20223.84],
    22: [9334.08, 6015.296, 3111.3599999999997, 19912.703999999998],
    30: [15260.48, 12815.84, 9037.76, 20816.48],
    48: [15556.8, 14001.119999999999, 8296.96, 18927.440000000002],
    52: [17216.192, 11823.168, 10060.063999999998, 21157.248],
}    

# Protocol labels
protocols = ['5 Coefficients', '10 Coefficients', '20 Coefficients', "ANP"]
num_protocols = len(protocols)

# Prepare data
nodes = list(t.keys())
x = np.arange(len(nodes))  # the label locations
width = 0.15  # the width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(num_protocols):
    values = [t[node][i] for node in nodes]
    ax.bar(x + i * width, values, width, label=protocols[i])

# Labels and title
ax.set_xlabel('Node')
ax.set_ylabel('Bytes')
ax.set_title('Average Bytes Within The Cluster')
ax.set_xticks(x + width)
ax.set_xticklabels([str(node) for node in nodes])
ax.legend()

# Show plot
plt.tight_layout()
plt.savefig("messages_rnp")