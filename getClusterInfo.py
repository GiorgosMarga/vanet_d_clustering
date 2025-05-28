import re
from collections import defaultdict

# Input file
filename = './graph_info/graph.info'

# Track per-node history of (cluster_head, full_cluster_list)
node_cluster_history = defaultdict(list)

# Match snapshot headers like "cars_10"
snapshot_pattern = re.compile(r'^.*cars_(\d+).*$', re.MULTILINE)

# Match full map[...] content (even if multiline)
map_block_pattern = re.compile(r'map\[(.*?)(?:\n\S|\Z)', re.DOTALL)

# Match cluster entries like 46:[46 33 27 51]
cluster_entry_pattern = re.compile(r'(\d+):\[(.*?)\]')

with open(filename, 'r') as f:
    content = f.read()

# Locate all snapshots
snapshot_indices = [(m.start(), m.group(1)) for m in snapshot_pattern.finditer(content)]
snapshot_indices.append((len(content), None))  # sentinel

# Process each snapshot's last map
for i in range(len(snapshot_indices) - 1):
    start_idx, snapshot_time = snapshot_indices[i]
    end_idx, _ = snapshot_indices[i + 1]
    snapshot_body = content[start_idx:end_idx]

    # Find all maps; take only the last one
    maps = map_block_pattern.findall(snapshot_body)
    if not maps:
        continue

    last_map = maps[-1]

    # Parse cluster entries
    for cluster_head_str, members_str in cluster_entry_pattern.findall(last_map):
        cluster_head = int(cluster_head_str)
        members = list(map(int, members_str.strip().split()))

        # Log full cluster under its head for each member
        for member in members:
            record = (cluster_head, members)
            if record not in node_cluster_history[member]:
                node_cluster_history[member].append(record)

# Output per-node cluster history with cluster heads
for node in sorted(node_cluster_history):
    print(f'Node {node}:')
    for i, (cluster_head, cluster_members) in enumerate(node_cluster_history[node], 1):
        print(f'  Cluster {i}: {cluster_head}: {cluster_members}')
    print()

while(True):
    node = input("Node: ")
    print(f'Node {node}:')
    for i, (cluster_head, cluster_members) in enumerate(node_cluster_history[int(node)], 1):
        print(f'  Cluster {i}: {cluster_head}: {cluster_members}')
    print()

