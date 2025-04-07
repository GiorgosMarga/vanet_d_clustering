#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p graph_images || { echo "Failed to create graph_images directory"; exit 1; }

# Check if graphviz directory exists
if [ ! -d "graphviz" ]; then
    echo "Error: graphviz directory not found in current directory"
    exit 1
fi

# Change to the graphviz directory
cd graphviz || { echo "Failed to change to graphviz directory"; exit 1; }

# Check if dot command exists
if ! command -v dot >/dev/null 2>&1; then
    echo "Error: Graphviz (dot command) not installed"
    exit 1
fi

# Check if any .dot files exist
if ! ls *.dot >/dev/null 2>&1; then
    echo "Error: No cars*.dot files found in graphviz directory"
    exit 1
fi

# Loop through the files
for file in *.dot; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .dot)
        if dot -Tpng "$file" -o "../graph_images/$filename.png"; then
            echo "Successfully generated ../graph_images/$filename.png"
        else
            echo "Failed to generate image for $file"
        fi
    fi
done