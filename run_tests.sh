#!/bin/bash

# Check if the snapshot argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <snapshot_name>"
    exit 1
fi

# Assign the snapshot from the first argument
SNAPSHOT=$1
# Define learning rate values
learning_rates=(0.001)

# Loop over each learning rate
for lr in "${learning_rates[@]}"; do
    echo "Running experiments with lr=$lr"

    # Run the Go program and redirect output
    go run main.go -g=$SNAPSHOT -lr=$lr > "output.txt"
    mkdir -p simple_lr_$lr
    mv output.txt cars_info simple_lr_$lr
    rm -rf output_folder
    echo "Finished simple with lr=$lr"

    go run main.go -g=$SNAPSHOT -rnp=0.5 -lr=$lr > "output.txt"
    mkdir -p rnp_0_50_lr_$lr
    mv output.txt cars_info rnp_0_50_lr_$lr
    echo "Finished rnp 0.5 with lr=$lr"

    go run main.go -g=$SNAPSHOT -rnp=0.75 -lr=$lr > "output.txt"
    mkdir -p rnp_0_75_lr_$lr
    mv output.txt cars_info rnp_0_75_lr_$lr
    echo "Finished rnp 0.75 with lr=$lr"

    go run main.go -g=$SNAPSHOT -pv2s=5 -pe=200 -lr=$lr > "output.txt"
    mkdir -p p_5_lr_$lr
    mv output.txt cars_info p_5_lr_$lr
    echo "Finished p_5 with lr=$lr"

    go run main.go -g=$SNAPSHOT -pv2s=10 -pe=200 -lr=$lr > "output.txt"
    mkdir -p p_10_lr_$lr
    mv output.txt cars_info p_10_lr_$lr
    echo "Finished p_10 with lr=$lr"

    go run main.go -g=$SNAPSHOT -pv2s=20 -pe=200 -lr=$lr > "output.txt"
    mkdir -p p_20_lr_$lr
    mv output.txt cars_info p_20_lr_$lr
    echo "Finished p_20 with lr=$lr"

done

echo "All experiments completed."