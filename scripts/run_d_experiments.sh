#!/bin/bash

# Loop through i values 2, 3, and 4
for i in 2 3 4
do
    # Run the Go program with the specified parameter
    go run main.go -d=$i
    
    # Check if cars_info folder exists
    if [ -d "cars_info" ]; then
        # Rename and move the folder to experiments
        mv cars_info xprmnts/epochs_d/epochs_d_$i
        echo "Moved cars_info to xprmnts/epochs_d/epochs_$i"
    else
        echo "Error: cars_info folder not found for i=$i"
    fi
done