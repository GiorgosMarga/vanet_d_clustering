#!/bin/bash

# Check if the snapshot argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <snapshot_name>"
    exit 1
fi
# Assign the snapshot from the first argument
SNAPSHOT=$1

# Define learning rate value
lr=0.01
# Define gap and lap values
gaps=(1 2 5)
laps=(1 2 5)
# Define data folders
# data_folders=("data_elec_usage" "data_humidity" "data_temperatures")
data_folders=("data_humidity" "data_temperatures")

# Loop over each data folder
for data_folder in "${data_folders[@]}"; do
    echo "Processing data folder: $data_folder"
    mkdir -p "results_$data_folder"
    # Loop over each gap value
    for gap in "${gaps[@]}"; do
        # Loop over each lap value
        for lap in "${laps[@]}"; do
            # Loop over each learning rate
            echo "Running experiments with lr=$lr, gap=$gap, lap=$lap in $data_folder"
            # Run the Go program and redirect output for simple configuration
            go run main.go -g=$SNAPSHOT -gap=$gap -lap=$lap -lr=$lr -data=$data_folder > "output.txt"
            mkdir -p results_${data_folder}/anp_gap_${gap}_lap_${lap}
            python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
            mv output.txt cars_info/ results_${data_folder}/anp_gap_${gap}_lap_${lap}
            echo "Finished simple with lr=$lr, gap=$gap, lap=$lap in $data_folder"

            # Run with rnp=0.5
            go run main.go -g=$SNAPSHOT -gap=$gap -lap=$lap -rnp=0.5 -lr=$lr -data=$data_folder > "output.txt"
            mkdir -p results_${data_folder}/rnp_0_50_gap_${gap}_lap_${lap}
            python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
            mv output.txt cars_info results_${data_folder}/rnp_0_50_gap_${gap}_lap_${lap}
            echo "Finished rnp 0.5 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

            # Run with rnp=0.75
            go run main.go -g=$SNAPSHOT -gap=$gap -lap=$lap -rnp=0.75 -lr=$lr -data=$data_folder > "output.txt"
            mkdir -p results_${data_folder}/rnp_0_75_gap_${gap}_lap_${lap}
            python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
            mv output.txt cars_info results_${data_folder}/rnp_0_75_gap_${gap}_lap_${lap}
            echo "Finished rnp 0.75 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

            # Run with pv2s=5, pe=300
            go run main.go -g=$SNAPSHOT -gap=$gap -lap=$lap -pv2s=5 -pe=300 -lr=$lr -data=$data_folder > "output.txt"
            mkdir -p results_${data_folder}/snp_5_gap_${gap}_lap_${lap}
            python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
            mv output.txt cars_info results_${data_folder}/snp_5_gap_${gap}_lap_${lap}
            echo "Finished snp_5 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

            # Run with pv2s=10, pe=300
            go run main.go -g=$SNAPSHOT -gap=$gap -lap=$lap -pv2s=10 -pe=300 -lr=$lr -data=$data_folder > "output.txt"
            mkdir -p results_${data_folder}/snp_10_gap_${gap}_lap_${lap}
            python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
            mv output.txt cars_info results_${data_folder}/snp_10_gap_${gap}_lap_${lap}
            echo "Finished snp_10 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

            # Run with pv2s=20, pe=300
            go run main.go -g=$SNAPSHOT -gap=$gap -lap=$lap -pv2s=20 -pe=300 -lr=$lr -data=$data_folder > "output.txt"
            mkdir -p results_${data_folder}/snp_20_gap_${gap}_lap_${lap}
            python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
            mv output.txt cars_info results_${data_folder}/snp_20_gap_${gap}_lap_${lap}
            echo "Finished snp_20 with lr=$lr, gap=$gap, lap=$lap in $data_folder"
        done
    done
done

echo "All experiments completed."