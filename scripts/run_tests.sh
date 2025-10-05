#!/bin/bash

# Define snapshots as an array
#
# "snapshots/cars_80_200m"
snapshots=("snapshots/cars_20_200m" "snapshots/cars_20_400m" "snapshots/cars_40_200m" "snapshots/cars_60_200m" "snapshots/cars_60_400m" "snapshots/cars_80_200m" "snapshots/cars_80_400m")
# snapshots=("snapshots/cars_40_400m" "snapshots/cars_80_400m" "snapshots/cars_80_200m")
# Define learning rate value
lr=0.01

# Define gap and lap values
gaps=(1 2 5)
laps=(1 2 5)

# Define data folders
data_folders=("data_temperatures")
# data_folders=("data_humidity" "data_temperatures")

# Loop over each snapshot
for SNAPSHOT in "${snapshots[@]}"; do
    echo "Processing snapshot: $SNAPSHOT"

    # Loop over each data folder
    for data_folder in "${data_folders[@]}"; do
        folder_name="results_${data_folder}_${SNAPSHOT##*/}"
        echo "Processing data folder: $folder_name"
        mkdir -p "$folder_name"

        # Loop over each gap value
        for gap in "${gaps[@]}"; do
            # Loop over each lap value
            for lap in "${laps[@]}"; do
                if (( lap > gap )); then
                  echo "Skipping lap=$lap since lap <= gap ($gap)"
                  continue
                fi

                echo "Running experiments with lr=$lr, gap=$gap, lap=$lap in $data_folder"

                # Run the Go program and redirect output for simple configuration
                go run main.go -g="$SNAPSHOT" -gap="$gap" -lap="$lap" -lr="$lr" -data="$data_folder" > "output.txt"
                mkdir -p "$folder_name/anp_gap_${gap}_lap_${lap}"
                python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
                mv output.txt cars_info "$folder_name/anp_gap_${gap}_lap_${lap}"
                echo "Finished simple with lr=$lr, gap=$gap, lap=$lap in $data_folder"

                # Run with rnp=0.5
                go run main.go -g="$SNAPSHOT" -gap="$gap" -lap="$lap" -rnp=0.5 -lr="$lr" -data="$data_folder" > "output.txt"
                mkdir -p "$folder_name/rnp_0_50_gap_${gap}_lap_${lap}"
                python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
                mv output.txt cars_info "$folder_name/rnp_0_50_gap_${gap}_lap_${lap}"
                echo "Finished rnp 0.5 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

                # Run with rnp=0.75
                go run main.go -g="$SNAPSHOT" -gap="$gap" -lap="$lap" -rnp=0.75 -lr="$lr" -data="$data_folder" > "output.txt"
                mkdir -p "$folder_name/rnp_0_75_gap_${gap}_lap_${lap}"
                python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
                mv output.txt cars_info "$folder_name/rnp_0_75_gap_${gap}_lap_${lap}"
                echo "Finished rnp 0.75 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

                # Run with pv2s=5, pe=300
                go run main.go -g="$SNAPSHOT" -gap="$gap" -lap="$lap" -pv2s=5 -pe=300 -lr="$lr" -data="$data_folder" > "output.txt"
                mkdir -p "$folder_name/snp_5_gap_${gap}_lap_${lap}"
                python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
                mv output.txt cars_info "$folder_name/snp_5_gap_${gap}_lap_${lap}"
                echo "Finished snp_5 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

                # Run with pv2s=10, pe=300
                go run main.go -g="$SNAPSHOT" -gap="$gap" -lap="$lap" -pv2s=10 -pe=300 -lr="$lr" -data="$data_folder" > "output.txt"
                mkdir -p "$folder_name/snp_10_gap_${gap}_lap_${lap}"
                python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
                mv output.txt cars_info "$folder_name/snp_10_gap_${gap}_lap_${lap}"
                echo "Finished snp_10 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

                # Run with pv2s=20, pe=300
                go run main.go -g="$SNAPSHOT" -gap="$gap" -lap="$lap" -pv2s=20 -pe=300 -lr="$lr" -data="$data_folder" > "output.txt"
                mkdir -p "$folder_name/snp_20_gap_${gap}_lap_${lap}"
                python3 ./python_utils/cleanCarsInfo.py cars_info/ cars_info/
                mv output.txt cars_info "$folder_name/snp_20_gap_${gap}_lap_${lap}"
                echo "Finished snp_20 with lr=$lr, gap=$gap, lap=$lap in $data_folder"

            done
        done
    done
done

echo "All experiments completed."
