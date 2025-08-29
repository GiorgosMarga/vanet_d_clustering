#!/bin/bash

# Common values
gaps=("gap_1" "gap_2" "gap_5")
laps=("lap_1" "lap_2" "lap_5")

# rnp and snp concentrations
rnp_concs=("0_50" "0_75")
snp_concs=("5" "10" "20")

# === Process ALL groups ===
echo "Processing all anp folders..."
python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "anp"

echo "Processing all rnp folders..."
python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "rnp"

echo "Processing all snp folders..."
python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "snp"

# === Process all anp combinations ===
echo "Plotting for ANP"
for gap in "${gaps[@]}"; do
    echo "Plotting for $gap"
    python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "anp" "$gap"
done
for lap in "${laps[@]}"; do
    echo "Plotting for $lap"
    python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "anp" "$lap"
done 

# === Process all rnp combinations ===
echo "Processing rnp concentrations with all gaps and laps..."
for conc in "${rnp_concs[@]}"; do
    for gap in "${gaps[@]}"; do
        echo "Plotting for RNP_$conc $gap"
        python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "rnp_$conc" "$gap"
    done
    for lap in "${laps[@]}"; do
        echo "Plotting for RNP_$conc $lap"
        python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "rnp_$conc" "$lap"
    done
done

# === Process all snp combinations ===
echo "Processing snp concentrations with all gaps and laps..."
for conc in "${snp_concs[@]}"; do
    for gap in "${gaps[@]}"; do
        echo "Plotting for SNP_$conc $gap"

        python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "snp_$conc" "$gap"
    done
    for lap in "${laps[@]}"; do
        echo "Plotting for SNP_$conc $lap"

        python3 ./python_utils/plotAverageAccuracies.py 250m_cars10_results_data_temperatures "snp_$conc" "$lap"
    done
done