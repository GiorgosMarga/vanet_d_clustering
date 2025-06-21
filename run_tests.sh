#!/bin/bash

set -e

# Check for required destination argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <destination_folder>"
  exit 1
fi

SNAPSHOT="$1"
# Run the Go program and redirect output
go run main.go -g=$SNAPSHOT> "output.txt"

# Create destination folder if it doesn't exist
mkdir -p simple

# Move output file and folder into the destination
mv output.txt cars_info simple

rm -rf output_folder

echo "Finished simple"


# Run the Go program and redirect output
go run main.go -g=$SNAPSHOT -rnp=0.5 > "output.txt"

# Create destination folder if it doesn't exist
mkdir -p rnp_0_50

# Move output file and folder into the destination
mv output.txt cars_info rnp_0_50

echo "Finished rnp 0.5"

go run main.go -g=$SNAPSHOT -rnp=0.75 > "output.txt"

# Create destination folder if it doesn't exist
mkdir -p rnp_0_75

# Move output file and folder into the destination
mv output.txt cars_info rnp_0_75

echo "Finished rnp 0.75"

# parseval 

go run main.go -g=$SNAPSHOT -pv2s=5 -pe=200 > "output.txt"

# Create destination folder if it doesn't exist
mkdir -p p_5

# Move output file and folder into the destination
mv output.txt cars_info p_5

echo "Finished p_5."


go run main.go -g=$SNAPSHOT -pv2s=10 -pe=200 > "output.txt"

# Create destination folder if it doesn't exist
mkdir -p p_10

# Move output file and folder into the destination
mv output.txt cars_info p_10

echo "Finished p_10."


go run main.go -g=$SNAPSHOT -pv2s=20 -pe=200 > "output.txt"

# Create destination folder if it doesn't exist
mkdir -p p_20

# Move output file and folder into the destination
mv output.txt cars_info p_20

echo "Finished p_20."

