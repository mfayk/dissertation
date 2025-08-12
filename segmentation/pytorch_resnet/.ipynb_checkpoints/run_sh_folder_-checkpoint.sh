#!/bin/bash

# Path to the folder containing shell scripts
script_folder="/home/mfaykus/dissertation/segmentation/pytorch_resnet/scripts"

# Change to the directory containing the scripts
cd "$script_folder" || { echo "Directory not found: $script_folder"; exit 1; }

# Loop through all .sh files in the folder
for script in *.sh; do
    if [[ -f "$script" ]]; then
        echo "Running $script..."
        sbatch "$script"
        echo "Finished running $script."
        echo
    else
        echo "No .sh files found in $script_folder."
    fi
done

echo "All scripts in the folder have been executed."