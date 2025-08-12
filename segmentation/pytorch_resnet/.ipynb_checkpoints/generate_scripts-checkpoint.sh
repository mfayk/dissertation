#!/bin/bash

# Directory to store the generated shell scripts
output_dir="/home/mfaykus/dissertation/segmentation/pytorch_resnet/scripts"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through a range of numbers to generate scripts
for i in {05,06,08,09,11,12}; do
for j in {0005,01}; do
for r in {2,999}; do
for z in {50,10111}; do
  # Define the filename for the new shell script
  filename="$output_dir/script_${i}_${z}_${j}_${r}.sh"

  # Write the new shell script
  cat <<EOF > "$filename"
#!/bin/bash

#SBATCH --job-name train_resnet
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 64gb
#SBATCH --time 48:00:00
#SBATCH --constraint interconnect_hdr

export OMP_NUM_THREADS=8

module load anaconda3

source activate diss

cd /home/mfaykus/dissertation/segmentation/pytorch_resnet

python train.py $i $z $j $r
EOF

  # Make the generated script executable
  chmod +x "$filename"

  # Print a message to indicate the script was created
  echo "Created: $filename"
done
done
done
done

# Notify completion
echo "All scripts have been generated in the \"$output_dir\" folder."
