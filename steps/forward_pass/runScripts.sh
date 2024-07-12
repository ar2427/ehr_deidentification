#!/bin/bash

# Define the file paths
file_path="thresholds.txt"
nohup_file_path="nohup.out"

# Check if the file exists
if [[ ! -f "$file_path" ]]; then
  echo "File not found!" >> "$nohup_file_path"
  exit 1
fi

# Log the start time
echo "Script started at $(date)" >> "$nohup_file_path"

# Read the file line by line
while IFS= read -r line; do
  # Skip lines that start with '#'
  if [[ $line == \#* ]]; then
    continue
  fi

  # Form commands to run
  command_to_run_part_one_of_three="python forwardPassCustomInput.py /prj0124_gpu/akr4007/data/currently_relevant_data/full_csv_most_relevant_decoded_encoded_512_token_length_notes_per_person_part_one_of_three.csv ./run/i2b2/predict_i2b2_with_threshold_max.json False $line"
  command_to_run_part_two_of_three="python forwardPassCustomInput.py /prj0124_gpu/akr4007/data/currently_relevant_data/full_csv_most_relevant_decoded_encoded_512_token_length_notes_per_person_part_two_of_three.csv ./run/i2b2/predict_i2b2_with_threshold_max.json False $line"
  command_to_run_part_three_of_three="python forwardPassCustomInput.py /prj0124_gpu/akr4007/data/currently_relevant_data/full_csv_most_relevant_decoded_encoded_512_token_length_notes_per_person_part_three_of_three.csv ./run/i2b2/predict_i2b2_with_threshold_max.json False $line"

  # Run the commands and redirect their output to nohup file
  echo "Command Running: $command_to_run_part_one_of_three" >> "$nohup_file_path"
  $command_to_run_part_one_of_three >> "$nohup_file_path" 2>&1
  echo "Finished command: $command_to_run_part_one_of_three" >> "$nohup_file_path"

  echo "Command Running: $command_to_run_part_two_of_three" >> "$nohup_file_path"
  $command_to_run_part_two_of_three >> "$nohup_file_path" 2>&1
  echo "Finished command: $command_to_run_part_two_of_three" >> "$nohup_file_path"

  echo "Command Running: $command_to_run_part_three_of_three" >> "$nohup_file_path"
  $command_to_run_part_three_of_three >> "$nohup_file_path" 2>&1
  echo "Finished command: $command_to_run_part_three_of_three" >> "$nohup_file_path"
done < "$file_path"

# Log the end time
echo "Script finished at $(date)" >> "$nohup_file_path"
