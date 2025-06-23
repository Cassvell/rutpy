#!/bin/bash

# Check if year argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a year as argument (YYYY format)"
    echo "Usage: $0 YYYY"
    exit 1
fi

year="$1"


data_dir="/home/isaac/datos/gics_obs/${year}/"

declare -a stations=( "QRO" "LAV" "RMY" "MZT")




# Ensure daily directories and process CSV files for each station
for station in "${stations[@]}"; do
    station_dir="${data_dir}${station}"
    daily_dir="${station_dir}/daily"

    # Create daily directory (ignore if already exists)
    if mkdir -p "$daily_dir" 2>/dev/null; then
        echo "Created directory: ${daily_dir}"
    fi

    # Process CSV files
    for csv_file in "${station_dir}"/*.csv; do
        [[ -f "$csv_file" ]] || continue  # Skip if no files match

        echo "Processing: ${csv_file}"
	output_file="${csv_file}.dat"
        echo "${output_file}"
	 #python3 preprocess_gicfiles.py "${year}" "${station}" "${csv_file}"
	 python3 dailifrag_gic_week.py "${year}" "${station}" "${output_file}"
            
    done
done

