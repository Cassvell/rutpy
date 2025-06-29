#!/bin/bash

#download_dir="$HOME/snap/firefox/common"
download_dir2="$HOME/Descargas"

# If no argument is provided, use the current year
if [ -z "$1" ]; then
  year=$(date +"%Y")
else
  year="$1"
fi
echo "${year}"
data_dir="$HOME/datos/gics_obs/$year"

if [[ ! -e $data_dir ]]; then
	mkdir $data_dir
fi	

declare -a stations=( "QRO" "LAV" "RMY" "MZT")

# Process files for each station
for station in "${stations[@]}"; do
#	ls "${download_dir2}/"*${station}.csv 
 # Check in download_dir
	if ls "${download_dir2}/"*${station}.csv; then # > /dev/null; then
		#ls "${download_dir2}/"*${station}.csv
    # Ensure subdirectory exists
   # 		mkdir -p "$data_dir/$station"

    # Move files safely
    		mv "${download_dir2}/"*${station}.csv "$data_dir/$station/"
    		echo "${station} files moved from download_dir2."

  	else
    		echo "No new files for station ${station}."
  	fi
done

echo "preparing gic files..."


#!/bin/bash

# Ensure daily directories and process CSV files for each station
for i in "${!stations[@]}"; do
  station_dir="$data_dir/${stations[$i]}"
  daily_dir="$station_dir/daily"
  echo "$daily_dir"
  # Create the daily directory if it doesn't exist
	if [[ ! -d "$daily_dir" ]]; then
	    mkdir -p "$daily_dir"
  	fi

  # Check if there are any CSV files before proceeding
	  csv_files=("$station_dir"/*.csv)
	  if [[ ! -e "${csv_files[0]}" ]]; then
	    echo "No CSV files found for station ${stations[$i]} in $station_dir."
    		continue
 	fi


# Process last CSV file	 
	    latest_csv=$(ls -t "$station_dir"/*.csv | head -n 1)
		if [[ -n "$latest_csv" ]]; then
		    output_file="${latest_csv}.dat"
		    echo "Processing latest file: ${latest_csv}"
		    python3 preprocess_gicfiles.py "${year}" "${stations[$i]}" "${latest_csv}" "${output_file}"
		else
		    echo "No CSV files found in $station_dir"
		fi

done
echo "done"

echo "generating daily files..."

for i in ${!stations[@]}; do
	python3 dailifrag_gic_week.py $year ${stations[$i]}
done	
echo "done"

echo "enter idate [yyyymmdd]"
echo ">"
read idate

echo "enter fdate[yyyymmdd]"
echo ">"
read fdate

echo "enter geomagnetic station (coe, teo, itu...)"
echo ">"
read geo_stat

echo "ejecutando gráfica de gics"


python3 gic_graph.py $geo_stat $idate $fdate


