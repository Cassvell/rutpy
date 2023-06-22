#!/bin/bash
download_dir="/home/isaac/snap/firefox/common"
data_dir="$HOME/MEGAsync/datos/gics_obs/2023"
#find $fdir -type f -name "*QRO.csv"
declare -a st=("QRO" "LAV" "RMY" "MZT")

for i in ${!st[@]};do
	if ls ${download_dir}/*${st[$i]}.csv &>/dev/null 
	then
	#	if [ -d ${data_dir}/${st[$i]}]; then 
			mv ${download_dir}/*${st[$i]}.csv ${data_dir}/${st[$i]}
	#	else
	#		mkdir ${data_dir}/${st[$i]}	
	#	fi
		echo "${st[$i]} moved"
	else
		echo "there is no new files from ${st[$i]} station"
	fi
	
done

