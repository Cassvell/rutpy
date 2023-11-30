#!/bin/bash

download_dir="$HOME/snap/firefox/common"
download_dir2="$HOME/Descargas"

echo "enter year [yyyy]"

read year
data_dir="$HOME/MEGAsync/datos/gics_obs/$year"

if [[ ! -e $data_dir ]]; then
	mkdir $data_dir
fi	

#find $fdir -type f -name "*QRO.csv"
declare -a st=( "LAV" "QRO" "RMY" "MZT" "MTZ")




for i in ${!st[@]};do
        if ls ${download_dir2}/*${st[$i]}.csv &>/dev/null
        then
               if [[ -e ${data_dir}/${st[$i]} ]]
	       then 
                        mv ${download_dir2}/*${st[$i]}.csv ${data_dir}/${st[$i]}
               else
                       mkdir ${data_dir}/${st[$i]}     
               fi
                echo "${st[$i]} moved"

	elif ls ls ${download_dir}/*${st[$i]}.csv &>/dev/null
	then
		mv ${download_dir}/*${st[$i]}.csv ${data_dir}/${st[$i]}
	 	echo "${st[$i]} moved"	 
        else
                echo "there is no new files from ${st[$i]} station"
        fi

done

echo "preparing gic files..."

#gic_dir="/home/isaac/MEGAsync/datos/gics_obs/$year"

for i in ${!st[@]};do
	if [[ ! -e $data_dir/${st[$i]}/daily ]]; then
		mkdir "$data_dir/${st[$i]}/daily/"
	fi		
#	echo "$gic_dir/${st[$i]}"
	#for j in "$data_dir/${st[$i]}/*.csv" ;do cat $j >> $data_dir/${st[$i]}/gic_${st[$i]}.output ;done
	for j in $data_dir/${st[$i]}/*.csv ;do 
	awk '{if ($1 ~ /^[2]...-..-../){gsub(/[T]/, " "); gsub(/[Z]/, ""); print} \
	else if ($1 ~ /^"2...-..-../){gsub(/['\"']/, ""); print}}' "$j" >> "$j.dat"
	#rm $data_dir/${st[$i]}/gic_${st[$i]}.output	
	done
done

echo "done"

echo "generating daily files..."

for i in ${!st[@]}; do
	python3 dailifrag_gic_week.py $year ${st[$i]}
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


python3 gic_graph.py coe $idate $fdate


