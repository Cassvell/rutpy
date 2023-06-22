#!/bin/bash

download_dir="/home/isaac/snap/firefox/common"
data_dir="$HOME/MEGAsync/datos/gics_obs/2023"
#find $fdir -type f -name "*QRO.csv"
declare -a st=("QRO" "LAV" "RMY" "MZT")

for i in ${!st[@]};do
        if ls ${download_dir}/*${st[$i]}.csv &>/dev/null
        then
        #       if [ -d ${data_dir}/${st[$i]}]; then 
                        mv ${download_dir}/*${st[$i]}.csv ${data_dir}/${st[$i]}
        #       else
        #               mkdir ${data_dir}/${st[$i]}     
        #       fi
                echo "${st[$i]} moved"
        else
                echo "there is no new files from ${st[$i]} station"
        fi

done

echo "preparing gic files..."

#gic_dir="/home/isaac/MEGAsync/datos/gics_obs/2023"

for i in ${!st[@]};do
#	echo "$gic_dir/${st[$i]}"
	for j in "$data_dir/${st[$i]}/*.csv" ;do cat $j >> $data_dir/${st[$i]}/gic_${st[$i]}.output ;done
	awk '{if ($1 ~ /^[2]...-..-../){gsub(/[T]/, " "); gsub(/[Z]/, ""); print} \
	else if ($1 ~ /^"2...-..-../){gsub(/['\"']/, ""); print}}' $data_dir/${st[$i]}/gic_${st[$i]}.output >> $data_dir/${st[$i]}/gic_${st[$i]}.dat
	rm $data_dir/${st[$i]}/gic_${st[$i]}.output
done

echo "done"

#for i in "$gic_dir/*.csv" ;do cat $i >> $gic_dir/gic_$gic_st.output ;done

#awk '{if ($1 ~ /^[2]...-..-../){gsub(/[T]/, " "); gsub(/[Z]/, ""); print} \
#else if ($1 ~ /^"2...-..-../){gsub(/['\"']/, ""); print}}' $gic_dir/gic_$gic_st.output >> $gic_dir/gic_$gic_st.dat 

#rm $gic_dir/gic_$gic_st.output

 
#echo "done" 
