#!/bin/bash

gic_dir="/home/isaac/MEGAsync/datos/gics_obs"
declare -a gic_st=("LAV" "RMY" "QRO" "MZT")

for i in ${!gic_st[@]};do
#	echo "$gic_dir/${gic_st[$i]}"
	for j in "$gic_dir/${gic_st[$i]}/*.csv" ;do cat $j >> $gic_dir/${gic_st[$i]}/gic_${gic_st[$i]}.output ;done
	awk '{if ($1 ~ /^[2]...-..-../){gsub(/[T]/, " "); gsub(/[Z]/, ""); print} \
	else if ($1 ~ /^"2...-..-../){gsub(/['\"']/, ""); print}}' $gic_dir/${gic_st[i]}/gic_${gic_st[$i]}.output >> $gic_dir/${gic_st[$i]}/gic_${gic_st[$i]}.dat
	rm $gic_dir/${gic_st[$i]}/gic_${gic_st[$i]}.output
done

echo "done"

#for i in "$gic_dir/*.csv" ;do cat $i >> $gic_dir/gic_$gic_st.output ;done

#awk '{if ($1 ~ /^[2]...-..-../){gsub(/[T]/, " "); gsub(/[Z]/, ""); print} \
#else if ($1 ~ /^"2...-..-../){gsub(/['\"']/, ""); print}}' $gic_dir/gic_$gic_st.output >> $gic_dir/gic_$gic_st.dat 

#rm $gic_dir/gic_$gic_st.output

 
#echo "done" 
