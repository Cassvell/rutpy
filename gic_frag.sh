
year=2023
data_dir="$HOME/MEGAsync/datos/gics_obs/$year/LAV"
echo $data_dir
if [[ ! -e $data_dir ]]; then
        mkdir $data_dir
fi


echo "preparing gic files..."

#gic_dir="/home/isaac/MEGAsync/datos/gics_obs/$year"

#for i in ${!st[@]};do
#       echo "$gic_dir/${st[$i]}"
        for j in $data_dir/${st[$i]}/*.csv ;do 
	
        awk '{if ($1 ~ /^[2]...-..-../){gsub(/[T]/, " "); gsub(/[Z]/, ""); print} \
        else if ($1 ~ /^"2...-..-../){gsub(/['\"']/, ""); print}}' "$j" >> "$j.dat"
        
	done
#done

echo "done"

