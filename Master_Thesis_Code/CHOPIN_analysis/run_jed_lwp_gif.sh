#!/bin/bash
# $1 is file path
# $2 is the name of the gif to save


save_dir="/scratch/waseem/gif_making"
temp_dir="${save_dir}/gif_images"

if [ -d $temp_dir ]; then
  rm -r $temp_dir
fi

mkdir -p $temp_dir
file_path=$1
gif_name=$2

python /home/waseem/CHOPIN_analysis/jed_liquid_water_column_gif.py --file-path "$file_path" --temp-path "$temp_dir" --gif-save-path "${save_dir}/${gif_name}"