#!/bin/bash

# $1 is wrfout file path
# $2 is the output path
# $3 is the parameters file path
# $4 is number of cpus

wrfout_path=$( realpath $1 )
output_path=$( realpath $2 )
parameters_path=$( realpath $3 )
cpus="$4"  

echo $wrfout_path
echo $output_path
echo $parameters_path 

cpu_string=
if [ -n "$cpus" ] ; then
    echo "Using $cpus cpus"
    cpu_string="--num-cpus $cpus"
fi

start=`date +%s.%N`
python /home/waseem/scripts/crsim_multi_runner.py --wrfout-file-path "$wrfout_path" --output-dir "$output_path" --parameters-file-path "$parameters_path" --quiet $cpu_string
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "srun script took $runtime"