#!/bin/bash

# $1 is wrfout file path
# $2 is the output path
# $3 is the parameters file path

wrfout_path="/scratch/waseem/CHOPIN_nov11-14/wrfout_CHPN_d03_2024-11-11_LARGE_SUBGRID_radar_temp.nc"
output_path="/scratch/waseem/radar_testing/grid_test"
parameters_path="/home/waseem/template_files/crsim/PARAMETERS_MIRA_GRID"  

echo $wrfout_path
echo $output_path
echo $parameters_path 

start=`date +%s.%N`
export OMP_NUM_THREADS=1
bash /home/waseem/scripts/crsim_single_runner.sh -p "$parameters_path" "$output_path" "$wrfout_path" 300
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "srun script took $runtime"
echo $OMP_NUM_THREADS

# rm ${output_path}/*