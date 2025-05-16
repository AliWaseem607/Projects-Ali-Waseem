#!/bin/bash

# This script is to launch the radar sim for a wrf simulation

#######################################################
# Help command
#######################################################
Help () {
    echo "usage: bash crsim_runner.sh simulation_directory wrfout_d03_gridcell_filename"
    echo
    echo "Simulation Directory"
    echo "The simulation directory is where the wrf case was run."
    echo "This is usually in the scratch folder."
    echo
    echo "wrfout d03 filename"
    echo "This is the netCDF file that represents one gridcell's worth of data"
    echo "From the WRF simulation."
    echo 
    echo "Options"
    echo "  -h, --help      		Print out help information"
	echo "  -p, --parameters-file	Supply a different PARAMETERS file for crsim"
	echo "  -id, --identifier		identifier to be added to the crsim folder name"
    echo 
}

#######################################################
# Command line Parsing
#######################################################

parametersFile=
id=
while :; do
    case $1 in 
    -h|--help)
        Help
        exit
        ;;
	-p|--parameters-file)
		if [ "$2" ]; then
			parametersFile=$2
			shift
		else
			echo 'ERROR: "--parameters-file" requires a non-empty option argument.'
			exit 1
		fi
		shift
		;;
	-id|--identifier)
		if [ "$2" ]; then
			id=$2
			shift
		else
			echo 'ERROR: "--identifier" requires a non-empty option argument.'
			exit 1
		fi
		shift
		;;
    *) # Anything Else
        break
    esac
done


###############################################################
#   Main Script
###############################################################

# $1 simulation directory
# $2 Name of wrfout file
start=`date +%s.%N`
set -e  # exit on errors

current_working_dir=$( pwd )

# Move to dir to run radar sim
cd "/work/lapi/CRSIM/crsim-3.32/share/crsim/test/"

simulation_dir=$( realpath "$1")

echo "Starting CRSIM simulations"
echo
echo "Making required folders and files"

# Make crsim dir
if [ "$id" ]; then
	id="_$id"
fi

crsim_dir="$simulation_dir/crsim${id}"
output_dir="$crsim_dir/out"

mkdir -p "$crsim_dir"
mkdir -p "$output_dir"
# cp /home/waseem/template_files/crsim/PARAMETERS "$crsim_dir"
if [ -z "${parametersFile}" ]; then
	parametersFile="/home/waseem/template_files/crsim/PARAMETERS"
fi
parametersFileName=$( echo $parametersFile | awk -F / '{print $NF}' ) 

cp "$parametersFile" "$crsim_dir"

binDir="/work/lapi/CRSIM/crsim-3.32/bin"

# Log File
log_OutFile="$crsim_dir/crsim.log"

# Names of the input PARAMETERS file
InputParamFile="$crsim_dir/$parametersFileName"
InputWRFFile="$simulation_dir/$2"
TempWRFFile="$simulation_dir/temp_radar_file.nc"
pattern="Time = .*([0-9]+).*)"

# Extract the number of timesteps from wrfout using ncdump
echo "${InputWRFFile}"
ntimes=$(ncdump -h "${InputWRFFile}" | grep "Time =" | grep -oE "[0-9]+")

echo "Running CRSIM on $InputWRFFile"
echo "PARAMETERS file used is $InputParamFile"
echo "Outputting results to $output_dir"
echo "ntimes = $ntimes"
export OMP_NUM_THREADS=1

# Loop over the timesteps
for ((i=280; i<$ntimes; i++))
do
  echo "Running crsim with i=$i" 
  cp "$InputParamFile" "$crsim_dir/tempfile"
  
  # Replace the value of "it" in PARAMETERS with timestep
  sed -i "s/^it.*/$i/" "$crsim_dir/tempfile"
  
  InputTempFile="$crsim_dir/tempfile"
  
  # Run crsim with the updated PARAMETERS file and output file name
  # ${binDir}/crsim ${InputTempFile} ${InputWRFFile} ${output_dir}/Output$i.nc &> ${log_OutFile}

  # Run on cluster
  # srun -c 1 -t 00:01:00 "${binDir}/crsim" "${InputTempFile}" "${InputWRFFile}" "${output_dir}/Output$i.nc" &> "${log_OutFile}"

  # Run on cpu
  "${binDir}/crsim" "${InputTempFile}" "${InputWRFFile}" "${output_dir}/Output$i.nc" &> "${log_OutFile}"

done

# # run all
# # rm $output_dir/*
# start=`date +%s.%N`
# srun -c 1 -t 00:20:00 /home/waseem/scripts/run_all_crsim.sh "${binDir}/crsim" "${InputParamFile}" "${InputWRFFile}" "${output_dir}" "${log_OutFile}" $crsim_dir $ntimes
# end=`date +%s.%N`
# runtime=$( echo "$end - $start" | bc -l )
# echo "srun script took $runtime"

ExitCode=$?

 if [ "${ExitCode}" -eq 0 ] ; then
     nt_status=0
     echo '-->> CRSIM successfully executed'
     cd "$current_working_dir"
 else
     echo '-->> CRSIM cannot be executed'
     cd "$current_working_dir"
 fi
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "srun BASTA script took $runtime"