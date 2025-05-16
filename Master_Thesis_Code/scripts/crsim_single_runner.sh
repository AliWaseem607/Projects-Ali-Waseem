#!/bin/bash

# This script is to launch the radar sim for a single time step wrf simulation

#######################################################
# Help command
#######################################################
Help () {
    echo "usage: bash crsim_runner.sh output_path wrfout_d03_gridcell_file_path timestep"
    echo
    echo "output directory"
    echo "The directory where the output should be placed."
    echo "This is usually in the scratch folder."
    echo
    echo "wrfout d03 file"
    echo "This is the netCDF file that will be analyzed."
    echo 
    echo "Options"
    echo "  -h, --help      		Print out help information"
    echo "  -q, --quiet             works in quiet mode"
	echo "  -p, --parameters-file	Supply a different PARAMETERS file for crsim"
    echo "  -e, --elevation         Change the elevation of the radar, must have decimal place"
    echo 
}

#######################################################
# Command line Parsing
#######################################################

elevation=
parametersFile=
quiet=
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
    -e|--elevation)
        if [ "$2" ]; then
            elevation=$2
            shift
        else
			echo 'ERROR: "--elevation" requires a non-empty option argument.'
			exit 1
		fi
		shift
		;;
    -q|--quiet)
        quiet=0
        shift
        ;;   
    *) # Anything Else
        break
    esac
done


###############################################################
#   Main Script
###############################################################

# $1 output directory
# $2 Path of wrfout file
# $3 timestep

set -e  # exit on errors

current_working_dir=$( pwd )

# Move to dir to run radar sim
cd "/work/lapi/CRSIM/crsim-3.32/share/crsim/test/"

expected_args=3

if [ $# -ne $expected_args ]; then
    echo "Invalid number of arguments."
    echo "Run bash crsim_single_runner --help for more information"
    exit 1
fi

output_dir=$( realpath "$1")
InputWRFFile=$( realpath "$2")
timestep=$3

mkdir -p "$output_dir"

if [ -z "${parametersFile}" ]; then
	parametersFile="/home/waseem/template_files/crsim/PARAMETERS"
fi
parametersFileName=$( echo $parametersFile | awk -F / '{print $NF}' ) 

# Names of the input PARAMETERS file
InputParamFile="$output_dir/$parametersFileName"

if [ ! -f $InputParamFile ]; then
    cp -f "$parametersFile" "$output_dir"
fi



binDir="/work/lapi/CRSIM/crsim-3.32/bin"

# Log File
log_OutFile="$output_dir/crsim${timestep}.log"



pattern="Time = .*([0-9]+).*)"

# Extract the number of timesteps from wrfout using ncdump
if [ -z $quiet ]; then
    echo "${InputWRFFile}"
fi
ntimes=$(ncdump -h "${InputWRFFile}" | grep "Time =" | grep -oE "[0-9]+")

if [ -z $quiet ]; then
    echo "Running CRSIM on $InputWRFFile"
    echo "PARAMETERS file used is $InputParamFile"
    echo "Outputting results to $output_dir"
fi

export OMP_NUM_THREADS=1

if [ -z $quiet ]; then
    echo "Running crsim with i=$timestep" 
fi

cp "$InputParamFile" "$output_dir/tempfile${timestep}"

# Replace the value of "it" in PARAMETERS with timestep
sed -i "s/^it.*/$timestep/" "$output_dir/tempfile${timestep}"
# adjust the elevation
if [ -n "$elevation" ]; then
	sed -i "s/[0-9.]*d0 *# elevation/${elevation}d0   # elevation/" "$InputParamFile"
fi

InputTempFile="$output_dir/tempfile${timestep}"

# Run crsim with the updated PARAMETERS file and output file name
# ${binDir}/crsim ${InputTempFile} ${InputWRFFile} ${output_dir}/Output$timestep.nc &> ${log_OutFile}

# Run on cluster
# srun -c 1 -t 00:01:00 "${binDir}/crsim" "${InputTempFile}" "${InputWRFFile}" "${output_dir}/Output$timestep.nc" &> "${log_OutFile}"

# Run on cpu
start=`date +%s.%N`
"${binDir}/crsim" "${InputTempFile}" "${InputWRFFile}" "${output_dir}/Output$timestep.nc" &> "${log_OutFile}"
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )

if [ -z $quiet ]; then
    echo "crsim with i=$timestep took $runtime seconds" 
fi

# clean up
rm "$output_dir/tempfile${timestep}"

ExitCode=$?

 if [ "${ExitCode}" -eq 0 ] ; then
     nt_status=0
     if [ -z $quiet ]; then 
        echo '-->> CRSIM successfully executed'
     fi
     rm "${log_OutFile}"
     cd "$current_working_dir"
 else
     if [ -z $quiet ]; then
        echo '-->> CRSIM cannot be executed'
     fi
     cd "$current_working_dir"
 fi
