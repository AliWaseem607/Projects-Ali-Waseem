# !/bin/bash

# This is a bash script designed to set up a directory 
# Inputs: 
#   1.  simulation_dir: Location where the simulation will be run
#       Usually /scratch/<username>/<new_folder_name>
#   2. wrfout_name this is used if a wrfout needs to be specified

#######################################################
# Help command
#######################################################
Help () {
    echo "usage: bash post_process.sh simulation_directory [wrfout d03 filename]"
    echo
    echo "Simulation Directory"
    echo "The simulation directory is where the wrf case was run."
    echo "This is usually in the scratch folder."
    echo
    echo "wrfout d03 filename"
    echo "Optional argument to directly specify a file for the wrfout d03 results" 
    echo "The default is to obtain the d03 domain from the naming pattern in the"
    echo "namelist.input file. NOTE that this must NOT be a path"
    echo 
    echo "Options"
    echo "  -h, --help      Print out help information"
    echo "  -SIP            Add that SIP used in sim"
    echo "  -nSIP           Add that no SIP where used in sim"
    echo 
}

#######################################################
# Function Declarations 
#######################################################
get_namelist_variable () {
    # requires that the working directory is the 
    # wrf output directory
    # $1 : name of variable to be matched
    local variable=$(
        grep --max-count=1 "$1" namelist.input | 
        sed -r "s/$1 *= *(.*,)+/\1/" | # Find the value after the equals sign
        # remove quotes and commas, the g at the end means global replacement
        sed -e "s/'//g" -e 's/"//g' -re "s/, +/ /g" -e "s/,/ /g" |
        sed -r "s/^ //" # Remove leading space if any
    )

    echo "$variable"
}

get_nth_value() {
    # utility function to treat strings with spaces as lists
    # it is able to take the string from std in or command line
    # $1 : string to parse
    # $2 : index to be retrieved, starts at 1

    # Check for too many inputs
    if (( ${#} >2 )) ;  then
        echo "Too many inputs have been given"
        exit 1
    fi

    # assign string_to_search and index
    if (( ${#} < 2 )) ; then    # if there are less than two outputs get string from std in

        if [ -p /dev/stdin ]; then # if data was piped
            read in
            local string_to_search="$in"
            local index=$1
        else
            echo "Not enough inputs given"
            exit 1
        fi

    else                        # Else the string has been provided
        local string_to_search=$1
        local index=$2
    fi

    
    # obtain value
    echo $( echo "$string_to_search" | cut -d " " -f $index )
}

error_message() {
    echo "Invalid number of arguments"
    echo "run with '--help' for more information."

}

#######################################################
# Command line Parsing
#######################################################
SIP=
nSIP=
while :; do
    case $1 in 
    -h|--help)
        Help
        exit
        ;;
    -SIP)
        SIP=1
        shift
        ;;
    -nSIP)
        nSIP=1
        shift
        ;;
    *) # Anything else
        break
    esac
done

if [[ -n $SIP ]] && [[ -n $nSIP ]] ; then 
    echo "ERROR: cannot have both SIP and nSIP flags"
    exit
fi

#######################################################
# Main 
#######################################################

set -e
shopt -s extglob

wrfout_d03_filename=
maximum_args=2
minimum_args=1

# Check for correct number of arguments
if [ $# -lt $minimum_args ]; then
    error_message
    exit 1
fi

if [ $# -gt $maximum_args ]; then
    error_message
    exit 1
fi

wrfoutput_dir=$( realpath $1 )
# Check to see that the wrfoutput_dir exists
if [ ! -d $wrfoutput_dir ]; then
    echo "ERROR: folder $wrfoutput_dir does not exist"
    exit 1
fi

current_dir=$( pwd ) 

# Check if wrfout file name was given
if [ -n "$2" ] ; then
    wrfout_d03_filename=$2
fi

# move to wrf output dir
cd $wrfoutput_dir

wrfout_naming_pattern=$( get_namelist_variable "history_outname" )

# obtain $wrfout_d03_pattern_name
if [ -n "$wrfout_d03_filename" ] ; then
    # Get date data from file name if possible
    echo "Using File: $wrfout_d03_filename"
    start_year=$( echo $wrfout_d03_filename | sed -E "s~.*([0-9]{4})-[0-9]{2}-[0-9]{2}.*~\1~" )
    start_month=$( echo $wrfout_d03_filename | sed -E "s~.*[0-9]{4}-([0-9]{2})-[0-9]{2}.*~\1~" )
    start_day=$( echo $wrfout_d03_filename | sed -E "s~.*[0-9]{4}-[0-9]{2}-([0-9]{2}).*~\1~" )
else
    # Read namelist input for pattern
    echo "Obtaining data from namelist.input..."
    start_year=$( get_namelist_variable "start_year" | get_nth_value 3 )
    start_month=$( get_namelist_variable "start_month" | get_nth_value 3)
    start_day=$( get_namelist_variable "start_day" | get_nth_value 3)
    start_hour=$( get_namelist_variable "start_hour" | get_nth_value 3)
    start_minute=$( get_namelist_variable "start_minute" | get_nth_value 3)
    start_second=$( get_namelist_variable "start_second" | get_nth_value 3)

    wrfout_d03_pattern=$( 
        echo $wrfout_naming_pattern | 
        sed "s/<domain>/03/" |
        sed "s/<date>/${start_year}-${start_month}-${start_day}_${start_hour}:${start_minute}:${start_second}/" |
        sed "s~./~~"
    )
    wrfout_d03_filename=$( ls "$wrfoutput_dir" | grep "$wrfout_d03_pattern" )
fi

# check to see if wrfout_d03_filename exists
if [ ! -e "$wrfout_d03_filename" ]; then
    echo "ERROR: invalid wrfout d03 file"
    echo "found/given name: $wrfout_d03_filename"
    echo "${wrfoutput_dir}/${wrfout_d03_filename} does not exist."
    exit 1
fi
echo $wrfout_d03_filename
echo "Adding metadata to d03"

# Check if SIP is already in file:
SIP_attr_in_file=$( ncdump -h ${wrfoutput_dir}/${wrfout_d03_filename} | ( grep :SIP || echo "" ) )
if [[ -z $SIP_attr_in_file ]] ; then 
    if [[ -n $SIP ]] ; then
        ncatted "${wrfout_d03_filename}" -a SIP,global,c,i,1 -a demott,global,c,i,0 -a hallett_mossop,global,c,i,1 -a breakup,global,c,i,1 -a breakup_type,global,c,c,"Phillips" -a dropshatter,global,c,i,1 -a sublbreakup,global,c,i,1
    fi
    if [[ -n $nSIP ]] ; then
        ncatted "${wrfout_d03_filename}" -a SIP,global,c,i,0
    fi
else
    SIP_code=$(echo $SIP_attr_in_file | sed -E "s~ *:(.*);~\1~")
    echo "SIP code was already found in file"
    echo $SIP_code
    echo 
fi

L_SUBGRID_file_name=$( echo "$wrfout_d03_filename" | sed -re "s~(.*).nc~\1_LARGE_SUBGRID.nc~" -re "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~~" )
SUBGRID_file_name=$( echo "$wrfout_d03_filename" | sed -re "s~(.*).nc~\1_SUBGRID.nc~" -re "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~~" )
NPRK_out_file_name=$( echo "$wrfout_d03_filename" | sed -re "s~(.*).nc~\1_NPRK.nc~" -re "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~~" )
SPRK_out_file_name=$( echo "$wrfout_d03_filename" | sed -re "s~(.*).nc~\1_SPRK.nc~" -re "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~~" )
HAC_out_file_name=$( echo "$wrfout_d03_filename" | sed -re "s~(.*).nc~\1_HAC.nc~" -re "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~~" )

echo "Extracting data for radar locations..."
echo "Extracting LARGE SUBGRID"
# extract a sub grid to make further extractions faster
ncks -d south_north,163,171 -d south_north_stag,163,172 -d west_east,148,156 -d west_east_stag,148,157  "$wrfout_d03_filename" "$L_SUBGRID_file_name"
echo "SUBGRID data extracted to $wrfoutput_dir/$L_SUBGRID_file_name"

echo "Extracting smaller subgrid"
ncks -d south_north,1,4  -d south_north_stag,1,5 -d west_east,4,5 -d west_east_stag,4,6  --overwrite "$L_SUBGRID_file_name" "$SUBGRID_file_name"
echo "SUBGRID data extracted to $wrfoutput_dir/$SUBGRID_file_name"

# Note that double inputs are kept below so that the dimension is not removed
# extract HAC locations: -d south_north,164,164 -d west_east,153,153 -d west_east_stag,153 -d south_north_stag,165
echo "Extracting HAC" 
ncks -d south_north,0,0 -d south_north_stag,0,1 -d west_east,1,1 -d west_east_stag,1,2  --overwrite "$SUBGRID_file_name" "$HAC_out_file_name"
echo "HAC data extracted to $wrfoutput_dir/$HAC_out_file_name"

# extract SPRK locations: -d south_north,167,167 -d west_east,153,153 -d west_east_stag,153 -d south_north_stag,167
echo "Extracting SPRK"
ncks -d south_north,3,3 -d south_north_stag,3,4 -d west_east,1,1 -d west_east_stag,1,2  --overwrite "$SUBGRID_file_name" "$SPRK_out_file_name"
echo "SPRK data extracted to $wrfoutput_dir/$SPRK_out_file_name"

# extract NPRK locations: -d south_north,167,167 -d west_east,152,152 -d west_east_stag,153 -d south_north_stag,167
echo "Extracting NPRK"
ncks -d south_north,3,3 -d south_north_stag,3,4 -d west_east,0,0 -d west_east_stag,0,1  --overwrite "$SUBGRID_file_name" "$NPRK_out_file_name"
echo "NPRK data extracted to $wrfoutput_dir/$NPRK_out_file_name"

if [[ -d "$wrfoutput_dir/crsim_MIRA" ]] ; then
    rm -r "$wrfoutput_dir/crsim_MIRA"
fi

if [[ -d "$wrfoutput_dir/crsim_BASTA" ]] ; then
    rm -r "$wrfoutput_dir/crsim_BASTA"
fi

echo "Starting Radar Analysis"
mkdir -p $wrfoutput_dir/crsim_MIRA
mkdir -p $wrfoutput_dir/crsim_BASTA

# extract subsection for radar
echo "Extracting Radar temporary file"
radar_ncfile_name=$(echo "$NPRK_out_file_name" | sed "s~.nc~_radar_tmp.nc~")
ncks -v RDX,RDY,XLAT,XLONG,U,V,W,PB,P,T,PHB,PH,QVAPOR,QCLOUD,QRAIN,QICE,QSNOW,QGRAUP,QNRAIN,QNICE,QNSNOW,Times "$NPRK_out_file_name" "$radar_ncfile_name"

echo "Running Radar Simulations..."
echo "Running MIRA..."

# srun -c 1 -t 00:20:00 /home/waseem/scripts/crsim_runner.sh -p "/home/waseem/template_files/crsim/PARAMETERS_MIRA" -id "MIRA" "$wrfoutput_dir" "$radar_ncfile_name" # %> "crsim_MIRA/crsim_runner.log"
MIRA_results_path="$wrfoutput_dir/crsim_MIRA/"
mkdir -p $MIRA_results_path
srun -c 15 -t 00:08:00 /home/waseem/scripts/run_crsim_multi_runner.sh "$wrfoutput_dir/$radar_ncfile_name" "$MIRA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_MIRA"

echo "Running BASTA..."
# srun -c 1 -t 00:20:00 /home/waseem/scripts/crsim_runner.sh -p "/home/waseem/template_files/crsim/PARAMETERS_BASTA" -id "BASTA" "$wrfoutput_dir" "$radar_ncfile_name" # %> "crsim_BASTA/crsim_runner.log"
BASTA_results_path="$wrfoutput_dir/crsim_BASTA/"
mkdir -p $BASTA_results_path
srun -c 15 -t 00:05:00 /home/waseem/scripts/run_crsim_multi_runner.sh "$wrfoutput_dir/$radar_ncfile_name" "$BASTA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_BASTA"


echo "Adding reflectivites to NPRK..."
python /home/waseem/scripts/add_reflectivities.py --BASTA-results-path "$BASTA_results_path" --MIRA-results-path "$MIRA_results_path" --wrfout-path "$NPRK_out_file_name"


echo "Removing temporary files"
rm "$wrfoutput_dir/$radar_ncfile_name"

# echo "Begin compression of wrf results..."

# declare -a domains=("01" "02" "03")
# for domain in "${domains[@]}" 
# do  
#     mkdir -p "download_data/d${domain}_zipped"
#     wrfout_zip_pattern=$( 
#         echo $wrfout_naming_pattern | 
#         sed "s/<domain>/${domain}/" | # remove domain keyword
#         sed "s/<date>/\*/"    # replace date keyword with 20* so we can glob them
#     )
#     for file in $wrfout_zip_pattern 
#     do 
#         zipped_path=$(
#             echo "download_data/d${domain}_zipped/$file" | 
#             sed -r "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~~" | # get rid of the hour, minute, seconds
#             sed "s~\./~~" | # remove the ./ directory specifier
#             sed "s/\.nc/\.zip/" # replace the extension with .zip
#         )
        
#         zip -s 5g "$zipped_path" "$file"
#     done
# done

# echo "Begin compression of radar results..."
# zip -q "download_data/crsim_outputs.zip" crsim*/out/*

# cp crsim*/*.npy "download_data/"
# cp "namelist.input" "download_data/"


echo "Finished Post Processing"
