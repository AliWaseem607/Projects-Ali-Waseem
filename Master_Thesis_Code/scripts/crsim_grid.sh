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

L_SUBGRID_file_name=$( echo "$wrfout_d03_filename" | sed -re "s~(.*).nc~\1_LARGE_SUBGRID.nc~" -re "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~~" )

echo "Setting up Radar folders..."
MIRA_results_path="$wrfoutput_dir/crsim_MIRA_grid/"
BASTA_results_path="$wrfoutput_dir/crsim_BASTA_grid/"
mkdir -p $MIRA_results_path
mkdir -p $BASTA_results_path

cp /home/waseem/template_files/crsim/run_grid_sbatch $MIRA_results_path
cp /home/waseem/template_files/crsim/run_grid_sbatch $BASTA_results_path

test=",this,is,that,path"
job_name=${wrfoutput_dir##*/} # ## means to take the shortest string from the end that matches the pattern */
sed -i -e "s~wrfout_path~\"$wrfoutput_dir/$L_SUBGRID_file_name\"~" \
    -e "s~output_path~\"$MIRA_results_path\"~" \
    -e "s~parameters_path~\"/home/waseem/template_files/crsim/PARAMETERS_MIRA_GRID\"~" \
    -e "s~jobname~$job_name-MIRA~" \
    "$MIRA_results_path/run_grid_sbatch"

sed -i -e "s~wrfout_path~\"$wrfoutput_dir/$L_SUBGRID_file_name\"~" \
    -e "s~output_path~\"$BASTA_results_path\"~" \
    -e "s~parameters_path~\"/home/waseem/template_files/crsim/PARAMETERS_BASTA_GRID\"~" \
    -e "s~jobname~$job_name-BASTA~" \
    "$BASTA_results_path/run_grid_sbatch"


echo "Launching Jobs"
cd $MIRA_results_path
sbatch run_grid_sbatch

cd $wrfoutput_dir
cd $BASTA_results_path
sbatch run_grid_sbatch

