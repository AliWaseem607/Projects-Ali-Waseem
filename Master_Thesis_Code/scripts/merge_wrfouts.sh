# Code to programmaticaly merge two .nc files
# $1 first file (chronologically)
# $2 second file (choronologically)

#######################################################
# Help command
#######################################################
Help () {
    echo "usage: bash merge_wrfouts.sh [Options] wrfout1 wrfout2"
    echo
    echo "wrfout1 must be chronologically earlier than wrfout2"
    echo
    echo "Options"
    echo "  -t, --time-steps    the number of time steps that should be taken in each"
    echo "                      cut of wrfout2 to merge with wrfout1, default is 10"
    echo "  -n, --no-overlap    If the files do not share a time overlap"
    echo
}

#######################################################
# Command line Parsing
#######################################################

time_steps=10
overlap=0
while :; do
    case $1 in 
    -h|--help)
        Help
        exit
        ;;

    -t|--time-steps)
        if [ "$2" ] ; then
            # check if int
            if ! [[ $2 =~ ^[0-9]+$ ]] ; then
                echo 'ERROR: "--time-steps" requires a non-empty integer option argument.'
                exit 1 
            fi
            time_steps=$2
            shift
        else
            echo 'ERROR: "--time-steps" requires a non-empty integer option argument.'
            exit 1
        fi
        shift
        ;;
    -n|--no-overlap)
        overlap=1
        shift
        ;;
    *)
        break
    esac
done

#######################################################
# Main 
#######################################################

expected_args=2
set -e

# Check for correct number of arguments
if [ $# -ne $expected_args ]; then
    echo "Invalid number of arguments."
    echo "Run bash merge_wrfout.sh --help for more information"
    exit 1
fi

# Check that both files exist
if [ ! -e "$1" ] ; then
    echo "ERROR: file: $1 does not exist"
    exit 1
fi

if [ ! -e "$2" ] ; then
    echo "ERROR: file: $2 does not exist"
    exit 1
fi

if [ $overlap -eq 0 ]; then
    first_file_times=$( ncdump -v "Times" "$1" | grep -Eo '^ *"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2}"' )
    second_file_times=$( ncdump -v "Times" "$2" | grep -Eo '^ *"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2}"' )

    first_file_number_times=$( ncdump -h "$1" | grep -E 'Time =.*[0-9]+' | grep -Eo "[0-9]+" )
    first_file_end_time=$( echo $first_file_times | cut -d " " -f $first_file_number_times )

    time_start_line=$( ncdump -v "Times" "$2" | grep -n "Times =" | cut -d ":" -f 1 )
    first_file_end_time_in_second_file_line=$( ncdump -v "Times" "$2" | grep -n $first_file_end_time | cut -d ":" -f 1 )
    num_skip_time_steps=$(( $first_file_end_time_in_second_file_line - $time_start_line - 1 )) # want to get the index to the value of the overlap -1 because we start 1 step before times begin
fi

if [ $overlap -eq 1 ]; then
    num_skip_time_steps=0
fi

second_file_number_times=$( ncdump -h "$2" | grep -E  'Time =.*[0-9]+' | grep -Eo "[0-9]+" )

start=
cut_number=1

echo "Start Time:"
date "+%R"
echo "Cutting file: $2"
echo
for stop in $( eval echo {$num_skip_time_steps..$second_file_number_times..$time_steps} ) ;
do 
    if [[ -n $start ]] ; then
        printf -v cut_id "%02d" $cut_number
        echo "Creating Cut $cut_number from $start to $stop"
        cut_file_name=$( echo "$2" | sed "s~.nc~_cut_${cut_id}_${start}_${stop}.nc~"  )
        ncks -d Time,$start,$stop "$2" "$cut_file_name"
        cut_number=$(( $cut_number + 1 ))
    fi
    start=$(( $stop + 1 ))
    
done

if [[ $stop -lt $second_file_number_times ]] ; then 
    printf -v cut_id "%02d" $cut_number
    cut_file_name=$( echo "$2" | sed "s~.nc~_cut_$cut_id.nc~"  )
    ncks -d Time,$start, "$2" "$cut_file_name"
fi

echo "Combining Files..."

cut_file_pattern=$( echo "$2" | sed "s~.nc~_cut*~" )
output_file_name=$( echo "$1" | sed "s~.nc~_gathered.nc~")
ncrcat "$1" $cut_file_pattern "$output_file_name"
rm $cut_file_pattern