# !/bin/bash

# This is a bash script designed to set up a directory 
# Inputs: 
#   1.  simulation_dir: Location where the simulation will be run
#       Usually /scratch/<username>/<new_folder_name>
#   2.  setup_dir: This is a directory that contains the WPS and WRF directorys


#######################################################
# Help command
#######################################################
Help () {
    echo "usage: bash set_up_sim_dir.sh [Options] simulation_directory setup_dictory"
    echo
    echo "Simulation Directory"
    echo "The simulation directory is where the wrf case will be run."
    echo "This is usually in the scratch folder."
    echo
    echo "Setup Directory"
    echo "The setup directory is the folder that contains the WPS and WRF"
    echo "directories that will be used to run the analysis."
    echo 
    echo "Options"
    echo "  -st, --start-time  time to start simulation provided as: YYYY-MM-DD_hh:mm:ss"
    echo "  -et, --end-time    time to end simulation provided as YYYY-MM-DD_hh:mm:ss"
    echo "  NOTE: start and end time should be provided together if used"
    echo "  -id                This is the the wrfout will have, default value is CHPN"
    echo "  --job-id           This is an id that will be added to jobs launched on SCITAS"
    echo
    echo ""
}

#######################################################
# Function Declarations 
#######################################################
Finish_Instructions () {
    echo "Final steps are to:"
    echo "    1. copy a namelist.input file into the directory and adjust as needed if start and end time not specified"
    echo "    2. run real.exe by running command 'sbatch run_real_JED'"
    echo "    3. After job has finished check that you have created a wrfbdy file, and n wrfinput and wrflow files with n the number of domains"
    echo "    4. run wrf.exe by running command 'sbatch run_wrf_JED'"
    echo "Good luck on your simulations!!!"
}

Finish_Instructions_2 () {
    echo "Final steps are to:"
    echo "    1. wait for run_real_JED to finish"
    echo "    2. check that run_real_JED completed successfully and that you have created a wrfbdy file, and n wrfinput and wrflow files with n the number of domains"
    echo "    3. run wrf.exe by running command 'sbatch run_wrf_JED'"
    echo "Good luck on your simulations!!!"
}

check_date () {   
    if ! [[ "$1" =~ [0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2} ]] ; then 
        echo 1
        return 0
    fi

    local month=$( echo $1 | grep -Po "[0-9]{4}-\K([0-9]{2})" | sed -r "s/^0//")
    local day=$( echo $1 | grep -Po "[0-9]{4}-[0-9]{2}-\K([0-9]{2})" | sed -r "s/^0//")
    local hour=$( echo $1 | grep -Po "[0-9]{4}-[0-9]{2}-[0-9]{2}_\K([0-9]{2})" | sed -r "s/^0//")
    local minute=$( echo $1 | grep -Po "[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:\K([0-9]{2})" | sed -r "s/^0//")
    local second=$( echo $1 | grep -Po "[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:\K([0-9]{2})" | sed -r "s/^0//")

    if [[ $month -gt 12 ]] ; then
        echo 1
        return 0
    elif [[ $day -gt 31 ]] ; then
        echo 1
        return 0
    elif [[ $hour -gt 23 ]] ; then
        echo 1
        return 0
    elif [[ $minute -gt 59 ]] ; then
        echo 1
        return 0
    elif [[ $second -gt 59 ]] ; then
        echo 1
        return 0
    fi

    echo 0
    return 0
}

copy_and_set_namelist_input() {
    # $1 simulation directory
    # $2 start date
    # $3 end date
    # $4 save identifier
    #
    # dates must be in form YYYY-MM-DD_hh:mm:ss

    local simulation_dir=$1
    local start_date=$2
    local end_date=$3
    local save_id=$4

    cp /home/waseem/template_files/namelist.input "$simulation_dir"
    
    local namelist_path="${simulation_dir}/namelist.input"
    
    # Parse through date
    local start_year=$( echo "$start_date" | cut -d "-" -f 1)
    local start_month=$( echo "$start_date" | cut -d "-" -f 2)
    local start_day=$( echo "$start_date" | cut -d "-" -f 3 | cut -d "_" -f 1)
    local start_hour=$( echo "$start_date" | cut -d ":" -f 1 | cut -d "_" -f 2)
    local start_minute=$( echo "$start_date" | cut -d ":" -f 2 )
    local start_second=$( echo "$start_date" | cut -d ":" -f 3 )

    local end_year=$( echo "$end_date" | cut -d "-" -f 1)
    local end_month=$( echo "$end_date" | cut -d "-" -f 2)
    local end_day=$( echo "$end_date" | cut -d "-" -f 3 | cut -d "_" -f 1)
    local end_hour=$( echo "$end_date" | cut -d ":" -f 1 | cut -d "_" -f 2)
    local end_minute=$( echo "$end_date" | cut -d ":" -f 2 )
    local end_second=$( echo "$end_date" | cut -d ":" -f 3 )

    # find runtime hours
    local start_date_datum=$( date -d "$start_year-$start_month-$start_day $start_hour:$start_minute:$start_second" "+%s" )
    local end_date_datum=$( date -d "$end_year-$end_month-$end_day $end_hour:$end_minute:$end_second" "+%s" )
    local runtime_hours=$(( ($end_date_datum - $start_date_datum)/(60*60) ))

    # set save path
    sed -Ei "s~(history_outname *= *) .*,~\1 './wrfout_${save_id}_d<domain>_<date>.nc',~" "$namelist_path"

    # set start time variables
    sed -Ei "s/(start_year *= *).*/\1$start_year, $start_year, $start_year,/" "$namelist_path"
    sed -Ei "s/(start_month *= *).*/\1$start_month,   $start_month,   $start_month,/" "$namelist_path"
    sed -Ei "s/(start_day *= *).*/\1$start_day,   $start_day,   $start_day,/" "$namelist_path"
    sed -Ei "s/(start_hour *= *).*/\1$start_hour,   $start_hour,   $start_hour,/" "$namelist_path"
    sed -Ei "s/(start_minute *= *).*/\1$start_minute,   $start_minute,   $start_minute,/" "$namelist_path"
    sed -Ei "s/(start_second *= *).*/\1$start_second,   $start_second,   $start_second,/" "$namelist_path"

    # set end time variables
    sed -Ei "s/(end_year *= *).*/\1$end_year, $end_year, $end_year,/" "$namelist_path"
    sed -Ei "s/(end_month *= *).*/\1$end_month,   $end_month,   $end_month,/" "$namelist_path"
    sed -Ei "s/(end_day *= *).*/\1$end_day,   $end_day,   $end_day,/" "$namelist_path"
    sed -Ei "s/(end_hour *= *).*/\1$end_hour,   $end_hour,   $end_hour,/" "$namelist_path"
    sed -Ei "s/(end_minute *= *).*/\1$end_minute,   $end_minute,   $end_minute,/" "$namelist_path"
    sed -Ei "s/(end_second *= *).*/\1$end_second,   $end_second,   $end_second,/" "$namelist_path"

    # set run hours
    sed -Ei "s/(run_hours *= *).*/\1$runtime_hours,/" "$namelist_path"
}

get_first_letter_of_month(){
    month_num=$1
    if [ $month_num -eq 12 ] ; then 
        month_str="dec"
    elif [ $month_num -eq 11 ] ; then 
        month_str="nov"
    elif [ $month_num -eq 10 ] ; then 
        month_str="oct"
    elif [ $month_num -eq 9 ] ; then 
        month_str="sep"
    elif [ $month_num -eq 8 ] ; then 
        month_str="aug"
    elif [ $month_num -eq 7 ] ; then 
        month_str="jul"
    elif [ $month_num -eq 6 ] ; then 
        month_str="jun"
    elif [ $month_num -eq 5 ] ; then 
        month_str="may"
    elif [ $month_num -eq 4 ] ; then 
        month_str="apr"
    elif [ $month_num -eq 3 ] ; then 
        month_str="mar"
    elif [ $month_num -eq 2 ] ; then 
        month_str="feb"
    elif [ $month_num -eq 1 ] ; then 
        month_str="jan"
    fi

    echo $month_str
}

get_job_name_string() {
    local start_month_num=$(( $1 ))
    local start_day_num=$(( $2 ))
    local end_month_num=$(( $3 ))
    local end_day_num=$(( $4 ))
    local job_id="$5"

    start_month_str=$( get_first_letter_of_month $start_month_num )

    if [[ -n $job_id ]] ; then 
        job_id="-${job_id}"
    fi

    if [ $start_month_num -eq $end_month_num ] ; then
        echo "${start_month_str}${start_day_num}-${end_day_num}${job_id}"
    else
        end_month_str=$( get_first_letter_of_month $end_month_num )
        echo "${start_month_str}${start_day_num}-${end_month_num}${end_day_num}${job_id}"
    fi

}


#######################################################
# Command line Parsing
#######################################################

start_time=
end_time=
save_id=
job_id=

while :; do
    case $1 in 
    -h|--help)
        Help
        exit
        ;;

    -st|--start-time)
        if [ "$2" ]; then
            if [ $( check_date $2 ) -eq 1 ] ; then
                echo "'$2' is an incorrect date"
                exit 1
            fi
            start_time=$2
            shift
        else
            echo 'ERROR: "--start-time" requires a non-empty option argument.'
            exit 1
        fi
        shift
        ;;
    
    -et|--end-time)
        if [ "$2" ]; then
            if [ $( check_date $2 ) -eq 1 ] ; then
                echo "'$2' is an incorrect date"
                exit 1
            fi
            end_time=$2
            shift
        else
            echo 'ERROR: "--end-time" requires a non-empty option argument.'
            exit 1
        fi
        shift
        ;;
    -id)
        if [ "$2" ] ; then
            save_id="$2"
            shift
        else
            echo 'ERROR "-id" requires a non-empty option argument.'
            exit 1
        fi
        shift
        ;;
    --job-id)
        if [ "$2" ] ; then
            job_id="$2"
            shift
        else
            echo 'ERROR "--job-id" requires a non-empty option argument.'
            exit 1
        fi
        shift
        ;;
    *)
        break
    esac
done

if [[ "$start_time" ]] || [[ "$end_time" ]] ; then
    if ! [[ "$start_time" ]] || ! [[ "$end_time" ]] ; then
        echo "must provide both --start-time and --end-time"
        exit 1
    fi
fi

if ! [ "$save_id" ] ; then
    save_id="CHPN"
fi


#######################################################
# Main 
#######################################################


expected_args=2
set -e
shopt -s extglob
echo $1
echo $2
# Check for correct number of arguments
if [ $# -ne $expected_args ]; then
    echo "Invalid number of arguments."
    echo "Run bash set_up_sim_dir.sh --help for more information"
    exit 1
fi

simulation_dir=$( realpath $1 )
setup_dir=$( realpath $2 )
current_dir=$(pwd)

# Check to see that the setp_dir exists
if [ ! -d $setup_dir ]; then
    echo "ERROR: folder $setup_dir does not exist"
    exit 1
fi

# Create simulation_dir if not exist, else delete old directory
if test -d $simulation_dir; then
    echo "$simulation_dir already exists, deleting directory"
    rm -r $simulation_dir
fi

mkdir $simulation_dir

# Begin setting up simulation directory
echo "Setting up Simulation Directory $simulation_dir" 
cd $simulation_dir
mkdir pl
mkdir sfc

# Copy and edit namelist.wps
cp "$setup_dir/WPS-4.5/namelist.wps" .
# Change the time of the namelist.wps
if [[ "$start_time" ]] || [[ "$end_time" ]] ; then
    start_replacement_string="start_date = '${start_time}','${start_time}','${start_time}',"
    sed -iE "s/start_date *=.*/${start_replacement_string}/" namelist.wps 
    end_replacement_string="end_date = '${end_time}','${end_time}','${end_time}',"
    sed -iE "s/end_date *=.*/${end_replacement_string}/" namelist.wps 
fi
# Change opt_geogrid_tbl_path
sed -iE "s~opt_geogrid_tbl_path *=.*~opt_geogrid_tbl_path = '${setup_dir}/WPS-4.5/geogrid'~" namelist.wps

# Link and copy rest of files
cp namelist.wps ./sfc/.
cp namelist.wps ./pl/.
cp /home/waseem/template_files/run_real_JED .
cp /home/waseem/template_files/run_wrf_JED .
ln -s "$setup_dir/WPS-4.5/geogrid.exe" .
ln -s "$setup_dir/WPS-4.5/geogrid/GEOGRID.TBL" .
ln -s "$setup_dir/WPS-4.5/ungrib.exe" ./pl/.
ln -s "$setup_dir/WPS-4.5/ungrib.exe" ./sfc/.
ln -s "$setup_dir/WPS-4.5/link_grib.csh" ./pl/.
ln -s "$setup_dir/WPS-4.5/link_grib.csh" ./sfc/.
ln -s "$setup_dir/WPS-4.5/metgrid.exe" .
ln -s "$setup_dir/WPS-4.5/metgrid" .
ln -s "$setup_dir/WPS-4.5/ungrib/Variable_Tables/Vtable.ECMWF" ./sfc/Vtable
ln -s "$setup_dir/WPS-4.5/ungrib/Variable_Tables/Vtable.ECMWF" ./pl/Vtable


# Run geogrid
echo "Running Geogrid"
./geogrid.exe

# Link ERA5 data and ungrib in pl and sfc, may need to change ERA5 data location at somepoint
echo "Linking and ungribbing ERA5 data"
cd pl
# ./link_grib.csh /work/lapi/georgaka/WRFinput/ERA5/HELMOS/pl/ERA5_pl_2021-12-*
./link_grib.csh /work/lapi/waseem/ERA5/HELMOS_CHOPIN_ERA5/*_pl.grib
sed -i -E "s/(prefix = ')[a-zA-Z0-9]+(',)/\1PL\2/" namelist.wps
./ungrib.exe >& ungrib.output

cd ../sfc
# ./link_grib.csh /work/lapi/georgaka/WRFinput/ERA5/HELMOS/sfc/ERA5_sfc_2021-12-*
./link_grib.csh /work/lapi/waseem/ERA5/HELMOS_CHOPIN_ERA5/*_sfc.grib
sed -i -E "s/(prefix = ')[a-zA-Z0-9]+(',)/\1SFC\2/" namelist.wps
./ungrib.exe >& ungrib.output

cd $simulation_dir
ln -s sfc/SFC:* .
ln -s pl/PL:* .


# run metgrid
echo "Running metgrid.exe"
sed -i -E "s/fg_name = [a-zA-Z0-9']+,/fg_name = 'SFC','PL'/" namelist.wps
./metgrid.exe >& metgrid.output


# Linking WRF files
echo "Linking WRF files from $setup_dir/WRF"
ln -sf $setup_dir/WRF/run/!(namelist*) .

# set up namelist.input if start and end dates are given
if [[ "$start_time" ]] || [[ "$end_time" ]] ; then
    echo "Setting up namelist.input"
    copy_and_set_namelist_input "$simulation_dir" "$start_time" "$end_time" "$save_id"
fi

#DONE
echo 
echo "Finished setting up WRF simulation dir $simulation_dir"
if [ -z $start_time ] || [ -z $end_time ] ; then 
    Finish_Instructions 
    exit 0
fi


echo "Start data and end data provided, adjusting run_real_JED and run_wrf_JED"
start_month=$( echo "$start_time" | cut -d "-" -f 2)
start_day=$( echo "$start_time" | cut -d "-" -f 3 | cut -d "_" -f 1)

end_month=$( echo "$end_time" | cut -d "-" -f 2)
end_day=$( echo "$end_time" | cut -d "-" -f 3 | cut -d "_" -f 1)

job_name=$( get_job_name_string $start_month $start_day $end_month $end_day $job_id )
sed -i "s/#SBATCH --job-name real/#SBATCH --job-name r-$job_name/" ${simulation_dir}/run_real_JED
sed -i "s/#SBATCH --time 70:00:00/#SBATCH --time 50:00:00/" ${simulation_dir}/run_wrf_JED
sed -i "s/#SBATCH --job-name wrf/#SBATCH --job-name wrf-$job_name/" ${simulation_dir}/run_wrf_JED

echo "Launching run_real_JED"

cd $simulation_dir
sbatch run_real_JED

Finish_Instructions_2

