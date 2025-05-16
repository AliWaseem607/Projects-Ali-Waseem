# !/bin/bash

# This is a bash script designed to set up a directory 
# Inputs: 
#   1.  simulation_dir: Location where the simulation will be run
#       Usually /scratch/<username>/<new_folder_name>
#   2.  setup_dir: This is a directory that contains the WPS and WRF directorys


args=("$@")
expected_args=2
set -e
shopt -s extglob

# Check for correct number of arguments
if [ $# -ne $expected_args ]; then
    echo "Invalid number of arguments. Expected: $expected_args"
    echo 
    echo "First input should be the simulation directory to be created"
    echo "Second input should be the setup directory that contains the WPS and WRF directories"
    exit 1
fi

simulation_dir=$( realpath ${args[0]} )
setup_dir=$( realpath ${args[1]} )
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


# Create copies and links
cp "$setup_dir/WPS-4.5/namelist.wps" .
cp "$setup_dir/WPS-4.5/namelist.wps" ./sfc/.
cp "$setup_dir/WPS-4.5/namelist.wps" ./pl/.
cp /home/waseem/run_real_JED .
cp /home/waseem/run_wrf_JED .
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


#Linking WRF files
echo "Linking WRF files from $setup_dir/WRF"
ln -sf $setup_dir/WRF/run/!(namelist*) .


#DONE
echo 
echo "Finished setting up WRF simulation dir $simulation_dir"
echo "Final steps are to:"
echo "    1. copy a namelist.input file into the directory and adjust as needed"
echo "    2. run real.exe by running command 'sbatch run_real_JED'"
echo "    3. After job has finished check that you have created a wrfbdy file, and n wrfinput and wrflow files with n the number of domains"
echo "    4. run wrf.exe by running command 'sbatch run_wrf_JED'"
echo "Good luck on your simulations!!!"
