set -e

arr=("/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-03_NPRK.nc"
"/home/waseem/CHOPIN_analysis/data/wrfout/wrfout_CHPN_d03_2024-11-03_MYNN_NPRK.nc"
"/home/waseem/CHOPIN_analysis/data/envelopment_period_1/wrfout/wrfout_CHPN_d03_2024-10-17_NPRK_MYNN.nc"
"/home/waseem/CHOPIN_analysis/data/rain_period_1/wrfout/wrfout_CHPN_d03_2024-10-06_NPRK_THMP.nc"
"/home/waseem/CHOPIN_analysis/data/clear_period_1/wrfout/wrfout_CHPN_d03_2024-10-27_NPRK_YSU.nc"
"/home/waseem/CHOPIN_analysis/data/clear_period_1/wrfout/wrfout_CHPN_d03_2024-10-27_NPRK_MYNN.nc")

for path in ${arr[@]} ; 
do 
    echo $path
    echo "Running Radar Simulations..."
    echo "Running MIRA..."
    # srun -c 1 -t 00:20:00 /home/waseem/scripts/crsim_runner.sh -p "/home/waseem/template_files/crsim/PARAMETERS_MIRA" -id "MIRA" "$wrfoutput_dir" "$radar_ncfile_name" # %> "crsim_MIRA/crsim_runner.log"
    MIRA_results_path="/scratch/waseem/radar_redo/crsim_MIRA/"
    mkdir -p $MIRA_results_path
    # srun -c 10 -t 00:15:00 /home/waseem/scripts/run_crsim_multi_runner.sh "$path" "$MIRA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_MIRA"
    /home/waseem/scripts/run_crsim_multi_runner.sh "$path" "$MIRA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_MIRA" 5

    echo "Running BASTA..."
    # srun -c 1 -t 00:20:00 /home/waseem/scripts/crsim_runner.sh -p "/home/waseem/template_files/crsim/PARAMETERS_BASTA" -id "BASTA" "$wrfoutput_dir" "$radar_ncfile_name" # %> "crsim_BASTA/crsim_runner.log"
    BASTA_results_path="/scratch/waseem/radar_redo/crsim_BASTA/"
    mkdir -p $BASTA_results_path
    # srun -c 10 -t 00:15:00 /home/waseem/scripts/run_crsim_multi_runner.sh "$path" "$BASTA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_BASTA"
    /home/waseem/scripts/run_crsim_multi_runner.sh "$path" "$BASTA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_BASTA" 5


    echo "Adding reflectivites to NPRK..."
    python /home/waseem/scripts/add_reflectivities.py --BASTA-results-path "$BASTA_results_path" --MIRA-results-path "$MIRA_results_path" --wrfout-path "$path"

    rm -r "$MIRA_results_path"
    rm -r "$BASTA_results_path"


    echo
done



# echo "Running Radar Simulations..."
# echo "Running MIRA..."
# # srun -c 1 -t 00:20:00 /home/waseem/scripts/crsim_runner.sh -p "/home/waseem/template_files/crsim/PARAMETERS_MIRA" -id "MIRA" "$wrfoutput_dir" "$radar_ncfile_name" # %> "crsim_MIRA/crsim_runner.log"
# MIRA_results_path="$wrfoutput_dir/crsim_MIRA/"
# mkdir -p $MIRA_results_path
# srun -c 15 -t 00:08:00 /home/waseem/scripts/run_crsim_multi_runner.sh "$wrfoutput_dir/$radar_ncfile_name" "$MIRA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_MIRA"

# echo "Running BASTA..."
# # srun -c 1 -t 00:20:00 /home/waseem/scripts/crsim_runner.sh -p "/home/waseem/template_files/crsim/PARAMETERS_BASTA" -id "BASTA" "$wrfoutput_dir" "$radar_ncfile_name" # %> "crsim_BASTA/crsim_runner.log"
# BASTA_results_path="$wrfoutput_dir/crsim_BASTA/"
# mkdir -p $BASTA_results_path
# srun -c 15 -t 00:08:00 /home/waseem/scripts/run_crsim_multi_runner.sh "$wrfoutput_dir/$radar_ncfile_name" "$BASTA_results_path" "/home/waseem/template_files/crsim/PARAMETERS_BASTA"


# echo "Adding reflectivites to NPRK..."
# python /home/waseem/scripts/add_reflectivities.py --BASTA-results-path "$BASTA_results_path" --MIRA-results-path "$MIRA_results_path" --wrfout-path "$NPRK_out_file_name"
