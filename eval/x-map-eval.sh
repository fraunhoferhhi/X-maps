#!/bin/bash

# This script calculates different depth map estimation algorithms
# and then performs the evaluation of the results.

# exit on errors, unreferenced variables, and pipe errors
set -euo pipefail

# may want to disable these, once computed, to avoid lengthy recomputations
compute_mc3d=false
compute_esl=false

compute_xmaps=true

# data_folder="/workspaces/CVG_EB3D_python/ESL"
data_folder="/ESL_data"
static_folder="$data_folder/static"
mkdir -p $static_folder

esl_data_url="https://rpg.ifi.uzh.ch/data/esl/static"

echo "Downloading and extracting data to ${static_folder} ..."

for seq_names in "seq1 book_duck" "seq2 plant" "seq3 city_of_lights" "seq4 desk" "seq5 chair" "seq6 room" "seq7 cycle" "seq8 heart" "seq9 david"
do
    tuple=( $seq_names );
    full_url="${esl_data_url}/${tuple[1]}/scans_np.zip"
    dest_folder="${static_folder}/${tuple[0]}/"
    wget --no-clobber "$full_url" -P "$dest_folder"
    unzip -q -u "$dest_folder/scans_np.zip" -d "$dest_folder"
done

echo "Downloading calibration..."
wget --no-clobber "https://raw.githubusercontent.com/uzh-rpg/ESL/734bf8e88f689db79a0b291b1fb30839c6dd4130/data/calib.yaml" -P "$data_folder"
calib_yaml="${data_folder}/calib.yaml"

for seq_id in 1 2 3 4 5 6 7 8 9
do

    seq_folder="${static_folder}/seq$seq_id/"

    echo "Processing sequence $seq_id @ $seq_folder"

    num_scans=$(ls $seq_folder/scans_np/*.npy | wc -l)
    echo "Number of scans: $num_scans"

    if [ $compute_mc3d = true ]; then
        echo "Running MC3D baseline in parallel..."
        seq 0 $((num_scans-1)) | parallel --no-notice --bar --eta "python3 python/eval/mc3d_baseline.py -object_dir '${seq_folder}' -num_scans 1 -calib ${calib_yaml} -start_scan {} > /dev/null 2>&1"

        # if the silent parallel work is not producing outputs, check manually:
        # python3 python/eval/mc3d_baseline.py -object_dir ${seq_folder} -num_scans ${num_scans} -calib ${calib_yaml}
    fi
    
    if [ $compute_esl = true ]; then
        echo "Running ESL in parallel..."
        seq 0 $((num_scans-1)) | parallel --no-notice --bar --eta "python3 python/esl/compute_depth.py -object_dir '${seq_folder}' -num_scans 1 -calib ${calib_yaml}  -start_scan {}  > /dev/null 2>&1"

        # if the parallel work is not producing outputs, check manually:
        # python3 python/esl/compute_depth.py -object_dir ${seq_folder} -num_scans ${num_scans} -calib ${calib_yaml}
    fi

    if [ $compute_xmaps = true ]; then
        # don't run in parallel, setup takes a bit, but frame computation is very fast
        echo "Running X-maps..."
        python3 python/eval/compute_depth_x_maps.py -object_dir ${seq_folder} -num_scans ${num_scans} -calib ${calib_yaml} > /dev/null 2>&1
    fi

done

echo "Running evaluation script to compare results..."
python3 python/eval/create_evaluation_table.py -object_dir ${static_folder} -max_depth 500
