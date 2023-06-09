#!/bin/bash

# exit on errors, unreferenced variables, and pipe errors
set -euo pipefail

data_folder="/ESL_data"
static_folder="$data_folder/static"
mkdir -p $static_folder

esl_data_url="https://rpg.ifi.uzh.ch/data/esl/static"

echo "Downloading and extracting data to ${static_folder} ..."

for seq_names in "seq1 book_duck" "seq2 plant" "seq3 city_of_lights" "seq4 desk" "seq5 chair" "seq6 room" "seq7 cycle" "seq8 heart" "seq9 david"
do
    tuple=( $seq_names );
    raw_url="${esl_data_url}/${tuple[1]}/data.raw"
    bias_url="${esl_data_url}/${tuple[1]}/data.bias"
    dest_folder="${static_folder}/${tuple[0]}/"
    wget --no-clobber "$raw_url" -P "$dest_folder"
    wget --no-clobber "$bias_url" -P "$dest_folder"
done
