#!/usr/bin/env bash

mkdir -p "output/"

input_directory_original="Webis-CausalQA-22-v-1.0/input/original-splits/"
model_directory_original="Webis-CausalQA-22-v-1.0/models/original-splits/"

input_directory_random="Webis-CausalQA-22-v-1.0/input/random-splits/"
model_directory_random="Webis-CausalQA-22-v-1.0/models/random-splits/"
datasets=("paq" "gooaq" "msmarco" "naturalquestions" "eli5" "searchqa" "squad2" "newsqa" "hotpotqa" "triviaqa")


get_batch_size() {
    if [ "paq" = "$1" ]; then
        echo "256"
    elif [ "gooaq" = "$1" ]; then
        echo "256"
    elif [ "msmarco" = "$1" ]; then
        echo "16"
    elif [ "eli5" = "$1" ]; then
        echo "32"
    else
        echo "1"
    fi
}

# Base model
for dataset in "${datasets[@]}"
do
    input_file="${input_directory_original}${dataset}_valid_original_split.csv"
    batch_size=$(get_batch_size $dataset)
    python evaluate_unifiedqa.py --valid_file "$input_file" --batch_size "$batch_size"
done

# Fine-tuned model
for dataset in "${datasets[@]}"
do
    input_file_original="${input_directory_original}${dataset}_valid_original_split.csv"
    input_file_random="${input_directory_random}${dataset}_valid_random_split.csv"
    model_original="${model_directory_original}${dataset}-original-split/"
    model_random="${model_directory_random}${dataset}-random-split/"

    batch_size=$(get_batch_size $dataset)
    python evaluate_unifiedqa.py --checkpoint "$model_original" \
                                 --valid_file "$input_file_original" \
                                 --batch_size "$batch_size"

    python evaluate_unifiedqa.py --checkpoint "$model_random" \
                                 --valid_file "$input_file_random" \
                                 --batch_size "$batch_size" \
                                 --output_directory "output/random-splits/"
done
