# Fine-tuning UnifiedQA

Scripts to fine-tune UnifiedQA to reproduce the results of Table 4.

## Installation

Clone the repository:
```
git clone https://github.com/webis-de/coling22-benchmark-for-causal-question-answering.git
```

Install the required dependencies. (You might have to update the torch settings depending on your system)

With `Anaconda3`:
```
conda env create -f environment.yml
```

With `pip`:
```
python -m venv causalqa
source causalqa/bin/activate
pip install -r requirements.txt
```

Download the input data and pretrained models:
```
wget https://zenodo.org/record/7186761/files/Webis-CausalQA-22-v-1.0.zip
```

## Evaluation

### Automatically

To reproduce all results run:
```
./reproduce_evaluation.sh
```
(Depending on your GPU you might have to adjust the batch sizes, the experiments were run with one A100 with 40GB)

### Manually

The `evaluate_unifiedqa.py` script can be used as follows:
```
python evaluate_unifiedqa.py --checkpoint "Webis-CausalQA-22-v-1.0/models/original-splits/squad2_original_split" \
                             --valid_file "Webis-CausalQA-22-v-1.0/input/original-splits/squad2_valid_original_split.csv" \
                             --batch_size 1
```

More usage information:
```
usage: evaluate_unifiedqa.py [-h] [--checkpoint CHECKPOINT] [--valid_file VALID_FILE]
                             [--tokenizer TOKENIZER] [--batch_size BATCH_SIZE] [--num_procs NUM_PROCS]
                             [--seed SEED] [--output_folder OUTPUT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Checkpoint to evaluate.
  --valid_file VALID_FILE
  --tokenizer TOKENIZER
  --batch_size BATCH_SIZE
  --num_procs NUM_PROCS
                        Number of processes for dataset loading.
  --seed SEED
  --output_directory OUTPUT_DIRECTORY
```

## Training

The `train_unifiedqa.py` script can be used as follows:
```
python train_unifiedqa.py --checkpoint "allenai/unifiedqa-v2-t5-base-1363200" \
                          --valid_file "Webis-CausalQA-22-v-1.0/input/original-splits/squad2_train_original_split.csv" \
                          --source_length 2048 \
                          --target_length 100 \
                          --batch_size 1
```

More usage information:
```
usage: train_unifiedqa.py [-h] [--checkpoint CHECKPOINT] [--train_file TRAIN_FILE] [--steps STEPS]
                          [--source_length SOURCE_LENGTH] [--target_length TARGET_LENGTH]
                          [--batch_size BATCH_SIZE] [--seed SEED] [--num_procs NUM_PROCS]
                          [--output_folder OUTPUT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Checkpoint to start fine-tuning from.
  --train_file TRAIN_FILE
  --steps STEPS         Number of train steps.
  --source_length SOURCE_LENGTH
                        Maximum length of the input sequences (question + context).
  --target_length TARGET_LENGTH
                        Maximum length of the output sequences.
  --batch_size BATCH_SIZE
  --seed SEED
  --num_procs NUM_PROCS
                        Number of processes for dataset loading.
  --output_directory OUTPUT_DIRECTORY
```
