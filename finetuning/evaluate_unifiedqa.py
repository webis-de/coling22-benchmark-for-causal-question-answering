import argparse
import json
import measures
import os
import torch

from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default='allenai/unifiedqa-v2-t5-base-1363200',
                        help='Checkpoint to evaluate.')
    parser.add_argument("--valid_file", type=str,
                        default='Webis-CausalQA-22-v-1.0/input/original-splits/squad2_valid_original_split.csv')
    parser.add_argument("--tokenizer", type=str, default='allenai/unifiedqa-v2-t5-base-1363200')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8,
                        help='Number of processes for dataset loading.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_directory", type=str, default='output/original-splits')
    return parser.parse_args()


# concatenate question+context with \\n as a separator
def build_input(batch):
    input_ = [(question + ' \\n ' + context if context is not None else question)
              for question, context in zip(batch['question_processed'], batch['context_processed'])]
    batch['input'] = input_
    return batch


def run_model(batch, model, tokenizer, args):
    if 'naturalquestions' in args.valid_file:
        encoded_inputs = tokenizer(batch, max_length=10000, padding='max_length',
                                   truncation=True, return_tensors="pt").to(DEVICE)
    else:
        encoded_inputs = tokenizer(batch, padding='longest', return_tensors="pt").to(DEVICE)
    res = model.generate(**encoded_inputs, max_length=500)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


def evaluate_unifiedqa(args):
    set_seed(args.seed)
    data = load_dataset('csv', data_files=args.valid_file)['train']
    data = data.map(build_input, batched=True, load_from_cache_file=False, num_proc=args.num_procs)
    data = data.remove_columns(['context', 'context_processed'])

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(DEVICE)

    loader = DataLoader(data, shuffle=False, num_workers=0, batch_size=args.batch_size)
    predictions = []
    for batch in tqdm(loader):
        batch_predictions = run_model(batch['input'], model, tokenizer, args)
        predictions.extend(batch_predictions)

    answers = data['answer']
    answers = [answer.split('\t') for answer in answers]

    result = {}
    result['checkpoint'] = args.checkpoint
    result['metrics'] = measures.all_metrics(predictions, answers)
    result['predictions'] = predictions

    start_index = args.valid_file.rfind("/") + 1
    end_index = args.valid_file.find("_")
    if 'allenai/unifiedqa-v2-t5' in args.checkpoint:
        output_file = f'base_{args.valid_file[start_index:end_index]}.json'
    else:
        output_file = f'finetuned_{args.valid_file[start_index:end_index]}.json'
    os.makedirs(args.output_directory, exist_ok=True)
    with open(os.path.join(args.output_directory, output_file), 'w+') as file:
        json.dump(result, file, indent=4)


if __name__ == '__main__':
    evaluate_unifiedqa(parse_args())
