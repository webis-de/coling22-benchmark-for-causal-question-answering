import argparse
from datasets import load_dataset
from transformers import (
        T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, set_seed,
        Seq2SeqTrainer, Seq2SeqTrainingArguments
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='allenai/unifiedqa-v2-t5-base-1363200',
                        help='Checkpoint to start fine-tuning from.')
    parser.add_argument("--train_file", type=str,
                        default='Webis-CausalQA-22-v-1.0/input/original-splits/squad2_train_original_split.csv')
    parser.add_argument("--steps", type=int, default=6000,
                        help='Number of train steps.')
    parser.add_argument("--source_length", type=int, default=2048,
                        help='Maximum length of the input sequences (question + context).')
    parser.add_argument("--target_length", type=int, default=100,
                        help='Maximum length of the output sequences.')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_procs", type=int, default=8,
                        help='Number of processes for dataset loading.')
    parser.add_argument("--output_directory", type=str, default='models/original-splits/')
    return parser.parse_args()


# concatenate question+context with \\n as a separator
def build_input(batch):
    input_ = [(question + ' \\n ' + context if context is not None else question)
              for question, context in zip(batch['question_processed'], batch['context_processed'])]
    batch['input'] = input_
    return batch


def train_unifiedqa(args: argparse.ArgumentParser):
    set_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)

    def tokenize_function_train(batches):
        encoded_inputs = tokenizer(batches['input'],
                                   max_length=args.source_length, padding='max_length',
                                   truncation=True)
        encoded_answers = tokenizer(batches['answer'],
                                    max_length=args.target_length, padding='max_length',
                                    truncation=True)
        encoded_inputs['labels'] = [
                    [(a if a != tokenizer.pad_token_id else -100) for a in ans]
                    for ans in encoded_answers["input_ids"]
                ]
        return encoded_inputs

    train_dataset = load_dataset('csv', data_files=args.train_file)['train']

    train_dataset = train_dataset.map(build_input, batched=True,
                                      load_from_cache_file=False, num_proc=args.num_procs)
    train_dataset = train_dataset.remove_columns(['context', 'context_processed'])
    train_dataset = train_dataset.map(tokenize_function_train, batched=True,
                                      load_from_cache_file=False, num_proc=args.num_procs)
    train_dataset = train_dataset.remove_columns(['input', 'answer'])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    log_steps = args.steps // 10

    train_args = Seq2SeqTrainingArguments('models',
                                          per_device_train_batch_size=args.batch_size,
                                          max_steps=args.steps,
                                          seed=args.seed,
                                          save_strategy='no',
                                          logging_strategy='steps',
                                          logging_steps=log_steps,
                                          save_total_limit=1
                                          )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    _ = trainer.train()

    start_index = args.train_file.rfind("/") + 1
    end_index = args.train_file.find("_")
    trainer.save_model(args.output_directory + args.train_file[start_index:end_index])


if __name__ == '__main__':
    train_unifiedqa(parse_args())
