import os
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# -------------------------
# CLI ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="styal/very-smollm-corpus-0.5M",
                    help="HuggingFace dataset name or local path")

parser.add_argument("--tokenizer", type=str, default="PleIAs/Monad",
                    help="HuggingFace tokenizer name")

parser.add_argument("--num_proc", type=int, default=4,
                    help="workers for tokenization")

parser.add_argument("--num_proc_load_dataset", type=int, default=4,
                    help="workers for loading dataset")

parser.add_argument("--test_size", type=float, default=0.0005,
                    help="validation split size")

parser.add_argument("--out_dir", type=str, default="data_bin",
                    help="output directory")
parser.add_argument("--block_size", type=int, default=1024)

args = parser.parse_args()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, num_proc=args.num_proc_load_dataset)

    # if dataset only has train split
    if "train" in dataset and "validation" not in dataset:
        split_dataset = dataset["train"].train_test_split(
            test_size=args.test_size,
            seed=2357,
            shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
    else:
        split_dataset = dataset
        
    split_dataset["train"] = split_dataset["train"].select(range(50000))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have eos_token")

    eos = tokenizer.eos_token_id

    # -------------------------
    # TOKENIZATION
    # -------------------------
    def process(batch):
        ids = tokenizer(
            batch["text"],
            add_special_tokens=False,
            #batched=True,
        )["input_ids"]
    
        # append EOS token
        #ids = [x + [eos] for x in ids]
    
        return {
            "ids": ids,
            "len": [len(x) for x in ids]
        }

    print("Tokenizing dataset...")

    tokenized = split_dataset.map(
        process,
        remove_columns=split_dataset["train"].column_names,
        desc="tokenizing",
        num_proc=args.num_proc,
        batched=True,
        batch_size=512*4,
    )

    # -------------------------
    # WRITE BIN FILES
    # -------------------------
    block_size = args.block_size
    
    for split, dset in tokenized.items():
    
        print(f"Building blocks for {split}...")
    
        all_tokens = np.concatenate([np.array(x, dtype=np.uint32) for x in dset["ids"]])
    
        total_tokens = len(all_tokens)
        num_blocks = total_tokens // block_size
    
        all_tokens = all_tokens[:num_blocks * block_size]
    
        blocks = all_tokens.reshape(num_blocks, block_size)
    
        filename = os.path.join(args.out_dir, f"{split}.bin")
    
        dtype = np.uint16 if tokenizer.vocab_size < 2**16 else np.uint32
    
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=blocks.shape)
    
        arr[:] = blocks[:]
        arr.flush()
    
        print(f"{split}: {num_blocks} blocks written")

    print("Done.")