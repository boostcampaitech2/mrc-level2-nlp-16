from pprint import pprint
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
from argparse import ArgumentParser


import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from datasets import concatenate_datasets, load_from_disk
from datasets import Value, Features, Dataset
from transformers import AutoTokenizer

from rank_bm25 import BM25Okapi


def delete_duplicate(org_context, contexts, fuzz_ratio, neg_samples):
    fancy_index = []

    for idx, context in enumerate(contexts):
        if fuzz.ratio(org_context, context) > fuzz_ratio:
            continue
        fancy_index.append(idx)
        if len(fancy_index) == neg_samples:
            break

    return fancy_index


def make_triplet_dataset(args):
    dataset = load_from_disk(args.train_dataset)
    dataset = concatenate_datasets([dataset["train"], dataset["validation"]])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    contexts = list(
        dict.fromkeys([context.replace("\\n", " ") for context in dataset["context"]])
    )
    print(f"Lengths of unique contexts : {len(contexts)}")
    ids = list(range(len(contexts)))

    # Tokenize contexts
    tokenize_fn = tokenizer.tokenize
    tokenized_corpus = []
    print("Tokenize corpus....")
    for context in tqdm(contexts):
        tokenized_context = tokenize_fn(context)
        tokenized_corpus.append(tokenized_context)

    bm25 = BM25Okapi(corpus=tokenized_corpus, k1=1.5, b=0.75)

    topk = 30
    scores, indices = [], []
    print("Retrieve documents for negative sampling")
    for question in tqdm(dataset["question"]):
        tokenized_question = tokenize_fn(question)
        doc_score = bm25.get_scores(tokenized_question)
        sorted_result = np.argsort(doc_score.squeeze())[::-1]
        scores.append(doc_score.squeeze()[sorted_result].tolist()[:topk])
        indices.append(sorted_result.tolist()[:topk])

    scores, indices = np.array(scores), np.array(indices)
    contexts = np.array(contexts)

    triplet_datasets = pd.DataFrame(
        columns=[
            "qid",
            "qry_text",
            "pos_did",
            "pos_doc_text",
            "neg_did",
            "neg_doc_text",
        ]
    )

    print("setting dataframe...")
    for idx, data in enumerate(tqdm(dataset)):

        org_context = data["context"]

        # retrieved contexts
        ret_contexts = contexts[indices[idx]]
        fancy_index = delete_duplicate(
            org_context, ret_contexts, args.fuzz_ratio, args.neg_samples
        )
        fancy_index = np.array(fancy_index)

        ret_contexts = ret_contexts[fancy_index]
        assert len(ret_contexts) == args.neg_samples, "neg_samples의 개수가 부족합니다."

        # triplet_datasets["negative"].append(ret_contexts[0])
        for j, ret_context in enumerate(ret_contexts):
            row = {
                "qid": data["id"],
                "qry_text": data["question"],
                "pos_did": str(data["document_id"]),
                "pos_doc_text": data["context"].replace("\\n", " "),
                "neg_did": str(data["document_id"]) + "_" + str(j),
                "neg_doc_text": ret_context,
            }
            triplet_datasets = triplet_datasets.append(row, ignore_index=True)

        if idx == 0:
            pprint(triplet_datasets)

    os.makedirs(args.save_dir, exist_ok=True)

    triplet_datasets.to_csv(
        os.path.join(args.save_dir, "triplet_dataset.tsv"),
        index=False,
        header=False,
        sep="\t",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--fuzz_ratio", type=int, required=True)
    parser.add_argument("--neg_samples", type=int, required=True)
    args = parser.parse_args()

    make_triplet_dataset(args)
