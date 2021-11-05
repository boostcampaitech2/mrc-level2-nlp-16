import os
import json
import time
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Optional, Union

from datasets import Dataset
from preprocess import wiki_preprocess


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrievalCOIL:
    def __init__(
        self,
        encode_fn,
        retriever_backbone,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.encode_fn = encode_fn
        self.retriever_backbone = retriever_backbone

        coil_model_name = f"model.pt"
        self.coil_model_path = os.path.join(
            "./retriever/coil/models", self.retriever_backbone, coil_model_name
        )

        assert (
            os.path.isfile(self.coil_model_path) == True
        ), "COIL retriever 모델을 먼저 학습하세요."

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([wiki_preprocess(v["text"]) for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.data_path = os.path.join(data_path, "coil_dataset")
        os.makedirs(self.data_path, exist_ok=True)

        wiki_path = os.path.join(self.data_path, context_path)

        if not os.path.isfile(wiki_path):

            self.wiki_df = pd.DataFrame(columns=["id", "text"])
            self.wiki_df["id"] = self.ids
            self.wiki_df["text"] = self.contexts

            # wikipedia documents to token_ids
            print("##### Wikipedia documents to token ids...")
            self._query_or_context_to_token_ids(
                wiki_path, self.wiki_df, self.encode_fn, max_length=512
            )

    def _query_or_context_to_token_ids(self, path, df, encode_fn, max_length):
        def encode_one_entry(entry, encode_fn, max_length):
            encoded = encode_fn(
                entry["text"],
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
            return json.dumps({"pid": entry["id"], "psg": encoded})

        with open(path, "w") as jfile:
            for i, row in tqdm(df.iterrows(), total=df.shape[0]):
                e = encode_one_entry(row, encode_fn, max_length)
                jfile.write(e + "\n")

    def encoding(self, query_dataset: Dataset) -> NoReturn:

        self.encoded_query_path = "./retriever/coil/encoded_query"
        self.encoded_wiki_path = "./retriever/coil/encoded_wiki"

        if os.path.isdir(self.encoded_query_path):
            print("Encoded query exists")
            pass
        else:
            print("Encoding query starts")
            self.query_df = pd.DataFrame(columns=["id", "text"])

            # mrc-1-000653 형태의 query_id를 '000653' + '1' = '0006531' 형태로 변형합니다.
            self.query_df["id"] = list(
                map(lambda x: x.split("-")[2] + x.split("-")[1], query_dataset["id"])
            )
            self.query_df["text"] = query_dataset["question"]

            query_path = os.path.join(self.data_path, "test.json")

            print("Queries to token ids...")
            self._query_or_context_to_token_ids(
                query_path, self.query_df, self.encode_fn, max_length=64
            )

            os.makedirs("./retriever/coil/encoded_query", exist_ok=True)
            os.system(
                f"python retriever/coil/run_macro.py \
                    --output_dir ./retriever/coil/encoded_query \
                    --model_name_or_path ./retriever/coil/models/{self.retriever_backbone} \
                    --tokenizer_name {self.retriever_backbone} \
                    --cls_dim 768 \
                    --token_dim 32 \
                    --do_encode \
                    --p_max_len 64 \
                    --fp16 \
                    --no_sep \
                    --pooling max \
                    --per_device_eval_batch_size 16 \
                    --encode_in_path {query_path} \
                    --encoded_save_path ./retriever/coil/encoded_query"
            )

        if os.path.isdir(self.encoded_wiki_path):
            print("Encoded wikipedia context exists")
            pass
        else:
            print("Encoding wikipedia context starts")
            os.makedirs("./retriever/coil/encoded_wiki/split00", exist_ok=True)
            os.system(
                f"python retriever/coil/run_macro.py \
                    --output_dir ./retriever/coil/encoded_wiki \
                    --model_name_or_path ./retriever/coil/models/{self.retriever_backbone} \
                    --tokenizer_name {self.retriever_backbone} \
                    --cls_dim 768 \
                    --token_dim 32 \
                    --do_encode \
                    --no_sep \
                    --p_max_len 512 \
                    --pooling max \
                    --fp16 \
                    --per_device_eval_batch_size 16 \
                    --encode_in_path ../data/coil_dataset/wikipedia_documents.json \
                    --encoded_save_path ./retriever/coil/encoded_wiki/split00"
            )
            print("##### Encoding wikipedia context ends #####")

    def retrieve(
        self,
        dataset: Dataset,
        retrieval_dataset: Dataset,
        topk: Optional[int] = 1,
        delimiter: str = " ",
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.encoded_query_path is not None, "encoding() 메소드를 먼저 수행하세요."

        # build document index shards
        print("building document index shards...")
        os.system(
            f"python retriever/coil/retrieve/sharding.py \
                --n_shards 10 \
                --shard_id 0 \
                --dir {self.encoded_wiki_path} \
                --save_to ./retriever/coil/index \
                --use_torch"
        )

        # reformat encoded query
        print("reformating encoded query...")
        os.system(
            f"python retriever/coil/retrieve/format-query.py \
                --dir {self.encoded_query_path} \
                --save_to ./retriever/coil/reformed_query \
                --as_torch"
        )

        # retrieve using retriever-fast
        ## setting for retriever-fast
        print("setting for retriever-fast...")
        now_path = os.getcwd()
        os.chdir("./retriever/coil/retrieve/retriever_ext")
        os.system(f"python setup.py build_ext --inplace")
        os.chdir(now_path)

        if isinstance(retrieval_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    retrieval_dataset["id"], k=topk
                )
            for idx, example in enumerate(tqdm(dataset, desc="COIL retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": delimiter.join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        # retrieve using retriever-fast
        print("retrieve using retriever-fast...")
        os.makedirs("./retriever/coil/score/intermediate", exist_ok=True)
        os.system(
            f"python retriever/coil/retrieve/retriever-fast.py \
                --query ./retriever/coil/reformed_query \
                --doc_shard ./retriever/coil/index/shard_00 \
                --top 100 \
                --save_to ./retriever/coil/score/intermediate/shard_00.pt \
                --batch_size 512"
        )

        # merge scores from all shards
        os.system(
            f"python retriever/coil/retrieve/merger.py \
                --score_dir ./retriever/coil/score/intermediate \
                --query_lookup ./retriever/coil/reformed_query/cls_ex_ids.pt \
                --depth 100 \
                --save_ranking_to {self.data_path}/rank.txt"
        )

        with open(os.path.join(self.data_path, "rank.txt"), "r") as file:
            rank = file.readlines()

        doc_scores = []
        doc_indices = []
        for idx, query in enumerate(queries):
            temp_rank = rank[idx * 100 : idx * 100 + k]
            topk_score = list(map(lambda x: float(x.split("\t")[2].strip()), temp_rank))
            topk_pid = list(map(lambda x: int(x.split("\t")[1]), temp_rank))
            doc_scores.append(topk_score)
            doc_indices.append(topk_pid)
        return doc_scores, doc_indices
