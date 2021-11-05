import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from datasets import Dataset

from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Optional, Union
from preprocess import wiki_preprocess


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrievalBM25:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys(
                [wiki_preprocess(v["text"]).replace("\\n", " ") for v in wiki.values()]
            )
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Tokenize contexts
        self.tokenize_fn = tokenize_fn
        tokenized_corpus = []
        for context in tqdm(self.contexts):
            tokenized_context = self.tokenize_fn(context)
            tokenized_corpus.append(tokenized_context)

        # BM25Okapi,BM25Plus,BM25L 사용가능
        self.bm25 = BM25Okapi(corpus=tokenized_corpus, k1=1.5, b=0.75)

    def get_relevant_doc(self, query, k=1):
        tokenized_query = self.tokenize_fn(query)
        doc_score = self.bm25.get_scores(tokenized_query)

        sorted_result = np.argsort(doc_score.squeeze())[::-1]
        doc_score = doc_score.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries, k=1):
        doc_scores = []
        doc_indices = []
        for query in tqdm(queries):
            tokenized_query = self.tokenize_fn(query)
            doc_score = self.bm25.get_scores(tokenized_query)
            sorted_result = np.argsort(doc_score.squeeze())[::-1]
            doc_scores.append(doc_score.squeeze()[sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        retrieval_dataset: Dataset,
        topk: Optional[int] = 1,
        delimiter: str = " ",
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        print("topk", topk)

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]

                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": delimiter.join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)

            return cqas
