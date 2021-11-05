import argparse
import json
import os
import time
from subprocess import Popen, PIPE, STDOUT

from datasets import (
    load_from_disk,
    Value,
    Features,
    Dataset,
    DatasetDict,
)
from elasticsearch import Elasticsearch
from tqdm import tqdm
from pprint import pprint as pp
import json
import re
import pandas as pd


"""
ElasticSearch를 활용하기 위한 코드 입니다.
대부분의 로직은 ES API를 활용하여 ES서버와 통신하는 구조 입니다.
"""


def set_index_and_server():
    config = {"host": "localhost", "port": 9200}
    es = Elasticsearch([config])

    index_name = "mrc_index"
    index_settings = {
        "settings": {
            "analysis": {
                "filter": {"my_stop_filter": {"type": "stop"}},
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter": ["my_stop_filter"],
                    }
                },
            }
        },
        "mappings": {
            "properties": {
                "document_text": {"type": "text", "analyzer": "nori_analyzer"}
            }
        },
    }

    print("elastic serach ping :", es.ping())
    if es.indices.exists(index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=index_settings)

    train_context = set_data()
    train_context = preproceessing_data(train_context)

    for i in tqdm(range(len(train_context))):
        es.index(index=index_name, id=i, body=train_context[i])

    return es


def set_data():
    with open("/opt/ml/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
        datasets = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    return datasets


def preproceessing_data(datasets):
    output = []
    for context in tqdm(datasets):
        preprocessing_text = context.replace("\n\n", " ")
        preprocessing_text = preprocessing_text.replace("\\n", " ")
        preprocessing_text = preprocessing_text.replace("\n", " ")
        preprocessing_text = preprocessing_text.replace("*", " ")
        preprocessing_text = preprocessing_text.replace("#", " ")

    output.append(preprocessing_text)
    datasets = [{"document_text": output[i]} for i in range(len(output))]
    return datasets


def make_addquery_datadict(es, query_count):
    df_list = pd.DataFrame(columns=["id", "question"])
    f_type = set_feature_type()
    combine_list = load_addquery_dataset(es, query_count)

    for combine in tqdm(combine_list):
        Data = Features({"id": combine[0], "question": combine[1]})
        df_list = df_list.append(Data, ignore_index=True)

    datasets = DatasetDict(
        {"validation": Dataset.from_pandas(df_list, features=f_type)}
    )
    datasets.save_to_disk("/opt/ml/data/test_dataset_addquery_" + query_count)
    return datasets


def set_feature_type():
    f = Features(
        {
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    return f


def load_addquery_dataset(es, query_count):
    addquery_count_list = {
        50: 53.5,
        100: 45.8,
        150: 40,
        200: 36.6,
        250: 33.74,
        300: 30.8,
    }
    combine_list = []
    test_dataset = load_test_dataset()

    for test in tqdm(test_dataset):
        es_results = es.search(index="mrc_index", q=test["question"], size=1)
        result = es_results["hits"]["hits"][0]
        combine = []
        if result["_score"] > addquery_count_list[query_count]:
            combine.append(test["id"])
            combine.append(
                test["question"]
                + " "
                + result["_source"]["document_text"][: 128 - len(test["question"]) - 1]
            )
        else:
            combine.append(test["id"])
            combine.append(test["question"])

    combine_list.append(combine)
    return combine_list


def load_test_dataset():
    test_dataset = load_from_disk("/opt/ml/data/test_dataset")
    test_dataset = test_dataset["validation"]
    return test_dataset
