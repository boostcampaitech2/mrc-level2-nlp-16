import random
import datasets
from collections import defaultdict
from typing import Union, List, Callable, Dict

from torch.utils.data import Dataset

from arguments import DataArguments
from transformers import PreTrainedTokenizer, BatchEncoding, EvalPrediction


class GroupedMarcoTrainDataset(Dataset):
    query_columns = ["qid", "query"]
    document_columns = ["pid", "passage"]

    def __init__(
        self,
        args: DataArguments,
        path_to_tsv: Union[List[str], str],
        tokenizer: PreTrainedTokenizer,
    ):
        self.nlp_dataset = datasets.load_dataset(
            "json",
            data_files=path_to_tsv,
            ignore_verifications=False,
            # features=datasets.Features(
            #     {
            #         "qry": {
            #             "qid": datasets.Value("string"),
            #             "query": [datasets.Value("int32")],
            #         },
            #         "pos": [
            #             {
            #                 "pid": datasets.Value("string"),
            #                 "passage": [datasets.Value("int32")],
            #             }
            #         ],
            #         "neg": [
            #             {
            #                 "pid": datasets.Value("string"),
            #                 "passage": [datasets.Value("int32")],
            #             }
            #         ],
            #     }
            # ),
        )["train"]

        self.tok = tokenizer
        self.flips = defaultdict(lambda: random.random() > 0.5)
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation="only_first",
            return_attention_mask=False,
            max_length=self.args.q_max_len if is_query else self.args.p_max_len,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        group = self.nlp_dataset[item]
        group_batch = []
        qid, qry = (group["qry"][k] for k in self.query_columns)
        encoded_query = self.create_one_example(qry, is_query=True)
        _, pos_psg = [random.choice(group["pos"])[k] for k in self.document_columns]
        group_batch.append(self.create_one_example(pos_psg))
        if len(group["neg"]) < self.args.train_group_size - 1:
            negs = random.choices(group["neg"], k=self.args.train_group_size - 1)
        else:
            negs = random.sample(group["neg"], k=self.args.train_group_size - 1)
        for neg_entry in negs:
            _, neg_psg = [neg_entry[k] for k in self.document_columns]
            group_batch.append(self.create_one_example(neg_psg))

        return encoded_query, group_batch


class MarcoPredDataset(Dataset):
    columns = ["qid", "pid", "qry", "psg"]

    def __init__(
        self,
        path_to_json: List[str],
        tokenizer: PreTrainedTokenizer,
        q_max_len=16,
        p_max_len=128,
    ):
        print(path_to_json, flush=True)
        self.nlp_dataset = datasets.load_dataset(
            "json",
            data_files=path_to_json,
        )["train"]
        self.tok = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len

        print("dataset loaded", flush=True)

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        qid, pid, qry, psg = (self.nlp_dataset[item][f] for f in self.columns)
        encoded_qry = self.tok.encode_plus(
            qry,
            truncation="only_first",
            return_attention_mask=False,
            max_length=self.q_max_len,
            return_token_type_ids=False,
        )
        encoded_psg = self.tok.encode_plus(
            psg,
            max_length=self.p_max_len,
            truncation="only_first",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return encoded_qry, encoded_psg


class MarcoEncodeDataset(Dataset):
    columns = ["pid", "psg"]

    def __init__(
        self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, p_max_len=128
    ):
        self.nlp_dataset = datasets.load_dataset(
            "json",
            data_files=path_to_json,
        )["train"]
        self.tok = tokenizer
        self.p_max_len = p_max_len

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        pid, psg = (self.nlp_dataset[item][f] for f in self.columns)
        encoded_psg = self.tok.encode_plus(
            psg,
            max_length=self.p_max_len,
            truncation="only_first",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return encoded_psg
