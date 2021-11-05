import random
import pandas as pd
import os
import re
import pickle
import json
from datasets import (
    Dataset,
    Features,
    Value,
    Sequence,
    load_from_disk,
    concatenate_datasets,
)
from koeda import AEDA
from tqdm import tqdm
from typing import Optional, List
from pororo import Pororo

FEATURES = Features(
    {
        "__index_level_0__": Value(dtype="int64", id=None),
        "answers": {
            "answer_start": Sequence(
                feature=Value(dtype="int64", id=None), length=-1, id=None
            ),
            "text": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
        },
        "context": Value(dtype="string", id=None),
        "document_id": Value(dtype="int64", id=None),
        "id": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
        "title": Value(dtype="string", id=None),
    }
)

# ranomly choose number
def random_idx(data_len, num):
    idx_list = []
    while len(idx_list) < num:
        idx_num = random.randrange(data_len)
        if idx_num not in idx_list:
            idx_list.append(idx_num)
    return idx_list


# select data from dataset according to random index
def select(dataset, idx_list):
    selected = []
    for idx in idx_list:
        select = dataset[idx]
        selected.append(select)
    selected_pd = pd.DataFrame(selected)
    selected_result = Dataset.from_pandas(selected_pd, features=FEATURES)
    return selected_result


class customAEDA(AEDA):
    """koEDA의 AEDA과정의 속도 향상을 위한 버전"""

    def __init__(
        self,
        data_path: str = "../data/train_dataset/train",
        data_ratio: float = 0.1,
        punc_ratio: float = 0.3,
        punctuations: List[str] = None,
    ):
        super().__init__(
            morpheme_analyzer="Okt", punc_ratio=punc_ratio, punctuations=punctuations
        )
        self.dataset = load_from_disk(data_path)
        self.data_ratio = data_ratio
        self.length = int(len(self.dataset) * self.data_ratio)

    def aeda(self, data, p=None):
        SPACE_TOKEN = "\u241F"

        def replace_space(text):
            return text.replace(" ", SPACE_TOKEN)

        def revert_space(text):
            clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
            return clean

        if p is None:
            p = self.ratio

        split_words = self.morpheme_analyzer.morphs(replace_space(data))
        words = self.morpheme_analyzer.morphs(data)

        new_words = []
        q = random.randint(1, int(p * len(words) + 1))
        qs_list = [
            index
            for index in range(len(split_words))
            if split_words[index] != SPACE_TOKEN
        ]
        qs = random.sample(qs_list, q)

        for j, word in enumerate(split_words):
            if j in qs:
                new_words.append(SPACE_TOKEN)
                new_words.append(
                    self.punctuations[random.randint(0, len(self.punctuations) - 1)]
                )
                new_words.append(SPACE_TOKEN)
                new_words.append(word)
            else:
                new_words.append(word)

        augmented_sentences = revert_space(new_words)
        return augmented_sentences

    def run_augmentation(self):
        print("\n#### Starting AEDA augmentation ####")
        print("Length of augmented data: " + str(self.length))
        idx_list = random_idx(len(self.dataset), self.length)
        aug_list = []
        for idx in tqdm(idx_list):
            aug_data_dic = {
                "__index_level_0__": self.dataset["__index_level_0__"][idx],
                "answers": self.dataset["answers"][idx],
                "context": self.dataset["context"][idx],
                "document_id": self.dataset["document_id"][idx],
                "id": self.dataset["id"][idx],
                "question": self.aeda(self.dataset["question"][idx]),
                "title": self.dataset["title"][idx],
            }
            aug_list.append(aug_data_dic)

        aug_df = pd.DataFrame(aug_list)
        aug_dataset = Dataset.from_pandas(aug_df, features=FEATURES)
        return aug_dataset


class QuestionGeneration:
    def __init__(
        self,
        data_path: Optional[str] = "../data/aug_dataset/question_generation/",
        data_ratio: float = 0.1,
    ):
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.dataset = load_from_disk(data_path)
        self.length = int(len(self.dataset) * self.data_ratio)

    def get_aug_dataset(self):
        if os.path.exists(self.data_path):
            print(f"Loading existing Question Generation dataset")
            print("Length of augmented data: " + str(self.length))
            candidate = load_from_disk(self.data_path)
            candidate_idx = random_idx(len(candidate), self.length)
            dataset = select(candidate, candidate_idx)
            print(dataset)
            return dataset
        else:
            print(f"Making Question Generation dataset")
            candidate = self.make_aug_dataset()
            print(f"Loading Question Generation dataset")
            print("Length of augmented data: " + str(self.length))
            candidate_idx = random_idx(len(candidate), self.length)
            dataset = select(candidate, candidate_idx)
            print(dataset)
            return dataset

    def make_aug_dataset(self):
        train_dataset = load_from_disk("../data/train_dataset/train")
        qg = Pororo(task="qg", lang="ko")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        new_questions = []
        error_idx = []
        for idx in tqdm(range(len(train_dataset))):
            answer = train_dataset["answers"][idx]["text"][0]
            context = re.sub(r"\\n", "  ", train_dataset["context"][idx])
            try:
                new_question = qg(answer, context)
                new_questions.append(new_question)
            except:
                error_idx.append(idx)
                new_questions.append("")

        manual_questions = [
            "평소 백인들의 오만함에 대해 비판적이었던 윤치호가 생각한 미국의 가장 핵심적인 특징은?",
            "미국의 특징이 인종주의라는 생각을 갖게 된 윤치호에게 영향을 준 인물은?",
            "조선에서도 윤치호를 노골적으로 왕따 취급한 인물은?",
            "흑인을 차별해야 한다는 설교를 하여 윤치호에게 충격을 준 인물은?",
        ]

        for idx in error_idx:
            new_questions[idx] = manual_questions[idx]

        with open("../data/aug_dataset/qg_questions.pkl", "wb") as f:
            pickle.dump(new_questions, f)

        total = []
        for idx, example in enumerate(tqdm(train_dataset)):
            tmp = {
                "__index_level_0__": example["__index_level_0__"],
                "answers": example["answers"],
                "context": example["context"],
                "document_id": example["document_id"],
                "id": example["id"],
                "question": new_questions[idx],
                "title": example["title"],
            }
            total.append(tmp)
        new_df = pd.DataFrame(total)
        train_dataset_qg = Dataset.from_pandas(new_df, features=FEATURES)
        train_dataset_qg.save_to_disk(f"../data/aug_dataset/question_generation")

        with open("../data/aug_dataset/dataset_dict.json", "rb") as f:
            dataset_dict = json.load(f)

        dataset_dict["splits"].append("question_generation")

        with open("../data/aug_dataset/dataset_dict.json", "wb") as f:
            json.dump(dataset_dict, f)

        return train_dataset_qg

    def run_augmentation(self):
        print("\n#### Starting Question Generation augmentation ####")
        new_data = self.get_aug_dataset()
        return new_data


class BackTranslation:
    def __init__(
        self,
        data_path: Optional[str] = "../data/aug_dataset/backtranslation/",
        data_ratio: float = 0.1,
    ):
        self.data_path = data_path
        self.dataset = load_from_disk(data_path)
        self.data_ratio = data_ratio
        self.length = int(len(self.dataset) * self.data_ratio)

    def get_aug_dataset(self):
        if os.path.exists(self.data_path):
            print(f"Loading existing Backtranslation dataset")
            print("Length of augmented data: " + str(self.length))
            candidate = load_from_disk(self.data_path)
            candidate_idx = random_idx(len(candidate), self.length)
            dataset = select(candidate, candidate_idx)
            print(dataset)
            return dataset
        else:
            print(f"Making backtranslation dataset")
            candidate = self.make_aug_dataset()
            print(f"Loading Backtranslation dataset")
            print("Length of augmented data: " + str(self.length))
            candidate_idx = random_idx(len(candidate), self.length)
            dataset = select(candidate, candidate_idx)
            print(dataset)
            return dataset

    def make_aug_dataset(self):
        train_dataset = load_from_disk("../data/train_dataset/train")
        mt = Pororo(task="translation", lang="multi")

        bt_questions = []
        for idx in tqdm(range(len(train_dataset))):
            question = train_dataset["question"][idx]
            try:
                ja_question = mt(question, src="ko", tgt="ja")
                bt_question = mt(ja_question, src="ja", tgt="ko")
                bt_questions.append(bt_question)
            except:
                print(f"Error in data {idx}")
                bt_questions.append("")

        with open("../data/aug_dataset/bt_questions.pkl", "wb") as f:
            pickle.dump(bt_questions, f)

        total = []
        for idx, example in enumerate(tqdm(train_dataset)):
            tmp = {
                "__index_level_0__": example["__index_level_0__"],
                "answers": example["answers"],
                "context": example["context"],
                "document_id": example["document_id"],
                "id": example["id"],
                "question": bt_questions[idx],
                "title": example["title"],
            }
            total.append(tmp)

        new_df = pd.DataFrame(total)
        train_dataset_bt = Dataset.from_pandas(new_df, features=FEATURES)
        train_dataset_bt.save_to_disk(f"../data/aug_dataset/backtranslation")

        with open("../data.aug_dataset/dataset_dict.json", "rb") as f:
            dataset_dict = json.load(f)

        dataset_dict["splits"].append(self.dataset_name)

        with open("../data.aug_dataset/dataset_dict.json", "wb") as f:
            json.dump(dataset_dict, f)

        return train_dataset_bt

    def run_augmentation(self):
        print("\n#### Starting Backtranslation augmentation ####")
        new_data = self.get_aug_dataset()
        return new_data


_augmentation_entrypoints = {
    "question_generation": QuestionGeneration,
    "backtranslation": BackTranslation,
    "aeda": customAEDA,
}


def augmentation_entrypoint(augmentation_name):
    return _augmentation_entrypoints[augmentation_name]


def is_augmentation(augmentation_name):
    return augmentation_name in _augmentation_entrypoints


def create_augmentation(args, **kwang):
    result_dataset = None
    for aug, ratio in zip(args.augmentation, args.augmentation_ratio):
        if is_augmentation(aug):
            if ratio > 0:
                create_fn = augmentation_entrypoint(aug)
                augmentation = create_fn(data_ratio=ratio, **kwang)
                result = augmentation.run_augmentation()
                if result_dataset == None:
                    result_dataset = result
                else:
                    result_dataset = concatenate_datasets(
                        [result_dataset.flatten_indices(), result.flatten_indices()]
                    )
        else:
            raise RuntimeError("Unknown augmentation (%s)" % aug)
    # K-Fold 사용시는 train data 가 memory에서 불러오기 때문에 return 값을,
    # 아닐때는 저장하여 train.py에서 load하여 사용하게 됩니다.
    if result_dataset is not None:
        result_dataset.save_to_disk("../data/aug_dataset/selected/")
        print("#### Augmentation Process Complete! ####")
        return result_dataset
    else:
        return None
