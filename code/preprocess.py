import re
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# context에 존재하는 개행문자를 삭제하여 줍니다.
def context_process(dataset):
    print("\n#### Context Pre-Process ####")
    data_in_df = pd.DataFrame(dataset)
    output = []
    for context in tqdm(data_in_df["context"]):
        new_context = context.replace("\\n", "  ")
        output.append(new_context)

    # post-processing
    process = pd.DataFrame(
        {
            "__index_level_0__": dataset["__index_level_0__"],
            "answers": dataset["answers"],
            "context": output,
            "document_id": dataset["document_id"],
            "id": dataset["id"],
            "question": dataset["question"],
            "title": dataset["title"],
        }
    )
    print("#### Context Pre-Process Complete ####")
    return Dataset.from_pandas(process)


def wiki_preprocess(text):
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]",
        " ",
        text,
    )
    return text
