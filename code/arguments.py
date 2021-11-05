from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    reader_name: str = field(
        default="conv",
        metadata={
            "help": "Custom reader for Question Answering [base, conv, lstm, lstm_conv, rnn_conv]"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="klue/roberta-large",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    retriever_backbone: str = field(
        default="klue/bert-base",
        metadata={"help": "used retriever_backbone model from huggingface.co/models"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    retrieval_dataset_name: Optional[str] = field(
        default="../data/test_dataset",
        metadata={"help": "The name of the dataset to use for retrieval"},
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=5,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    train_retrieval: bool = field(default=False)
    retriever_type: str = field(
        default="coil",
        metadata={"help": "The type of retriever to use. [tfidf, bm25, coil]"},
    )
    augmentation: Optional[List[str]] = field(
        default_factory=lambda: ["question_generation", "backtranslation", "aeda"],
        metadata={
            "help": "Data augmentation to use [question_generation, backtranslation, aeda]"
        },
    )
    augmentation_ratio: Optional[List[float]] = field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        metadata={
            "help": "Data augmentation ratio to use (must match with the order of augmentation to use)"
        },
    )
    masking_prob: float = field(
        default=0.0,
        metadata={"help": "probability for question token masking"},
    )
    masking_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio of masked token in question"},
    )
    context_join_delimiter: str = field(
        default=" @@@ ",
        metadata={"help": "joining delimiter between topk contexts"},
    )


@dataclass
class TrainArguments(TrainingArguments):
    """
    TrainingArguments를 상속받아 custom 할수있게 제작된 argument입니다.
    """

    output_dir: str = field(default="./models/")
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "Define the number of epoch to run during training"},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Select evaluation strategy[no, steps, epoch]"},
    )
    eval_steps: int = field(default=500)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    logging_first_step: bool = field(default=False)
    logging_steps: int = field(default=100)
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={
            "help": "Select evaluation strategy[linear, cosine, cosine_with_restarts, polynomial, constant, constant with warmup]"
        },
    )
    warmup_steps: int = field(default=500)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_exact_match")
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ["start_positions", "end_positions"]
    )


@dataclass
class KFoldArguments:
    do_kfold: bool = field(default=False, metadata={"help": "enable K-Fold process"})
    split: int = field(default=5, metadata={"help": "number of splits in K-Fold"})
    prob_threshold: float = field(
        default=0.0,
        metadata={"help": "prediction probability threshold for softvoting"},
    )


@dataclass
class WandBArguments:
    """
    wandb 사용을 위한 Argument입니다
    """

    wandb_entity: str = field(
        default="nlprime", metadata={"help": "Entity name in wandb"}
    )
    wandb_project: str = field(default="MRC")
    wandb_group: str = field(default="reader_kfold")
    wandb_name: str = field(default="conv_batch_16")
