import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from reader.ConvModel import ConvModel
from reader.LSTMModel import LSTMModel
from reader.LSTMConvModel import LSTMConvModel
from reader.RNNConvModel import RNNConvModel


_reader_entrypoints = {
    "conv": ConvModel,
    "lstm": LSTMModel,
    "lstm_conv": LSTMConvModel,
    "rnn_conv": RNNConvModel,
}


def reader_entrypoint(reader_name):
    return _reader_entrypoints[reader_name]


def is_reader(reader_name):
    return reader_name in _reader_entrypoints


def create_reader(reader_name, **kwargs):
    if is_reader(reader_name):
        create_fn = reader_entrypoint(reader_name)
        reader = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown reader (%s)" % reader_name)
    return reader
