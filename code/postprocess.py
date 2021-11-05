import os
import collections
import json
import pandas as pd

from scipy.stats import mode
from konlpy.tag import Mecab

# inference후 hard-voting을 k-fold용으로 구현한것입니다.
# 2개 이상의 답의 횟수가 동률일경우엔 n_prediction에서 확률이 높은 경우를 도출합니다.
def ensemble_hardvoting_kfold(model_dir):
    print("\n#### Ensemble Hard-Voting ####")
    pred_list = []
    dir_list = os.listdir(model_dir)
    for model in dir_list:
        if model.startswith("fold_"):
            path = os.path.join(model_dir, model)
            if os.path.isdir(path):
                prediction = pd.read_json(
                    os.path.join(path, "nbest_predictions.json"), orient="index"
                )
                pred_list.append(prediction[0])

    mrc_id_list = pred_list[0].index.tolist()
    prediction_result = collections.OrderedDict()
    for mrc_id in mrc_id_list:
        output = []
        best_prob = 0.0
        best_ans = ""
        for pred in pred_list:
            answer = pred[mrc_id]["text"]
            output.append(answer)
            if pred[mrc_id]["probability"] > best_prob and answer != ".":
                best_prob = pred[mrc_id]["probability"]
                best_ans = answer
        mode_result = mode(output)
        if mode_result[1][0] == 1:
            prediction_result[mrc_id] = best_ans
        else:
            prediction_result[mrc_id] = mode_result[0][0]
    ensemble_output_dir = os.path.join(model_dir, "ensemble")
    if not os.path.exists(ensemble_output_dir):
        os.mkdir(ensemble_output_dir)
    with open(
        os.path.join(ensemble_output_dir, "predictions.json"), "w", encoding="utf-8"
    ) as writer:
        writer.write(json.dumps(prediction_result, indent=4, ensure_ascii=False) + "\n")
    print("#### Ensemble Hard-Voting Complete! ####\n")


def ensemble_softvoting_kfold(model_dir, prob_threshold=0.0):
    print("\n#### Ensemble Soft-Voting ####")
    pred_list = []
    dir_list = os.listdir(model_dir)
    for model in dir_list:
        if model.startswith("fold_"):
            path = os.path.join(model_dir, model)
            if os.path.isdir(path):
                nbest_prediction = pd.read_json(
                    os.path.join(path, "nbest_predictions.json"),
                    orient="index",
                )
                pred_list.append(nbest_prediction)

    mrc_id_list = pred_list[0][0].index.tolist()
    prediction_result = collections.OrderedDict()

    for mrc_id in mrc_id_list:
        best_answer = ""
        best_prob = 0

        prediction_result[mrc_id] = collections.defaultdict(float)
        for pred in pred_list:
            for answer in pred.loc[mrc_id]:
                if answer == None:
                    continue
                text = answer["text"]
                prob = answer["probability"]
                if text != ".":
                    if prob >= prob_threshold:
                        prediction_result[mrc_id][text] += prob
                    else:
                        if prob > best_prob:
                            best_answer = text

        # prob_threshold 를 넘는 prediction이 없는 경우 prob이 가장 높은 단일 best_answer 사용
        if len(prediction_result[mrc_id]) == 0:
            prediction_result[mrc_id] = best_answer
        else:
            prediction_result[mrc_id] = sorted(
                prediction_result[mrc_id].items(), key=lambda x: x[1], reverse=True
            )[0][0]

    ensemble_output_dir = os.path.join(model_dir, "ensemble")
    if not os.path.exists(ensemble_output_dir):
        os.mkdir(ensemble_output_dir)
    with open(
        os.path.join(ensemble_output_dir, "softvoting_predictions.json"),
        "w",
        encoding="utf-8",
    ) as writer:
        writer.write(json.dumps(prediction_result, indent=4, ensure_ascii=False) + "\n")
    print("#### Ensemble Soft-Voting Complete! ####\n")


def ensemble_hardvoting(candidate_dir):
    file_list = os.listdir(candidate_dir)
    candidate_list = []
    for idx in range(len(file_list)):
        predict_name = "predictions" + str(idx) + ".json"
        candidate = pd.read_json(
            os.path.join(candidate_dir, predict_name), orient="index"
        )
        candidate_list.append(candidate)
    index_list = candidate_list[0].index.tolist()
    prediction_result = collections.OrderedDict()
    for index in index_list:
        output = []
        for json_file in candidate_list:
            output.append(json_file.loc[index][0])
        prediction_result[index] = mode(output)[0][0]
    with open(
        os.path.join(candidate_dir, "ensemble_result.json"), "w", encoding="utf-8"
    ) as writer:
        writer.write(json.dumps(prediction_result, indent=4, ensure_ascii=False) + "\n")


def json_postprocess(file_dir):
    pred_file = pd.read_json(os.path.join(file_dir, "predictions.json"), orient="index")
    mecab = Mecab()
    pred_result = collections.OrderedDict()
    pred_index = pred_file.index.tolist()
    for index in pred_index:
        word = pred_file[0][index]
        if len(word.split()) == 1:
            word_mecab = mecab.pos(word)
            if len(word_mecab) == 1:
                if (
                    word_mecab[0][1][0] != "N"
                    and word_mecab[0][1][0:2] != "SN"
                    and word_mecab[0][1][0:2] != "SL"
                ):
                    word = "."

        else:
            word_mecab = mecab.pos(word)
            if word_mecab[-1][1][0] == "J":
                word = word[: -(len(word_mecab[-1][0]))]

        pred_result[index] = word

    with open(
        os.path.join(file_dir, "prediction_postprocess.json"), "w", encoding="utf-8"
    ) as writer:
        writer.write(json.dumps(pred_result, indent=4, ensure_ascii=False) + "\n")


def ensemble_softvoting_multimodel(candidate_dir, prob_threshold=0.0):
    print("\n#### Ensemble Soft-Voting ####")
    candidate_list = os.listdir(candidate_dir)
    pred_list = []
    for candidate in candidate_list:
        candidate_path = os.path.join(candidate_dir, candidate)
        model_list = os.listdir(candidate_path)
        for model in model_list:
            if model.startswith("fold_"):
                model_path = os.path.join(candidate_path, model)
                if os.path.isdir(model_path):
                    nbest_prediction = pd.read_json(
                        os.path.join(model_path, "nbest_predictions.json"),
                        orient="index",
                    )
                    pred_list.append(nbest_prediction)

    mrc_id_list = pred_list[0][0].index.tolist()
    prediction_result = collections.OrderedDict()

    for mrc_id in mrc_id_list:
        prediction_result[mrc_id] = collections.defaultdict(float)
        for pred in pred_list:
            for answer in pred.loc[mrc_id]:
                if answer == None:
                    continue
                text = answer["text"]
                prob = answer["probability"]
                if text != "." and prob >= prob_threshold:
                    prediction_result[mrc_id][text] += prob

        prediction_result[mrc_id] = sorted(
            prediction_result[mrc_id].items(), key=lambda x: x[1], reverse=True
        )[0][0]

    ensemble_output_dir = os.path.join(candidate_dir, "ensemble")
    if not os.path.exists(ensemble_output_dir):
        os.mkdir(ensemble_output_dir)
    with open(
        os.path.join(ensemble_output_dir, "multi_softvoting_predictions.json"),
        "w",
        encoding="utf-8",
    ) as writer:
        writer.write(json.dumps(prediction_result, indent=4, ensure_ascii=False) + "\n")
    print("#### Ensemble Soft-Voting Complete! ####\n")
