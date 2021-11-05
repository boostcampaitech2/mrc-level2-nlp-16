```
cd /opt/ml/code/retriever/coil
```

# Make triplet_dataset for training
coil 모델을 학습하기 위해서는 make_triplet.py 를 실행하여 `query + positive sample + negative samples` 로 이루어진 triplet_dataset 을 생성해야 합니다.
- save_dir : triplet_dataset.tsv 저장경로
- train_dataset : 기존 train dataset 위치
- tokenizer_name : bm25 에 사용되는 토크나이저(from huggingface)
- fuzz_ratio : postive sample 과 negative_sample 의 최대 유사도
- neg_samples : negative sample 생성 개수

```
cd /opt/ml/code/retriever/coil
```

```
# 예시
python data_helper/make_triplet.py --save_dir /opt/ml/data/coil_dataset --train_dataset /opt/ml/data/train_dataset --tokenizer_name klue/bert-base --fuzz_ratio 90 --neg_samples 7
```

triplet_dataset.tsv 를 생성한 후, build_train_from_triplet.py 를 통해 trainer 가 학습할 수 있는 형태로 만들어 줍니다.
- tokenizer_name : 인코딩에 사용되는 tokenizer(from huggingface)
- file : triplet_dataset.tsv 저장된 경로
- truncate : truncate에 사용되는 max_length (long document의 경우 512)
- json_dir : coil_datset.json 저장경로

```
# 예시
python data_helper/build_train_from_triplet.py --tokenizer_name klue/bert-base --file /opt/ml/data/coil_dataset/triplet_dataset.tsv --truncate 512 --json_dir /opt/ml/data/coil_dataset
```


# Training

```
# 예시
python run_macro.py --output_dir ./models/klue/bert-base --model_name_or_path klue/bert-base --do_train --save_steps 2000 --train_path /opt/ml/data/coil_dataset/triplet_dataset.json --q_max_len 64 --p_max_len 512 --fp16 --per_device_train_batch_size 4 --train_group_size 8 --cls_dim 768 --token_dim 32 --num_train_epochs 5 --overwrite_output_dir --no_sep --pooling max
```
