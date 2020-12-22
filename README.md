# Pretrain with adapter

## How to Setup
First create a conda virtual environment with python >= 3.6, install pytorch >= 1.1.0. Then,
```
git clone https://github.com/seungwon1/pretrain_adapter.git
cd pretrain_adapter
pip install -r requirements.txt
```

##### (All experiments in the paper were performed with below dependencies)
- python=3.6.8
- torch=1.7.0
- adapter-transformer=1.0.1
- sklearn=0.23.2
- tensorboard=2.4.0

#### PreTraining

#### FineTuning

##### Baseline Roberta / TAPT
```
python finetune/new_train.py \
  --do_train \
  --do_eval \
  --data_dir datasets/hyperpartisan_news/ \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir results/baseline/hyperpartisan_news \
  --task_name hyperpartisan_news \
  --do_predict \
  --model_name_or_path roberta-base \
  --metric macro \
  --overwrite_output_dir \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --seed 7701 \
  --patience 10 \
```

For baseline roberta, set model_name_or_path as roberta-base. To finetune TAPT, set model_name_or_path as the path where the pretrained model is saved.

##### Adapter
```
python finetune/new_train.py \
  --do_train \
  --do_eval \
  --data_dir datasets/hyperpartisan_news/ \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --output_dir results/adapter/hyperpartisan_news/ \
  --task_name hyperpartisan_news \
  --do_predict \
  --load_best_model_at_end \
  --train_adapter \
  --model_name_or_path pt_adapter/hyperpartisan_news/ \
  --adapter_config pfeiffer \
  --metric macro \
  --overwrite_output_dir \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --sanity_check \
  --seed 7701 \
  --patience 10 \
```
For pre-trained adapter, set model_name_or_path as the path where the pretrained model is saved. To evaluate raw adapter, set model_name_or_path as roberta-base.



