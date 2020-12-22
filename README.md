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

## PreTraining

#### 1. TAPT
```
python pretrain.run_language_modeling --train_data_file datasets/chemprot/train.txt \
  --line_by_line \
  --output_dir tapt/roberta-tapt-acl-TAPT \
  --model_type roberta-base \
  --eval_data_file=datasets/citation_intent/test.txt \
  --tokenizer_name roberta-base \
  --mlm \
  --per_gpu_train_batch_size 8 \
  --gradient_accumulation_steps 32  \
  --model_name_or_path roberta-base \
  --do_eval \
  --evaluate_during_training  \
  --do_train \
  --num_train_epochs 100  \
  --learning_rate 0.0001 \
  --logging_steps 900 \
```

#### 2. Adapter
```
python pretrain.run_language_modeling_with_adapters --train_data_file datasets/scierc/train.txt \
  --line_by_line \
  --output_dir tapt-adapter/scierc/ \
  --model_type roberta-base \
  --tokenizer_name roberta-base \
  --mlm \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 64  \
  --model_name_or_path roberta-base \
  --do_eval \
  --do_train \
  --num_train_epochs 100  \
  --learning_rate 0.0001 \
  --logging_steps 900 \
  --adapter_name=scierc \
  --overwrite_output_dir \
  --evaluate_during_training  \
  --eval_data_file=datasets/scierc/dev.txt \
```

## FineTuning
After executing below commands, metrics such as loss or evaluation scores can be easily visualized using tensorboard.
```
tensorboard --logdir=runs/
```

#### 1. Baseline Roberta / TAPT
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

Args
```
evaluation_strategy : when to evaluate the model during training
load_best_model_at_end, metric_for_best_model: load best model based on the metric specified from the checkpoints that are saved with respect to evaluation_strategy
patience: the number of epochs which the metric get worse to be considered to execute early stopping
```
For more information about arguments, see
https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/training_args.py

#### 2. Adapter (pretrained/raw)
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
  --seed 7701 \
  --patience 10 \
```
For pre-trained adapter, set model_name_or_path as the path where the pretrained model is saved. To evaluate raw adapter, set model_name_or_path as roberta-base.

For more information about arguments, see
https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapter_training.py

#### 3. Adapter fusion
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
  --output_dir results/fusion/hyperpartisan_news/ \
  --task_name hyperpartisan_news \
  --do_predict \
  --load_best_model_at_end \
  --train_fusion \
  --model_name_or_path roberta-base \
  --adapter_config pfeiffer \
  --metric macro \
  --fusion_adapter_path1 pt_adapter/hyperpartisan_news/ \
  --fusion_adapter_path2 pt_adapter/imdb/ \
  --overwrite_output_dir \
  --evaluation_strategy epoch \
  --metric_for_best_model f1 \
  --seed 7701 \
  --patience 10 \
```



