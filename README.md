# Pretrain with adapter

## How to Setup
First create a virtual conda environment with python >= 3.6, install pytorch >= 1.1.0. Then,
```
git clone https://github.com/seungwon1/pretrain_adapter.git
cd pretrain_adapter
pip install -r requirements.txt
```

All experiments in the paper were performed with below dependencies.
- python=3.8.5
- torch=1.7.1
- adapter-transformer=1.1.0
- sklearn=0.23.2
- tensorboard=2.4.0
- datasets=1.4.1

Hardware Specification: NVIDIA RTX 3090

## PreTraining
After executing the below commands, the perplexity of LM evaluated after each epoch can be easily visualized using tensorboard.
```
tensorboard --logdir=runs/
```

#### 1. TAPT
```
python run_language_modeling.py --train_data_file datasets/rct-20k/train.txt \
  --line_by_line \
  --output_dir tapt/roberta-tapt-rct-20k \
  --model_type roberta-base \
  --eval_data_file=datasets/rct-20k/test.txt \
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
python run_language_modeling_with_adapters.py --train_data_file datasets/scierc/train.txt \
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
After executing the below commands, metrics such as loss or evaluation scores can be easily visualized using tensorboard.
```
tensorboard --logdir=runs/
```

#### 1. Baseline Roberta / TAPT
```
python new_train.py \
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
  --patience 10 \
  --adapter_reduction_factor 16 \
  --save_total_limit 3 \
  --logging_dir runs/baseline/hyperpartisan_news \
```

For baseline roberta, set model_name_or_path as roberta-base. To finetune TAPT, set model_name_or_path as the path where the pretrained model is saved.

Args
```
evaluation_strategy : when to evaluate the model during training
load_best_model_at_end, metric_for_best_model: load best model based on the metric specified from the checkpoints that are saved with respect to evaluation_strategy
patience: the number of epochs in which the metric get worse to be considered to execute early stopping
adapter_reduction_factor: reduction_factor to determine the network size of adapter layers
save_total_limit: total number of checkpoints saved after validation 
logging_dir: Tensorboard directory 
```
For more information about arguments, see
https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/training_args.py

#### 2. Adapter (pretrained/raw)
```
python new_train.py \
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
  --patience 10 \
  --adapter_reduction_factor 16 \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --save_total_limit 3 \
  --logging_dir runs/adapter/hyperpartisan_news \
```
For pre-trained adapter, set model_name_or_path as the path where the pretrained model is saved. To evaluate raw adapter, set model_name_or_path as roberta-base.

For more information about arguments, see
https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapter_training.py

#### 3. Adapter fusion
```
python new_train.py \
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
  --patience 10 \
  --adapter_reduction_factor 16 \
  --evaluation_strategy epoch \
  --metric_for_best_model f1 \
  --overwrite_output_dir \
  --save_total_limit 3 \
  --logging_dir runs/fusion/hyperpartisan_news \
```


