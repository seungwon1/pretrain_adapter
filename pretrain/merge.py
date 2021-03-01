import argparse
import subprocess
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                        type=str)

parser.add_argument('--cuda',
                        type=str)

parser.add_argument('--etype',
                        type=str)

parser.add_argument('--pat',
                        type=str)

parser.add_argument('--lr',
                        type=str)

parser.add_argument('--eps',
                        type=str)

parser.add_argument('--rdf',
                        default="16",
                        type=str)

parser.add_argument('--rep',
                        type=int)

parser.add_argument('--output_key',
                        default="",
                        type=str)

parser.add_argument('--save_limit',
                        default=1,
                        type=int)

parser.add_argument('--stage',
                        default='ft',
                        type=str)

args = parser.parse_args()

cuda = args.cuda
#os.environ["CUDA_VISIBLE_DEVICES"]= cuda

dataset, etype, lr, epochs, patience, reduction_factor, out_key = args.dataset, args.etype, args.lr, args.eps, args.pat, args.rdf, args.output_key
stage = args.stage
save_limit = args.save_limit

if len(out_key) !=0:
    out_key = "/"+out_key+"/"
else:
    out_key = "/"

if dataset == "rct-20k" or dataset == "chemprot":
    f1 = "micro"
else:
    f1 = "macro"

dataset_dir = dataset
if dataset == "helpfulness":
    dataset_dir = "amazon"
elif dataset == "agnews":
    dataset_dir = "ag"

for _ in range(args.rep):
    seed = str(random.randint(0, 1e5))

    if etype == "baseline":
        commands = ["python ",
            "new_train" + cuda + ".py ",
            "--do_train ",
            "--do_eval ",
            "--data_dir datasets/" + dataset_dir,
            " --max_seq_length 512",
            " --per_device_train_batch_size 16",
            " --gradient_accumulation_steps 1",
            " --per_device_eval_batch_size 16",
            " --learning_rate " + lr,
            " --num_train_epochs " + epochs,
            " --output_dir results/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr,
            " --task_name "+ dataset,
            " --do_predict",
            " --model_name_or_path roberta-base",
            " --metric " + f1,
            " --overwrite_output_dir",
            " --evaluation_strategy epoch",
            " --load_best_model_at_end",
            " --metric_for_best_model f1",
            " --seed " + seed,
            " --patience " + patience,
            " --save_total_limit " + str(save_limit),
            " --logging_dir runs/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr]

    elif etype == "tapt":
        tapt_dir = dataset_dir
        if dataset == "agnews":
            tapt_dir = "ag"
        elif dataset == "helpfulness":
            tapt_dir = "helpfulness"#/checkpoint-47000/"

        commands = ["python ",
                    "new_train" + cuda + ".py ",
                    "--do_train ",
                    "--do_eval ",
                    "--data_dir datasets/" + dataset_dir,
                    " --max_seq_length 512",
                    " --per_device_train_batch_size 16",
                    " --gradient_accumulation_steps 1",
                    " --per_device_eval_batch_size 16",
                    " --learning_rate " + lr,
                    " --num_train_epochs " + epochs,
                    " --output_dir results/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr,
                    " --task_name " + dataset,
                    " --do_predict",
                    " --model_name_or_path " + "pt_tapt/"+tapt_dir + "/",
                    " --metric " + f1,
                    " --overwrite_output_dir",
                    " --evaluation_strategy epoch",
                    " --load_best_model_at_end",
                    " --metric_for_best_model f1",
                    " --seed " + seed,
                    " --patience " + patience,
                    " --save_total_limit " + str(save_limit),
                    " --logging_dir runs/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr]

    elif etype == "raw_adapter":
        commands = ["python ",
                    "new_train" + cuda + ".py ",
                    "--do_train ",
                    "--do_eval ",
                    "--data_dir datasets/" + dataset_dir,
                    " --max_seq_length 512",
                    " --per_device_train_batch_size 16",
                    " --per_device_eval_batch_size 16",
                    " --gradient_accumulation_steps 1",
                    " --learning_rate " + lr,
                    " --num_train_epochs " + epochs,
                    " --output_dir results/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr,
                    " --task_name " + dataset,
                    " --do_predict",
                    " --train_adapter_wop",
                    " --adapter_config pfeiffer",
                    " --model_name_or_path roberta-base",
                    " --metric " + f1,
                    " --overwrite_output_dir",
                    " --evaluation_strategy epoch",
                    " --load_best_model_at_end",
                    " --metric_for_best_model f1",
                    " --seed " + seed,
                    " --patience " + patience,
                    " --adapter_reduction_factor " + reduction_factor,
                    " --save_total_limit " + str(save_limit),
                    " --logging_dir runs/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr]

    elif etype == "adapter":
        adapter_dir = dataset_dir
        if dataset == "agnews":
            adapter_dir = "ag"
        elif dataset == "helpfulness":
            adapter_dir = "helpfulness"#/40epoch"

        commands = ["python ",
                   "new_train" + cuda + ".py ",
                   "--do_train ",
                   "--do_eval ",
                   "--data_dir datasets/" + dataset_dir,
                   " --max_seq_length 512",
                   " --per_device_train_batch_size 16",
                   " --per_device_eval_batch_size 16",
                   " --gradient_accumulation_steps 1",
                   " --learning_rate " + lr,
                   " --num_train_epochs " + epochs,
                   " --output_dir results/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr,
                   " --task_name " + dataset,
                   " --do_predict",
                   " --train_adapter",
                   " --adapter_config pfeiffer",
                   " --model_name_or_path " + "pt_adapter/" + adapter_dir + "/",
                   " --metric " + f1,
                   " --overwrite_output_dir",
                   " --evaluation_strategy epoch",
                   " --load_best_model_at_end",
                   " --metric_for_best_model f1",
                   " --seed " + seed,
                   " --patience " + patience,
                   " --save_total_limit " + str(save_limit),
                   " --logging_dir runs/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr]

    elif etype == "fusion":
        adapter_dir = dataset_dir
        if dataset == "agnews":
            adapter_dir = "ag"

        commands = ["python ",
                    "new_train" + cuda + ".py ",
                    "--do_train ",
                    "--do_eval ",
                    "--data_dir datasets/" + dataset_dir,
                    " --max_seq_length 512",
                    " --per_device_train_batch_size 8",
                    " --per_device_eval_batch_size 8",
                    " --gradient_accumulation_steps 2",
                    " --learning_rate " + lr,
                    " --num_train_epochs " + epochs,
                    " --output_dir results/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr,
                    " --task_name " + dataset,
                    " --do_predict",
                    " --train_fusion",
                    " --adapter_config pfeiffer",
                    " --model_name_or_path roberta-base",
                    " --fusion_adapter_path1 pt_for_fusion/chemprot/1/",
                    " --fusion_adapter_path2 pt_for_fusion/rct-20k/1/",
                    " --fusion_adapter_path3 pt_for_fusion/citation_intent/1/",
                    " --fusion_adapter_path4 pt_for_fusion/scierc/1/",
                    " --fusion_adapter_path5 pt_for_fusion/helpfulness/2/",
                    " --fusion_adapter_path6 pt_for_fusion/imdb/1/",
                    " --fusion_adapter_path7 pt_for_fusion/hyperpartisan_news/1/",
                    " --fusion_adapter_path8 pt_for_fusion/agnews/1/",
                    " --metric " + f1,
                    " --overwrite_output_dir",
                    " --evaluation_strategy epoch",
                    " --load_best_model_at_end",
                    " --metric_for_best_model f1",
                    " --seed " + seed,
                    " --patience " + patience,
                    " --save_total_limit " + str(save_limit),
                    " --logging_dir runs/" + etype + "/" + dataset + out_key + seed + "_e" + epochs + "p" + patience + "lr" + lr]

    if "test" in out_key:
        commands.append(" --sanity_check")

    if stage == 'ft':
        subprocess.run(" ".join(commands), shell=True, check=True)
