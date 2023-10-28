import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("wandb_name", type=str, default="")
args = parser.parse_args()

python_dir = "python"
wandb_name = args.wandb_name

all_algs = ["emal", "uncertain", "mlp", "galaxy", "weak"]
uncertain_algs = ["emal", "uncertain", "galaxy"]
datasets = list(zip(["car", "coco", "voc", "celeb"], [500, 5000, 500, 10000], [20, 10, 20, 10]))[:1]

num_processes = 4
for alg, sub_name in list(zip(["random", "random_meta", "random_meta", "random_meta", "random_meta", "random_meta", "thompson_pos", "thompson_div", "albl_meta", "random_meta", "random_meta", "albl_meta", "thompson_div"],
                              [None, ["emal"], ["uncertain"], ["mlp"], ["galaxy"], ["weak"], all_algs, all_algs, all_algs, all_algs, uncertain_algs, uncertain_algs, uncertain_algs]))[-1:]:
    for data, batch_size, num_batch in datasets:
        processes = []
        for seed in np.linspace(1234, 9999999, num=num_processes, dtype=int):
            command = [python_dir, "main.py", str(seed), data, wandb_name, "resnet18", "--alg", alg, "--batch_size",
                       str(batch_size), "--num_batch", str(num_batch)]
            if sub_name is not None:
                command.append("--sub_procedure")
                command = command + sub_name
            processes.append(subprocess.Popen(command))
        for p in processes:
            p.wait()

