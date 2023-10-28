import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("wandb_name", type=str, default="")
args = parser.parse_args()

python_dir = "python3"
wandb_name = args.wandb_name

all_algs = ["galaxy", "confidence", "badge", "mlp", "similar"]
uncertain_algs = ["galaxy", "confidence", "badge"]
datasets = list(zip(["caltech", "kuzushiji", "cifar10_imb_2", "svhn_imb_2", "cifar100_imb_10", "imagenet"],
                    [1000, 1000, 2000, 7000, 2000, 20000], [20, 10, 20, 10, 20, 7]))[-1:]
algorithms = list(
    zip(["random", "random_meta", "random_meta", "random_meta", "random_meta", "random_meta", "random_meta",
         "albl_meta", "thompson_div", "random_meta", "albl_meta", "thompson_div"],
        [None, ["galaxy"], ["confidence"], ["mlp"], ["similar"], ["badge"], all_algs, all_algs, all_algs,
         uncertain_algs, uncertain_algs, uncertain_algs]))[-1:]

# for alg, sub_name in algorithms:
#     for data, batch_size, num_batch in datasets:
#         processes = []
#         for i, seed in list(enumerate(np.linspace(1234, 9999999, num=num_processes, dtype=int))):
#             command = [python_dir, "main.py", str(seed), data, wandb_name, "resnet18", "--alg", alg, "--batch_size",
#                        str(batch_size), "--num_batch", str(num_batch)]
#             if sub_name is not None:
#                 command.append("--sub_procedure")
#                 command = command + sub_name
#             processes.append(subprocess.Popen(command))
#             if i == 1 or i == 3:
#                 for p in processes:
#                     p.wait()
#                 processes = []

for alg, sub_name in algorithms:
    for data, batch_size, num_batch in datasets:
        num_processes = 1 if data == "imagenet" else 4
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

# for alg, sub_name in algorithms:
#     for data, batch_size, num_batch in datasets:
#         for seed in np.linspace(1234, 9999999, num=num_processes, dtype=int)[2:]:
#             processes = []
#             command = [python_dir, "main.py", str(seed), data, wandb_name, "resnet18", "--alg", alg, "--batch_size",
#                        str(batch_size), "--num_batch", str(num_batch)]
#             if sub_name is not None:
#                 command.append("--sub_procedure")
#                 command = command + sub_name
#             processes.append(subprocess.Popen(command))
#             for p in processes:
#                 p.wait()
