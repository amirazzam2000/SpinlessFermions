# import sys
# import os

# model_name = "./partial_data/model-A3-V5"
# load_name = "./partial_data/model-A3"
# directory= "./results/energy/data/A3_no_pre"
# os.system("python3 run.py -V 5 -S 0.3 -N 3 --epochs 100000 --preepochs 0 -DIR {}".format(directory))

# directory = "./results/energy/data/A3_pre"

# os.system("python3 run.py -V 5 -S 0.3 -N 2 -LM {} -M {} --epochs 100000".format(load_name, model_name))

# os.system("python3 run.py -V 5 -S 0.3 -N 3 -LM {} --epochs 100000 --preepochs 0 -F 1 -DIR {}".format(model_name, directory))

import sys
import os
import time

directory = "./results/energy/data/A3_new_stopping6-experimental"
model_name = "./partial_data/model-A3-V5-es6.new"

if os.path.exists(model_name):
    os.remove(model_name)

os.system("python3 run.py -V 5 -S 0.3 -N 2 -M {} -DIR {} --epochs 100000".format(model_name, directory))
# os.system("python3 run.py -V 5 -S 0.3 -N 3 -M {} -DIR {} --epochs 100000 --preepochs 0".format(model_name, directory))

# t0 = time.time()
# os.system("python3 run.py -V 5 -S 0.3 -N 3 -LM {} -DIR {} --epochs 100000 --preepochs 0".format(model_name, directory))
# print(time.time() - t0)

# Time taken:  0.1581332510104403

# t0 = time.time()
# os.system("python3 run.py -V 5 -S 0.3 -N 3 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))
# print(time.time() - t0)

# Time taken:  0.15845419198740274

directory = "./results/energy/data/higher_N_no_ES"

t0 = time.time()
os.system("python3 run.py -V 5 -S 0.3 -N 4 -LM {} -DIR {} --epochs 100000 --preepochs 0 -NoES".format(model_name, directory))
print(time.time() - t0)
# Number of parameters:     8712
# Epoch: 100000 | Energy: 8.8474 +/- 0.4033 | CI: 9.7266 | Walltime: 2.14e-01 (s) | window loss difference: 0.000336 | avg overlap : 0.000000

# Time taken:  0.2144414649810642  (accumulated wall time)
#          22570.24262905121 (recorded time)

t0 = time.time()
os.system("python3 run.py -V 5 -S 0.3 -N 5 -LM {} -DIR {} --epochs 100000 --preepochs 0 -NoES".format(model_name, directory))
print(time.time() - t0)
# Number of parameters:     8778
# Epoch: 100000 | Energy: 14.0148 + /- 0.9321 | CI: 15.4063 | Walltime: 2.96e-01 (s) | window loss difference: 0.000941 | avg overlap: 0.000000

# Time taken:  0.2957860060269013  (accumulated wall time)
# 27859.32918214798 (recorded time)

directory = "./results/energy/data/higher_N_ES"
t0 = time.time()
os.system("python3 run.py -V 5 -S 0.3 -N 4 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))
print(time.time() - t0)
# reduced number of parameters is:  264
# Epoch: 100000 | Energy: 8.8651 +/- 0.8032 | CI: 9.7266 | Walltime: 2.83e-01 (s) | window loss difference: 0.001732 | avg overlap : 0.000000

# Time taken:  0.21609469200484455  (accumulated wall time)
#          22709.9884557724 (recorded time)

t0 = time.time()
os.system("python3 run.py -V 5 -S 0.3 -N 5 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))
print(time.time() - t0)
# reduced number of parameters is:  330
# Epoch: 100000 | Energy: 14.0678 + /- 1.1999 | CI: 15.4063 | Walltime: 4.41e-01 (s) | window loss difference: 0.005070 | avg overlap: 0.000000

# Time taken:  0.4411039029946551  (accumulated wall time)
#           90712.18101072311 (recorded time)
