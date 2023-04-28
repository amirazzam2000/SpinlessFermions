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

# directory = "./results/energy/data/A3_new_stopping6-experimental"
model_name = "./partial_data/model-A3-V5-es6.new"

# if os.path.exists(model_name):
#     os.remove(model_name)

# os.system("python3 run.py -V 5 -S 0.3 -N 2 -M {} -DIR {} --epochs 100000".format(model_name, directory))
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

t0 = time.time()
os.system("python3 run.py -V 5 -S 0.3 -N 5 -LM {} -DIR {} --epochs 100000 --preepochs 0 -NoES".format(model_name, directory))
print(time.time() - t0)

directory = "./results/energy/data/higher_N_ES"
t0 = time.time()
os.system("python3 run.py -V 5 -S 0.3 -N 4 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))
print(time.time() - t0)

t0 = time.time()
os.system("python3 run.py -V 5 -S 0.3 -N 5 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))
print(time.time() - t0)
