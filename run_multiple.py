import sys, os

# model_name = "partial_data/model-transfer"
directory = "results/energy/data/weighted_MH"

# os.system("python3 run.py -V 10 -S 0.3 -M {} -DIR {} --epochs 100000".format(model_name, directory))

# os.system("python3 run.py -V 0 -S 0.3  -LM {} -M {} -DIR {} --epochs 100000 --preepochs 0".format(model_name, model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -LM {} -M {} -DIR {} --epochs 100000 --preepochs 0".format(model_name, model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 3 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 8 -W 8192 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F -NoES".format(model_name, directory))
os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1 -LL 1".format(directory))

os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1000 -LL 100".format(directory))

os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 500 -LL 100".format(directory))

os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 100 -LL 10".format(directory))
