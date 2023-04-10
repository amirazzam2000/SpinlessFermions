import sys, os

model_name = "./partial_data/model-pre-twice"
directory = "results/energy/data/pre_twice"

os.system("python3 run.py -V 10 -S 0.3 -M {} -DIR {} --epochs 10000".format(model_name, directory))

os.system("python3 run.py -V 0 -S 0.3  -LM {} -M {} -DIR {} --epochs 10000 --preepochs 0".format(model_name, model_name, directory))

os.system("python3 run.py -V 5 -S 0.3 -LM {} -DIR {} --epochs 10000 --preepochs 0".format(model_name, directory))
