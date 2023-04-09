import sys, os

model_name = "./partial_data/model-1"
directory = "results/energy/data/pre_twice"

os.system("python3 run.py -V 10 -S 0.3 -M {} -DIR {}".format(model_name, directory))

os.system("python3 run.py -V 0 -S 0.3 -M {} -DIR {}".format(model_name, directory))

os.system("python3 run.py -V 5 -S 0.3 -LM {} --epochs 100000 -DIR {}".format(model_name, directory))