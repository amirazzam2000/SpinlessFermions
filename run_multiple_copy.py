import sys, os

model_name = "./partial_data/model-1-test"
os.system("python3 run.py -V 10 -S 0.3 -N 2 -M {} --epochs 100000".format(model_name))


os.system("python3 run.py -V 10 -S 0.3 -N 3 -LM {} --epochs 100000".format(model_name))
