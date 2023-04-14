import sys
import os

model_name = "./partial_data/model-A3"
os.system("python3 run.py -V 10 -S 0.3 -N 2 -M {} --epochs 100000".format(model_name))


os.system("python3 run.py -V 10 -S 0.3 -N 3 -LM {} --epochs 100000 --preepochs 0 -F 1".format(model_name))
