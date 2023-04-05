import sys, os

model_name = "./partial_data/model-1"
os.system("python3 run.py -V 10 -S 0.3 -M {}".format(model_name))


os.system("python3 run.py -V 5 -S 0.3 -LM {} -M {}".format(model_name, model_name))