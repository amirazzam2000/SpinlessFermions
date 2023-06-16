import sys
import os

directory_base = "results/energy/data/ES_NoES_test"

out_dir = "./out/ES_NoES_test"

model_name_tag = "-ES-NoES-test"

if not os.path.exists(directory_base):
    os.system("mkdir {}".format(directory_base))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))



directory = directory_base + "/ES_then_NoES"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-ES-then-NoES" + model_name_tag
os.system("python3 run.py -N 6 -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES' > {}/ES.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'NoES' -NoES > {}/NoES.txt".format(model_name, directory, out_dir))


directory = directory_base + "/just_NoES"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-ES-then-NoES" + model_name_tag
os.system("python3 run.py -N 6 -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'NoES_first'-NoES > {}/NoES_first.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'NoES_second' -NoES > {}/NoES_second.txt".format(model_name, directory, out_dir))
