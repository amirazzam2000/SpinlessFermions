import sys
import os


Enable_ES = True
num_particles = 4
num_pre_particles = 2
num_epochs = 100000
tag = "Single_ES_A4_New_loss"
model_name_tag = "-Single-ES-A4-NL"
directory_base = "results/energy/data/" + tag
out_dir = "./out/" + tag


noes = "" if Enable_ES else "-NoES"

if not os.path.exists(directory_base):
    os.system("mkdir {}".format(directory_base))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))

# w WMH no trans w freezing

directory = directory_base + "/w_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-w-MH-C-" + model_name_tag
os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
    num_pre_particles, model_name, directory, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs 100000  --preepochs 0 -F {} -T {} > {}/no_trans_w_freezing_w_MH.txt".format(
    num_particles, model_name, directory, noes, tag, out_dir))
