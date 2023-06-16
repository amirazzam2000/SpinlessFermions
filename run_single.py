import sys
import os


Enable_ES = True
num_particles = 4
num_pre_particles = 2
num_epochs = 10000
tag = "Single_ES_A4"
model_name_tag = "-Single-ES-A4"
directory_base = "results/energy/data/" + tag
out_dir = "./out/" + tag


noes = "" if Enable_ES else "-NoES"

if not os.path.exists(directory_base):
    os.system("mkdir {}".format(directory_base))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))

# no WMH no trans no freezing

directory = directory_base + "/no_WMH_no_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 1 -LL 1 --epochs {} --preepochs 0 {} -T {} > {}/no_trans_no_freezing_no_MH.txt".format(num_particles, directory, num_epochs, noes, tag, out_dir))
