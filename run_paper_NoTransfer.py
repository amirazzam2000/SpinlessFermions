import sys
import os


tag = "PT_21_Nov_noTransfer_nofreezing"
Enable_ES = True
num_epochs = 1000000
num_particles = 4
num_pre_particles = 2
directory_base = "results/energy/data/" + tag
out_dir = "./out/" + tag
model_name_tag = tag


noes = "" if Enable_ES else "-NoES"

if not os.path.exists(directory_base):
    os.system("mkdir {}".format(directory_base))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))

# w WMH no trans no freezing w part

directory = directory_base + "/w_WMH_no_trans_no_freezing_no_part"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))


model_name = "partial_data/model-A2-no-freezing-no-transfer-w-MH-C-" + model_name_tag
os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
    num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

num_particles_list = [4, 6, 8, 10, 12]
load_model_name = "partial_data/model-A2-no-freezing-w-MH-C-" + model_name_tag

for i in num_particles_list:

    model_name = f"partial_data/model-A{i}-no-freezin-no-transfer-w-MH-C-" + model_name_tag
    os.system("python3 run.py -N {} -V 10 -S 0.5 -M {} -DIR {} -UL 100 -LL 1 --epochs {} {} -T {} -SW 40 -W 8000 > {}/no_trans_no_part_{}_no_freezing_w_MH.txt".format(
        i, model_name, directory, num_epochs, noes, tag, out_dir, i))
    load_model_name = model_name

