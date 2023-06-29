import sys
import os


tag = "ES_29_Jun"
Enable_ES = True
num_epochs = 100000
num_particles = 4
num_pre_particles = 2
directory_base = "results/energy/data/" + tag
out_dir = "./out/" + tag
model_name_tag = "-ES-29-Jun"


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

os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} > {}/no_trans_no_freezing_no_MH.txt".format(
    num_particles, directory, num_epochs, noes, tag, out_dir))

# no WMH no trans w freezing

directory = directory_base + "/no_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing" + model_name_tag
os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} > {}/dumb.txt".format(
    num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/no_trans_w_freezing_no_MH.txt".format(
    num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# no WMH w trans no freezing

directory = directory_base + "/no_WMH_w_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
    num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_no_freezing_no_MH.txt".format(
    num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# w WMH no trans no freezing

directory = directory_base + "/w_WMH_no_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} > {}/no_trans_no_freezing_w_MH.txt".format(
    num_particles, directory, num_epochs, noes, tag + 'WMH', out_dir))

# w WMH no trans w freezing

directory = directory_base + "/w_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-w-MH-C-" + model_name_tag
os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
    num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/no_trans_w_freezing_w_MH.txt".format(
    num_particles, model_name, directory, num_epochs, noes, tag, out_dir))


# w WMH w trans no freezing

directory = directory_base + "/w_WMH_w_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
    num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_no_freezing_w_MH.txt".format(
    num_particles, model_name, directory, num_epochs, noes, tag, out_dir))


# w WMH w trans w freezing

directory = directory_base + "/w_WMH_w_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-for-trans-w-MH" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
    num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

model_name_f = "partial_data/model-A2-after-freezing-for-trans-w-MH" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/w_trans_w_freezing_w_MH_part1.txt".format(
    num_particles, model_name_f, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_w_freezing_w_MH_part2.txt".format(
    num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing", out_dir))
os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} >> {}/w_trans_w_freezing_w_MH_part2.txt".format(
    num_particles, model_name_f, directory, num_epochs, noes,  tag + "_w_trans_w_freezing_w_freezing", out_dir))


# no WMH w trans w freezing

directory = directory_base + "/no_WMH_w_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-for-trans-no-MH" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
    num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

model_name_f = "partial_data/model-A2-after-freezing-for-trans-no-MH" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/w_trans_w_freezing_no_MH_part1.txt".format(
    num_particles, model_name_f, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_w_freezing_no_MH_part2.txt".format(
    num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing", out_dir))
os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} >> {}/w_trans_w_freezing_no_MH_part2.txt".format(
    num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing_w_freezing", out_dir))
