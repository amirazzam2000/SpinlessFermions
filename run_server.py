import sys
import os

import sys
import os


Enable_ES = True
num_particles = 4
num_pre_particles = 2
num_epochs = 100000
tag = "Single_ES_Trans2"
model_name_tag = "-Single-ES-Trans2"
directory_base = "results/energy/data/" + tag
out_dir = "./out/" + tag


noes = "" if Enable_ES else "-NoES"

if not os.path.exists(directory_base):
    os.system("mkdir {}".format(directory_base))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))


################################################################### Interaction Transfer testing sequence #################################################################################


directory = directory_base + "/without_ES"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans_NoES" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
    num_particles, model_name, directory, "-NoES", tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/without_ES.txt".format(
    num_particles, model_name, directory, noes, tag, out_dir))


directory = directory_base + "/with_ES"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans_ES3" + model_name_tag
os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
    num_particles, model_name, directory, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/with_ES.txt".format(
    num_particles, model_name, directory, noes, tag, out_dir))





################################################################################# old 33###########################################################33
# directory_base = "results/energy/data/ES_NoES_test"

# out_dir = "./out/ES_NoES_test"

# model_name_tag = "-ES-NoES-test"

# if not os.path.exists(directory_base):
#     os.system("mkdir {}".format(directory_base))

# if not os.path.exists("./out"):
#     os.system("mkdir {}".format("./out"))

# if not os.path.exists(out_dir):
#     os.system("mkdir {}".format(out_dir))



# directory = directory_base + "/ES_then_NoES"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-ES-then-NoES" + model_name_tag
# os.system("python3 run.py -N 6 -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES' > {}/ES.txt".format(model_name, directory, out_dir))

# os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'NoES' -NoES > {}/NoES.txt".format(model_name, directory, out_dir))


# directory = directory_base + "/just_NoES"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-ES-then-NoES" + model_name_tag
# os.system("python3 run.py -N 6 -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'NoES_first'-NoES > {}/NoES_first.txt".format(model_name, directory, out_dir))

# os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'NoES_second' -NoES > {}/NoES_second.txt".format(model_name, directory, out_dir))
