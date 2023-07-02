import sys
import os


tag = "ES_Server_spikes_2"
Enable_ES = False
num_epochs = 50000
num_particles = 4
num_pre_particles = 2
directory_base = "results/energy/data/" + tag
out_dir = "./out/" + tag
model_name_tag = tag #"-ES-Server-Debug-New-Loss2"


noes = "" if Enable_ES else "-NoES"

if not os.path.exists(directory_base):
    os.system("mkdir {}".format(directory_base))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))


# no WMH no trans no freezing

directory = directory_base + "/NoES_no_WMH_no_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} > {}/no_trans_no_freezing_no_MH.txt".format(
    num_particles, directory, num_epochs, noes, tag, out_dir))

# # no WMH no trans w freezing

# directory = directory_base + "/no_WMH_no_trans_w_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A2-for-freezing" + model_name_tag
# os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} > {}/dumb.txt".format(
#     num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/no_trans_w_freezing_no_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# # no WMH w trans no freezing

# directory = directory_base + "/no_WMH_w_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_no_freezing_no_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# w WMH no trans no freezing

# directory = directory_base + "/w_WMH_no_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} > {}/no_trans_no_freezing_w_MH.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH', out_dir))

# # w WMH no trans w freezing

# directory = directory_base + "/w_WMH_no_trans_w_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A2-for-freezing-w-MH-C-" + model_name_tag
# os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/no_trans_w_freezing_w_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))


# # w WMH w trans no freezing

# directory = directory_base + "/w_WMH_w_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_no_freezing_w_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))


# # w WMH w trans w freezing

# directory = directory_base + "/w_WMH_w_trans_w_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A2-for-freezing-for-trans-w-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# model_name_f = "partial_data/model-A2-after-freezing-for-trans-w-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/w_trans_w_freezing_w_MH_part1.txt".format(
#     num_particles, model_name_f, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_w_freezing_w_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing", out_dir))
# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} >> {}/w_trans_w_freezing_w_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes,  tag + "_w_trans_w_freezing_w_freezing", out_dir))


# # no WMH w trans w freezing

# directory = directory_base + "/no_WMH_w_trans_w_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A2-for-freezing-for-trans-no-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# model_name_f = "partial_data/model-A2-after-freezing-for-trans-no-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} > {}/w_trans_w_freezing_no_MH_part1.txt".format(
#     num_particles, model_name_f, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 {} -T {} > {}/w_trans_w_freezing_no_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing", out_dir))
# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} >> {}/w_trans_w_freezing_no_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing_w_freezing", out_dir))





































# import sys
# import os

# import sys
# import os


# Enable_ES = True
# num_particles = 4
# num_pre_particles = 2
# num_epochs = 100000
# tag = "Single_ES_Trans2"
# model_name_tag = "-Single-ES-Trans2"
# directory_base = "results/energy/data/" + tag
# out_dir = "./out/" + tag


# noes = "" if Enable_ES else "-NoES"

# if not os.path.exists(directory_base):
#     os.system("mkdir {}".format(directory_base))

# if not os.path.exists("./out"):
#     os.system("mkdir {}".format("./out"))

# if not os.path.exists(out_dir):
#     os.system("mkdir {}".format(out_dir))


# ################################################################### Interaction Transfer testing sequence #################################################################################


# directory = directory_base + "/without_ES"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans_NoES" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, "-NoES", tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/without_ES.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))


# directory = directory_base + "/with_ES"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans_ES3" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/with_ES.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))





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
