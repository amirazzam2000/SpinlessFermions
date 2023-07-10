import sys
import os


Enable_ES = True
num_particles = 4
num_pre_particles = 2
num_epochs = 100000
tag = "Single_Freeze_no_Freeze"
model_name_tag = tag #"Single_ES_Trans_test-29-Jun"
directory_base = "results/energy/data/" + tag
out_dir = "./out/" + tag


noes = "" if Enable_ES else "-NoES"

if not os.path.exists(directory_base):
    os.system("mkdir {}".format(directory_base))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))


# # w WMH no trans no freezing

# directory = directory_base + "/w_WMH_no_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} -SW 30 > {}/no_trans_no_freezing_w_MH.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH', out_dir))


# w WMH no trans w freezing

directory = directory_base + "/w_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-w-MH-C-" + model_name_tag
os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
    num_pre_particles, model_name, directory, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs 100000  --preepochs 0 -F {} -T {} -SW 40 -W 8000> {}/no_trans_w_freezing_w_MH.txt".format(
    num_particles, model_name, directory, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs 100000  --preepochs 0 {} -T {} -SW 40 -W 8000> {}/no_trans_w_freezing_w_MH.txt".format(
    num_particles, model_name, directory, noes, "NoFreezing" + tag, out_dir))


# # w WMH w trans no freezing

# directory = directory_base + "/w_WMH_w_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} -SW 20 -W 8000 > {}/w_trans_no_freezing_w_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# # w WMH w trans no freezing

# directory = directory_base + "/w_WMH_w_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} -SW 30 > {}/w_trans_no_freezing_w_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# # no WMH no trans no freezing

# directory = directory_base + "/no_WMH_no_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} -SW 30 > {}/no_trans_no_freezing_no_MH.txt".format(
#     num_particles, directory, num_epochs, noes, tag, out_dir))

# no WMH no trans w freezing

directory = directory_base + "/no_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing" + model_name_tag
os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} > {}/dumb.txt".format(
    num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} -SW 40 -W 8000 > {}/no_trans_w_freezing_no_MH.txt".format(
    num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 {} -T {} -SW 40 -W 8000 > {}/no_trans_w_freezing_no_MH.txt".format(
    num_particles, model_name, directory, num_epochs, noes, "NoFreezing" + tag, out_dir))

# # no WMH w trans no freezing

# directory = directory_base + "/no_WMH_w_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 {} -T {} -SW 30 > {}/w_trans_no_freezing_no_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))


# # w WMH w trans w freezing

# directory = directory_base + "/w_WMH_w_trans_w_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A2-for-freezing-for-trans-w-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# model_name_f = "partial_data/model-A2-after-freezing-for-trans-w-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} -SW 30 > {}/w_trans_w_freezing_w_MH_part1.txt".format(
#     num_particles, model_name_f, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} -SW 30 > {}/w_trans_w_freezing_w_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing", out_dir))
# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} -SW 30 >> {}/w_trans_w_freezing_w_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes,  tag + "_w_trans_w_freezing_w_freezing", out_dir))


# # no WMH w trans w freezing

# directory = directory_base + "/no_WMH_w_trans_w_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A2-for-freezing-for-trans-no-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# model_name_f = "partial_data/model-A2-after-freezing-for-trans-no-MH" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} -SW 30 > {}/w_trans_w_freezing_no_MH_part1.txt".format(
#     num_particles, model_name_f, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 {} -T {} -SW 30 > {}/w_trans_w_freezing_no_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing", out_dir))
# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} -SW 30 >> {}/w_trans_w_freezing_no_MH_part2.txt".format(
#     num_particles, model_name_f, directory, num_epochs, noes, tag + "_w_trans_w_freezing_w_freezing", out_dir))


######################################################33

# directory = directory_base + "/w_WMH_no_trans_w_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A2-for-freezing-w-MH-C-" + model_name_tag
# os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_pre_particles, model_name, directory, num_epochs, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 -F {} -T {} -LR 0.000005 > {}/no_trans_w_freezing_w_MH.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag, out_dir))


# directory = directory_base + "/w_WMH_w_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs {}  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag , out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 1 --epochs {}  --preepochs 0 {} -T {} -LR 0.000001 > {}/w_trans_no_freezing_w_MH_10_IM.txt".format(
#     num_particles, model_name, directory, num_epochs, noes, tag , out_dir))




# directory = directory_base + "/w_WMH_no_trans_no_freezing_inner_mean_04"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.04 -IM > {}/w_WMH_no_trans_no_freezing_inner_mean_04.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH', out_dir))

# directory = directory_base + "/w_WMH_no_trans_no_freezing_outer_mean_01"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.01 -ST 1 > {}/w_WMH_no_trans_no_freezing_outer_mean_01.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH1', out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.01 -ST 2 > {}/w_WMH_no_trans_no_freezing_normal_01.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH2', out_dir))
# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.005 -ST 1 > {}/w_WMH_no_trans_no_freezing_outer_mean_04.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH1', out_dir))

# directory = directory_base + "/w_WMH_no_trans_no_freezing_inner_mean_1"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.1 -IM > {}/w_WMH_no_trans_no_freezing_inner_mean_1.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH2', out_dir))

# directory = directory_base + "/w_WMH_no_trans_no_freezing_outer_mean_1"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.1 > {}/w_WMH_no_trans_no_freezing_outer_mean_1.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH3', out_dir))

################################################################### Interaction Transfer testing sequence #################################################################################


# directory = directory_base + "/without_ES"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans_NoES" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  -NoES -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 -T {} > {}/without_ES.txt".format(
#     num_particles, model_name, directory, tag, out_dir))


# directory = directory_base + "/with_ES"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans_ES2" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 -T {} > {}/with_ES.txt".format(
#     num_particles, model_name, directory, tag, out_dir))




################################################################### Interaction Transfer testing sequence #################################################################################

# directory = directory_base + "/no_WMH_w_trans_no_freezing"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/w_trans_no_freezing_no_MH.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))






# ################################################################### Freezing testing sequence #################################################################################

# envelope_path = "results/energy/checkpoints/ES_Loss_with_stability_A04_MH001_H064_L02_D01_Tanh_W4096_P010000_V1.00e+01_S5.00e-01_Adam_PT_False_device_cuda_dtype_float64_freeze_False_trans_False_chkp.pt"


# model_name = "partial_data/model-A2-for-no-freezing-w-MH-C-" + model_name_tag
# os.system("python3 run.py -N {} -V 10 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} > {}/og.txt".format(
#     num_pre_particles, model_name, directory_base, noes, tag, out_dir))


# # no WMH no env no freezing

# test_name = "no_RMH_no_env_no_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))
 

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir, test_name))

# # no WMH no env w freezing

# test_name = "no_RMH_no_env_w_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))
 

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 -F {} -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir, test_name))

# # w WMH no env no freezing

# test_name = "w_RMH_no_env_no_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))
 

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir, test_name))

# # w WMH no env w freezing

# test_name = "w_RMH_no_env_w_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))
 

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -UL 100 -LL 1 --epochs 100000  --preepochs 0 -F {} -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir, test_name))

# # no WMH w env no freezing

# test_name = "no_RMH_w_env_no_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))
 

# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -LE {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, envelope_path, noes, tag, out_dir, test_name))

# # no WMH w env w freezing

# test_name = "no_RMH_w_env_w_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))


# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -LE {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 -F {} -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, envelope_path, noes, tag, out_dir, test_name))

# # w WMH w env no freezing

# test_name = "w_RMH_w_env_no_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))


# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -LE {} -UL 100 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, envelope_path, noes, tag, out_dir, test_name))


# # w WMH w env w freezing

# test_name = "w_RMH_w_env_w_freezing"

# directory = directory_base + "/" + test_name
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))


# os.system("python3 run.py -N {} -V 10 -S 0.5 -LM {} -DIR {} -LE {} -UL 100 -LL 1 --epochs 100000  --preepochs 0 {} -F -T {} > {}/{}.txt".format(
#     num_particles, model_name, directory, envelope_path, noes, tag, out_dir, test_name))

