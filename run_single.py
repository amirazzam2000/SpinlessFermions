import sys
import os


Enable_ES = True
num_particles = 4
num_pre_particles = 2
num_epochs = 50000
tag = "Single_schedule_8"
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



# directory = directory_base + "/w_WMH_no_trans_no_freezing_inner_mean_04"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.04 -IM > {}/w_WMH_no_trans_no_freezing_inner_mean_04.txt".format(
#     num_particles, directory, num_epochs, noes, tag + 'WMH', out_dir))

directory = directory_base + "/w_WMH_no_trans_no_freezing_outer_mean_01"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.01 -ST 1 > {}/w_WMH_no_trans_no_freezing_outer_mean_01.txt".format(
    num_particles, directory, num_epochs, noes, tag + 'WMH1', out_dir))

os.system("python3 run.py -N {} -V 10 -S 0.5  -DIR {} -UL 100 -LL 1 --epochs {}  {} -T {}  -STD 0.01 -ST 2 > {}/w_WMH_no_trans_no_freezing_normal_01.txt".format(
    num_particles, directory, num_epochs, noes, tag + 'WMH2', out_dir))
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
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, "-NoES", tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/without_ES.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))


# directory = directory_base + "/with_ES"
# if not os.path.exists(directory):
#     os.system("mkdir {}".format(directory))

# model_name = "partial_data/model-A4-for-trans_ES2" + model_name_tag
# os.system("python3 run.py -N {} -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000  {} -T {} >> {}/dumb.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))

# os.system("python3 run.py -N {} -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000  --preepochs 0 {} -T {} > {}/with_ES.txt".format(
#     num_particles, model_name, directory, noes, tag, out_dir))




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

