import sys, os

directory_base = "results/energy/data/No_Early_Stopping_Testing_complete"

out_dir = "./out/NoES_complete"

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

os.system("python3 run.py -N 4 -V 5 -S 0.3  -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'NoES_' -NoES > {}/no_trans_no_freezing_no_MH.txt".format(directory, out_dir))

# no WMH no trans w freezing

directory = directory_base +  "/no_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing"
os.system("python3 run.py -N 2 -V 5 -S 0.3  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'NoES_' -NoES > {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -T 'NoES_' -NoES > {}/no_trans_w_freezing_no_MH.txt".format(model_name, directory, out_dir))

# no WMH w trans no freezing

directory = directory_base + "/no_WMH_w_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans"
os.system("python3 run.py -N 4 -V 0 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'NoES_' -NoES >> {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'NoES_' -NoES > {}/w_trans_no_freezing_no_MH.txt".format(model_name, directory, out_dir))

# w WMH no trans no freezing

directory = directory_base + "/w_WMH_no_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

os.system("python3 run.py -N 4 -V 5 -S 0.3  -DIR {} -UL 10 -LL 100 --epochs 100000 -T 'NoES_' -NoES > {}/no_trans_no_freezing_w_MH.txt".format(directory, out_dir))

# w WMH no trans w freezing

directory = directory_base + "/w_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-w-MH"
os.system("python3 run.py -N 2 -V 5 -S 0.3  -M {} -DIR {} -UL 10 -LL 100 --epochs 100000 -T 'NoES_' -NoES >> {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 10 -LL 100 --epochs 100000 --preepochs 0 -F -T 'NoES_' -NoES > {}/no_trans_w_freezing_w_MH.txt".format(model_name, directory, out_dir))


# w WMH w trans no freezing

directory = directory_base + "/w_WMH_w_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans"
os.system("python3 run.py -N 4 -V 0 -S 0.5  -M {} -DIR {} -UL 10 -LL 100 --epochs 100000 -T 'NoES_' -NoES >> {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 10 -LL 100 --epochs 100000 --preepochs 0 -T 'NoES_' -NoES > {}/w_trans_no_freezing_w_MH.txt".format(model_name, directory, out_dir))


# w WMH w trans w freezing

directory = directory_base + "/w_WMH_w_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-for-trans-w-MH"
os.system("python3 run.py -N 2 -V 0 -S 0.5  -M {} -DIR {} -UL 10 -LL 100 --epochs 100000 -T 'NoES_' -NoES >> {}/dumb.txt".format(model_name, directory, out_dir))

model_name_f = "partial_data/model-A2-after-freezing-for-trans-w-MH"
os.system("python3 run.py -N 4 -V 0 -S 0.5   -M {} -LM {} -DIR {} -UL 10 -LL 100 --epochs 100000 --preepochs 0 -F -T 'NoES_' -NoES > {}/w_trans_w_freezing_w_MH_part1.txt".format(model_name_f, model_name, directory, out_dir))

os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 10 -LL 100 --epochs 100000 --preepochs 0 -T 'NoES_' -NoES > {}/w_trans_w_freezing_w_MH_part2.txt".format(model_name_f, directory, out_dir))
os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 10 -LL 100 --epochs 100000 --preepochs 0 -F -T 'NoES_' -NoES > {}/w_trans_w_freezing_w_MH_part2.txt".format(model_name_f, directory, out_dir))


# no WMH w trans w freezing

directory = directory_base + "/no_WMH_w_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-for-trans-no-MH"
os.system("python3 run.py -N 2 -V 0 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'NoES_' -NoES >> {}/dumb.txt".format(model_name, directory, out_dir))

model_name_f = "partial_data/model-A2-after-freezing-for-trans-no-MH"
os.system("python3 run.py -N 4 -V 0 -S 0.5   -M {} -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -T 'NoES_' -NoES > {}/w_trans_w_freezing_no_MH_part1.txt".format(model_name_f, model_name, directory, out_dir))

os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'NoES_' -NoES > {}/w_trans_w_freezing_no_MH_part2.txt".format(model_name_f, directory, out_dir))
os.system("python3 run.py -N 4 -V 5 -S 0.3   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -T 'NoES_' -NoES > {}/w_trans_w_freezing_no_MH_part2.txt".format(model_name_f, directory, out_dir))




# os.system("python3 run.py -V 5 -S 0.3 -LM {} -M {} -DIR {} --epochs 100000 --preepochs 0".format(model_name, model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 3 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 8 -W 8192 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F -NoES".format(model_name, directory))

#################################### Testing Weighted MH ########################################################
# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1 -LL 1".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1000 -LL 100".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 500 -LL 100".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 100 -LL 10".format(directory))
