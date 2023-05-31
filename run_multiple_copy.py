import sys
import os

directory_base = "results/energy/data/ES_A6_test"

out_dir = "./out/ES_A6_test"

model_name_tag = "-ES-A6-test"

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

os.system("python3 run.py -N 6 -V 10 -S 0.5  -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES_16_' > {}/no_trans_no_freezing_no_MH.txt".format(directory, out_dir))

# no WMH no trans w freezing

directory = directory_base + "/no_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing" + model_name_tag
os.system("python3 run.py -N 4 -V 10 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES_16_' > {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -T 'ES_16_' > {}/no_trans_w_freezing_no_MH.txt".format(model_name, directory, out_dir))

# no WMH w trans no freezing

directory = directory_base + "/no_WMH_w_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans" + model_name_tag
os.system("python3 run.py -N 6 -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES_16_' >> {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'ES_16_' > {}/w_trans_no_freezing_no_MH.txt".format(model_name, directory, out_dir))

# w WMH no trans no freezing

directory = directory_base + "/w_WMH_no_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

os.system("python3 run.py -N 6 -V 10 -S 0.5  -DIR {} -UL 100 -LL 10 --epochs 100000 -T 'ES_16_Weighted_' > {}/no_trans_no_freezing_w_MH.txt".format(directory, out_dir))

# w WMH no trans w freezing

directory = directory_base + "/w_WMH_no_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-w-MH" + model_name_tag
os.system("python3 run.py -N 6 -V 10 -S 0.5  -M {} -DIR {} -UL 100 -LL 10 --epochs 100000 -T 'ES_16_' >> {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -F -T 'ES_16_' > {}/no_trans_w_freezing_w_MH.txt".format(model_name, directory, out_dir))


# w WMH w trans no freezing

directory = directory_base + "/w_WMH_w_trans_no_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A4-for-trans" + model_name_tag
os.system("python3 run.py -N 6 -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 10 --epochs 100000 -T 'ES_16_' >> {}/dumb.txt".format(model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -T 'ES_16_' > {}/w_trans_no_freezing_w_MH.txt".format(model_name, directory, out_dir))


# w WMH w trans w freezing

directory = directory_base + "/w_WMH_w_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-for-trans-w-MH" + model_name_tag
os.system("python3 run.py -N 4 -V 5 -S 0.5  -M {} -DIR {} -UL 100 -LL 10 --epochs 100000 -T 'ES_16_' >> {}/dumb.txt".format(model_name, directory, out_dir))

model_name_f = "partial_data/model-A2-after-freezing-for-trans-w-MH"
os.system("python3 run.py -N 6 -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -F -T 'ES_16_' > {}/w_trans_w_freezing_w_MH_part1.txt".format(model_name_f, model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -T 'ES_16_' > {}/w_trans_w_freezing_w_MH_part2.txt".format(model_name_f, directory, out_dir))
os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -F -T 'ES_16_' >> {}/w_trans_w_freezing_w_MH_part2.txt".format(model_name_f, directory, out_dir))


# no WMH w trans w freezing

directory = directory_base + "/no_WMH_w_trans_w_freezing"
if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

model_name = "partial_data/model-A2-for-freezing-for-trans-no-MH" + model_name_tag
os.system("python3 run.py -N 4 -V 5 -S 0.5  -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES_16_' >> {}/dumb.txt".format(model_name, directory, out_dir))

model_name_f = "partial_data/model-A2-after-freezing-for-trans-no-MH"
os.system("python3 run.py -N 6 -V 5 -S 0.5   -M {} -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -T 'ES_16_' > {}/w_trans_w_freezing_no_MH_part1.txt".format(model_name_f, model_name, directory, out_dir))

os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'ES_16_' > {}/w_trans_w_freezing_no_MH_part2.txt".format(model_name_f, directory, out_dir))
os.system("python3 run.py -N 6 -V 10 -S 0.5   -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -T 'ES_16_' >> {}/w_trans_w_freezing_no_MH_part2.txt".format(model_name_f, directory, out_dir))

