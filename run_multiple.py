import sys, os

directory = "results/energy/data/Early_Stopping_Testing"

if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists("./out/NoES"):
    os.system("mkdir {}".format("./out/NoES"))

# no pre no freezing 
os.system("python3 run.py -V 5 -S 0.3 -DIR {} -UL 1 -LL 1 --epochs 100000 -NoES > out/NoES/no_trans_no_freezing_no_MH.txt".format(directory))

model_name = "partial_data/model-OG"
os.system("python3 run.py -V 0 -S 0.5 -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -NoES > out/NoES/dumb.txt".format(model_name, directory))

# transfer learning (no freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -NoES > out/NoES/w_trans_no_freezing_no_MH.txt".format(model_name, directory))

# transfer learning (freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -NoES > out/NoES/w_trans_w_freezing_no_MH.txt".format(model_name, directory))

# weighted MH [no trans] (no freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 -NoES > out/NoES/no_trans_no_freezing_w_MH.txt".format(model_name, directory))

# with transfer learning :

# weighted MH (no freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -NoES > out/NoES/w_trans_no_freezing_w_MH.txt".format(model_name, directory))

# weighted MH (freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -F -NoES > out/NoES/w_trans_w_freezing_w_MH.txt".format(model_name, directory))





# os.system("python3 run.py -V 5 -S 0.3 -LM {} -M {} -DIR {} --epochs 100000 --preepochs 0".format(model_name, model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 3 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 8 -W 8192 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F -NoES".format(model_name, directory))

#################################### Testing Weighted MH ########################################################
# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1 -LL 1".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1000 -LL 100".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 500 -LL 100".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 100 -LL 10".format(directory))
