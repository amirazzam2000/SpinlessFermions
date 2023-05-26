import sys, os

directory = "results/energy/data/Early_Stopping_Testing_New_slope"

out_dir = "./out/ES2"

if not os.path.exists(directory):
    os.system("mkdir {}".format(directory))

if not os.path.exists("./out"):
    os.system("mkdir {}".format("./out"))

if not os.path.exists(out_dir):
    os.system("mkdir {}".format(out_dir))

# no pre no freezing 
os.system("python3 run.py -V 5 -S 0.3 -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES_' > {}/no_trans_no_freezing_no_MH.txt".format(directory, out_dir))

model_name = "partial_data/model-OG"
os.system("python3 run.py -V 0 -S 0.5 -M {} -DIR {} -UL 1 -LL 1 --epochs 100000 -T 'ES_' > {}/dumb.txt".format(model_name, directory, out_dir))

# transfer learning (no freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -T 'ES_' > {}/w_trans_no_freezing_no_MH.txt".format(model_name, directory, out_dir))

# transfer learning (freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 1 -LL 1 --epochs 100000 --preepochs 0 -F -T 'ES_' > {}/w_trans_w_freezing_no_MH.txt".format(model_name, directory, out_dir))

# weighted MH [no trans] (no freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 -T 'ES_' > {}/no_trans_no_freezing_w_MH.txt".format(model_name, directory, out_dir))

# with transfer learning :

# weighted MH (no freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -T 'ES_' > {}/w_trans_no_freezing_w_MH.txt".format(model_name, directory, out_dir))

# weighted MH (freezing)
os.system("python3 run.py -V 5 -S 0.3  -LM {} -DIR {} -UL 100 -LL 10 --epochs 100000 --preepochs 0 -F -T 'ES_' > {}/w_trans_w_freezing_w_MH.txt".format(model_name, directory, out_dir))





# os.system("python3 run.py -V 5 -S 0.3 -LM {} -M {} -DIR {} --epochs 100000 --preepochs 0".format(model_name, model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 3 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F".format(model_name, directory))

# os.system("python3 run.py -V 5 -S 0.3 -N 8 -W 8192 -LM {} -DIR {} --epochs 100000 --preepochs 0 -F -NoES".format(model_name, directory))

#################################### Testing Weighted MH ########################################################
# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1 -LL 1".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 1000 -LL 100".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 500 -LL 100".format(directory))

# os.system("python3 run.py -V 5 -S 0.3 -DIR {} --epochs 50000 --preepochs 0 -NoES -UL 100 -LL 10".format(directory))
