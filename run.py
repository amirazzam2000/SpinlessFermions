import pandas as pd
import torch
from torch import nn, Tensor
import numpy as np
import pickle

import os, sys, time

torch.manual_seed(0)
torch.set_printoptions(4)
torch.backends.cudnn.benchmark=True
torch.set_default_dtype(torch.float64)

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
dtype = str(torch.get_default_dtype()).split('.')[-1]

sys.path.append("./src/")

from Models import vLogHarmonicNet
from Samplers import MetropolisHastings
from Hamiltonian import HarmonicOscillatorWithInteraction1D as HOw1D
from Pretraining import HermitePolynomialMatrix 

from utils import load_dataframe, load_model, count_parameters, get_groundstate, load_envelope
from utils import get_params, sync_time, clip, calc_pretraining_loss

import argparse

parser = argparse.ArgumentParser(prog="SpinlessFermions",
                                 usage='%(prog)s [options]',
                                 description="A Neural Quantum State (NQS) solution to one-dimensional fermions interacting in a Harmonic trap",
                                 epilog="and fin")


def add_bool_arg(parser, name, short, help="", default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-' + short, '--' + name,
                       dest=name, action='store_true', help="")
    group.add_argument('-no-' + short, '--no-' + name,
                       dest=name, action='store_false')
    parser.set_defaults(**{name: default})


parser.add_argument("-N", "--num_fermions", type=int,   default=2,     help="Number of fermions in physical system")
parser.add_argument("-UL", "--upper_lim", type=int,   default=100,     help="the upper limit to the weighted MH")
parser.add_argument("-LL", "--lower_lim", type=int,   default=10,     help="the upper limit to the weighted MH")
parser.add_argument("-H", "--num_hidden",   type=int,   default=64,    help="Number of hidden neurons per layer")
parser.add_argument("-L", "--num_layers",   type=int,   default=2,     help="Number of layers within the network")
parser.add_argument("-D", "--num_dets",     type=int,   default=1,     help="Number of determinants within the network's final layer")
parser.add_argument("-V", "--V0",           type=float, default=0.,    help="Interaction strength (in harmonic units)")
parser.add_argument("-S", "--sigma0",       type=float, default=0.5,   help="Interaction distance (in harmonic units")
parser.add_argument("--preepochs",          type=int,   default=10000, help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,   default=10000, help="Number of epochs for the energy minimisation phase")
parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")
add_bool_arg(parser, 'freeze', 'F', help="freeze the first layers of the neural network when it's loaded.")
add_bool_arg(parser, 'no_early_stopping', 'NoES', help="disable early stopping")
parser.add_argument("-M", "--model_name",       type=str,   default=None,     help="The path of the output model")
parser.add_argument("-W", "--num_walkers",       type=int,   default=4096,     help="Number of walkers for the metrapolis hasting")
parser.add_argument("-LM", "--load_model_name",       type=str,   default=None,     help="The name of the input model")
parser.add_argument("-LE", "--load_envelope_name",       type=str,   default=None,     help="The path to the input model to extract the envelope from")
parser.add_argument("-DIR", "--dir",       type=str,   default=None,     help="The name of the output directory")
parser.add_argument("-T", "--tag",       type=str,   default="",     help="tag the name of the file")

args = parser.parse_args()

nfermions = args.num_fermions #number of input nodes
upper_lim = args.upper_lim #upper limit for MH
lower_lim = args.lower_lim #lower limit for MH
num_hidden = args.num_hidden  #number of hidden nodes per layer
num_layers = args.num_layers  #number of layers in network
num_dets = args.num_dets      #number of determinants (accepts arb. value)
model_name = args.model_name      #the name of the model
load_model_name = args.load_model_name      #the name of the model
load_envelope_name = args.load_envelope_name  # the name of the model
freeze = args.freeze    
nwalkers = args.num_walkers
early_stopping_active = not args.no_early_stopping
func = nn.Tanh()  #activation function between layers
pretrain = True   #pretraining output shape?

tag = args.tag

directory = args.dir 

nwalkers = args.num_walkers
n_sweeps=10 #n_discard
std=1.#0.02#1.
target_acceptance=0.5

V0 = args.V0
sigma0 = args.sigma0

pt_save_every_ith=1000
em_save_every_ith=1000

nchunks=1

preepochs=args.preepochs
epochs=args.epochs

net = vLogHarmonicNet(num_input=nfermions,
                      num_hidden=num_hidden,
                      num_layers=num_layers,
                      num_dets=num_dets,
                      func=func,
                      pretrain=pretrain)
net=net.to(device)

sampler = MetropolisHastings(network=net,
                             dof=nfermions,
                             nwalkers=nwalkers,
                             target_acceptance=target_acceptance)




calc_elocal = HOw1D(net=net, V0=V0, sigma0=sigma0, nchunks=nchunks)

HO = HermitePolynomialMatrix(num_particles=nfermions)

lr = 1e-4
optim = torch.optim.Adam(params=net.parameters(), lr=lr) 

gs_CI = get_groundstate(A=nfermions, V0=V0, datapath="groundstate/")

print("Network     | A: %4i | H: %4i | L: %4i | D: %4i " % (nfermions, num_hidden, num_layers, num_dets))
print("Sampler     | B: %4i | T: %4i | std: %4.2f | targ: %s" % (nwalkers, n_sweeps, std, str(target_acceptance)))
print("Hamitlonian | V0: %4.2f | S0: %4.2f" % (V0, sigma0))
print("Pre-epochs: | %6i" % (preepochs))
print("Epochs:     | %6i" % (epochs))
print("Number of parameters: %8i\n" % (count_parameters(net)))


###############################################################################################################################################
#####                                           PRE-TRAINING LOOP                                                                         #####
###############################################################################################################################################

model_path_pt = "results/pretrain/checkpoints/%s_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                 (tag,nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, \
                  optim.__class__.__name__, True, device, dtype)
filename_pt = "results/pretrain/data/%s_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s.csv" % \
                 (tag, nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, \
                  optim.__class__.__name__, True, device, dtype)

net.pretrain = True #check pretrain

writer_pt = load_dataframe(filename_pt)
output_dict = load_model(model_path=model_path_pt, device=device, net=net, optim=optim, sampler=sampler)

start=output_dict['start'] #unpack dict
net=output_dict['net']
optim=output_dict['optim']
sampler=output_dict['sampler']

#Pre-training
for preepoch in range(start, preepochs+1):
    stats={}
    
    start=sync_time()

    x, _ = sampler(n_sweeps=n_sweeps)
    
    network_orbitals = net(x)
    target_orbitals = HO(x) #no_grad op
    
    mean_preloss, stddev_preloss = calc_pretraining_loss(network_orbitals, target_orbitals)

    optim.zero_grad()
    mean_preloss.backward()  
    optim.step()

    end = sync_time()

    stats['epoch'] = [preepoch] 
    stats['loss_mean'] = mean_preloss.item()
    stats['loss_std'] = stddev_preloss.item()
    stats['proposal_width'] = sampler.sigma.item()
    stats['acceptance_rate'] = sampler.acceptance_rate if not isinstance(
        sampler.acceptance_rate, Tensor) else sampler.acceptance_rate.item()
    
    stats['walltime'] = end-start

    writer_pt(stats) #push data to Writer

    if(preepoch % pt_save_every_ith == 0):
        torch.save({'epoch':preepoch,
                    'model_state_dict':net.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'loss':mean_preloss.item(),
                    'chains':sampler.chains.detach(),
                    'log_prob':sampler.log_prob.detach(),
                    'sigma':sampler.sigma.item()},
                    model_path_pt)
        writer_pt.write_to_file(filename_pt)
        #write data here?

    sys.stdout.write("Epoch: %6i | Loss: %6.4f +/- %6.4f | Walltime: %4.2e (s)      \r" % (preepoch, mean_preloss, stddev_preloss, end-start))
    sys.stdout.flush()

print("\n")

###############################################################################################################################################
#####                                           ENERGY-MINIMISATION LOOP                                                                  #####
###############################################################################################################################################

net.pretrain = False #check it's false
optim = torch.optim.Adam(params=net.parameters(), lr=1e-4) #new optimizer

model_path = "results/energy/checkpoints/%s_A%02i_MH%03i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_freeze_%s_trans_%s_chkp.pt" % \
    (tag, nfermions, upper_lim, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0,
     optim.__class__.__name__, False, device, dtype, freeze, (load_model_name is not None)) if model_name is None else model_name

filename = "results/energy/data/%s_A%02i_MH%03i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_freeze_%s_lr_%4.2e_trans_%s.csv" % \
    (tag, nfermions, upper_lim, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0,
     optim.__class__.__name__, False, device, dtype, freeze, lr, (load_model_name is not None)) if directory is None else directory.rstrip('\\') + \
    "/%s_A%02i_MH%03i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_freeze_%s_lr_%4.2e_trans_%s.csv" % \
    (tag, nfermions, upper_lim, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0,
     optim.__class__.__name__, False, device, dtype, freeze, lr, (load_model_name is not None))


time_filename = "results/energy/timing_data/%s_A%02i_MH%03i_freeze_%s_trans_%s_wait_data.csv" % ( tag, 
    nfermions, upper_lim, freeze, (load_model_name is not None)) if directory is None else directory.rstrip('\\') + "/%s_A%02i_MH%03i_freeze_%s_trans_%s_wait_data.csv" % (tag,
    nfermions, upper_lim, freeze, (load_model_name is not None))

print("saving model at:", model_path)

writer_t = load_dataframe(time_filename)
writer = load_dataframe(filename)

if load_model_name is not None:
    output_dict = load_model(model_path=load_model_name, device=device,
                             net=net, optim=optim, sampler=sampler, fix_size=True, freeze=freeze)
    sampler = MetropolisHastings(network=net,
                                 dof=nfermions,
                                 nwalkers=nwalkers,
                                 target_acceptance=target_acceptance)
    optim = torch.optim.Adam(params=net.parameters(), lr=lr)
    start =0
    print()
    print('loading pre-trained model')
    print("freezing = ", freeze)
    if freeze:
        print("reduced number of parameters is: ",  count_parameters(net))
    
    if load_envelope_name is not None:
        net = load_envelope(load_envelope_name, device=device, net=net, freeze=True)

else :
    output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)
    start = output_dict['start']  # unpack dict
    optim = output_dict['optim']
    sampler = output_dict['sampler']

net=output_dict['net']

the_last_loss = 100
patience = 4
trigger_times = 0
num_iterations = 0
delta = 1e-3
var_delta = 3e-3
std_delta = 3e-3
error_tolerance = 0

window_size = 1000
mean_energy_list = []
var_energy_list = []
weights_list = []
sliding_window_loss = 0
last_window_loss = 0
avg_loss_diff = 0
avg_coverage = 0
old_slope = 0 

total_time = 0

print("early stopping = ", early_stopping_active)
print()

time_stats = {}
time_stats['MH_time'] = []

weighted_ratio = 0
old_logabs = 1
wait_epochs= 0
waited_epochs = 0

# wait_data['waited_epochs'] = []
# wait_data['ratio'] = []
# wait_data['wait_threshold'] = []

#Energy Minimisation
t0 = sync_time()

x, _ = sampler(n_sweeps)


weights_filename = "results/energy/data/weights.pkl" if directory is None else directory.rstrip('\\') + "/weights.pkl"

f=open(weights_filename, 'wb')

for epoch in range(start, epochs+1):
    waited_epochs += 1
    wait_data = {}
    stats={}

    start=sync_time()

    if waited_epochs >= wait_epochs:
        t_MH = sync_time()  # time.time()
        x, _ = sampler(n_sweeps)
        time_stats['MH_time'] = sync_time() - t_MH

        sign, logabs = net(x)
     
        # print()
        # print("updating samples: steps taken =", waited_epochs, "threshold=", wait_epochs)
        # print()

        old_logabs = logabs.clone()
        waited_epochs = 0


    else: 
        time_stats['MH_time'] = 0
        sign, logabs = net(x)

    elocal, k, p, g = calc_elocal(x, return_all=True)
    elocal = clip(elocal, clip_factor=5)
    k = clip(k, clip_factor=5)
    p = clip(p, clip_factor=5)
    g = clip(g, clip_factor=5)



    with torch.no_grad():
        ratio_no_mean = torch.exp(2 * (logabs - old_logabs))   
        
        # ratio_no_mean = 0 if ratio_no_mean < 0 else ratio_no_mean
        # ratio_no_mean = 2 if ratio_no_mean > 2 else ratio_no_mean
        
        weighted_ratio = torch.mean(ratio_no_mean).item()
        wait_epochs = upper_lim - (upper_lim - lower_lim) * np.abs(1 - weighted_ratio)
        wait_epochs = wait_epochs if wait_epochs > 0 else 0

        wait_data['waited_epochs'] = [waited_epochs]
        wait_data['ratio'] = weighted_ratio
        wait_data['wait_threshold'] = wait_epochs
       

        r_mean = torch.mean(ratio_no_mean)  
        energy_mean = torch.mean(elocal * ratio_no_mean) / r_mean  # sqrt(var/ num_walkers)

        k = torch.mean(k)  # sqrt(var/ num_walkers)
        p = torch.mean(p)  # sqrt(var/ num_walkers)
        g = torch.mean(g)  # sqrt(var/ num_walkers)

        energy_var = torch.mean((elocal - energy_mean )**2 * ratio_no_mean) / r_mean  # sqrt(var/ num_walkers)
        energy_var = torch.sqrt(energy_var / elocal.shape[0]) 

        # normalize the ratio 
        ratio_no_mean = ratio_no_mean / torch.sum(ratio_no_mean)
        r_mean = torch.mean(ratio_no_mean)  

        mean_elocal = torch.mean(elocal * ratio_no_mean) 
    
    # loss_elocal = 2.*((elocal - torch.mean(elocal)).detach() * logabs)
    
    # loss_elocal = 2.*((elocal - torch.mean(elocal)).detach() * (logabs - torch.mean(logabs)))

    # loss1 = (ratio_no_mean * (elocal - mean_elocal)).detach() * logabs
    # loss2 = (ratio_no_mean * (elocal - mean_elocal)).detach() * torch.mean(ratio_no_mean.detach() * logabs)
    loss1 = ((ratio_no_mean * elocal - mean_elocal) / r_mean).detach() * logabs
    loss2 = ((ratio_no_mean * elocal - mean_elocal) / r_mean).detach() * torch.mean(ratio_no_mean.detach() * logabs)/ r_mean.detach()

    # loss=torch.mean(loss_elocal)  

    # ratio_no_mean_test = torch.exp(2 * (logabs - (old_logabs).detach()))
    # loss = torch.mean(loss_elocal * ratio_no_mean_test) / torch.mean(ratio_no_mean_test)

    loss = torch.mean(loss1) - torch.mean(loss2)
    # loss = clip(loss, clip_factor=5)
     
    
    optim.zero_grad()
    t_MH = sync_time()
    loss.backward()  #populates leafs with grads
    stats['back_time'] = sync_time() - t_MH

    t_MH = sync_time()
    optim.step()
    stats['opt_time'] = sync_time() - t_MH

    end = sync_time()

    net_time = net.pop_time_records()

    ## log temporal values
    for key, value in time_stats.items():
        stats[key] = value
    for key, value in net_time.items():
        stats[key] = np.average(value)

    total_time = end - start
    stats['epoch'] = [epoch] #must pass index
    stats['loss'] = loss.item() 
    stats['energy_mean'] = energy_mean.item() 
    stats['energy_std'] = np.sqrt(energy_var.item() / nwalkers) #energy_var.sqrt().item() 
    stats['kinetic'] = k.item()
    stats['potential'] = p.item()
    stats['gaussian'] = g.item()
    stats['CI'] = gs_CI
    stats['proposal_width'] = sampler.sigma.item() 
    stats['acceptance_rate'] = sampler.acceptance_rate if not isinstance(
        sampler.acceptance_rate, Tensor) else sampler.acceptance_rate.item()

    stats['walltime'] = end-start

    the_current_loss = loss.item()
    mean_energy_list.append(energy_mean.item())
    var_energy_list.append(np.sqrt(energy_var.item() / nwalkers))

    loss_diff = np.abs(the_current_loss - the_last_loss)

    stats['loss_diff'] = loss_diff
    stats['window_loss'] = avg_loss_diff
    stats['overlap'] = avg_coverage
    
    writer(stats)
    writer_t(wait_data)

    # for i in net.log_envelope.log_envs[0].parameters():
    #     weights_list.append(i.cpu().detach().numpy())
    pickle.dump(net.state_dict(), f)
    


    if(epoch % em_save_every_ith == 0):
        torch.save({'epoch':epoch,
                    'model_state_dict':net.state_dict(),
                    'optim_state_dict':optim.state_dict(),
                    'loss':loss,
                    'energy':energy_mean,
                    'energy_std':energy_var.sqrt(),
                    'chains':sampler.chains.detach(),
                    'log_prob':sampler.log_prob.detach(),
                    'sigma':sampler.sigma},
                    model_path)
        writer.write_to_file(filename)
        writer_t.write_to_file(time_filename)

    sys.stdout.write("Epoch: %6i | Energy: %6.4f +/- %6.4f | Loss: %6.4f | CI: %6.4f | Walltime: %4.2e (s) | epochs to wait: %6.6f | weight ratio: %6.6f | waited epochs: %6i \r" %
                     (epoch, energy_mean, np.sqrt(energy_var.item() / nwalkers), the_current_loss, gs_CI, end-start, wait_epochs, weighted_ratio, waited_epochs))
    sys.stdout.flush()

    if len(mean_energy_list) > window_size:
        # remove outliers
        mean_energy_list = np.array(mean_energy_list)
        mean_energy_list = mean_energy_list[abs(mean_energy_list - np.mean(mean_energy_list)) < 2 * np.std(mean_energy_list)]

        # Compute the weighted average validation loss over sliding window
        sliding_window_loss = np.mean(mean_energy_list) # / np.sum(1/np.array(var_energy_list)) 
        avg_var = np.mean(var_energy_list)
        ene_std = np.std(mean_energy_list, ddof=1)

        slope, _ = np.polyfit(range(len(mean_energy_list)), mean_energy_list, 1)

        mean_energy_list = []
        var_energy_list = []


        avg_loss_diff = np.abs(sliding_window_loss - last_window_loss)
        slop_diff = np.abs(old_slope - slope)

        if slope < 3e-7 and old_slope < 3e-7 and slop_diff < 3e-6:
            if avg_loss_diff < delta and avg_var < var_delta and ene_std < std_delta:
                trigger_times += 1
                
                if trigger_times == 1:
                    error_tolerance = 0

                if trigger_times >= patience:
                    if early_stopping_active:
                        print('\nEarly stopping!')
                        break

        else:
            error_tolerance += 1
            if error_tolerance >= 2:
                trigger_times = 0
                error_tolerance = 0

        last_window_loss = sliding_window_loss
        old_slope = slope 

    the_last_loss = the_current_loss

t1 = sync_time() - t0

writer.write_to_file(filename)
writer_t.write_to_file(time_filename)






print("\nDone")
print("\nNumber of epochs:", epoch)
print("Time taken: ", total_time, " (accumulated wall time)\n\t", t1, "(recorded time)")


num_samples = 1000
m_ene = 0
v_ene = 0

for i in range(num_samples):
    x, _ = sampler(n_sweeps)
    sign, logabs = net(x)
    elocal = calc_elocal(x)
    elocal = clip(elocal, clip_factor=5)

    with torch.no_grad():
        m = torch.mean(elocal)
        v_ene += torch.mean((elocal - m)**2)
        m_ene += m

m_ene = m_ene/num_samples
v_ene = v_ene/num_samples

print("mean energy = ", m_ene.item() )
print("variance in energy = ", v_ene.item() )
