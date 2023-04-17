import torch
from torch import nn, Tensor
import numpy as np

import os, sys, time

torch.manual_seed(0)
torch.set_printoptions(4)
torch.backends.cudnn.benchmark=True
torch.set_default_dtype(torch.float32)

device = torch.device('cpu')# if not torch.cuda.is_available() else torch.device('cuda')
dtype = str(torch.get_default_dtype()).split('.')[-1]

sys.path.append("./src/")

from Models import vLogHarmonicNet
from Samplers import MetropolisHastings
from Hamiltonian import HarmonicOscillatorWithInteraction1D as HOw1D
from Pretraining import HermitePolynomialMatrix 

from utils import load_dataframe, load_model, count_parameters, get_groundstate
from utils import get_params, sync_time, clip, calc_pretraining_loss

import argparse

parser = argparse.ArgumentParser(prog="SpinlessFermions",
                                 usage='%(prog)s [options]',
                                 description="A Neural Quantum State (NQS) solution to one-dimensional fermions interacting in a Harmonic trap",
                                 epilog="and fin")

parser.add_argument("-N", "--num_fermions", type=int,   default=2,     help="Number of fermions in physical system")
parser.add_argument("-H", "--num_hidden",   type=int,   default=64,    help="Number of hidden neurons per layer")
parser.add_argument("-L", "--num_layers",   type=int,   default=2,     help="Number of layers within the network")
parser.add_argument("-D", "--num_dets",     type=int,   default=1,     help="Number of determinants within the network's final layer")
parser.add_argument("-V", "--V0",           type=float, default=0.,    help="Interaction strength (in harmonic units)")
parser.add_argument("-S", "--sigma0",       type=float, default=0.5,   help="Interaction distance (in harmonic units")
parser.add_argument("--preepochs",          type=int,   default=10000, help="Number of pre-epochs for the pretraining phase")
parser.add_argument("--epochs",             type=int,   default=10000, help="Number of epochs for the energy minimisation phase")
parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")
parser.add_argument("-F", "--freeze",       type=bool,   default=0,     help="freeze the first layers of the neural network when it's loaded.")
parser.add_argument("-M", "--model_name",       type=str,   default=None,     help="The path of the output model")
parser.add_argument("-LM", "--load_model_name",       type=str,   default=None,     help="The name of the input model")
parser.add_argument("-DIR", "--dir",       type=str,   default=None,     help="The name of the output directory")

args = parser.parse_args()

nfermions = args.num_fermions #number of input nodes
num_hidden = args.num_hidden  #number of hidden nodes per layer
num_layers = args.num_layers  #number of layers in network
num_dets = args.num_dets      #number of determinants (accepts arb. value)
model_name = args.model_name      #the name of the model
load_model_name = args.load_model_name      #the name of the model
freeze = True if args.freeze != 0 else False      #the name of the model
func = nn.Tanh()  #activation function between layers
pretrain = True   #pretraining output shape?

directory = args.dir 

nwalkers=4096
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

model_path_pt = "results/pretrain/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                 (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, \
                  optim.__class__.__name__, True, device, dtype)
filename_pt = "results/pretrain/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s.csv" % \
                 (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, \
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

model_path = "results/energy/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_freeze_%s_chkp.pt" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim.__class__.__name__, False, device, dtype, freeze) if model_name is None else model_name
filename = "results/energy/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_freeze_%s_lr_%4.2e.csv" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, \
                 optim.__class__.__name__, False, device, dtype, freeze, lr) if directory is None else directory.rstrip('\\') + "/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_freeze_%s_lr_%4.2e.csv" % \
                (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0,
                 optim.__class__.__name__, False, device, dtype, freeze, lr)


print("saving model at:", model_path)

writer = load_dataframe(filename)

if load_model_name is not None:
    output_dict = load_model(model_path=load_model_name, device=device, net=net, optim=optim, sampler=sampler, fix_size=True, freeze = freeze)
    sampler = MetropolisHastings(network=net,
                                 dof=nfermions,
                                 nwalkers=nwalkers,
                                 target_acceptance=target_acceptance)
    optim = torch.optim.Adam(params=net.parameters(), lr=lr)
    start =0
    print('loading pre-trained model')
else :
    output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)
    start = output_dict['start']  # unpack dict
    optim = output_dict['optim']
    sampler = output_dict['sampler']

net=output_dict['net']

the_last_loss = 100
patience = 5
trigger_times = 0
num_iterations = 0
delta = 1e-4
error_tolerance = 0

window_size = 10
best_loss = float('inf')
no_improvement_counter = 0
validation_losses = []
sliding_window_loss = 0

print("early stopping active")
#Energy Minimisation
for epoch in range(start, epochs+1):
    stats={}

    start=sync_time()

    x, _ = sampler(n_sweeps)

    elocal = calc_elocal(x)
    elocal = clip(elocal, clip_factor=5)

    _, logabs = net(x)

    loss_elocal = 2.*((elocal - torch.mean(elocal)).detach() * logabs)
    
    with torch.no_grad():
        energy_var, energy_mean = torch.var_mean(elocal, unbiased=True)

    loss=torch.mean(loss_elocal)  
     
    
    optim.zero_grad()
    loss.backward()  #populates leafs with grads
    optim.step()

    end = sync_time()

    stats['epoch'] = [epoch] #must pass index
    stats['loss'] = loss.item() 
    stats['energy_mean'] = energy_mean.item() 
    stats['energy_std'] = energy_var.sqrt().item() 
    stats['CI'] = gs_CI
    stats['proposal_width'] = sampler.sigma.item() 
    stats['acceptance_rate'] = sampler.acceptance_rate if not isinstance(
        sampler.acceptance_rate, Tensor) else sampler.acceptance_rate.item()

    stats['walltime'] = end-start

    the_current_loss = loss.item()
    validation_losses.append(the_current_loss)

    loss_diff = np.abs(the_current_loss - the_last_loss)

    stats['loss diff'] = loss_diff
    stats['widnow_loss'] = sliding_window_loss
    
    writer(stats)

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


    sys.stdout.write("Epoch: %6i | Energy: %6.4f +/- %6.4f | CI: %6.4f | Walltime: %4.2e (s) | loss difference: %6.6f        \r" % (epoch, energy_mean, energy_var.sqrt(), gs_CI, end-start, loss_diff))
    sys.stdout.flush()

    if len(validation_losses) > window_size:
        # Compute average validation loss over sliding window
        sliding_window_loss = sum(validation_losses[-window_size:]) / window_size

        if sliding_window_loss < best_loss:
            # Update best loss and reset counter
            best_loss = sliding_window_loss
            no_improvement_counter = 0
        else:
            # Increment counter
            no_improvement_counter += 1

        # Check if training should stop
        if no_improvement_counter == patience:
            print("Validation loss has not improved for",
                  patience, "sliding windows. Stopping training.")
            break

    # if loss_diff < delta:
    #     trigger_times += 1
    #     if trigger_times == 1:
    #         error_tolerance = 0

    #     if trigger_times >= patience:
    #         print('Early stopping!')
    #         break

    # else:
    #     error_tolerance += 1
    #     if error_tolerance >= 2:
    #         trigger_times = 0
    #         error_tolerance = 0

    # the_last_loss = the_current_loss

print("\nDone")
