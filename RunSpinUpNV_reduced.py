#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:38:37 2019

@author: Tobias Haug (tobias.haug@u.nus.edu)

Quantum control of NV centers with driving, optimization using PPO

Instructions for code are found below at instructions around line 404
"""

activateSpinningup=1




if(activateSpinningup==1):
    import tensorflow as tf
    import spinup
    from spinup import ppo_tf1
    from spinup import sac_tf1

    
    

import os
import sys




import time

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

import GymEnvNVnew

import pickle

from myutil import *

def flatten (l):
    return [item for sublist in l for item in sublist]


#To calculate moving average while plottin only the skip-th entry
def moving_averageAvgSkip(a, n=3, skip=1):
    return np.array([np.average(a[i*skip:i*skip+n]) for i in range((len(a)-n+1)//skip)])

#moving max over array a
def moving_max(a, n=3,skip=1):
    #b = pd.DataFrame(a)
    #return pd.rolling_max(b, n)
    return np.array([np.amax(a[i*skip:i*skip+n]) for i in range((len(a)-n+1)//skip)])


#run optimization algorithm
def runNVOptimize():
    global env,countfigure
    starttime=time.time()
    res=[]
    locreslist=[]
    if(optimizer==0 or optimizer==2):
        numsteps=env.totalsteps
        env_fn = lambda : env
        hidden_sizes=np.ones(circuitdepth)*nneurons #size of neural network
        ac_kwargs = dict(hidden_sizes=hidden_sizes, activation=tf.nn.relu)
        logger_kwargs = dict(output_dir=output_dir+dataset+"/", exp_name=dataset) #this logger is not used, we have our own
        if(optimizer==0): #Use PPO
            with tf.Graph().as_default():
                ppo_tf1(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=(numsteps)*iterations_per_epoch, pi_lr=pi_lr,
                    vf_lr=vf_lr,train_pi_iters=iterations_per_epoch, train_v_iters=iterations_per_epoch,epochs=big_epochs,lam=0.99, 
                    logger_kwargs=logger_kwargs,target_kl=100, save_freq=100,clip_ratio=clip_ratio)
        elif(optimizer==2): #Use SAC, does not work properly yet
            with tf.Graph().as_default():
                startsteps=10000
                sac_tf1(env_fn, ac_kwargs=ac_kwargs, seed=0, steps_per_epoch=(numsteps)*iterations_per_epoch, 
                           epochs=big_epochs, replay_size=1000000, gamma=0.99, polyak=0.995, lr=sac_lr, alpha=0.05, batch_size=500, 
                           start_steps=startsteps, max_ep_len=1000,logger_kwargs=logger_kwargs, save_freq=100)
    
    
        tf.reset_default_graph()
    
    elif(optimizer==1): #Use DIRECCT algorithm
        from scipydirect import minimize
    
        print("start DIRECT")
        bounds=np.transpose([env.action_space.low[0]*np.ones(env.action_spaceLength*env.totalsteps),env.action_space.high[0]*np.ones(env.totalsteps*env.action_spaceLength)])
        
        def gymWrapper(x): #Wrap gym environment into format understood by optimizer
            env.reset()
            actions=np.reshape(x,[env.totalsteps,env.action_spaceLength])
    
            for i in range(np.shape(actions)[0]):
                env.step(actions[i])
                
    
            return -env.fidel_val
        
        
    
        
        res = minimize(gymWrapper, bounds,fglobal=-1,maxf=maxiterations,algmethod=1)
        
        
        actions=np.reshape(res["x"],[env.totalsteps,env.action_spaceLength])
        print(res)
        print('DIRECT result',-res["fun"],actions)
    elif(optimizer==3): #Use nealder-mead
        steadytime=False #Keep time per bin constant, but can vary time overall
        if(len(t_var)>0 and steadytime==True):
            actionspace=(env.action_spaceLength-1)*env.totalsteps+1
        else:
            actionspace=env.action_spaceLength*env.totalsteps
        #action is bounded betwen -0.5 and 0.5
        bounds=np.transpose([env.action_space.low[0]*np.ones(actionspace),env.action_space.high[0]*np.ones(actionspace)])
        print("start scipy optimize")
        
        def reshapeActions(x): 
            if(len(t_var)>0 and steadytime==True):
                xtime=x[0] #time per bin, mapped between -0.5 and 0.5
                xrest=np.reshape(x[1:],[env.totalsteps,(env.action_spaceLength-1)])
                actions=np.zeros([env.totalsteps,env.action_spaceLength])
                for i in range(env.totalsteps):
                    actions[i]=list(xrest[i])+[xtime]
            else:
                actions=np.reshape(x,[env.totalsteps,env.action_spaceLength])
                
            return actions
        
        def gymWrapper(x):
            env.reset()
            actions=reshapeActions(x)
            totalreward=env.fidel_val
            for i in range(np.shape(actions)[0]):
                _,reward,done,_=env.step(actions[i])
                totalreward+=reward
                    
                
            #totalreward=env.fidel_val
            return -totalreward
        
        
        for i in range(repeatOptimize): #Repeat optimization repeatOptimize times, with random initial state
            x0=(np.random.rand(actionspace)-0.5)*0.5 #Random initial parameters
            res=sp.optimize.minimize(gymWrapper,x0,method="Nelder-Mead",bounds=bounds,options={"maxiter": maxiterations,"adaptive":True})
                    
            actions=reshapeActions(res["x"])
            print(res)
            print('scipy result',-res["fun"],actions)
            locreslist.append(res)
            if(-res["fun"]>0.99):
                break
            
        
    
    
    print(dataset)
    
    #Get results from logger in gym environment
    finalfidelresults=[env.logger[i][0] for i in range(len(env.logger))] #Fidelity at end
    #fidelresults=[env.logger[i][1] for i in range(len(env.logger))]
    penaltyresults=[np.sum(env.logger[i][3]) for i in range(len(env.logger))] #Penalty for going out of bounds, decreases to zero over training
    rewardresults=[finalfidelresults[i]-penaltyresults[i] for i in range(len(env.logger))] #Reward including penalty, e..g. reward=fidelity-penalty
    
    omegaresults=[env.logger[i][2].flatten() for i in range(len(env.logger))] #Driving parameters found during training

    argmax=np.argmax(rewardresults) #Get best result
    print(rewardresults[argmax],omegaresults[argmax])
    


    #Re/run best result found and store it in variable maxdata
    env.reset()
    fidelmaxlist=[] #Best fidelity found
    fidelmaxlist.append(env.fidel_val)

    omegamax=np.reshape(omegaresults[argmax],[env.totalsteps,env.action_spaceLength]) #Best driving parameters found, parameters are mapped between -0.5 and +0.5
    for i in range(np.shape(omegamax)[0]): #Run system with driving parameters and save fidelity
        env.step(omegamax[i])
        fidelmaxlist.append(env.fidel_val)
    
    tlist=env.tlist#np.linspace(0,t_max,num=N_bins+1) #Time
    #tlistOmega=env.tlist[1:]#np.linspace(0,t_max,num=N_bins)
    
    actualparammax=env.logger[argmax][6] #driving parameters in real values
    maxdata=[finalfidelresults[argmax],penaltyresults[argmax],omegaresults[argmax],tlist,actualparammax]
    
    
        
    totaltime=time.time()-starttime
    print(time.time()-starttime) #Total runtime
    return env.logger,totaltime,maxdata,locreslist #Return result



#Goes through the eneural network after training, and evaluate for all theta,phi
def validate(dataset,datasetLoad,nplay=11,thetas=[],phis=[],doCosTheta=False,randomParam=[]):
    global env,finalrewardlist,finalpenaltylist,finalactionlist
    #sess=tf.get_default_session()
    
    
    if(doCosTheta==False):
        thetalabel="$\\theta/\pi$"
    else:
        thetalabel="cos($\\theta$)"
        
    philabel="$\phi/(2\pi)$"
    with tf.Graph().as_default():
        #Load neural network using dataset name
        sess = tf.Session()
        model=spinup.utils.logx.restore_tf_graph(sess, output_dir+datasetLoad+"/tf1_save/")
        action_op = model['pi']
        get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]
        
        env.randomize_state=-1#to set manually target state
        if(len(thetas)==0):
            thetalist=np.linspace(0,1,num=nplay)
            #thetalist=np.linspace(-0.5,1.5,num=nplay)
        else:
            thetalist=thetas
        if(len(phis)==0):
            philist=np.linspace(0,1,num=nplay)
            #philist=np.linspace(-0.5,1.5,num=nplay)
        else:
            philist=phis
        
        thetalistplot=np.array(thetalist)
        if(doCosTheta==False):
            thetalistplot=np.array(thetalist)
        else:
            thetalistplot=(np.array(thetalist)-0.5)*2

            
        rewardlist=[]
        penaltylist=[]
        actionlist=[]
        finaltimelist=[]
        actualparamlist=[]
        #finalstatelist=[]
        
        #envthetalist=[]
        for i in range(len(thetalist)):
            for j in range(len(philist)):
                env.set_param(thetalist[i],philist[j],doCosTheta=env.useCosTheta)


                        
                        
                #if(doCosTheta==False):
                #    env.set_theta_val(thetalist[i])
                #else:
                #    env.set_cos_theta_val(thetalist[i])
                #env.phi_val=philist[j]
                
                obs=env.reset()
                done=False
                totalreward=0
                while(not(done)):
                    action=get_action(obs)
                    actionlist.append(action)
                    #print(action)
                    obs, reward, done, _ = env.step(action)
                    totalreward+=reward
                
                
                rewardlist.append(env.fidelList[-1])
                penaltylist.append(np.sum(env.penaltyList))
                finaltimelist.append(env.tlist[-1])
                actualparamlist.append(env.actualParamList)
                #finalstatelist.append(env.rho.data.toarray()[0,0])
                #envthetalist.append(env.theta_val)
                
    
    finalrewardlist=np.reshape(rewardlist,[len(thetalist),len(philist)])
    finalpenaltylist=np.reshape(penaltylist,[len(thetalist),len(philist)])  
    finalactionlist=np.reshape(actionlist,[len(thetalist),len(philist),env.totalsteps,len(action)])  
    actualparamlist=np.reshape(actualparamlist,[len(thetalist),len(philist),env.totalsteps,env.actualParamLength])
    finaltimelist=np.reshape(finaltimelist,[len(thetalist),len(philist)])
    #finalstatelist=np.reshape(finalstatelist,[len(thetalist),len(philist)])
    #envthetalist=np.reshape(envthetalist,[len(thetalist),len(philist)])
    

    if(len(philist)>1 and len(thetalist)>1):
        plot2D(finalrewardlist,philist,thetalistplot,"",philabel,thetalabel,saveto+dataset+"/",dataset,"fidel",shading='flat')
        plot2D(finaltimelist,philist,thetalistplot,"",philabel,thetalabel,saveto+dataset+"/",dataset,"time",shading='flat')
        if(env.totalsteps==1 or (env.totalsteps==2 and env.action_spaceLength==1)):
            plot2D(actualparamlist[:,:,0,0],philist,thetalistplot,"",philabel,thetalabel,saveto+dataset+"/",dataset,"Omega00",shading='flat')
            plot2D(actualparamlist[:,:,0,1],philist,thetalistplot,"",philabel,thetalabel,saveto+dataset+"/",dataset,"Omega01",shading='flat')
        if((env.totalsteps==2 and env.action_spaceLength==1) and (env.mode==4 or env.mode==5)):
            plot2D(actualparamlist[:,:,0,0],philist,thetalistplot,"",philabel,thetalabel,saveto+dataset+"/",dataset,"Omega00",shading='flat')
            plot2D(actualparamlist[:,:,1,1],philist,thetalistplot,"",philabel,thetalabel,saveto+dataset+"/",dataset,"Omega01",shading='flat')
        
        #plot2D(np.abs(finalstatelist)**2,philist,thetalist,"","$\phi$",thetalabel,".",dataset,"gsfidelity",shading='flat')
        #plot2D(envthetalist,philist,thetalist,"","$\phi$",thetalabel,".",dataset,"theta",shading='flat')
    else:
        if(len(philist)>1):
            plot1D(finalrewardlist[0,:],philist,"",philabel,"fidelity",saveto+dataset+"/",dataset,"fidel")
            plot1D(finaltimelist[0,:],philist,"",philabel,"time",saveto+dataset+"/",dataset,"time")
        if(len(thetalist)>1):
            plot1D(finalrewardlist,thetalistplot,"",thetalabel,"fidelity",saveto+dataset+"/",dataset,"fidel")
            plot1D(finaltimelist,thetalistplot,"",thetalabel,"time",saveto+dataset+"/",dataset,"time")
            
    print("Validate mean",np.mean(finalrewardlist),"validate min",np.min(finalrewardlist),"validate std",np.std(finalrewardlist) )

    
    env.randomize_state=randomize_state
    return thetalist,philist,finalrewardlist,finalpenaltylist,finalactionlist,actualparamlist
    #print(totalreward,np.sum(env.penaltyList))
   
    
def getDataset(prefix=""): #Get name of dataset
    dataset=prefix+expname+"N"+str(N_bins)+"T"+str(t_max)+"i"+str(observationtype)+"m"+str(mode)+"O"+str(Omega_max)+"o"+str(Omega_min)+"D"+str(detuning1)+"d"+str(detuning2)+"n"+str(nneurons)+"d"+str(circuitdepth)+"R"+str(randomize_state)+"S"+str(target_state[0])+"s"+str(target_state[1])+"O"+str(optimizer)
    if(magneticfield!=0):
        dataset+="B"+str(magneticfield)
    

    if(clip_ratio!=0.1):
        dataset+="D"+str(clip_ratio)
    if(doclipAction==1):
        dataset+="C"+str(doclipAction)
    if(boundsPenaltyFactor!=0):
        dataset+="P"+str(boundsPenaltyFactor)
    
    if(optimizer==0):
        dataset+="p"+str(pi_lr)+"v"+str(vf_lr)
    if(len(t_var)>0):
        dataset+="t"+str(t_var[0])+"T"+str(t_var[1])
    if(tfix!=0):
        dataset+="TFIX"+str(tfix)
    if(mapvariables!=0):
        dataset+="v"+str(mapvariables)

    if(randomize_state==3 or randomize_state==4):
        if(sample_angles!=0):
            dataset+="S"+str(sample_angles)
        if(sample_offset!=0.5):
            dataset+="o"+str(sample_offset)
        if(sample_grid!=20):
            dataset+="n"+str(sample_grid)


    if(forceUseCosTheta==1):
        dataset+="COS"
    if(optimizer==1 or optimizer==3):
        dataset+="i"+str(maxiterations)
        dataset+="r"+str(repeatOptimize)
    if(param!=""):
        dataset=param+dataset
        dataset+="d"+str(datapoints)+"p"+str(paramlist[0])+"P"+str(paramlist[-1])
    if(param2!=""):
        dataset=param2+dataset
        dataset+="D"+str(datapoints2)+"p"+str(paramlist2[0])+"P"+str(paramlist2[-1])
    if(iterations_per_epoch!=100):
        dataset+="e"+str(iterations_per_epoch)
    if(big_epochs!=8000):
        dataset+="E"+str(big_epochs)
    if(optimizer==1):
        dataset+="DIRECT"
    if(optimizer==2):
        dataset+="s"+str(sac_lr)
        dataset+="SAC"
    if(optimizer==3):
        dataset+="NM"
    


    
    dataset=dataset.replace(".","_")
    print(dataset)
    return dataset


def createEnv(): #Create environment according to a varaibles set
    global t_var
    if(len(t_var)>0):
        t_var=[t_max/N_bins*minmulttvar,t_max/N_bins*maxmulttvar]
    
    
    env = GymEnvNVnew.GymEnv(N_bins,t_max,observationtype,mode,Omega_min,Omega_max,detuning1,detuning2,forceUseCosTheta=forceUseCosTheta,sample_angles=sample_angles,sample_grid=sample_grid,sample_offset=sample_offset,mapvariables=mapvariables,t_var=t_var,magneticfield=magneticfield,doclipAction=doclipAction,boundsPenaltyFactor=boundsPenaltyFactor,randomize_state=randomize_state,target_state=target_state,tfix=tfix,customLog=customLog)  
    return env
    


saveto='./'

    
    
output_dir='./'



"""
Instructions:
#This code trains control of NV centers (via driving) using Proximal Policy optimization. Implementation via SpinningUp library.

 ##To re-create results from paper, we present here three template options to train.
 #Read further instructions therein
   """

##To re-create results from paper, we present here three template options to get data
##predefinedTemplates defines three main results
#predefinedTemplates=0 #drive NVcenter for n=600 neurons and train with PPO (main Figure of paper), runtime up to 1 day
#predefinedTemplates=1 #drive NV center and optimize with NealerMead. May run several days

#This template recreates the supplemental materials, where a two level sytem is controled by spin rotations
#Use this as a test example to play with nneurons
predefinedTemplates=0 #drive two level system to generate superposition state |0>+Exp(i phi)|1> from initial state |0>. Use this to test basic features of PPO (supplemental result - Spin system). Runs in 10 minutes or so


param=""
param2=""

if(predefinedTemplates==2):
    
    ##Uncomment next part to see this case
    """
    ##This driving restriction has no jump in fidelity
    Omega_max=1.571
    Omega_min=0
    """


    ##comment next part to see other case
    #"""
    ##This driving restriction has a jump at phi=pi in fidelity, less fidelity as driving parameters have to change at this point. Neural network tries to interpolate, however fails. 

    Omega_max=0
    Omega_min=-1.571
    #"""
    
    
    nneurons=15 #Number of neurons in neural network. With increasing neurons, the width of the dip in fidelity will become smaller!
    circuitdepth=2 #Depth of  fully connected neural network
    

    
    pi_lr=5e-4 #Learning rate for policy network
    vf_lr=1e-4 #Learning rate for value network
    
    #pi_lr=5e-5 #Learning rate for policy network
    #vf_lr=1e-5 #Learning rate for value network
    
    clip_ratio=0.05 #Clip ratio for PPO policy training

    iterations_per_epoch=100 #Iterations per big_epoch, this is the size of memory for past runs
    #big_epochs=2500 #Runs, total number of epochs is iterations_per_epoch*big_epochs
    big_epochs=150 #Runs, total number of epochs is iterations_per_epoch*big_epochs
    
    maxiterations=10000 #Iterations for Nealder Mead
    
    
    
    expname='NV2' #Name of dataset
    
    

    detuning1=0  #Detuning of first laser, not used here

    detuning2=0 #Detuning of second laser, not used here
    N_bins=2 #number of timesteps, keep at two

    t_max=2 #Duration of driving



    
    #Not used for this example
    sample_angles=4000
    sample_offset=0.5
    sample_grid=10
    

    magneticfield=0 #Not used here as well
    
    
    
    #mapvariables=-2 #1: theta runs from 0 to pi, to have periodic function, 2:Add more entries for theta, phi in state
    mapvariables=-2 #Keep as it
    
    
    minmulttvar=0.5 #Not used here
    maxmulttvar=2 #Not used here
    
    

    t_var=[] #fixed bin timesteps, keep as it
    

    
    
    
    randomize_state=6 # How to randomize target state theta,phi #0: use specific target state, 1: random theta, 2: random theta and phi, 3: random theta with sampling, 4: random phi and theta with sampling, 5: random phi with sampling, 6: random phi
    target_state=[0.5,0] #Target state, first index is theta, second phi.  randomize_state=6 means that phi is randomized over, thus entry for phi value is ignored
    
    observationtype=2 #Input to neural network is #0: previously chosen driving parameters 1: density matrix with absolute value and phase 2:Density matrix with real/imag part

    
    repeatOptimize=5 #Repeat optimizer for better results for Nealder Mead
    optimizer=0 #0: PPO, 1: DIRECT, 2: soft actor critic, 3: nealder-mead
    
    mode=4 #0: Full open system NV center 1: effective closed system (8 level), 2: closed Lambda system (3 level approximation), 3: two level system, 4: two level system where x rotation is followed by z rotation

    #To ensure driving parameters chosen by neural network are within specific bounds, use penalty to enforce the restriction
    #boundsPenaltyFactor=0.2 #>0 punish boundary, 0: hard clip, -1: do nothing
    boundsPenaltyFactor=5



elif(predefinedTemplates==0 or predefinedTemplates==1):

    nneurons=600 #Number of neurons in neural network
    circuitdepth=2 #Depth of  fully connected neural network
    
    
    pi_lr=1e-5 #Learning rate for policy network
    vf_lr=1e-5 #Learning rate for value network
    
    clip_ratio=0.05 #Clip ratio for PPO policy training

    iterations_per_epoch=100 #Iterations per big_epoch, this is the size of memory for past runs
    big_epochs=6500 #Runs, total number of epochs is iterations_per_epoch*big_epochs
    
    maxiterations=10000 #Iterations for Nealder Mead
    
    
    
    expname='NV4' #Name of dataset
    

    
    
    

    
    detuning1=50 #Detuning of first laser
    detuning2=0 #Detuning of second laser
    N_bins=9 #Number of timesteps
    t_max=0.4 #Time of quantum evolution
    
    
    #t_var sets minimal and maximal time of evolution. At each step, the neural network selects the duration of the timestep, the minimal and maximal value is given by t_var
    minmulttvar=0.5 #minimal time multiplicative factor
    maxmulttvar=2 #maximal time multiplicative factor
    
    t_var=[t_max/N_bins*minmulttvar,t_max/N_bins*maxmulttvar] #min and max bound of deltat. this line is repeated below when running
    #t_var=[] #if empty list use a fixed bin timesteps
    
    
    Omega_max=20 #Maximal driving strength, #Changed behavior if detuning1=detuning2, now set Omega2=Omega1
    
    Omega_min=-Omega_max #Minmal driving strength #Negative Omega gives better results than more positive
    
    
    
    
    magneticfield=0.15 #Magnetic field
    #magneticfield=0
    
    #To ensure driving parameters chosen by neural network are within specific bounds, use penalty to enforce the restriction
    boundsPenaltyFactor=0.2 #>0 punish boundary, 0: hard clip, -1: do nothing
    
    
    
    
    
    
        

    #target_state=[0.5,0.25]
    target_state=[0.5,0] #which target state to optimized, used only if randomize_state=0 as else its selected randomly
    
    observationtype=2 #Input to neural network is #0: previously chosen driving parameters 1: density matrix with absolute value and phase 2:Density matrix with real/imag part
    
    repeatOptimize=5 #Repeat optimizer for better results for Nealder Mead
    
    
    #What system to evolve
    mode=1 #0: Full open system NV center 1: effective closed system (8 level), 2: closed Lambda system (3 level approximation), 3: two level system, 4: two level system where x rotation is followed by z rotation
    
    
    mapvariables=-2 #-2: target states are mapped to theta and phi 2: Target states mapped to theta, phi, phi is mapped to cos(phi) and sin(phi) to create periodic phi in neural network
    
    
    #When randomize_state=2 or randomize_state=4, target states are not chosen complelty random, but areas of low fidelity are sampled preferentially. These are the hyperparameters
    #Sample not randomly, but adjust sampling to focus on areas which havelow fidelity.
    sample_angles=10000 #How many samples to use in a rolling basis
    sample_offset=0.5 #How much to deviate from random sampling, real number between 0 and 1. 1: just random sampling, 0: sample just according to areas of low fidelity
    sample_grid=20 #Grid size to determine fidelity 
    
    



    if(predefinedTemplates==1):
        randomize_state=0 # How to randomize target state theta,phi #0: use specific target state, 1: random theta, 2: random theta and phi, 3: random theta with sampling, 4: random phi and theta with sampling, 5: random phi with sampling, 6: random phi
        optimizer=3 #0: PPO, 1: DIRECT, 2: soft actor critic, 3: nealder-mead
        param="target2"
        param2="targetCos"
    else:
        randomize_state=4 # How to randomize target state theta,phi #0: use specific target state, 1: random theta, 2: random theta and phi, 3: random theta with sampling, 4: random phi and theta with sampling, 5: random phi with sampling, 6: random phi
        optimizer=0 #0: PPO, 1: DIRECT, 2: soft actor critic, 3: nealder-mead
    


#Other parameters that shouldnt be changed
special=0


doclipAction=0
tfix=0 #default 0, fix time in first step when set to 1

sac_lr=0.001 #Learning rate for SAC only

forceUseCosTheta=0 #keep at 0 Use cos Theta for evaluation

customLog=1

if(optimizer==0):
    customLog=1

if(randomize_state>0 and (optimizer==1 or optimizer==3)):
    raise NameError("Bad combination, optimizer and random")


#Train for various parameters

datapoints=11 #Number of datapoints to go over for param if param!=""
datapoints2=11 #Number of datapoints to go over for param2 if param2!=""
if(param=="t_max"):
    paramlist=np.linspace(0,1.,num=datapoints)[1:]
if(param=="Omega"):
    paramlist=np.linspace(0,30,num=datapoints)[1:]
elif(param=="N_bins"):
    paramlist=np.arange(1,datapoints+1)
elif(param=="target"):
    paramlist=np.linspace(0,1,num=datapoints)
elif(param=="target2"):
    paramlist=np.linspace(0,1,num=datapoints)
elif(param=="detuning1"):
    paramlist=np.linspace(0,100,num=datapoints)
elif(param=="magneticfield"):
    paramlist=np.linspace(0,0.1,num=datapoints)


if(param2=="targetCos"):
    paramlist2=np.linspace(0,1,num=datapoints2)
    forceUseCosTheta=1
elif(param2=="t_max"):
    paramlist2=np.linspace(0,1,num=datapoints2)[1:]


if(param==""):
    datapoints=1
else:
    datapoints=len(paramlist)
    
if(param2==""):
    datapoints2=1
else:
    datapoints2=len(paramlist2)





loggerlist=[] #Stores logged data
calctime=[]
maxdatalist=[] #Stores best results
reslist=[]

dataset=getDataset() #Get dataset name



countfigure=100
#Run for all selected parameters
for j in range(datapoints2): #Go over param2
    if(param2!=""):
        if(param2=="targetCos"):
            target_state=[paramlist2[j],target_state[0]]
        else:
            vars()[param]=paramlist2[j]
    
    for i in range(datapoints): #Go over param1
        timedatastart=time.time()
        print("doing datapoint",j,i)
        dataset=getDataset()
        if(param!=""):
            if(param=="Omega"):
                Omega_max=paramlist[i]
                Omega_min=-paramlist[i]
            elif(param=="target"):
                target_state=[paramlist[i],target_state[1]]
            elif(param=="target2"):
                target_state=[target_state[0],paramlist[i]]
            else:
                vars()[param]=paramlist[i]
    
    
    
        env=createEnv() #Get environment

        result=runNVOptimize() #Run optimizer
        
        
        
        #Save data to lists
        if(datapoints2==1):
            loggerlist.append(result[0])
            
        
        calctime.append(result[1])
        maxdatalist.append(result[2])
        reslist.append(result[3])
        print("Finish datapoint",j,i, "with time",time.time()-timedatastart)


        



#Evaluate data

res=[]
maxfidellist=[]
maxomegalist=[]
maxtimelist=[]
averagelist=[]
for j in range(len(loggerlist)): #Go through all logged data
    if(optimizer==0 or optimizer==2):
        wheresave=saveto+dataset+"/"
    else:
        wheresave=saveto
    templog=loggerlist[j]
    finalfidelresults=[templog[i][0] for i in range(len(templog))] #Final fidelity
    fidelresults=[templog[i][1] for i in range(len(templog))] #Evolution of fidelity during time evoltuion
    penaltyresults=[np.sum(templog[i][3]) for i in range(len(templog))] #Penalty for driving parameters out of bound
    rewardresults=[finalfidelresults[i]-penaltyresults[i] for i in range(len(templog))] #Reward=fidelity-penalty
    
    nskip=100 
    stepavg=100
    if(len(rewardresults)<nskip):
        nskip=1
        stepavg=1
    averagelist.append(moving_averageAvgSkip(finalfidelresults,stepavg,skip=nskip)) #Moving average over final fidelity
    omegaresults=[templog[i][2].flatten() for i in range(len(templog))] #driving parameters
    
    
    #Get best result with highest reward
    argmax=np.argmax(rewardresults)
    print(rewardresults[argmax],omegaresults[argmax])
    maxfidellist.append(rewardresults[argmax])
    maxomegalist.append(omegaresults[argmax])
    maxtimelist.append(templog[argmax][4][-1])

    #
    if(datapoints2==1):
        #Plot reward over training epochs
        plt.figure(countfigure)
        countfigure+=1
        plt.plot(moving_averageAvgSkip(rewardresults,stepavg,skip=nskip))
        plt.plot(moving_max(rewardresults,stepavg,skip=nskip))
        print(moving_averageAvgSkip(rewardresults,stepavg,skip=nskip)[-1])
        
    
        
    
#            plt.figure(countfigure)
#            countfigure+=1
#            plt.plot(averagelist[-1])
#            plt.plot(moving_max(finalfidelresults,stepavg,skip=nskip))
#            plt.plot(-moving_max(-np.array(finalfidelresults),stepavg,skip=nskip))
        print(moving_averageAvgSkip(finalfidelresults,stepavg,skip=nskip)[-1])
        
        epochlist=np.arange(len(averagelist[-1]))*iterations_per_epoch/1000
        #Plot fidelity over training epochs
        plot1D([moving_max(finalfidelresults,stepavg,skip=nskip),averagelist[-1],-moving_max(-np.array(finalfidelresults),stepavg,skip=nskip)],epochlist,"","epochs $(10^3)$","fidelity",wheresave,dataset,"fidelEpoch",elements=3,scatter=True,xmin=np.amin(epochlist),xmax=np.amax(epochlist),xnbins=4,markersize=1,label=['max','avg','min'])
    


            
    
    
    


#Plot results for nealder mead
if(datapoints>1 and datapoints2>1):
    if(env.useCosTheta==False):
        thetalabel="$\\theta/\pi$"
    else:
        thetalabel="cos($\\theta$)"
        
    philabel="$\phi/(2\pi)$"
    
    if(param=="magneticfield"):
        thetalabel="$T$"
    if(param2=="t_max"):
        philabel="$B$"
        
    
    maxfidellist=np.reshape([maxdatalist[i][0] for i in range(len(maxdatalist))],[datapoints2,datapoints])
    maxtimelist=np.reshape([maxdatalist[i][3][-1] for i in range(len(maxdatalist))],[datapoints2,datapoints])


    #Maximal fidelity against theta and phi3
    plot2D(maxfidellist,paramlist,paramlist2,"",philabel,thetalabel,saveto,dataset,"time",shading='flat')
    plot2D(maxtimelist,paramlist,paramlist2,"",philabel,thetalabel,saveto,dataset,"time",shading='flat')
    if((env.totalsteps==2 and env.action_spaceLength==1) and (env.mode==4 or env.mode==5)):
        maxActualParamlist=np.reshape([maxdatalist[i][4] for i in range(len(maxdatalist))],[datapoints2,datapoints,np.shape(maxdatalist[0][4])[0],np.shape(maxdatalist[0][4])[1]])
        plot2D(maxActualParamlist[:,:,0,0],paramlist,paramlist2,"",philabel,thetalabel,saveto,dataset,"Omega00",shading='flat')
        plot2D(maxActualParamlist[:,:,1,1],paramlist,paramlist2,"",philabel,thetalabel,saveto,dataset,"Omega01",shading='flat')
    
    



#Here run trained neural network for all target states and get fidelities after training
env.customLog=0

if(randomize_state==1 or randomize_state==3): 
    res=validate(dataset,dataset,nplay=51,thetas=np.linspace(0,1,num=201),phis=[target_state[1]])
elif(randomize_state==2 or randomize_state==4):
    #env = GymEnvNV.GymEnv(N_bins,t_max,observationtype,mode,Omega_min,Omega_max,detuning1,detuning2,sample_angles=sample_angles,sample_grid=sample_grid,sample_offset=sample_offset,mapvariables=mapvariables,t_var=t_var,magneticfield=magneticfield,doclipAction=doclipAction,boundsPenaltyFactor=boundsPenaltyFactor,randomize_state=randomize_state,target_state=target_state,customLog=customLog)  
    #env.useCosTheta=1
    res=validate(dataset,dataset,nplay=201,doCosTheta=True) 
elif(randomize_state==5 or randomize_state==6): 
    res=validate(dataset,dataset,nplay=51,thetas=[target_state[0]],phis=np.linspace(0,1,num=201))
elif(randomParamType!=0):
    res=validate(dataset,dataset,nplay=51,doCosTheta=True,randomParam=randomParam)
env.customLog=customLog
print(maxfidellist)

#Save to file
if(optimizer==0):
    outfile=open( output_dir+"/"+"LogNew"+dataset+".pcl", "wb" )
    pickle.dump([loggerlist[0],[maxfidellist,maxomegalist,maxtimelist,averagelist],res], outfile)
    outfile.close()

if(env.totalsteps==2 and env.action_spaceLength==1):#Plot for two level system only
    thetalist,philist,finalrewardlist,finalpenaltylist,finalactionlist,actualparamlist=res
    if(Omega_min<0):
        sign=-1
    else:
        sign=1
    philabel="$\phi/(2\pi)$"
    linewidth=2.5
    plot1D(sign*actualparamlist[0,:,0,0]/np.pi*2,philist,"",philabel,"$\Omega_y$",saveto+dataset+"/",dataset,"drivey", elements=1,linewidth=[linewidth],ymin=-0.02,ymax=1.02,plot1DLinestyle=["solid","dashed","dotted","dashdot","solid","dashed","dotted","dashdot"])

    plot1D(sign*actualparamlist[0,:,1,1]/np.pi*2,philist,"",philabel,"$\Omega_z$",saveto+dataset+"/",dataset,"drivez", elements=1,linewidth=[linewidth],ymin=-0.02,ymax=1.02,plot1DLinestyle=["solid","dashed","dotted","dashdot","solid","dashed","dotted","dashdot"])


