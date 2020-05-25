# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:48:06 2019

@author: Tobias Haug (tobias.haug@u.nus.edu)

#Gym Environment to drive NV center
"""

import gym

from gym.spaces import Discrete, Box
import numpy as np
import scipy as sp
import NV_tenlevel_new as NV

import scipy.stats as stats


def getNearestIndex(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx



#Gym environment
class GymEnv(gym.Env):
    #Run when environment is initialized
    def __init__(self,N_bins,t_horizon,observationtype,mode,Omega_min,Omega_max,freq1,
                 freq2,forceUseCosTheta=0,
                 sample_angles=1000,sample_grid=20,sample_offset=0.5,mapvariables=0,t_var=[],magneticfield=0,
                 doclipAction=0,boundsPenaltyFactor=0,randomize_state=0,target_state=[1,0],
                 tfix=0,customLog=0):
        
        self.N_bins = N_bins #Number of timesteps
        self.t_horizon = t_horizon #Maximal time
        self.Omega_max = Omega_max #Maximal driving strenght
        self.Omega_min = Omega_min #Minimal driving strength
        
        self.OmegaZero=-self.Omega_min/(self.Omega_max-self.Omega_min ) #Define zero  Omega for the Omega parameter for remapping purposes
        
        self.freq1 = freq1 #Detuning 1
        self.freq2 = freq2 #Detunin g2
        self.mode = mode #0: Full open system NV center 1: effective closed system (8 level), 2: closed Lambda system (3 level approximation), 3: two level system, 4: two level system where x rotation is followed by z rotation

        self.customLog=customLog
        
        
        self.epsilon=10**-7
        
        
        self.magneticfield=magneticfield #Magnetic field
        
        #To ensure driving parameters chosen by neural network are within specific bounds, use penalty to enforce the restriction
        self.boundsPenaltyFactor=boundsPenaltyFactor #>0 punish boundary, 0: hard clip, -1: do nothing
        
        
        self.tfix=tfix#Fix time in first step if 1
        self.t_var=t_var #if time is variable, at every step neuralnetwork will choose timestep
        self.numtaction=0
        if(len(self.t_var)!=0):
            if(self.tfix==0):
                self.numtaction=1 #Set to one if neural network chooses timestep
                
            self.tbound_min=self.t_var[0]
            self.tbound_max=self.t_var[1]
        
        self.doclipAction=doclipAction
        self.target_state=target_state
        self.randomize_state=randomize_state

        
        if(self.randomize_state==2 or self.randomize_state==4 or forceUseCosTheta==1):
            self.useCosTheta=1
        else:
            self.useCosTheta=0
            

            
        self.mapvariables=mapvariables
            

        
        self.observationtype=observationtype
            

        
        
        # Total number of timesteps
        
        if(self.tfix==0):
            self.totalsteps = self.N_bins #number of time bins
        else:
            self.totalsteps=self.N_bins+1
        

        
        self.sample_angles=sample_angles
        self.sample_offset=sample_offset
        self.sample_grid=sample_grid
        
                
        self.countsampling=0
        self.Nredostatistic=int(np.ceil(sample_angles/100)) #redo statistics every self.Nredostatistic steps
        self.statistic=[]
        
        self.modTheta=1
        if(self.mapvariables==1 or self.mapvariables==-1):
            self.modTheta=2 #modifz theta to run from 0 to pi to be periodic in theta 
        

        # Space of possible actions at a given iteration
        
        
        self.NVcenter=NV.NV(freq1,freq2,mode,B=magneticfield) #Create NV center class
        
        #driveAction tells us how many driving paraeters per step are chosen
        if(self.freq1==self.freq2):
            self.driveAction=1
        else:
            self.driveAction=2
            
        if(self.mode==3):
            self.driveAction=2
        if(self.mode==4 or self.mode==5):
            self.driveAction=1
            
            
        self.action_spaceLength=self.driveAction+self.numtaction #update driving amplitudes of 2 lasers and 1 for time

            
            
        self.hilbertspace=self.NVcenter.hilbertspace
        
        
        self.centerAction=0
        self.widthAction=1
            
        self.action_space=Box(low=-np.ones(self.action_spaceLength)*self.widthAction/2+self.centerAction,high=np.ones(self.action_spaceLength)*self.widthAction/2+self.centerAction) #low and high define lower and upper bound of allowed values into neuronal network
        
        
        #This sets up the observation_space, e.g. the input to the neural network.
        #The target state is parameterized using angles theta,phi. One can feed them parameterized in different ways to the neural network.
        #mapvariables==-2 feeds in theta 4 times (parameterized betweeon 0 and 1), and phi 2 times (also between 0 and 1)
        if(self.mapvariables==0):
            self.startrho=3
            self.mapfactor=1
        elif(self.mapvariables==1):
            self.startrho=5
            self.mapfactor=1
        elif(self.mapvariables==-1):
            self.startrho=4
            self.mapfactor=1
        elif(self.mapvariables==2):
            self.mapfactor=2
            self.startrho=1+self.mapfactor*4
        elif(self.mapvariables==-2):
            self.mapfactor=2
            self.startrho=1+self.mapfactor*3
        elif(self.mapvariables==3):
            self.mapfactor=3
            self.startrho=1+self.mapfactor*4
        elif(self.mapvariables==-3):
            self.mapfactor=3
            self.startrho=1+self.mapfactor*3
        elif(self.mapvariables==-11):
            self.mapfactor=1
            self.startrho=1+self.mapfactor*2
        elif(self.mapvariables==-12):
            self.mapfactor=2
            self.startrho=1+self.mapfactor*4 #Includes input state

            



         #Input to neural network has length observation_spaceLength, includes the target state as well as the current wavefunction
        if(self.observationtype==0):
            self.observation_spaceLength = self.startrho +self.action_spaceLength*self.N_bins #Length of your state, e.g. the input to your neuronal network
        elif(self.observationtype==1 or self.observationtype==2):
            if(self.mode==0):
                self.observation_spaceLength = self.startrho + self.NVcenter.LVspace
            elif(self.mode==1 or self.mode==2 or self.mode==3 or self.mode==4 or self.mode==5):
                self.observation_spaceLength = self.startrho + self.hilbertspace*2       
        

        
        # Observation space (The space of possible states at a given iteration)
        self.observation_space=Box(low=np.zeros(self.observation_spaceLength),high=np.ones(self.observation_spaceLength)) #low and high define lower and upper bound of allowed values into neuronal network

    
        if(self.customLog==1):
            self.logger=[]
            
            
            

        self.reset()
        
    #Does randomized sampling modified such that areas of low fidelity are sampled with higher probability
    def sampling(self,x,y,z,statistic=[]):
        #input irregular data - obtained from neural network
        #N_grid =20
        nsample= 1 #1 random sample each time
        N_grid=self.sample_grid
        #grid averaging to smooth out huge fluctuations in data
        
        if(len(statistic)==0):
            statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values=z, statistic='mean',range=[[0, 1], [0, 1]], bins=(N_grid,N_grid))
            statistic[np.isnan(statistic)] = 0
        
        
        probDistr=(self.sample_offset/N_grid**2+(1-self.sample_offset)*statistic/np.sum(statistic))
        
        
        xedges=np.linspace(0,1,num=N_grid+1)
        yedges=np.linspace(0,1,num=N_grid+1)
        xcenter=(xedges[1:]+xedges[:-1])/2
        ycenter=(yedges[1:]+yedges[:-1])/2
        
        # generate the set of all x,y pairs represented by the pmf
        pairs=np.indices(dimensions=(N_grid,N_grid)).T # here are all of the x,y pairs 
        helpindex=np.arange(N_grid**2)
        
        # make n random selections from the flattened pmf without replacement
        # whether you want replacement depends on your application
        
        inds=np.random.choice(helpindex,p=probDistr.reshape(-1),size=nsample,replace=True)
        
        # inds is the set of n randomly chosen indicies into the flattened dist array...
        # therefore the random x,y selections
        # come from selecting the associated elements
        # from the flattened pairs array
        selections = pairs.reshape(-1,2)[inds]
        
        dx=xcenter[1]-xcenter[0]
        dy=ycenter[1]-ycenter[0]
        xresult=xcenter[selections[:,1]]+(np.random.rand(nsample)-0.5)*dx
        yresult=ycenter[selections[:,0]]+(np.random.rand(nsample)-0.5)*dy
    
        return(np.transpose([yresult,xresult])),statistic #set values back to [0,1]
        
        
        

    
    #Sets target state theta and phi. Sets both cos_theta and theta. Choose doCosTheta to sample with Haar measure
    def set_param(self,thetaIn,phi,doCosTheta=0):
        if(doCosTheta==0):
            self.theta_val = thetaIn #np.random.rand() #selects random theta from [0,pi)
            self.cos_theta_val = (np.cos(thetaIn*np.pi)+1)/2 #goes from 0 to 1
        else:
            self.cos_theta_val = thetaIn #sample with haar measure
            self.theta_val=np.arccos((thetaIn-0.5)*2)/np.pi
        
        self.phi_val = phi #np.random.rand() #selects random theta from [0,pi)

        self.inputTheta,self.inputPhi=[thetaIn,phi]
        
    def set_ini_param(self,ini_thetaIn,ini_phi,doCosTheta=0):
        if(doCosTheta==0):
            self.ini_theta_val = ini_thetaIn #np.random.rand() #selects random theta from [0,pi)
            self.ini_cos_theta_val = (np.cos(ini_thetaIn*np.pi)+1)/2 #goes from 0 to 1
        else:
            self.ini_cos_theta_val = ini_thetaIn #sample with haar measure,goes from 0 to 1
            self.ini_theta_val=np.arccos((ini_thetaIn-0.5)*2)/np.pi
        
        self.ini_phi_val = ini_phi #np.random.rand() #selects random theta from [0,pi)

        self.ini_inputTheta,self.ini_inputPhi=[ini_thetaIn,ini_phi]
        
        
        
        
    #runs after everytime one run is finished
    def reset(self):
        self.game_step=0
        self.done= False
        
        self.deltaTaction=None
        
        #Set target states randomly
        if(self.randomize_state==0): #set manually both theta and phi, no randomness
            self.set_param(self.target_state[0],self.target_state[1],doCosTheta=self.useCosTheta)

        elif(self.randomize_state==1): #set only theta randomly
            self.set_param(np.random.rand(),self.target_state[1],doCosTheta=self.useCosTheta)

        elif(self.randomize_state==2):#set both theta and phi randomly
            self.set_param(np.random.rand(),np.random.rand(),doCosTheta=self.useCosTheta)

        elif(self.randomize_state==6): #set only phi randomly
            self.set_param(self.target_state[0],np.random.rand(),doCosTheta=self.useCosTheta)


        elif(self.randomize_state==3 or self.randomize_state==4 or self.randomize_state==5): #sampling from custom distribution
            if len(self.logger) < self.sample_angles:
                if(self.randomize_state==3):
                    self.set_param(np.random.rand(),self.target_state[1],doCosTheta=self.useCosTheta)

                elif(self.randomize_state==4):
                    self.set_param(np.random.rand(),np.random.rand(),doCosTheta=self.useCosTheta)

                elif(self.randomize_state==5):
                    self.set_param(self.target_state[0],np.random.rand(),doCosTheta=self.useCosTheta)

            else:
                if(self.useCosTheta==0):
                    fids,thetas,phis = np.transpose([[self.logger[-i-1][0],self.logger[-i-1][5][0],self.logger[-i-1][5][1]] for i in range(self.sample_angles)]) #obtain angles from logger
                else:
                    fids,cos_thetas,phis = np.transpose([[self.logger[-i-1][0],self.logger[-i-1][5][2],self.logger[-i-1][5][1]] for i in range(self.sample_angles)]) #obtain angles from logger

                #print(phis,thetas,fids)
                
                if(self.countsampling%self.Nredostatistic==0):
                    self.statistic=[]
                if(self.randomize_state==3):
                    angles,self.statistic  = self.sampling(phis,thetas,1-fids,statistic=self.statistic)
                elif(self.randomize_state==4):
                    angles,self.statistic  = self.sampling(phis,cos_thetas,1-fids,statistic=self.statistic)
                self.countsampling+=1
                
                if(self.randomize_state==3):
                    theta_val, _ =angles[0]
                    self.set_param(theta_val,self.target_state[1],doCosTheta=self.useCosTheta)
                    #self.set_theta_val(theta_val)
                    #self.cos_theta_val = (np.cos(self.theta_val*np.pi)+1)/2
                elif(self.randomize_state==4):
                    cos_theta_val, phi_val =angles[0]
                    self.set_param(cos_theta_val,phi_val,doCosTheta=self.useCosTheta)
                    #self.set_phi_val(phi_val)
                    #self.set_cos_theta_val(cos_theta_val)
                    #self.theta_val=np.arccos((self.cos_theta_val-0.5)*2)/np.pi
                elif(self.randomize_state==5):
                    _, phi_val =angles[0]
                    self.set_param(self.target_state[0],phi_val,doCosTheta=self.useCosTheta)
                    #self.set_phi_val(phi_val)

        #Set initial state  default to theta=0, phi=0
        self.set_ini_param(0,0,doCosTheta=0)
            
                    
        #Set current wavefunction
        self.rho = self.NVcenter.getInitialstate(theta=self.ini_theta_val,phi=self.ini_phi_val) #initial state
        
        #get current fidelity
        self.fidel_val=self.NVcenter.calcFidelity(self.rho,self.theta_val*self.modTheta,self.phi_val)
        
        #initialize penality list
        self.penaltyList=[0]
    

        #list of all driving parameters, mapped between 0 and 1
        self.OmegaList=np.zeros([self.totalsteps,self.action_spaceLength])
        
        
        #makea  list of driving parameters in actual numbers
        self.actualParamLength=3
        self.actualParamList=np.zeros([self.totalsteps,self.actualParamLength])
        
        self.fidelList=np.zeros(self.totalsteps+1)
        
        #print(self.game_step)
        self.fidelList[0]=self.fidel_val
        
        self.tlist=[0]
        
        
        return self.constructState()
    

    
    #ConstructState, the input to the neural network. Includes the target quantum state (parametrized by theta and phi), as well as the current wavefunction
    def constructState(self):
        #construct state of environment
        
        state = np.zeros(self.observation_spaceLength) #initialise input as zeros
        startrho=self.startrho #at which index current wavefunction is put
        
        #Set target quantum state in state
        #There are various ways to do this
        if(self.mapvariables==0): #input theta and phi target states, each once into neural net
            state[0] =self.inputTheta
            #if(self.useCosTheta==1):
            #    state[0] = self.cos_theta_val 
            #else:
            #    state[0] = self.theta_val
            #state[1] = self.phi_val#(np.cos(2*np.pi*self.phi_val)+1)/2
            state[1]=self.inputPhi
            #state[2] = (np.sin(2*np.pi*self.phi_val)+1)/2
            
        elif(self.mapvariables==1):  # input theta and phi, both periodic by using sine/cosine. requires theta to go from 0 to 2pi
            state[0] = (np.cos(2*np.pi*self.inputTheta)+1)/2
            state[1] = (np.sin(2*np.pi*self.inputTheta)+1)/2
            state[2] = (np.cos(2*np.pi*self.inputPhi)+1)/2
            state[3] = (np.sin(2*np.pi*self.inputPhi)+1)/2
        elif(self.mapvariables==2 or self.mapvariables==3): #input theta and phi, phi is periodic via sine/cosine trick
            state[0:self.mapfactor*2] =self.inputTheta
            #if(self.useCosTheta==1):
            #    state[0:self.mapfactor*2] = self.cos_theta_val
            #else:
            #    state[0:self.mapfactor*2] = self.theta_val
            state[self.mapfactor*2:self.mapfactor*3] = (np.cos(2*np.pi*self.inputPhi)+1)/2
            state[self.mapfactor*3:self.mapfactor*4] = (np.sin(2*np.pi*self.inputPhi)+1)/2
            
        elif(self.mapvariables==-1): #
            state[0] = (np.cos(2*np.pi*self.inputTheta)+1)/2
            state[1] = (np.sin(2*np.pi*self.inputTheta)+1)/2
            state[2] = self.inputPhi
        elif(self.mapvariables==-2 or self.mapvariables==-3): #input theta, phi without periodicity
            state[0:self.mapfactor*2] =self.inputTheta
            #if(self.useCosTheta==1):
            #    state[0:self.mapfactor*2] = self.cos_theta_val
            #else:
            #    state[0:self.mapfactor*2] = self.theta_val
            state[self.mapfactor*2:self.mapfactor*3] = self.inputPhi
        #elif(self.mapvariables==-12 or self.mapvariables==-11):
        #    state[0:self.mapfactor] =self.inputTheta
        #    state[self.mapfactor:self.mapfactor*2] = self.inputPhi
        elif(self.mapvariables==-12):
            state[0:self.mapfactor] =self.ini_inputTheta
            state[self.mapfactor:self.mapfactor*2] = self.ini_inputPhi
            state[self.mapfactor*2:self.mapfactor*3] =self.inputTheta
            state[self.mapfactor*3:self.mapfactor*4] = self.inputPhi
            



            
            

        state[startrho-1]=self.game_step/self.totalsteps#set currnet time in state given to neural network

        if(self.observationtype==0): #use past driving information as input to neural network
        
            for i in range(self.action_spaceLength):
                state[startrho+i*self.totalsteps:startrho+(i+1)*self.totalsteps] = self.OmegaList[:,i]
            
        elif(self.observationtype==1): #Insert rho as state with phase angle and absolute value as information for neural network

            if(self.mode==0):
                rawDM=self.rho.data.toarray() #get density matrix as array from Qutip
                occupations=np.diag(rawDM)
                uppertriangindex=np.triu_indices(self.hilbertspace, 1)
                #print(np.abs(rawDM[uppertriangindex]),self.observation_spaceLength,3+hilbertspace,3+hilbertspace+len(uppertriangindex))
                state[startrho:startrho+self.hilbertspace]=np.abs(occupations)
                state[startrho+self.hilbertspace:startrho+self.hilbertspace+np.shape(uppertriangindex)[1]]=np.abs(rawDM[uppertriangindex])
                state[startrho+self.hilbertspace+np.shape(uppertriangindex)[1]:startrho+self.hilbertspace+2*np.shape(uppertriangindex)[1]]=(np.angle(rawDM[uppertriangindex])/np.pi+1)/2.
            elif(self.mode==1 or self.mode==2 or self.mode==3 or self.mode==4 or self.mode==5):
                wavefunction=self.rho.data.toarray()[:,0]
                state[startrho:startrho+self.hilbertspace]=np.abs(wavefunction)
                state[startrho+self.hilbertspace:startrho+2*self.hilbertspace]=(np.angle(wavefunction)/np.pi+1)/2.
        elif(self.observationtype==2): #Insert rho as state as real/imag as info for neural net
            if(self.mode==0):
                rawDM=self.rho.data.toarray() #get DM as array from Qutip
                occupations=np.diag(rawDM)
                uppertriangindex=np.triu_indices(self.hilbertspace, 1)
                #print(np.abs(rawDM[uppertriangindex]),self.observation_spaceLength,3+hilbertspace,3+hilbertspace+len(uppertriangindex))
                state[startrho:startrho+self.hilbertspace]=np.abs(occupations)
                state[startrho+self.hilbertspace:startrho+self.hilbertspace+np.shape(uppertriangindex)[1]]=np.real(rawDM[uppertriangindex])
                state[startrho+self.hilbertspace+np.shape(uppertriangindex)[1]:startrho+self.hilbertspace+2*np.shape(uppertriangindex)[1]]=np.imag(rawDM[uppertriangindex])
            elif(self.mode==1 or self.mode==2 or self.mode==3 or self.mode==4 or self.mode==5):
                wavefunction=self.rho.data.toarray()[:,0]
                state[startrho:startrho+self.hilbertspace]=np.real(wavefunction)
                state[startrho+self.hilbertspace:startrho+2*self.hilbertspace]=np.imag(wavefunction)
            
        return state

    # Do one step of environment with action
    def step(self,action):
        ##Do action on environment here, also update state of environment here
        reward=0
        #print(action)
        if(self.done==False):
            #run mesolve for 1 time bin
            self.previousFidel_val=self.fidel_val #Previous fidelity
            
            #Calculate penalty when going out of bounds for driving parameters
            penalty=0
            clipaction=np.array(action)
            if(self.doclipAction==1):
                clipaction=np.array([np.clip(action, self.action_space.low[i], self.action_space.high[i]) for i in range(self.action_spaceLength)])
            elif(self.boundsPenaltyFactor>0):
                for i in range(len(action)):
                    if(action[i]<self.action_space.low[i]):
                        penalty+=np.abs(action[i]-self.action_space.low[i])*self.boundsPenaltyFactor
                    if(action[i]>self.action_space.high[i]):
                        penalty+=np.abs(action[i]-self.action_space.high[i])*self.boundsPenaltyFactor
                
            self.penaltyList.append(penalty)
            
            
            #Map actions between 0 and 1
            actionMapped= (clipaction-self.centerAction+self.widthAction/2)/self.widthAction#Omegas mapped between 0 and 1 (optimally), from output of action, which is between -0.5 and 0.5
            
            
            #Map driving parameters to actual numbers
            if(self.tfix==1 and self.game_step==0):
                actualOmega1=0
                actualOmega2=0
            else:
                Omega1=actionMapped[0]
                if self.driveAction>1:
                    Omega2=actionMapped[1]
                else:
                    if(self.mode==4 or self.mode==5):

                        if(self.game_step%2==self.tfix):
                            Omega1=actionMapped[0]
                            Omega2=self.OmegaZero
                        else:
                            Omega1=self.OmegaZero
                            Omega2=actionMapped[0]

                    else:
                        Omega2=Omega1 #If no detuning, set Omega2=Omega1, #-self.Omega_min/(self.Omega_max-self.Omega_min )
                
                actualOmega1=(self.Omega_max-self.Omega_min )* Omega1+self.Omega_min
                actualOmega2=(self.Omega_max-self.Omega_min )* Omega2+self.Omega_min
            
            

            #Get time step length self.deltaTaction
            if(len(self.t_var)!=0): #map deltaT action between 0 and 1
                if(self.tfix==0):#Fix timestep in everz game step
                    if self.driveAction>1:
                        self.deltaTaction=actionMapped[2]
                    else:
                        self.deltaTaction=actionMapped[1]
                else:
                    if(self.game_step==0): #Use first step to determine timestep
                        self.deltaTaction=actionMapped[0]
            
            
            if(self.tfix==1 and self.game_step==0):
                tstart=0
                tend=0
            else:
                if(len(self.t_var)!=0):
                    tstart=self.tlist[-1]
                    tend=self.tlist[-1]+self.deltaTaction*(self.tbound_max-self.tbound_min)+self.tbound_min
                    if(tend<tstart):
                        tend=tstart+self.epsilon
                else:
                    tstart=self.game_step * self.t_horizon/self.N_bins
                    tend=(self.game_step+1)*self.t_horizon/self.N_bins
                    
                #Set current time
                self.tlist.append(self.tlist[-1]+tend-tstart)
                #Set current wavefunction
                self.rho = self.NVcenter.run(self.rho, tstart, tend, actualOmega1, actualOmega2)
        
                #Set current fielidty
                self.fidel_val = self.NVcenter.calcFidelity(self.rho,self.theta_val*self.modTheta,self.phi_val)
        
    
    
            reward = self.fidel_val-self.previousFidel_val -penalty#Reward given out at this timestep
            
            #Set current driving parameters mapped between -0.5 and 0.5
            self.OmegaList[self.game_step,:]=clipaction
            
            #Set actual driving parameters in real SI units
            self.actualParamList[self.game_step,:]=[actualOmega1,actualOmega2,tend-tstart]
            
            self.fidelList[self.game_step+1]=self.fidel_val
            
            self.game_step+=1
                                  
            #Do logging                
            if(self.game_step==self.totalsteps):
                self.done=True
                if(self.customLog==1):
                    self.logger.append([self.fidelList[-1],self.fidelList,self.OmegaList,self.penaltyList,self.tlist,[self.theta_val,self.phi_val,self.cos_theta_val],self.actualParamList])
        
        
        info=dict()
        

        return self.constructState(),reward,self.done,info
    