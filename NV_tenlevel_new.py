#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Tobias Haug (tobias.haug@u.nus.edu)
NV center level system

@author: Tobias Haug (tobias.haug@u.nus.edu), Mok Wai Keong

"""



from qutip import *
import numpy as np


class NV():
    
    def __init__(self,omega1,omega2,mode,B=0,eps_static=1):
        self.delta1 = omega1# - E_g #laser1 detuning
        self.delta2 = omega2# - E_g #laser2 detuning
        self.mode = mode #set system mode (0: full density matrix, 1: closed system)
        self.eps_static=eps_static
        self.B=B #Magnetic field in Tesla
        self.makeModel()
        
    def makeModel(self):
        #hbar is set to one, units in GHz
        #define system paramters
        D_gs = 2*np.pi * 2.88 #zero-field ground state splitting
        g_gs = 2.01 #lande g-factor of ground state
        mu_B = 2*np.pi * 14 #bohr magneton, in units of hbar and GHz
        #B = 0 #external magnetic field
        D_es = 2*np.pi * 1.42 #spin-spin interaction
        Delta = 2*np.pi * 1.55 #spin-spin interaction
        Delta_pp = 2*np.pi * 0.2 #spin-spin interaction
        lz = 2*np.pi * 5.3 #axial spin-orbit splitting
        g_es = 2.01 #lande g-factor of excited state
        #E_g = 2*np.pi * 4.71*10**5 #ZPL energy gap
        E_g = 0

        
        #define decay rates for open system
        gamma_34 = 1/24
        gamma_35 = 1/24
        gamma_38 = 1/24
        gamma_39 = 1/24
        gamma_14 = 1/31
        gamma_15 = 1/31
        gamma_18 = 1/31
        gamma_19 = 1/31
        gamma_24 = 1/104
        gamma_25 = 1/104
        gamma_28 = 1/104
        gamma_29 = 1/104
        gamma_M4 = 1/33
        gamma_M5 = 1/33
        gamma_M8 = 1/33
        gamma_M9 = 1/33
        gamma_26 = 1/13
        gamma_27 = 1/13
        gamma_36 = 1/666
        gamma_37 = 1/666
        gamma_16 = 1/666
        gamma_17 = 1/666
        gamma_M6 = 0
        gamma_M7 = 0
        gamma_2M = 1/303
        gamma_1M = 0
        gamma_3M = 0
        

        
        if self.mode == 0: #Full open quantum system
            #define states - NV ten levels
            s1 = basis(10,0) #ground state with m = -1
            s2 = basis(10,1) #ground state with m = 0
            s3 = basis(10,2) #ground state with m = 1
            s4 = basis(10,3) #excited state A2
            s5 = basis(10,4) #excited state A1
            s6 = basis(10,5) #excited state Ex
            s7 = basis(10,6) #excited state Ey
            s8 = basis(10,7) #excited state E1
            s9 = basis(10,8) #excited state E2
            s10 = basis(10,9) #metastable state
            

            
            #ground state hamiltonian
            H_gs = (D_gs - g_gs * mu_B * self.B) * s1 * s1.dag() + (D_gs + g_gs * mu_B * self.B) * s3 * s3.dag()
            
            #excited state hamiltonian (A2, A1)
            H_es1 = (Delta + 2*lz) * s4 * s4.dag() + (-Delta + 2*lz) * s5 * s5.dag() \
                  + (g_es * mu_B * self.B) * s4 * s5.dag() + (g_es * mu_B * self.B) * s5 * s4.dag()
            
            #excited state hamiltonian (Ex, Ey, E1, E2)
            H_es2 = (-D_es + lz) * s6 * s6.dag() + (-D_es + lz) * s7 * s7.dag() \
                  + Delta_pp * s6 * s9.dag() + Delta_pp * s9 * s6.dag() \
                  + 1j * Delta_pp * s7 * s8.dag() - 1j * Delta_pp * s8 * s7.dag() \
                  - (g_es * mu_B * self.B) * s8 * s9.dag() - (g_es * mu_B * self.B) * s9 * s8.dag()
                  
            #excited state hamiltonian (gs-es energy gap (ZPL))
            H_eg = E_g * ( s4 * s4.dag() + s5 * s5.dag() + s6 * s6.dag() + s7 * s7.dag() + s8 * s8.dag() + s9 * s9.dag() )
    
            V = 1j * s1 * s4.dag() - 1j * s4 * s1.dag() - 1j * s1 * s5.dag() + 1j * s5 * s1.dag() \
                    -1j * s1 * s8.dag() + 1j * s8 * s1.dag() - 1j * s1 * s9.dag() + 1j * s9 * s1.dag() \
                    + 2 * s2 * s7.dag() + 2 * s7 * s2.dag() - 1j * s3 * s4.dag() + 1j * s4 * s3.dag() \
                    -1j * s3 * s5.dag() + 1j * s5 * s3.dag() + 1j * s3 * s8.dag() - 1j * s8 * s3.dag() \
                    -1j * s3 * s9.dag() + 1j * s9 * s3.dag()

            #collapse operators
            c_ops = []
            c_ops.append(np.sqrt(gamma_34) * s3 * s4.dag())
            c_ops.append(np.sqrt(gamma_35) * s3 * s5.dag())
            c_ops.append(np.sqrt(gamma_38) * s3 * s8.dag())
            c_ops.append(np.sqrt(gamma_39) * s3 * s9.dag())
            c_ops.append(np.sqrt(gamma_14) * s1 * s4.dag())
            c_ops.append(np.sqrt(gamma_15) * s1 * s5.dag())
            c_ops.append(np.sqrt(gamma_18) * s1 * s8.dag())
            c_ops.append(np.sqrt(gamma_19) * s1 * s9.dag())
            c_ops.append(np.sqrt(gamma_24) * s2 * s4.dag())
            c_ops.append(np.sqrt(gamma_25) * s2 * s5.dag())
            c_ops.append(np.sqrt(gamma_28) * s2 * s8.dag())
            c_ops.append(np.sqrt(gamma_29) * s2 * s9.dag())
            c_ops.append(np.sqrt(gamma_M4) * s10 * s4.dag())
            c_ops.append(np.sqrt(gamma_M5) * s10 * s5.dag())
            c_ops.append(np.sqrt(gamma_M8) * s10 * s8.dag())
            c_ops.append(np.sqrt(gamma_M9) * s10 * s9.dag())
            c_ops.append(np.sqrt(gamma_26) * s2 * s6.dag())
            c_ops.append(np.sqrt(gamma_27) * s2 * s7.dag())
            c_ops.append(np.sqrt(gamma_36) * s3 * s6.dag())
            c_ops.append(np.sqrt(gamma_37) * s3 * s7.dag())
            c_ops.append(np.sqrt(gamma_16) * s1 * s6.dag())
            c_ops.append(np.sqrt(gamma_17) * s1 * s7.dag())
            c_ops.append(np.sqrt(gamma_M6) * s10 * s6.dag())
            c_ops.append(np.sqrt(gamma_M7) * s10 * s7.dag())
            c_ops.append(np.sqrt(gamma_2M) * s2 * s10.dag())
            c_ops.append(np.sqrt(gamma_1M) * s1 * s10.dag())
            c_ops.append(np.sqrt(gamma_3M) * s3 * s10.dag())

        elif self.mode == 1: #effective 8-level model (two gs, four es), closed system, valid on short times
            #define states - NV ten levels
            s1 = basis(8,0) #ground state with m = -1
            s3 = basis(8,1) #ground state with m = 1
            s4 = basis(8,2) #excited state A2
            s5 = basis(8,3) #excited state A1
            s6 = basis(8,4) #excited state Ex
            s7 = basis(8,5) #excited state Ey
            s8 = basis(8,6) #excited state E1
            s9 = basis(8,7) #excited state E2
            
            
            #ground state hamiltonian
            H_gs = (D_gs - g_gs * mu_B * self.B) * s1 * s1.dag() + (D_gs + g_gs * mu_B * self.B) * s3 * s3.dag()
            
            #excited state hamiltonian (A2, A1)
            H_es1 = (Delta + 2*lz) * s4 * s4.dag() + (-Delta + 2*lz) * s5 * s5.dag() \
                  + (g_es * mu_B * self.B) * s4 * s5.dag() + (g_es * mu_B * self.B) * s5 * s4.dag()
            
            #excited state hamiltonian (Ex, Ey, E1, E2)
            H_es2 = (-D_es + lz) * s6 * s6.dag() + (-D_es + lz) * s7 * s7.dag() \
                  + Delta_pp * s6 * s9.dag() + Delta_pp * s9 * s6.dag() \
                  + 1j * Delta_pp * s7 * s8.dag() - 1j * Delta_pp * s8 * s7.dag() \
                  - (g_es * mu_B * self.B) * s8 * s9.dag() - (g_es * mu_B * self.B) * s9 * s8.dag()
                  
            #excited state hamiltonian (gs-es energy gap (ZPL))
            H_eg = E_g * ( s4 * s4.dag() + s5 * s5.dag() + s6 * s6.dag() + s7 * s7.dag() + s8 * s8.dag() + s9 * s9.dag() )
    
            V = 1j * s1 * s4.dag() - 1j * s4 * s1.dag() - 1j * s1 * s5.dag() + 1j * s5 * s1.dag() \
                    -1j * s1 * s8.dag() + 1j * s8 * s1.dag() - 1j * s1 * s9.dag() + 1j * s9 * s1.dag() \
                    - 1j * s3 * s4.dag() + 1j * s4 * s3.dag() \
                    -1j * s3 * s5.dag() + 1j * s5 * s3.dag() + 1j * s3 * s8.dag() - 1j * s8 * s3.dag() \
                    -1j * s3 * s9.dag() + 1j * s9 * s3.dag()
        

        elif self.mode == 2: #three-level Lambda model
            #define states
            s1 = basis(3,0) #ground state with m = -1
            s3 = basis(3,1) #ground state with m = 1
            s4 = basis(3,2) #excited state A2

            #ground state hamiltonian
            H_gs = (D_gs - g_gs * mu_B * self.B) * s1 * s1.dag() + (D_gs + g_gs * mu_B * self.B) * s3 * s3.dag()
            
            #excited state hamiltonian (A2)
            H_es1 = (Delta + 2*lz) * s4 * s4.dag()
            
            #excited state hamiltonian (Ex, Ey, E1, E2)
            H_es2 =  qeye(s1.shape[0])*0
            
            #excited state hamiltonian (gs-es energy gap (ZPL))
            H_eg = E_g * s4 * s4.dag() 
    
            V = 1j * s1 * s4.dag() - 1j * s4 * s1.dag() - 1j * s3 * s4.dag() + 1j * s4 * s3.dag() 
 
        elif self.mode == 3 or self.mode==4: #two-level model
            #define states
            s1 = basis(3,0) #ground state with m = -1
            s3 = basis(3,1) #ground state with m = 1

            #ground state hamiltonian
            H_gs =  - self.B* s1 * s1.dag() +self.B* s3 * s3.dag()
            
            #excited state hamiltonian (A2)
            H_es1 = qeye(s1.shape[0])*0
            
            #excited state hamiltonian (Ex, Ey, E1, E2)
            H_es2 =  qeye(s1.shape[0])*0
            
            #excited state hamiltonian (gs-es energy gap (ZPL))
            H_eg = qeye(s1.shape[0])*0
    
            V =  -1j*s1 * s3.dag()  +1j* s3 * s1.dag()
            V2 =  2*(- s1*s1.dag() +s3 * s3.dag())
        elif(self.mode==5): #two-level model
            #define states
            s1 = basis(3,0) #ground state with m = -1
            s3 = basis(3,1) #ground state with m = 1

            #ground state hamiltonian
            H_gs =  - self.B* s1 * s1.dag() +self.B* s3 * s3.dag()
            
            #excited state hamiltonian (A2)
            H_es1 = qeye(s1.shape[0])*0
            
            #excited state hamiltonian (Ex, Ey, E1, E2)
            H_es2 =  qeye(s1.shape[0])*0
            
            #excited state hamiltonian (gs-es energy gap (ZPL))
            H_eg = qeye(s1.shape[0])*0
    
            V =  s1 * s3.dag()  +s3 * s1.dag()
            V2 =  (-1j*s1 * s3.dag()  +1j* s3 * s1.dag())
    
    
        self.state1=s1 #relevant two level subspace
        self.state3=s3 #relevant two level subspace
            
    
        #define initial state
        self.psi0 = s1 #make density matrix of initial state
        self.psi1 = s3 #make density matrix of initial state
        if self.mode == 0: #full dm
            self.rho0 = ket2dm(self.psi0) #Density matrix
            self.rho1 = ket2dm(self.psi1)
        
        
        self.hilbertspace=s1.shape[0]
        self.LVspace=self.hilbertspace**2
        
        
        #driving field hamiltonian
        def epsilon_x(t,args):
            Omega1 = args["Omega1"]
            Omega2 = args["Omega2"]
            return (Omega1 * np.cos(self.delta1*t) + Omega2 * np.cos(self.delta2*t))   
        
        def epsilon_1(t,args):
            Omega1 = args["Omega1"]
            return Omega1 * np.cos(self.delta1*t)
        
        def epsilon_2(t,args):
            Omega2 = args["Omega2"]
            return Omega2 * np.cos(self.delta2*t)
        
        
    
         #Get hamiltonian
        if(self.mode==3 or self.mode==4 or self.mode==5):
            H_drive = [V,epsilon_1]
            H_drive2 = [V2,epsilon_2]
            self.H_total = [H_gs,H_es1,H_es2,H_eg,H_drive,H_drive2] #Final Hamiltonian
            self.H_static = H_gs+H_es1+H_es2+H_eg+V*self.eps_static+V2*self.eps_static #Just used for debugging

        else:
            H_drive = [V,epsilon_x]
            #total hamiltonian
            self.H_total = [H_gs,H_es1,H_es2,H_eg,H_drive] #Final Hamiltonian
            self.H_static = H_gs+H_es1+H_es2+H_eg+V*self.eps_static #Just used for debugging
        

#        if(self.delta1!=0 or self.delta2!=0):
#            self.H_total = [H_gs,H_es1,H_es2,H_eg,H_drive]
#        else:
#            self.H_static = [H_gs,H_es1,H_es2,H_eg]
        
        
        #Define lindblad operators
        self.collapse = []
        if self.mode == 0:
            self.collapse = c_ops


    #Run quantum system under driving Omega1, Omega2, for time tstart to tend
    def run(self,rho,tstart,tend,Omega1,Omega2):   
        #solve master equation
        args={"Omega1": Omega1, "Omega2": Omega2}
        options = Options()
        options.nsteps = 10000
        tlist=np.linspace(tstart,tend,num=int(np.ceil((tend-tstart)))+1)
        output = mesolve(self.H_total,rho,tlist,self.collapse,[],args=args,options=options)
#        if(self.delta1!=0 or self.delta2!=0):
#            output = mesolve(self.H_total,rho,tlist,self.collapse,[],args=args,options=options)
#        else:
#            Hlocal=self.H_static+[self.drivingOp*(Omega1+Omega2)]
#            output = mesolve(Hlocal,rho,tlist,self.collapse,[],options=options)
        return output.states[-1] #Return evolved wavefunction

    def getInitialstate(self,theta=0,phi=0): #Get initial state to start with
    
        if self.mode == 0: #full dm
            return Qobj(np.cos(np.pi*theta/2) *self.rho0+ np.sin(np.pi*theta/2) * np.exp(1j*2*np.pi*phi)*self.rho1)
        elif (self.mode == 1 or self.mode == 2 or self.mode == 3 or self.mode==4 or self.mode==5): #state vector
            return Qobj(np.cos(np.pi*theta/2) *self.psi0+ np.sin(np.pi*theta/2) * np.exp(1j*2*np.pi*phi)*self.psi1)


    def calcFidelity(self,rho,theta,phi):#Calculate fidelity with target state
        target = np.cos(np.pi*theta/2) * self.state1 + np.sin(np.pi*theta/2) * np.exp(1j*2*np.pi*phi) *self.state3 #define target state

        fidel=fidelity(rho,target)**2 #calculate fidelity
        #print(fidel)
        return(fidel)    
       