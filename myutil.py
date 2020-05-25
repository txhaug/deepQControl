# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:52:38 2017

@author: Tobias Haug

Utility functions used for plotting etc.
"""





import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from matplotlib import cm
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import MaxNLocator





rcParams.update({'figure.autolayout': True})
rcParams['legend.handlelength'] = 1
rcParams['legend.labelspacing'] = 0.2
rcParams['legend.handletextpad']=0.2
rcParams['legend.borderpad']=0.15
rcParams['legend.borderaxespad']=0.15  


colormap=np.array([(56,108,176),(251,128,114),(51,160,44),(253,191,111),(227,26,28),(178,223,138),(166,206,227),(255,127,0),(202,178,214),(106,61,154)])/np.array([255.,255.,255.]) #self constructed from color brewer


figurenumber=1

#fsize=24
fsize=18
#rc('font',**{'family':'serif','serif':['Times']})
plt.rcParams['mathtext.fontset'] = 'cm'
#rc('text', usetex=True)
fsizeLabel=fsize+14
fsizeTitle=fsize
fsizeLegend=fsize+3
lwglobal=3
tickerWidth=1.2
minorLength=4
majorLength=8
dashdef=[3,2]
figsize2D=(6, 5)
        
xnbinsParam=3
ynbinsParam=4
cbarnbinsParam=2

floatformat='02.2f'




def execfile(filepath, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)

#Construct arange when on a ring. Order is left to right
def foldRange(start,end,length):
    startFold=np.mod(start,length)
    endFold=np.mod(end,length)
    if(startFold<endFold and end <length):
        return np.arange(startFold,endFold)
    else:
        return np.concatenate((np.arange(startFold,length),np.arange(0,endFold)))

def powerSignE(value,power):
    return np.sign(value)*np.power(np.abs(value),power)

powerSign=np.vectorize(powerSignE)

#map angle to space 0 ...2, input divided by pi already
def mapAngleE(phase):
    if(phase<0.):
        return phase+2
    elif(phase>=2.):
        return phase-2
    else:
        return phase

mapAngle=np.vectorize(mapAngleE)


def between(value, a, b,arfind=False,brfind=False):
    # Find and validate before-part.
    if(arfind==False):
        pos_a = value.find(a)
    else:
        pos_a = value.rfind(a)
    if pos_a == -1: return ""
    # Find and validate after part.
    adjusted_pos_a = pos_a + len(a)
    if(brfind==False):
        pos_b = value[adjusted_pos_a:].find(b)
    else:
        pos_b = value[adjusted_pos_a:].rfind(b)
    if pos_b == -1: return ""
    # Return middle part.

    if adjusted_pos_a >= pos_b+adjusted_pos_a: return ""
    return value[adjusted_pos_a:pos_b+adjusted_pos_a]


def chopE(value):
    epsilon=10**(-15)
    if(np.abs(value)<epsilon):
        return value*0.0
    else:
        return value
    
chop=np.vectorize(chopE)

def maptooneE(x):
    if(x==0):
        return 1
    else:
        return 0
    
maptoone=np.vectorize(maptooneE)


def getNearestIndex(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def createSublist(l,reorder=False,multiply=[]):
    ndataset=len(l)
    if(len(multiply)==0):
        multiply=np.array([1 for i in range(ndataset)])
    if(reorder==False):
        return [item for sublist in [[multiply[k]*l[k][j] for k in range(ndataset)] for j in range(len(l[0]))] for item in sublist]
    else:
        return [item for sublist in [[multiply[k]*l[k][j] for j in range(len(l[0]))] for k in range(ndataset)] for item in sublist]
    
    
def flatten(l):
    return [item for sublist in l for item in sublist]


def moving_average1D(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#Calculates moving average in 2D, special defines a special neighbor to average over
def moving_average(a, n=2,axis=0,special=[]):
    x=np.array(a)
    specialArr=np.array(special)
    for i in range(0,np.shape(a)[axis]):
        nextsites=np.arange(i,i+n)
        if(len(specialArr)>0 and n==2):
            result=np.where(specialArr[:,0]==i)[0]
            if(len(result)>0):
                nextsites=[i,specialArr[result[0],1]]
                
        if(nextsites[1]<np.shape(a)[axis]):
            if(axis==0):
                x[i,:]=np.mean(a[nextsites,:],axis=0)
            elif(axis==1):
                x[:,i]=np.mean(a[:,nextsites],axis=1)
    
    return x


#Calculates moving difference, special defines a special neighbor to average over
def moving_difference(a, n=2,axis=0,special=[]):
    x=np.array(a)
    specialArr=np.array(special)
    for i in range(0,np.shape(a)[axis]-n+1):
        nextsites=np.arange(i,i+n)
        if(len(specialArr)>0):
            result=np.where(specialArr[:,0]==i)[0]
            if(len(result)>0):
                nextsites=[i,specialArr[result[0],1]+n-2]
            if(n>=3):
                result=np.where(specialArr[:,0]==i+1)[0]
                if(len(result)>0):
                    nextsites=[i,specialArr[result[0],1]+n-3]
            if(n>=4):
                result=np.where(specialArr[:,0]==i+2)[0]
                if(len(result)>0):
                    nextsites=[i,specialArr[result[0],1]+n-4]
        if(axis==0):
            x[i,:]=a[nextsites[-1],:]-a[nextsites[0],:]
        elif(axis==1):
            x[:,i]=a[:,nextsites[-1]]-a[:,nextsites[0]]
    
    return x

#plotting program for 1D
def plot1D(data,x,title,xlabelstring,ylabelstring,saveto,dataset,dataname,anchor=(1.1,1.0),scientfic=False,plotratio=1.3,fsizeLegend=fsizeLegend,multicolor=[],clim=None,hist=False,histrange=None,bins="auto",scatter=False,dashdef=None,linewidth=None,multix=False,markerStyle=None,colorMarker=None,markevery=1,markersize=10,elements=1,marker=0,markerx="",markery="",error=False,errordata=[],errorsublist=[],errorcapsize=3,label="",outsidelabel=False,legendloc="best",ymin="",ymax="",xmin="",xmax="",logx=False,logy=False,dpi=300,xnbins=xnbinsParam,ynbins=ynbinsParam,plot1DLinestyle=None,colormap=colormap,cmap="brg",overrideColor="",customTicky=[],customMinorTicky=[],saveformat="pdf",close=False,extralines=[]):
    global figurenumber
    global fsize
    a=5

    plt.figure(figurenumber,(a*plotratio,a ))
    if(saveformat=="pdf"):
        pp = PdfPages(saveto+dataname+"1D"+dataset+'.pdf')

    ax = plt.gca()
    
    if(scientfic==True):
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    plt.ticklabel_format(useOffset=False)
    if(plot1DLinestyle==None):
        plot1DLinestyle=["solid" for i in range(0,elements)]
    if(markerStyle==None):
        if(scatter==True):
            markerStyle=["." for i in range(0,elements)]
        else:
            markerStyle=["" for i in range(0,elements)]
    linewidthMod=np.ones(elements)
    if(dashdef==None):
        dashdef=[]
        for i in range(0,elements):
            if(plot1DLinestyle[i]=="solid"):
                dashdef.append([1,1])
                linewidthMod[i]=1
            elif(plot1DLinestyle[i]=="dotted"):
                dashdef.append([0.2,1.7])
                linewidthMod[i]=1.7
            elif(plot1DLinestyle[i]=="dashed"):
                dashdef.append([3,2])
                linewidthMod[i]=1
            elif(plot1DLinestyle[i]=="dashdot"):
                dashdef.append([5,2.5,1.5,2.5])
                linewidthMod[i]=1
            else:
                dashdef.append([1,1])
                linewidthMod[i]=2
    if(np.all(linewidth==None)):
        linewidth=[lwglobal*linewidthMod[i] for i in range(0,elements)]
        
    if(error==True):
        if(len(errorsublist)==0):
            errorsublist=[True for i in range(elements)]
    else:
        errorsublist=[False for i in range(elements)]
        
    if(hist==False):
        if(elements==1):
            if(multix==True):
                x=x[0]
                data=data[0]
            if(scatter==True):
                if(len(multicolor)==0):
                    l=plt.scatter(x,data,color="black",s=markersize*1)
                else:
                    l=plt.scatter(x,data,cmap=cmap,c=multicolor,s=markersize)
                #plt.plot(x, data, markerStyle[0], ms=markersize, color='black') 
            else:
                if(overrideColor==""):
                    color=colormap[0]
                else:
                    color=colormap[np.mod(overrideColor[0],len(colormap))]
                if(error==False):
                    if(len(data)==len(x)):
                        l,=plt.plot(x,data,color=color,linewidth=linewidth[0],linestyle=plot1DLinestyle[0],marker=markerStyle[0],markevery=markevery,ms=markersize,dash_capstyle = "round")  
                    else:
                        l,=plt.plot(x,data[0],color=color,linewidth=linewidth[0],linestyle=plot1DLinestyle[0],marker=markerStyle[0],markevery=markevery,ms=markersize,dash_capstyle = "round")
                else:
                    if(len(data)==len(x)):
                        l=plt.errorbar(x,data,yerr=errordata,color=color,linewidth=linewidth[0],linestyle=plot1DLinestyle[0],marker=markerStyle[0],markevery=markevery,ms=markersize,dash_capstyle = "round", capsize=errorcapsize)  
                        l=l.lines[0]
                    else:
                        l=plt.errorbar(x,data[0],yerr=errordata[0],color=color,linewidth=linewidth[0],linestyle=plot1DLinestyle[0],marker=markerStyle[0],markevery=markevery,ms=markersize,dash_capstyle = "round", capsize=errorcapsize)
                        l=l.lines[0]
                
                if(plot1DLinestyle[i]=="dashed" or plot1DLinestyle[i]=="dotted" or plot1DLinestyle[i]=="dashdot"):
                    l.set_dashes(dashdef[0])
        else:
            if(scatter==True):
                for i in range(0,elements):
                    if(multix==False):
                        plt.scatter(x, data[i], marker=markerStyle[i],s=markersize, color=colormap[np.mod(i,len(colormap))]) 
                    else:
                        if(len(multicolor)==0):
                            plt.plot(x[i], data[i], markerStyle[0], ms=markersize, color=colormap[np.mod(i,len(colormap))])
                        else:
                            if(clim==None):
                                vmax=np.amax(multicolor)
                                vmin=np.amin(multicolor)
                            else:
                                vmin=clim[0]
                                vmax=clim[1]
                            #plt.plot(x[i], data[i], markerStyle[0],c=multicolor[i],cmap="cubehelix_r", ms=markersize)
                            plt.scatter(x[i],data[i],cmap=cmap,c=multicolor[i],vmin=vmin,vmax=vmax,marker=markerStyle[0],s=markersize)
                            
            else:
                for i in range(0,elements):
                    if(overrideColor==""):
                        color=colormap[np.mod(i,len(colormap))]
                    else:
                        color=colormap[np.mod(overrideColor[i],len(colormap))]
                        
                    
                    if(multix==False):
                        if(errorsublist[i]==False):
                            l,=plt.plot(x,data[i],color=color,linewidth=linewidth[i],linestyle=plot1DLinestyle[i],marker=markerStyle[i],markevery=markevery,ms=markersize,dash_capstyle = "round")
                        else:
                            l=plt.errorbar(x,data[i],yerr=errordata[i],color=color,linewidth=linewidth[i],linestyle=plot1DLinestyle[i],marker=markerStyle[i],markevery=markevery,ms=markersize,dash_capstyle = "round", capsize=errorcapsize)
                            l=l.lines[0]
                    else:
                        #print(i,len(x),len(data),len(linewidth),len(plot1DLinestyle),len(markerStyle))
                        if(errorsublist[i]==False):
                            l,=plt.plot(x[i],data[i],color=color,linewidth=linewidth[i],linestyle=plot1DLinestyle[i],marker=markerStyle[i],markevery=markevery,ms=markersize,dash_capstyle = "round")
                        else:
                            l=plt.plot(x[i],data[i],yerr=errordata[i],color=color,linewidth=linewidth[i],linestyle=plot1DLinestyle[i],marker=markerStyle[i],markevery=markevery,ms=markersize,dash_capstyle = "round", capsize=errorcapsize)
                            l=l.lines[0]
                    if(plot1DLinestyle[i]=="dashed" or plot1DLinestyle[i]=="dotted" or plot1DLinestyle[i]=="dashdot"):
                        l.set_dashes(dashdef[i])
                
                
        for i in range(0,marker):
            if(colorMarker!=None):
                color=colorMarker
            elif(overrideColor==""):
                color=colormap[np.mod(i,len(colormap))]
            else:
                color=colormap[np.mod(overrideColor[i],len(colormap))]
            if(multix==False):
                plt.scatter(markerx,markery[i],color=color,s=markersize)
            else:
                plt.scatter(markerx[i],markery[i],color=color,marker=markerStyle[0],s=markersize)
    else:
        if(elements==1):
            plt.hist(data,bins=bins,range=histrange)
        else:
            #for i in range(0,elements):
            #    plt.hist(data[i],bins=bins,range=histrange, color=colormap[np.mod(i,len(colormap))], alpha=0.7)
            plt.hist(data,bins=bins,range=histrange, alpha=0.7)
             

    if(scatter==True or hist==True):
        if(len(extralines)>0):
            for i in range(len(extralines)):
                extralinestyle=extralines[i][3]
                if(extralinestyle=="solid"):
                    extradashdef=[1,1]
                    extralinewdith=lwglobal/2
                elif(extralinestyle=="dotted"):
                    extradashdef=[0.2,1.7]
                    extralinewdith=1.7*lwglobal
                elif(extralinestyle=="dashed"):
                    extradashdef=[3,2]
                    extralinewdith=lwglobal
                elif(extralinestyle=="dashdot"):
                    extradashdef=[5,2.5,1.5,2.5]
                    extralinewdith=lwglobal
                else:
                    extradashdef=[1,1]
                    extralinewdith=lwglobal*2
                    
                    
                l,=plt.plot(extralines[i][0],extralines[i][1],color=extralines[i][2],linewidth=extralinewdith,linestyle=extralinestyle,dash_capstyle = "round")  #
                if(extralinestyle!="solid"):
                    l.set_dashes(extradashdef)


    if(logx==True):
        ax.set_xscale('log')
    if(logy==True):
        ax.set_yscale('log')
    if(xnbins!=None and logx==False):
        plt.locator_params(axis = 'x',nbins=xnbins)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if(ynbins!=None and logy==False):
        plt.locator_params(axis = 'y',nbins=ynbins)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
    plt.tick_params(axis ='both',which='both', width=tickerWidth)
    plt.tick_params(axis ='both',which='minor', length=minorLength)
    plt.tick_params(axis ='both', which='major', length=majorLength)
    if(len(customTicky)>0):
        #ax.yaxis.set_ticks(customTicky)
        plt.yticks(customTicky[0], customTicky[1])
    if(len(customMinorTicky)>0):
        ax.yaxis.set_minor_locator(FixedLocator(customMinorTicky))
    
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
                
    plt.xlabel(xlabelstring)
    plt.ylabel(ylabelstring)
    plt.title(title)  
    if(label!=""):
        if(outsidelabel==False):
            lgd=plt.legend(label,loc=legendloc, fontsize=fsizeLegend)
        else:
            lgd=plt.legend(label, fontsize=fsizeLegend,loc=legendloc, bbox_to_anchor=anchor)
    
        
        if(scatter==True):
            for i in range(len(label)):
                lgd.legendHandles[i]._sizes = [50]
                


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(fsizeLabel)
    for item in ([]+ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)
    if(ymin!="" and ymax==""):
        ax.set_ylim([ymin,np.amax(data)*1.1])
    elif(ymin=="" and ymax!=""):
        ax.set_ylim([np.amin(data)*0.9,ymax])
    elif(ymin!="" and ymax!=""):
        ax.set_ylim([ymin,ymax])
        
    if(xmin!="" and xmax==""):
        ax.set_xlim([xmin,np.amax(x)])
    elif(xmin=="" and xmax!=""):
        ax.set_xlim([np.amin(x),xmax])
    elif(xmin!="" and xmax!=""):
        ax.set_xlim([xmin,xmax])
        
    ax.set_aspect(1.0/ax.get_data_ratio()*1/plotratio)
    
    if(label!=""):
        if(saveformat=="pdf"):
            pp.savefig(plt.gcf() ,bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig(saveto+dataname+"1D"+dataset+"."+saveformat ,bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=dpi)
    else:
        if(saveformat=="pdf"):
            pp.savefig(plt.gcf() )
        else:
            plt.savefig(saveto+dataname+"1D"+dataset+"."+saveformat,dpi=dpi)
    if(saveformat=="pdf"):
        pp.close()
    figurenumber=figurenumber+1
    if(close==True):
        plt.close()


#shading="flat"
#Regrid deals with irregular spaced axis, by making a new grid with up to regrid entries. May slow down plot generation
def plot2D(griddata,gridx,gridy,title,xlabelstring,ylabelstring,saveto,dataset,dataname,customTicky=[],customTickx=[],customMinorTicky=[],customMinorTickx=[],regrid=500,hideaxis=False,figsize2D=figsize2D,extfig=None,savetype=".pdf",plot3D=False,zlabelstring="",fixaspect=False,camera=[45,30],zMin=None,zMax=None,cmap="RdYlBu_r",makecbar=True,customCbarTick=None,fsize=fsize,zlim="",logx=False,logy=False,Nplot1D=0,plotx1D=[],ploty1D=[],plot1DLinestyle=None,plot1DColor=None,xnbins=xnbinsParam,ynbins=ynbinsParam,cbarnbins=cbarnbinsParam,linewidth=lwglobal,shading='gouraud',fsizeTitle=fsizeTitle,inverty=False): #"YlOrBr" "cubehelix_r" "RdYlBu_r"
    global figurenumber
    #global fsize
    if(extfig==None):
        fig=plt.figure(figurenumber,figsize=figsize2D)
    else:
        fig=extfig
    if(savetype==".pdf"):
        pp = PdfPages(saveto+dataname+dataset+savetype)




    data=np.array(griddata)
    x=np.array(gridx)
    
    gridyIn=np.array(gridy)
    #if(inverty==True):
    #    gridyIn=np.flipud(gridyIn)
    y=np.array(gridyIn)

    doregrid=False
    if(regrid!=0):
        #Check if axis is irregular
        epsilon=10**(-9)
        regularx=np.linspace(gridx[0],gridx[-1],num=len(gridx))
        regulary=np.linspace(gridyIn[0],gridyIn[-1],num=len(gridyIn))
        if(np.sum(np.abs(regularx-gridx))>epsilon):
            x=np.linspace(gridx[0],gridx[-1],num=min(regrid,2*int(np.round((x[-1]-x[0])/np.amin(x[1:]-x[:-1])+1))))
            doregrid=True

        if(np.sum(np.abs(regulary-gridyIn))>epsilon):
            y=np.linspace(gridyIn[0],gridyIn[-1],num=min(regrid,2*int(np.round((y[-1]-y[0])/np.amin(y[1:]-y[:-1])+1))))
            doregrid=True
        if(doregrid==True):
            print('WARN: Doing regrid for',dataname,'with new length',len(x),len(y),',old length',len(gridx),len(gridyIn), 'as axis are irregular')
            newdata=np.zeros([len(y),len(x)],dtype=type(data[0][0]))
            for i in range(len(y)):
                supporty=getNearestIndex(gridyIn,y[i])
                for j in range(len(x)):
                    supportx=getNearestIndex(gridx,x[j])
                    newdata[i,j]=data[supporty,supportx]
            data=newdata
        
    if(zMin==None):
        minZ=np.amin(data)
    else:
        minZ=zMin
    
    if(zlim!=""):
        zMin=zlim
        
    if(zMax==None):
        maxZ=np.amax(data)
    else:
        maxZ=zMax
        

    if(plot3D==False):
        ax = plt.gca()
        deltax=(x[-1]-x[0])/len(x)
        deltay=(y[-1]-y[0])/len(y)
        if(shading=="flat"):
            addnum=1
        else:
            addnum=0
        plotx=np.linspace(x[0]-deltax/2,x[-1]+deltax/2,num=len(x)+addnum)
        ploty=np.linspace(y[0]-deltay/2,y[-1]+deltay/2,num=len(y)+addnum)
        #plotx=np.append(x,[x[-1]+deltax])-deltax/2
        #ploty=np.append(y,[y[-1]+deltay])-deltay/2
        
        plt.pcolormesh(plotx, ploty, data, vmin=minZ, vmax=maxZ,cmap=cmap,linewidth=0,rasterized=True,shading=shading,antialiased=False)
        #plt.axis('equal')
        ax.set_aspect(abs((x[-1]-x[0])/(y[-1]-y[0])))
        plt.axis([ plotx.min(), plotx.max(),ploty.min(), ploty.max()])
        if(makecbar==True and hideaxis==False):
            cbar=plt.colorbar(fraction=0.046, pad=0.04)
            cbar.locator = MaxNLocator( nbins = cbarnbins)
            cbar.ax.tick_params(labelsize=fsize)
            #cbar.ax.minorticks_on()
            cbar.ax.tick_params(which='minor', length=minorLength)
            cbar.ax.tick_params(which='both', width=tickerWidth)
            cbar.ax.tick_params(which='major', length=majorLength)
            cbar.ax.get_yaxis().set_minor_locator(AutoMinorLocator(2))
            if(customCbarTick!=None):
                cbar.set_ticks(customCbarTick)
            cbar.update_ticks()

            
        #plt.xticks(np.arange(int(min(x)), int(max(x))+1, 1.0))
        #plt.yticks(np.arange(int(min(y)), int(max(y))+1, 1.0))
        plt.locator_params(axis = 'x',nbins=xnbins)
        plt.locator_params(axis = 'y',nbins=ynbins)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tick_params(axis ='both',which='both', width=tickerWidth)
        plt.tick_params(axis ='both',which='minor', length=minorLength)
        plt.tick_params(axis ='both', which='major', length=majorLength)
        if(len(customTicky)>0):
            #ax.yaxis.set_ticks(customTicky)
            plt.yticks(customTicky[0], customTicky[1])
        if(len(customMinorTicky)>0):
            ax.yaxis.set_minor_locator(FixedLocator(customMinorTicky))
        if(len(customTickx)>0):
            #ax.yaxis.set_ticks(customTicky)
            plt.xticks(customTickx[0], customTickx[1])
        if(len(customMinorTickx)>0):
            ax.xaxis.set_minor_locator(FixedLocator(customMinorTickx))
        
    else:  
        meshx, meshy = np.meshgrid(x, y)
        #ax=fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')
        #ax.set_aspect('equal')
        #print(data)
        ax.plot_surface(meshx, meshy, data, rstride=1, cstride=1, cmap=cmap, linewidth=0)#cm.coolwarm

        ax.set_zlim(minZ, maxZ)
        ax.axis([ x.min(), x.max(),y.min(), y.max()])
        ax.set_zlabel(zlabelstring)
        ax.zaxis.label.set_fontsize(fsize)
        for item in ax.get_zticklabels():
            item.set_fontsize(fsize)
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        #ax.view_init(elev=camera[0], azim=camera[1]) #Camera angle in degree
        #ax.dist = 11
        plt.locator_params(axis = 'x',nbins=xnbins)
        plt.locator_params(axis = 'y',nbins=ynbins)
        plt.locator_params(axis = 'z',nbins=ynbins)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.zaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tick_params(axis ='both',which='both', width=tickerWidth)
        plt.tick_params(axis ='both',which='minor', length=minorLength)
        plt.tick_params(axis ='both', which='major', length=majorLength)
    
    if(Nplot1D>0 and plotx1D!=None):
        
        if(plot1DLinestyle==None):
            plot1DLinestyle=["dashed" for i in range(0,Nplot1D)]
            
        linewidthMod=np.ones(Nplot1D)
        dashdef=[]
        for i in range(0,Nplot1D):
            if(plot1DLinestyle[i]=="solid"):
                dashdef.append([1,1])
                linewidthMod[i]=1
            elif(plot1DLinestyle[i]=="dotted"):
                dashdef.append([0.2,1.7])
                linewidthMod[i]=1.7
            elif(plot1DLinestyle[i]=="dashed"):
                dashdef.append([3,2])
                linewidthMod[i]=1
            else:
                dashdef.append([1,1])
                linewidthMod[i]=1

        linewidthList=[linewidth*linewidthMod[i] for i in range(0,Nplot1D)]
        
        
        

    
        if(len(ploty1D)==0):
            ploty1D=[ploty for i in range(0,Nplot1D)]


        if(plot1DColor==None):
            plot1DColor=["k" for i in range(0,Nplot1D)]
        for i in range(0,Nplot1D):
            if(plot1DLinestyle[i]=="dashed"):
                l,=plt.plot(plotx1D[i],ploty1D[i],color=plot1DColor[i],linewidth=linewidthList[i],linestyle=plot1DLinestyle[i],dash_capstyle="round")
            else:
                l,=plt.plot(plotx1D[i],ploty1D[i],color=plot1DColor[i],linewidth=linewidthList[i],linestyle=plot1DLinestyle[i],dash_capstyle="round")
            if(plot1DLinestyle[i]=="dashed" or plot1DLinestyle[i]=="dotted"):
                l.set_dashes(dashdef[0])
    
    
    
    if(logx==True):
        #ax.set_xscale('log')
        plt.xscale("log")
    if(logy==True):
        #ax.set_yscale('log')
        plt.yscale("log")

    plt.xlabel(xlabelstring)
    plt.ylabel(ylabelstring)
            
    plt.title(title)      
    for item in ([ax.title]):
        item.set_fontsize(fsizeTitle)
    for item in ([ ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(fsizeLabel)
    for item in ([]+ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)
    
    if(hideaxis==True):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        
    if(inverty==True):
        plt.gca().invert_yaxis()
        #ylabels = ax.get_yticks().tolist()#[item.get_text() for item in ax.get_yticklabels()]
        #if(np.sum([np.abs(ylabels[i]-int(np.round(ylabels[i]))) for i in range(len(ylabels))])<10**-8):
        #    ylabels=[int(np.round(ylabels[i])) for i in range(len(ylabels))]
        #ax.set_yticklabels([ylabels[-i-1+len(ylabels)] for i in range(len(ylabels))])



    
    if(savetype==".pdf"):
        pp.savefig(plt.gcf(),bbox_inches='tight')
        pp.close()
    else:
        plt.savefig(saveto+dataname+dataset+savetype,dpi=300,bbox_inches='tight', transparent=True,pad_inches=0)
    #show()
    figurenumber=figurenumber+1
    
    