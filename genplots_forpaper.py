import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from itertools import permutations
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *
from AnalysisUtils import *
from mapdef_utils import *
import time
from run_euclidlike_analysis import *

     #plots for isw paper
#################################################################

#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def depthtest_plot_rhohist_forpaper(z0vals=np.array([.3,.6,.7,.8])):
    varname='rho'
    plotdir='output/plots_forpaper/'
    rhogrid=depthtest_read_rho_wfiles(z0vals,varname)
    Nreal=rhogrid.shape[1]
    rhopred=depthtest_get_expected_rho(z0vals,varname)
    plotname ='depthtest_{1:s}hist_r{0:05d}_forpaper'.format(Nreal,varname)
    reclabels=['$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]
    
    #plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    varstr=r'\rho'
    Nreal=rhogrid.shape[1]
    title=''#r'{0:s}: correlation coef. $\rho$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$'
    #plothist(varstr,rhogrid,reclabels,title,xtitle,plotdir,plotname,rhopred)
    predvals=rhopred
    Nbins=100
    Nrecs=rhogrid.shape[0]
    Nreal=rhogrid.shape[1]
    maxval=np.max(rhogrid)
    minval=np.min(rhogrid)
    vallim=(minval,maxval)
    #rholim=(0.,maxrho)
    #colors=['#1b9e77','#e7298a']
    colors=['#1b9e77','#d95f02','#e7298a','#7570b3','#66a61e','#e6ab02']
    plt.figure(0)
    plt.subplots_adjust(left=0.15, bottom=.17, right=.95, top=.95, wspace=0, hspace=0)
    #plt.title(plotname)
    plt.xlabel(xtitle,fontsize=26)
    plt.ylabel('Realizations',fontsize=26)
    plt.tick_params(axis='y', which='both', labelsize=16)
    plt.tick_params(axis='x', which='both', labelsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        colstr=colors[i%len(colors)]
        if len(predvals):
            predval=predvals[i]
            plt.axvline(predval,linestyle='-',color=colstr)
            label=r'{0:s}: $\langle {1:s}\rangle={2:0.3f}$'.format(reclabels[i],varstr,predval)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=vallim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.6)

    plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from simulation')
    plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='mean from theory')
    plt.legend(loc='upper left',frameon=False,fontsize=16)

    #plot window functions as inset
    ax=plt.axes([.21,.28,.33,.25])#lower left
    #ax=plt.axes([.2,.72,.33,.85])
    bins=depthtest_get_binmaps(z0vals,False) #just gal maps
    Nbins=len(bins)
    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title(r'Depth test: redshift distributions')
    plt.xlabel('Redshift z',fontsize=14)
    #plt.ylabel(r'arbitrary units',fontsize=14)
    plt.ylim(0,.7)
    plt.xlim(0,zmax)
    #ax.tick_params(axis='x', labelsize=18)
    for n in xrange(Nbins):
        m=bins[n]
        wgrid=m.window(zgrid)*m.nbar/1.e9
        colstr=colors[n%len(colors)]
        plt.plot(zgrid,wgrid,color=colstr,linestyle='-',linewidth=2)
    eqstr=r'$\frac{dn}{dz}\propto \,z^2 e^{-\left(z/z_0\right)^{1.5}}$'
    textbox=ax.text(.75, .6, eqstr,fontsize=20,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def bintest_plot_rhohist_forpaper(divstr=['6','222','111111']):
    varname='rho'
    plotdir='output/plots_forpaper/'
    rhogrid=bintest_read_rho_wfiles(divstr,varname=varname)
    Nreal=rhogrid.shape[1]
    _,rhopred=bintest_get_rhoexp(doplot=False,getdivs=divstr,saverho=False)
    plotname ='bintest_{1:s}hist_r{0:05d}_forpaper'.format(Nreal,varname)
    reclabels=['1 bin','3 bins','6 bins']
    
    #plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    varstr=r'\rho'
    Nreal=rhogrid.shape[1]
    title=''#r'{0:s}: correlation coef. $\rho$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$'
    #plothist(varstr,rhogrid,reclabels,title,xtitle,plotdir,plotname,rhopred)
    predvals=rhopred
    Nbins=100
    Nrecs=rhogrid.shape[0]
    Nreal=rhogrid.shape[1]
    maxval=np.max(rhogrid)
    minval=np.min(rhogrid)
    vallim=(minval,maxval)
    #rholim=(0.,maxrho)
    #colors=['#1b9e77','#e7298a']'#7570b3','#1b9e77'
    colors=['#e7298a','#e6ab02','#66a61e']
    plt.figure(0)
    plt.subplots_adjust(left=0.15, bottom=.17, right=.95, top=.95, wspace=0, hspace=0)
    #plt.title(plotname)
    plt.xlabel(xtitle,fontsize=26)
    plt.ylabel('Realizations',fontsize=26)
    plt.tick_params(axis='y', which='both', labelsize=16)
    plt.tick_params(axis='x', which='both', labelsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        colstr=colors[i%len(colors)]
        if len(predvals):
            predval=predvals[i]
            plt.axvline(predval,linestyle='-',color=colstr)
            label=r'{0:s}: $\langle {1:s}\rangle={2:0.3f}$'.format(reclabels[i],varstr,predval)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=vallim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.6)

    plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from simulation')
    plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='mean from theory')
    plt.legend(loc='upper left',frameon=False,fontsize=16)

    #plot window functions as inset
    ax=plt.axes([.21,.28,.33,.25])#lower left
    #ax=plt.axes([.2,.72,.33,.85])
    Nrecs=len(divstr)
    binsetlist=[]
    for n in xrange(Nrecs):
        bins=bintest_get_binmaps(getdivs=[divstr[n]],includeisw=False) #just gal maps
        binsetlist.append(bins)

    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title(r'Depth test: redshift distributions')
    plt.xlabel('Redshift z',fontsize=14)
    #plt.ylabel(r'arbitrary units',fontsize=14)
    ymax=.33
    plt.ylim(0,ymax)
    plt.xlim(0,zmax)
    #ax.tick_params(axis='x', labelsize=18)

    #plot isw kernel
    cosmparamfile='testparam.cosm'
    #plotname=plotname+'_iswkernel'
    cosm=Cosmology(cosmparamfile)
    cosm.tabulateZdep(zmax,nperz=nperz)
    cosmz=cosm.z_array
    kernel=(1.-cosm.f_array)*cosm.g_array #isw kernel, stripped of prefactor
    kernelmax=np.max(kernel)
    wantmax=.8*ymax
    scaleby=wantmax/kernelmax

    eqstr=r'$\frac{dn({\rm bin})}{dz}$'
    #plt.plot(np.array([]),np.array([]),color='black',label=eqstr,linewidth=2,linestyle='-')#dummy line
    plt.plot(cosmz,kernel*scaleby,color='grey',label='ISW kernel',linewidth=2,linestyle='--')
    #plt.legend(loc='upper right',fancybox=False, framealpha=0.,prop={'size':16},handlelength=3.5)
    
    #plot bin envelopes
    linewidths=[3,2,1]
    linestyles=['-','-','-']
    for n in xrange(Nrecs):
        for i in xrange(len(binsetlist[n])):#loop through individual bins
            m=binsetlist[n][i]
            wgrid=m.window(zgrid)*m.nbar/1.e9
            colstr=colors[n%len(colors)]
            plt.plot(zgrid,wgrid,color=colstr,linestyle=linestyles[n],linewidth=linewidths[n])

    textbox=ax.text(1.75, .25, eqstr,fontsize=20,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))
    
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()


#################################################################
if __name__=="__main__":
    #depthtest_plot_rhohist_forpaper(z0vals=np.array([.3,.6,.7,.8]))
    #bintest_plot_rhohist_forpaper()
    depthtest_TTscatter_forpaper()
    
