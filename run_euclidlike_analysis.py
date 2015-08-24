import numpy as np
import matplotlib.pyplot as plt
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *
from AnalysisUtils import *
from mapdef_utils import *
import time

#################################################################
# misc studies
#################################################################
def plot_isw_kernel(zmax=5):
    plotdir='output/plots/'
    cosmparamfile='testparam.cosm'
    cosm=Cosmology(cosmparamfile)
    cosm.tabulateZdep(zmax,nperz=100)
    z=cosm.z_array
    kernel=(1.-cosm.f_array)*cosm.g_array #isw kernel, stripped of prefactor
    plt.figure(0)
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    plt.title(r'ISW kernel (in units of $3H_0^2\Omega_m/c^2k^2$)')
    plt.ylabel(r'$\left(1-f(z)\right)\times D(z)$',fontsize=20)
    plt.xlabel(r'Redshift $z$',fontsize=20)
    #plt.ylim(0,.7)
    #plt.xlim(0,zmax)
    plt.plot(z,kernel)
    plotname='isw_kernel'
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#################################################################
# SURVEY PROPERTY TESTS
# Use same Cl for simulation and reconstruction, no calib errors
# Basically: in the best possible scenerio, how well can recon work?
#################################################################
# depth test - vary z0 for Euclid like fiducial survey
#        do this for one bin
#================================================================
# Generate Cl
#----------------------------------------------------------------
def depthtest_get_binmaps(z0vals=np.array([.3,.6,.7,.8]),includeisw=True):
    surveys=[get_Euclidlike_SurveyType(z0=z0,onebin=True,tag='eucz{0:02d}'.format(int(10*z0))) for z0 in z0vals]
    bins=[s.binmaps[0] for s in surveys] #all surveys have only one bin
    if includeisw:
        iswmaptype=get_fullISW_MapType(zmax=10)
        iswbins=iswmaptype.binmaps
        bins=iswbins+bins
    return bins

def depthtest_get_Cl(justread=True,z0vals = np.array([.3,.6,.7,.8])):
    bins=depthtest_get_binmaps(z0vals)
    zmax=max(m.zmax for m in bins)
    rundat = ClRunData(tag='depthtest',rundir='output/depthtest/',lmax=95,zmax=zmax)
    return getCl(bins,rundat,dopairs=['all'],DoNotOverwrite=justread)

#----------------------------------------------------------------
# Make maps and run reconstructions
#----------------------------------------------------------------
# get list of RecData objects descrbing isw reconstructions to do
def depththest_get_reclist(z0vals=np.array([.3,.6,.7,.8])):
    bins=depthtest_get_binmaps(z0vals)
    reclist=[]
    for i in xrange(len(bins)):
        bintag=bins[i].tag
        if bintag!='isw_bin0':
            includeglm=[bintag]
            inmaptag=bintag[:bintag.rfind('_bin0')]
            recdat=RecData(includeglm=includeglm,inmaptag=inmaptag)
            reclist.append(recdat)

#use cldat to generate glm, alm, and maps; saves maps but not alm
def depthtest_get_glm_and_rec(Nreal=1,z0vals=np.array([.3,.6,.7,.8]),minreal=0,justgetrho=0):
    t0=time.time()
    cldat=depthtest_get_Cl(justread=True)
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=depththest_get_reclist(z0vals)
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho)
    t1=time.time()
    print "total time for Nreal",Nreal,": ",t1-t0,'sec'

#get arrays of rho saved in .rho.dat files
def depthtest_read_rho_wfiles(z0vals=np.array([.3,.6,.7,.8])):
    mapdir='output/depthtest/map_output/'
    files=['iswREC.eucz{0:02d}.fid.fullsky.depthtest.rho.dat'.format(int(10*z0)) for z0 in z0vals]
    rhogrid=np.array([read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

#----------------------------------------------------------------
# Analysis: make plots
#----------------------------------------------------------------
def depthtest_plot_zwindowfuncs(z0vals=np.array([.3,.6,.7,.8])):
    bins=depthtest_get_binmaps(z0vals,False) #just gal maps
    plotdir='output/depthtest/plots/'
    Nbins=len(bins)
    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    plt.figure(0)
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    ax.set_yticklabels([])
    plt.title(r'Depth test: redshift distributions')
    plt.xlabel('Redshift z',fontsize=20)
    plt.ylabel('Source distribution (arbitrary units)',fontsize=20)
    plt.ylim(0,.7)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x', labelsize=18)
    for n in xrange(Nbins):
        m=bins[n]
        wgrid=m.window(zgrid)*m.nbar/1.e9
        colstr=colors[n%len(colors)]
        #plt.fill_between(zgrid,0,wgrid, facecolor=colstr,edgecolor='none',linewidth=2, alpha=0.3)
        plt.plot(zgrid,wgrid,color=colstr,linestyle='-',linewidth=2,label='{0:0.1f}'.format(z0vals[n]))
    plt.legend(title='$z_0$',loc='center right',fancybox=False, framealpha=0.,prop={'size':20},handlelength=3.5)
    eqstr=r'$\frac{dn}{dz}\propto \,z^2 e^{-\left(z/z_0\right)^{1.5}}$'
    textbox=ax.text(.75, .6, eqstr,fontsize=30,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))

    plotname='depthtest_zbins'
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

def depthtest_plot_rhohist(z0vals=np.array([.3,.6,.7,.8]),plotsampledist=True,rhocl=np.array([])):
    #if plotsampledist, plot curve of expected sample distribution
    #  if rhocl.size=z0vals.size, use those as input for rho in the sample dist
    #             otherwise just use mean

    plotdir='output/depthtest/plots/'
    Nrecs=z0vals.size
    Nbins=100
    rhogrid=depthtest_read_rho_wfiles(z0vals)
    Nreal=rhogrid.shape[1]
    maxrho=np.max(rhogrid)
    minrho=np.min(rhogrid)
    rholim=(minrho,maxrho)
    #rholim=(0.,maxrho)
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    plt.figure(0)
    plt.title(r'Depth test: correlation coef. $\rho$ for {0:g} rlzns'.format(Nreal))
    plt.xlabel(r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$',fontsize=16)
    plt.ylabel('realizations',fontsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        label=r'$z_0={0:0.1f}$: $\bar{{\rho}}={1:0.3f}$, $\sigma={2:0.3f}$'.format(z0vals[i],mean,sigma)
        colstr=colors[i%len(colors)]
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=rholim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.7)
        if plotsampledist:
            if rhocl.size==Nrecs:
                rhoval=rhocl[i]
            else:
                rhoval=mean
            Pr=rho_sampledist(evals,rhoval)
            plt.plot(evals,Pr,color=colstr,linestyle='--')
    plt.legend(loc='upper left')
    plotname='depthtest_rhohist_r{0:05d}'.format(Nreal)
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
#rho_sampledist(r,rho,NSIDE=32)

#================================================================
# binning test - vary binning strategy of fiducial Euclid like survey
#================================================================
# Generate Cl
#----------------------------------------------------------------

#----------------------------------------------------------------
# Make maps and run reconstructions
#----------------------------------------------------------------

#----------------------------------------------------------------
# Analysis: make plots
#----------------------------------------------------------------

#================================================================
# smoothing test - vary photoz uncertainty for fid Euclid like survey
#                  with a set number of bins
#================================================================


#================================================================
# photoz error test
#================================================================
#----------------------------------------------------------------


#################################################################
if __name__=="__main__":
    #plot_isw_kernel()
    depthtestz0=np.array([.3,.6,.7,.8])
    if 0: #compute Cl
        t0=time.time()
        depthtest_get_Cl(justread=False,z0vals=depthtestz0)
        t1=time.time()
        print "time:",str(t1-t0),"sec"
    if 0:
        depthtest_get_glm_and_rec(Nreal=10000,z0vals=depthtestz0,justgetrho=0,minreal=0)
    if 1: 
        #depthtest_plot_zwindowfuncs(depthtestz0)
        depthtest_plot_rhohist(depthtestz0,True)
        #WORKING HERE: tomorrow make plots of expected rec variance

    
