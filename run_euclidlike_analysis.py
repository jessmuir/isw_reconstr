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
def depthtest_get_glm_and_rec(Nreal=1,z0vals=np.array([.3,.6,.7,.8]),minreal=0,justgetrho=0):
    t0=time.time()
    cldat=depthtest_get_Cl(justread=True)
    bins=depthtest_get_binmaps(z0vals)
    redo=1
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)

    reclist=[]
    for i in xrange(len(bins)):
        bintag=bins[i].tag
        if bintag!='isw_bin0':
            includeglm=[bintag]
            inmaptag=bintag[:bintag.rfind('_bin0')]
            recdat=RecData(includeglm=includeglm,inmaptag=inmaptag)
            reclist.append(recdat)
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho)
    t1=time.time()
    print "total time for Nreal",Nreal,": ",t1-t0,'sec'
#----------------------------------------------------------------
# Analysis: make plots
#----------------------------------------------------------------
#NEED TO TEST
def depthtest_plot_zwindowfuncs(z0vals=np.array([.3,.6,.7,.8])):
    bins=depthtest_get_binmaps(z0vals,False) #just gal maps
    plotdir='output/depthtest/plots/'
    Nbins=len(bins)
    zmax=5.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    plt.figure(0)
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    ax.set_yticklabels([])
    plt.xlabel('Redshift z',fontsize=20)
    plt.ylabel('Source distribution (arbitrary units)',fontsize=20)
    plt.ylim(0.001,1.)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x', labelsize=18)
    for n in xrange(Nbins):
        m=bins[n]
        wgrid=m.window(zgrid)*m.nbar/1.e9
        colstr=colors[n%len(colors)]
        plt.fill_between(zgrid,0,wgrid, facecolor=colstr,edgecolor='none',linewidth=2, alpha=0.2)
        plt.plot(zgrid,wgridphoto,color=colstr,linestyle='-',linewidth=2,label='{0:0.1f}'.format(z0vals[n]))
    plt.legend(loc='upper right',fancybox=True, framealpha=0.,prop={'size':16},handlelength=3.5)
    plotname='depthtest_zbins'
    plt.savefig(plotdir+plotname+'.png')
    plt.close()

def depthtest_plot_rhohist():
    pass

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
    depthtestz0=np.array([.3,.6,.7,.8])
    if 0: #compute Cl
        t0=time.time()
        depthtest_get_Cl(justread=False,z0vals=depthtestz0)
        t1=time.time()
        print "time:",str(t1-t0),"sec"
    if 1:
        #WORKING HERE; run for a big number of realizations
        depthtest_get_glm_and_rec(Nreal=5,z0vals=depthtestz0,justgetrho=0,minreal=0)
    

    
