##################################################################
# This script is intended to run simulations similar to what is used in
# Manzotti and Dodelson's ISW rec paper, to be used as a cross check.
# 
##################################################################
from scipy.optimize import leastsq
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *
from AnalysisUtils import *
from mapdef_utils import *
import numpy as np
from run_euclidlike_analysis import *
##################################################################

def dndz_desMDlike(z):
    z0=.5
    result=1.5/(z0**3) #steeper version of dndzEuclidlike
    result*=z*z
    exponent=-1.*(z/z0)**1.7
    result*=np.exp(exponent)
    return result

def get_NVSSlike_SurveyType(tag=''):
    if not tag:
        tag='nvss'
    nbar=1.e9 #picking a big number, assuming shot noise will be negligible
    zedges=np.array([0.01,5.])
    bias=nobias
    dndz=dndz_NVSSlike
    longtag='NVSS-like survey with bias=1'
    sigz=.1 #only one bin, so this won't be used
    biasargs=[]
    dndzargs=[.32,0.36]
    return SurveyType(tag,zedges,sigz,nbar,dndz,bias,dndzargs=dndzargs,longtag=longtag)

def get_MDDESlike_SurveyType(tag='',nbins=2):
    if not tag:
        tag='desMD{0:d}bin'.format(nbins)
    nbar=1.e9 #picking a big number, assuming shot noise will be negligible
    if nbins==2:
        zedges=np.array([0.01,.5,1.6])
    elif nbins==3:
        zedges=np.array([0.1,.5,1.,1.6])
    bias=nobias
    dndz=dndz_desMDlike
    longtag='DES-like survey ala M&D with bias=1'
    sigz=.025 #only one bin, so this won't be used
    return SurveyType(tag,zedges,sigz,nbar,dndz,bias,longtag=longtag)

def MDtest_get_maptypelist(includeisw=False,Ndesbins=[2,3]):
    surveys=[]
    surveys.append(get_NVSSlike_SurveyType())
    print 'dndz nvss args:',surveys[0].dndzargs
    for n in Ndesbins:
        surveys.append(get_MDDESlike_SurveyType(nbins=n))
    if includeisw:
        surveys.append(get_fullISW_MapType(zmax=15))
    return surveys
    
def MDtest_get_binmaps(includeisw=True,Ndesbins=[2,3]):
    surveys=MDtest_get_maptypelist(Ndesbins=Ndesbins)
    bins=[]
    for survey in surveys:
        bins=bins+survey.binmaps
    if includeisw:
        iswmaptype=get_fullISW_MapType(zmax=15)
        iswbins=iswmaptype.binmaps
        bins=iswbins+bins
    return bins

def MDtest_get_Cl(justread=True,Ndesbins=[2,3]):
    bins=MDtest_get_binmaps(Ndesbins=Ndesbins)
    zmax=max(m.zmax for m in bins)
    rundat = ClRunData(tag='MDtest',rundir='output/MDchecks/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    pair=[]
    #pair up isw and each LSS maps, but not lss maps together 
    for b in bins:
        if b.isGal:
            pairs.append((b.typetag,'isw'))
    return getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)

def MDtest_get_reclist(Ndesbins=[2,3]):
    maptypes=MDtest_get_maptypelist(Ndesbins=Ndesbins)
    Nrec=len(maptypes)
    reclist=[]
    for i in xrange(Nrec):
        mtype=maptypes[i]
        inmaptag=mtype.tag #label in output glmdat
        includeglm=[b.tag for b in mtype.binmaps]
        recdat=RecData(includeglm=includeglm,inmaptag=inmaptag)
        reclist.append(recdat)
    return reclist

#use cldat to generate glm, alm, and maps; saves maps but not alm
def MDtest_get_glm_and_rec(Nreal=1,minreal=0,justgetrho=0,dorho=1,Ndesbins=[2,3]):
    cldat=MDtest_get_Cl(justread=True,Ndesbins=Ndesbins)
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=MDtest_get_reclist(Ndesbins=Ndesbins)
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorho=dorho)

#get arrays of rho saved in .rho.dat files or .s.dat
def MDtest_read_rho_wfiles(varname='rho',Ndesbins=[2,3]):
    maptypess=MDtest_get_maptypelist(Ndesbins=Ndesbins)  #list of LSS survey types
    mapdir='output/MDchecks/map_output/'
    files=['iswREC.{0:s}.fid.fullsky.MDtest.{1:s}.dat'.format(mtype.tag,varname) for mtype in maptypes]
    rhogrid=np.array([read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

######################
def MDtest_plot_zwindowfuncs(desNbins=3):
    maptypes=MDtest_get_maptypelist(includeisw=False,Ndesbins=[desNbins])
    Nrecs=len(maptypes)
    plotdir='output/MDchecks/plots/'
    plotname='MDtest_zbins'
    Nrecs=len(maptypes)
    binsetlist=[s.binmaps for s in maptypes]
    colors=['#ff7f00','#377eb8','#e7298a']#'#d95f02''#1b9e77'
    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    plt.figure(0)
    ax=plt.subplot()
    ax.set_yticklabels([])
    plt.subplots_adjust(bottom=.2)
    plt.subplots_adjust(left=.1)
    plt.subplots_adjust(right=.85)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.xlabel('Redshift z')
    plt.ylabel(r'$dn/dz$ (arb. units)')
    #ymax=.33
    #plt.ylim(0,ymax)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x')
    ax.set_yticklabels([])
    for n in xrange(Nrecs):
        colstr=colors[n%len(colors)]
        ntot=0
        for i in xrange(len(binsetlist[n])):
            ntot+=binsetlist[n][i].nbar
        for i in xrange(len(binsetlist[n])):#loop through individual bins
            m=binsetlist[n][i]
            if maptypes[n].tag=='nvss':
                wgrid=m.window(zgrid)
            else:
                wgrid=m.window(zgrid)*m.nbar/ntot
            if i==0:
                label=maptypes[n].tag
            else:
                label=''
            plt.plot(zgrid,wgrid,color=colstr,label=label)

    plt.legend()
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()


def MDchecks_plot_rhohist(varname='rho'):
    pass

#################################################################
if __name__=="__main__":
    MDtest_plot_zwindowfuncs()
