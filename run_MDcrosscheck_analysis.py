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
        zedges=np.array([0.1,.5,1.6])
    elif nbins==3:
        zedges=np.array([0.1,.5,1.,1.6])
    bias=nobias
    dndz=dndz_desMDlike
    longtag='DES-like survey ala M&D with bias=1'
    sigz=.025 #only one bin, so this won't be used
    return SurveyType(tag,zedges,sigz,nbar,dndz,bias,longtag=longtag)

def MDtest_get_maptypelist(includeisw=False,Ndesbins=[2,3],nvss=True):
    surveys=[]
    if nvss:
        surveys.append(get_NVSSlike_SurveyType())
    #print 'dndz nvss args:',surveys[0].dndzargs
    if includeisw:
        surveys.append(get_fullISW_MapType(zmax=15))   
    for n in Ndesbins:
        surveys.append(get_MDDESlike_SurveyType(nbins=n))
    return surveys
    
def MDtest_get_binmaps(includeisw=True,Ndesbins=[2,3],nvss=True):
    surveys=MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)
    bins=[]
    for survey in surveys:
        bins=bins+survey.binmaps
    if includeisw:
        iswmaptype=get_fullISW_MapType(zmax=15)
        iswbins=iswmaptype.binmaps
        bins=iswbins+bins
    return bins

def MDtest_get_Cl(justread=True,Ndesbins=[2,3],nvss=True):
    surveys=MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)
    bins=MDtest_get_binmaps(Ndesbins=Ndesbins,nvss=nvss)
    zmax=max(m.zmax for m in bins)
    rundat = ClRunData(tag='MDtest',rundir='output/MDchecks/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    pairs=[]
    #pair up isw and each LSS maps, but not lss maps together 
    for s in surveys:
        pairs.append((s.tag,'isw'))
        pairs.append((s.tag,s.tag))
    return getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)

def MDtest_get_reclist(Ndesbins=[2,3],lmin=3,lmax=80,nvss=True):
    maptypes=MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)
    Nrec=len(maptypes)
    reclist=[]
    for i in xrange(Nrec):
        mtype=maptypes[i]
        inmaptag=mtype.tag #label in output glmdat
        includeglm=[b.tag for b in mtype.binmaps]
        recdat=RecData(includeglm=includeglm,inmaptag=inmaptag,minl_forrec=lmin,maxl_forrec=lmax)
        reclist.append(recdat)
    return reclist

#use cldat to generate glm, alm, and maps; saves maps but not alm
def MDtest_get_glm_and_rec(Nreal=1,minreal=0,justgetrho=0,dorho=1,Ndesbins=[2,3],lmin=3,lmax=80,rhofiletag='',nvss=True):
    #adding this block of text fixed bug when passing two sets of analysis. why?
    maptypes=MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)
    mapsfor=[mt.tag for mt in maptypes] #tags for maps we want to make
    mapsfor.append('isw')
    allcldat=MDtest_get_Cl(justread=True,Ndesbins=Ndesbins,nvss=nvss)
    #cldat=get_reduced_cldata(allcldat,dothesemaps=mapsfor)#basically just reorders
    cldat=allcldat
    
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=MDtest_get_reclist(Ndesbins=Ndesbins,lmin=lmin,lmax=lmax,nvss=nvss)
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorho=dorho,dos=False,dochisq=False,rhofiletag=rhofiletag)

#use cldat to generate glm, no iswrec #WORKING HERE, bug is in this func
def MDtest_get_glm(Nreal=1,minreal=0,Ndesbins=[2,3],nvss=True):
    cldat=MDtest_get_Cl(justread=True,Ndesbins=Ndesbins,nvss=nvss)
    rlzns=np.arange(minreal,minreal+Nreal)
    #leaving reclist empty means just gen gal and isw maps
    getmaps_fromCl(cldat,rlzns=rlzns)

#assuming maps already generated, do reconstructions
def MDtest_iswrec(Nreal,minreal=0,justgetrho=0,dorho=1,Ndesbins=[2,3],lmin=3,lmax=80,rhofiletag='',nvss=True):
    rlzns=np.arange(minreal,minreal+Nreal)
    cldat=MDtest_get_Cl(justread=True,Ndesbins=Ndesbins,nvss=nvss)
    reclist=MDtest_get_reclist(Ndesbins=Ndesbins,lmin=lmin,lmax=lmax,nvss=nvss)
    dummyglm=get_glm(cldat,Nreal=0,runtag=cldat.rundat.tag)
    doiswrec_formaps(dummyglm,cldat,rlzns=rlzns,reclist=reclist,rhofiletag=rhofiletag,dos=False)
    
#get arrays of rho saved in .rho.dat files or .s.dat
def MDtest_read_rho_wfiles(varname='rho',Ndesbins=[2,3],lmin=3,lmax=80,rhofiletag='',nvss=True):
    maptypes=MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)  #list of LSS survey types
    mapdir='output/MDchecks/map_output/'
    if not rhofiletag:
        tagstr=''
    else:
        tagstr='_'+rhofiletag
    if lmax>0:
        files=['iswREC.{0:s}.fid.fullsky-lmin{2:02d}-lmax{3:02d}.MDtest{4:s}.{1:s}.dat'.format(mtype.tag,varname,lmin,lmax,tagstr) for mtype in maptypes]
    else:
        files=['iswREC.{0:s}.fid.fullsky-lmin{2:02d}.MDtest{3:s}.{1:s}.dat'.format(mtype.tag,varname,lmin,tagstr) for mtype in maptypes]
    rhogrid=np.array([read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

# get expectation values of rho or s, choose variable via varname
def MDtest_get_expected_rho(varname='rho',Ndesbins=[2,3],lmin=3,lmax=80,nvss=True):
    Nrec=len(Ndesbins)+nvss
    cldat=MDtest_get_Cl(Ndesbins=Ndesbins,nvss=nvss)
    reclist=MDtest_get_reclist(Ndesbins,lmin,lmax,nvss=nvss)
    rhopred=np.zeros(Nrec)
    for i in xrange(Nrec):
        rhopred[i]=compute_rho_fromcl(cldat,reclist[i],varname=varname)
    return rhopred

######################
def MDtest_plot_zwindowfuncs(desNbins=[3],nvss=True):
    maptypes=MDtest_get_maptypelist(includeisw=False,Ndesbins=desNbins,nvss=True)
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


def MDtest_plot_rhohist(varname='rho',Ndesbins=[2,3],lmin=3,lmax=80,getrhopred=True,firstNreal=-1,rhofiletag='',nvss=True):
    plotdir='output/MDchecks/plots/'
    rhogrid=MDtest_read_rho_wfiles(varname,Ndesbins,lmin,lmax,rhofiletag,nvss=nvss)
    #output order are nvss, then the des-like surveys in given Ndesbins order
    Nreal=rhogrid.shape[1]
    if firstNreal>0 and firstNreal<Nreal:
        Nreal=firstNreal
        rhogrid=rhogrid[:,:Nreal]
    testname='MDchecks'
    if getrhopred:
        rhopred=MDtest_get_expected_rho(varname,Ndesbins,lmin,lmax,nvss=nvss)
    else:
        rhopred=[]
    plotname ='MDtest_{1:s}hist_r{0:05d}'.format(Nreal,varname)
    reclabels=[]
    if nvss:
        reclabels.append('NVSS')
    for n in Ndesbins:
        reclabels.append('DES {0:d} bins'.format(n))


    if varname=='rho':
        plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='s':
        plot_shist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)

#################################################################
if __name__=="__main__":
    MDtest_plot_zwindowfuncs([2,3])
    Ndesbins=[2,3]
    lmin=3
    lmax=80
    
    Nreal=10#0000
    if 1:
        #rhofiletag=''
        rhofiletag='test'
        #MDtest_get_glm_and_rec(Nreal,justgetrho=False,dorho=1,Ndesbins=Ndesbins,lmin=lmin,lmax=lmax,rhofiletag=rhofiletag)
        donvss=False
        MDtest_get_glm(Nreal,Ndesbins=[3],nvss=donvss)#this seems to be where mixup is happening

        MDtest_iswrec(Nreal,Ndesbins=[2,3],nvss=1,lmin=lmin,lmax=lmax,rhofiletag=rhofiletag) #including more maps here doesn't mess up results
        #MDtest_iswrec(Nreal,Ndesbins=[2],lmin=lmin,lmax=lmax,rhofiletag=rhofiletag)


    MDtest_plot_rhohist('rho',Ndesbins=[2,3],nvss=1,lmin=lmin,lmax=lmax,firstNreal=Nreal,rhofiletag=rhofiletag)
