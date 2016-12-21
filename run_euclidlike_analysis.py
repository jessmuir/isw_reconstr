import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import host_subplot #to have two axis labels
import mpl_toolkits.axisartist as AA #to have two axis labels
from itertools import permutations
from scipy.optimize import leastsq
import MapParams as mp
import CosmParams as cp #[NJW 160606]
import ClRunUtils as clu
import genCrossCor as gcc
import genMapsfromCor as gmc
import AnalysisUtils as au
import mapdef_utils as mdu
import time
import healpy as hp
import os

#################################################################
# misc studies
#################################################################
def plot_isw_kernel(zmax=5):
    plotdir='output/plots/'
    cosmparamfile='testparam.cosm'
    cosm=cp.Cosmology(cosmparamfile)
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
    outname=plotdir+plotname+'.pdf'
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
    surveys=[mdu.get_Euclidlike_SurveyType(z0=z0,onebin=True,tag='eucz{0:02d}'.format(int(10*z0))) for z0 in z0vals]
    bins=[s.binmaps[0] for s in surveys] #all surveys have only one bin
    if includeisw:
        iswmaptype=mdu.get_fullISW_MapType(zmax=15)
        iswbins=iswmaptype.binmaps
        bins=iswbins+bins
    return bins

def depthtest_get_Cl(justread=True,z0vals = np.array([.3,.6,.7,.8]), outtag=''): #[added outtag to allow easy spec of different output directory]
    bins=depthtest_get_binmaps(z0vals)
    zmax=max(m.zmax for m in bins) #get highest max z of all the binmaps
    rundat = clu.ClRunData(tag='depthtest',rundir='output/depthtest'+outtag+'/',lmax=95,zmax=zmax) #keep info like lmax, output dir, tag, zmax, etc. and other general cl information we'll need for all maps
    return gcc.getCl(bins,rundat,dopairs=['all'],DoNotOverwrite=justread)

#----------------------------------------------------------------
# Make maps and run reconstructions
#----------------------------------------------------------------
# get list of RecData objects descrbing isw reconstructions to do
def depthtest_get_reclist(z0vals=np.array([.3,.6,.7,.8])):
    bins=depthtest_get_binmaps(z0vals)
    reclist=[]
    for i in xrange(len(bins)):
        bintag=bins[i].tag
        if bintag!='isw_bin0':
            includeglm=[bintag]
            inmaptag=bintag[:bintag.rfind('_bin0')]
            recdat = au.RecData(includeglm=includeglm,inmaptag=inmaptag)
            reclist.append(recdat)
    return reclist

# same as depthtest_get_reclist, but don't define includeglm so uses all bins instead of limiting to the current bin (i.e. multisurvey)
def multi_depthtest_get_reclist(z0vals=np.array([.3,.6,.7,.8]), multi = False):
    if multi == False:
        multi = range(z0vals) + [(0,1)]
    bins=depthtest_get_binmaps(z0vals) #list of bins, since each map is only one bin for depth test
    binlist=[] #bins but without isw_bin0, so corresponds to input z0vals
    for b in bins:
        if b.tag!='isw_bin0':
            binlist.append([b])
        print b.tag   #make list of lists, with each sublist a list of the bins in the map to use for recreation (only 1 each for now)
    Nrec = len(multi)
    reclist = [0]* Nrec
    for i,irec in enumerate(multi):
        includeglm=[]
        inmaptag = ''
        if type(irec) == int:
            irec = (irec,) #set to tuple so we can reuse upcoming for loop code
            mtag = ''
        else: mtag = '+'
        for m in irec: #for each survey in the list of surveys to use for recon
            for b in binlist[m]: #for each bin in the survey
                bintag=b.tag
                includeglm.append(bintag)
            inmaptag += mtag + bintag[:bintag.rfind('_bin')] #if single map, will be the standard bintag (without '_bin[X]'), if multi will be 'bintag0+bintag1'
        reclist[i] = au.RecData(includeglm=includeglm, inmaptag=inmaptag)
        print (irec,includeglm, inmaptag)
    return reclist

#use cldat to generate glm, alm, and maps; saves maps but not alm
#   does isw reconstruction automatically
def depthtest_get_glm_and_rec(Nreal=1,z0vals=np.array([.3,.6,.7,.8]),minreal=0,justgetrho=0,dorell=0,dorho=1,dos=1,dochisq=1,dochisqell=0, multi=False, outtag=''):
    t0=time.time()
    cldat=depthtest_get_Cl(justread=True,z0vals=z0vals,outtag=outtag)
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    if multi:
        reclist = multi_depthtest_get_reclist(z0vals, multi=multi) #try to use all the depth tests in recon
    else: reclist=depthtest_get_reclist(z0vals)
    au.getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorell=dorell,dorho=dorho,dos=dos,dochisq=dochisq,dochisqell=dochisqell)
    t1=time.time()
    print "total time for Nreal",Nreal,": ",t1-t0,'sec'

#do reconstructions based on exisitng LSS maps
# if domaps=False, just computes rho, s, etc without generating maps
def depthtest_iswrec(Nreal, cldat=False, z0vals=np.array([.3,.6,.7,.8]),minreal=0,dorell=0,dorho=1,dos=1,domaps=True, multi=False, outtag=''):#[160620=_NJW add option to pass cldat]
    if cldat==False:
        cldat=depthtest_get_Cl(z0vals=z0vals, outtag=outtag)
    rlzns=np.arange(minreal,minreal+Nreal)    
    if multi != False:
        reclist = multi_depthtest_get_reclist(z0vals, multi=multi) #try to use all the depth tests in recon
    else: reclist=depthtest_get_reclist(z0vals)
    dummyglm=gmc.get_glm(cldat,Nreal=0,matchClruntag=True)
    au.doiswrec_formaps(dummyglm,cldat,rlzns=rlzns,reclist=reclist,domaps=domaps,dorell=dorell,dos=dos)
    

#get arrays of rho saved in .rho.dat files or .s.dat
def depthtest_read_rho_wfiles(z0vals=np.array([.3,.6,.7,.8]),varname='rho'):
    mapdir='output/depthtest/map_output/'
    files=['iswREC.eucz{0:02d}.fid.fullsky-lmin02.depthtest.{1:s}.dat'.format(int(10*z0),varname) for z0 in z0vals]
    rhogrid=np.array([au.read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

#get arrays of rho saved in .rell.dat files
def depthtest_read_rell_wfiles(z0vals=np.array([.3,.6,.7,.8]),varname='rell'):
    mapdir='output/depthtest/map_output/'
    files=['iswREC.eucz{0:02d}.fid.fullsky-lmin02.depthtest.{1:s}.dat'.format(int(10*z0),varname) for z0 in z0vals]
    rellgrid=np.array([au.read_relldat_wfile(mapdir+f) for f in files])
    return rellgrid #[reconstruction,realization,ell]

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
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
    
#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def depthtest_plot_rhohist(z0vals=np.array([.3,.6,.7,.8]),getrhopred=True,varname='rho',firstNreal=-1,startrealat=0):
    plotdir='output/depthtest/plots/'
    testname="Depth test"
    rhogrid=depthtest_read_rho_wfiles(z0vals,varname)
    Nreal=rhogrid.shape[1]
    if firstNreal>0 and firstNreal<(Nreal-startrealat):
        Nreal=firstNreal
        rhogrid=rhogrid[:,startrealat:startrealat+Nreal]
    if getrhopred:
        rhopred=depthtest_get_expected_rho(z0vals,varname)
    else:
        rhopred=[]
    if startrealat==0:
        plotname ='depthtest_{1:s}hist_r{0:05d}'.format(Nreal,varname)
    else:
        plotname ='depthtest_{1:s}hist_r{0:05d}_startat{2:05d}'.format(Nreal,varname,startrealat)
    reclabels=['$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]

    if varname=='rho':
        au.plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='s':
        au.plot_shist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='chisq':
        au.plot_chisqhist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)

#--------------------------------
# plot r_ell values
def depthtest_plot_relldat(z0vals=np.array([.3,.6,.7,.8]),getpred=True,varname='rell'):
    plotdir='output/depthtest/plots/'
    testname="Depth test"
    rellgrid=depthtest_read_rell_wfiles(z0vals,varname=varname)
    print 'rellgrid.shape',rellgrid.shape
    Nreal=rellgrid.shape[1]
    if getpred:
        rellpred=depthtest_get_expected_rell(z0vals,varname)
    else:
        rellpred=[]
    plotname ='depthtest_{1:s}dat_r{0:05d}'.format(Nreal,varname)
    reclabels=['$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]

    au.plot_relldat(reclabels,testname,plotdir,plotname,rellgrid,rellpred,varname=varname)
        
#--------------------------------
# get expectation values of rho or s, choose variable via varname
def depthtest_get_expected_rho(z0vals=np.array([0.3,0.6,0.7,0.8]),varname='rho'):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rhopred=np.zeros_like(z0vals)
    for i in xrange(len(z0vals)):
        rhopred[i] = au.compute_rho_fromcl(cldat,reclist[i],varname=varname)
    return rhopred

def depthtest_get_expected_rell(z0vals=np.array([0.3,0.6,0.7,0.8]),varname='rell'):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rellpred=[]
    for i in xrange(z0vals.size):
        rellpred.append(au.compute_rell_fromcl(cldat,reclist[i],varname=varname))
    rellpred=np.array(rellpred)#[Nrec,Nell]
    #print rellpred
    return rellpred

#do Tisw-Trec scatter plot for a given realization r
def depthtest_TTscatter(r=0, z0vals=np.array([0.3,0.6,0.7,0.8]),savepngmaps=True,colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']):
    plotdir='output/depthtest/plots/'
    #colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    Nrec=z0vals.size
    #get map names and read in maps
    recmapfiles=[]
    recmaps=[]
    iswmapfiles=[]
    iswmaps=[]
    #get dummy glm and alm for filenames
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    glmdat=gmc.get_glm(cldat,Nreal=0,runtag=cldat.rundat.tag)
    almdat=au.get_dummy_recalmdat(glmdat,reclist,outruntag=glmdat.runtag)
    for i in xrange(Nrec):
        truemapf=glmdat.get_mapfile_fortags(r,reclist[i].zerotagstr)
        truemap= hp.read_map(truemapf,verbose=False)
        truemap= au.remove_lowell_frommap(truemap,lmin=2)
        iswmapfiles.append(truemapf)
        iswmaps.append(truemap)
        recmapf=almdat.get_mapfile(r,i,'fits')
        recmap=hp.read_map(recmapf,verbose=False)
        recmap=au.remove_lowell_frommap(recmap,lmin=2)
        recmapfiles.append(recmapf)
        recmaps.append(recmap)
        if savepngmaps:
            #set up color scheme for lss map
            mono_cm=cmx.Greys_r
            mono_cm.set_under("w") #set background to white
            lssmapfs=[]
            for glmstr in reclist[i].includeglm: #assumes default mod and mask
                lssmapfs.append(glmdat.get_mapfile_fortags(r,glmstr))
            for lssf in lssmapfs:
                lssm=hp.read_map(lssf,verbose=False)
                plotmax=0.7*np.max(np.fabs(lssm))
                lssfbase=lssf[lssf.rfind('/')+1:lssf.rfind('.fits')]
                hp.mollview(lssm,title=lssfbase,unit=r' $\delta\rho/\rho$',max=plotmax,min=-1*plotmax,cmap=mono_cm)
                plt.savefig(plotdir+'mapplot_'+lssfbase+'.pdf')
            maxtemp=np.max(truemap)
            maxtemp=max(maxtemp,np.max(recmap))
            plotmax=0.7*maxtemp
            truefbase=truemapf[truemapf.rfind('/')+1:truemapf.rfind('.fits')]
            hp.mollview(truemap,title=truefbase,unit='K',max=plotmax,min=-1*plotmax)
            plt.savefig(plotdir+'mapplot_'+truefbase+'.pdf')
            recfbase=recmapf[recmapf.rfind('/')+1:recmapf.rfind('.fits')]
            hp.mollview(recmap,title=recfbase,unit='K',max=plotmax,min=-1*plotmax)
            plt.savefig(plotdir+'mapplot_'+recfbase+'.pdf')

            
    #compute rho (could also read from file but this seams simplest)
    rhovals=[au.rho_onereal(iswmaps[n],recmaps[n]) for n in xrange(Nrec)]
    reclabels=['z0={0:0.1f}'.format(z0) for z0 in z0vals]

    #set up plot
    plotname='TrecTisw_scatter_depthtest.r{0:05d}'.format(r)
    au.plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir,plotname,colors)
    
    
#================================================================
# binning test - vary binning strategy of fiducial Euclid like survey
#================================================================
# binning and labeling utils
#----------------------------------------------------------------
# finestN - finest division of bins to use
#            last bin is always from z=2-5*z0 (tail of distrib)
#            the rest of the bins are evenly distributed in z=0-2
# getdivs - list of strings describing divisions (see label info below)
#           'all' means all possible divisions
#           'equal' means all possible div with equal zbins
# label map types like eucNbindivXXX
#    where N=finestN, XXX specifies how its divided
#       eg. six individual bins: euc6bindiv111111
#           six bins w first 2, last four combined: euc6bindiv24
#-----
#get z edges for finest division of z bins
def bintest_get_finest_zedges(finestN=6,z0=0.7):
    zedges=np.zeros(finestN+1)
    zedges[-1]=5.*z0
    if finestN>1:
        zedges[-2]=2.
        dz=2./(finestN-1)
        for n in xrange(finestN-1):
            zedges[n]=dz*n
    zedges[0]=.01 #don't go down all the way to zero
    return zedges
    
#-----
#get z edges for finest division of z bins, changed to be z0-dependent instead of hardcoded. [161216 NJW]
def bintest_get_finest_zedges_z0dep(finestN=6,z0=0.7):
    zedges=np.zeros(finestN+1)
    zedges[-1]=5.*z0 
    if finestN>1:
        zedges[-2]=2./0.7*z0 #2.  #scale by z0
        dz=zedges[-2]/(finestN-1)
        for n in xrange(finestN-1):
            zedges[n]=dz*n
    zedges[0]=.01 #don't go down all the way to zero
    return zedges
    
#-----
# get string tags for divisions for option "equal"
def bintest_get_divstr_equal(finestN=6):
    N=finestN
    slist=[]
    for d in xrange(N):#d=number of internal dividers
        bins=d+1
        if N%bins==0: #evenly divisible
            s=str(N/bins)*bins
            slist.append(s)
    return slist

def get_sumperm(n,N):
    #get all ordered combos of n numbers which add up to N
    if n>N: #no combos
        return []
    #will be recursive
    outlist=[]
    firstval=N-n+1
    nlist=np.ones(n,dtype=int)
    nlist[0]=firstval
    outlist.append(np.array(nlist))
    while firstval>1 and n>1:
        firstval-=1
        sublists=get_sumperm(n-1,N-firstval)
        for s in sublists:
            newlist=np.array([firstval]+list(s))
            outlist.append(newlist)
    return outlist
    
def bintest_get_divstr_all(finestN=6):
    slist=[]
    N=finestN
    for d in xrange(N): #d= number of divisions to place
        #get sets of d+1 numbers which add up to n
        perm=get_sumperm(d+1,N)
        for p in perm:
            slist.append(''.join([str(n) for n in p]))
    return slist

def bintest_get_zedgeslist(zedges,getdivs=['all'],returnstr=True):
    Nmax=zedges.size-1
    slist=getdivs
    if len(getdivs)==1:
        if getdivs[0]=='equal':
            slist=bintest_get_divstr_equal(Nmax)
        elif getdivs[0]=='all':
            slist=bintest_get_divstr_all(Nmax)
    zedgelist=[]#will be list of arrays, one for each binning strategy
#    print 'slist=',slist
    for s in slist:
        nlist=[int(x) for x in s]
        if np.sum(nlist)!=Nmax:
            print "wrong number of bins:sum(",s,")!=",Nmax
            continue
        zlist=[zedges[0]]
        zind=0
        for n in nlist:
            zind+=n
            zlist.append(zedges[zind])
        zedgelist.append(np.array(zlist))
    if returnstr:
        return zedgelist,slist
    else:
        return zedgelist

#----------------------------------------------------------------    
# Generate Cl
#----------------------------------------------------------------
def bintest_get_maptypelist(finestN=6,getdivs=['all'],z0=0.7,sigz=0.05,nbar=3.5e8,includeisw=True,survtype='euc'):
    #get zedges
    #zedges0 = bintest_get_finest_zedges(finestN,z0) #for finest division
    zedges0 = bintest_get_finest_zedges_z0dep(finestN,z0) # made it so edges scale with z0. z0=0.7 gives same results as orig above
#    print '\nin euc.bintest_get_maptyplest. (finestN, zedges,getdivs)=',(finestN,zedges0,getdivs)
    zedges,divstr=bintest_get_zedgeslist(zedges0,getdivs,True) 
    Ntypes = len(zedges)
    maptypes=[] #list of maptype objects, put finest div first
    maintag=survtype+'{0:d}bins{1:03d}div'.format(finestN,int(1000*sigz))
    if includeisw:
        iswmaptype = mdu.get_fullISW_MapType(zmax=15)
        maptypes.append(iswmaptype)
    for i in xrange(Ntypes):
        #print 'getting survey for zedges=',zedges[i]
        tag=maintag+divstr[i]
        if survtype=='euc':
            survey = mdu.get_Euclidlike_SurveyType(sigz=sigz,z0=z0,nbar=nbar,tag=tag,zedges=zedges[i])#0.7,tag=tag,zedges=zedges[i]) [NJW 160822]
            maptypes.append(survey)
        elif survtype=='spx':
            survey = mdu.get_Spherexlike_SurveyType(sigz=sigz,z0=z0,nbar=nbar,tag=tag,zedges=zedges[i])#0.7,tag=tag,zedges=zedges[i]) [NJW 160822]
            maptypes.append(survey)
        else: raise ValueError('Error: Only set up to take Euclidlike (\'euc\') or Spherexlike (\'spx\') surveys. You passed {0}'.format(survtype))
    return maptypes

def bintest_get_binmaps(finestN=6,getdivs=['all'],z0=0.7,sigz=0.05,includeisw=True,justfinest=False):
    if justfinest:
        getdivs=['1'*finestN]
    maptypes=bintest_get_maptypelist(finestN,getdivs,z0,sigz,includeisw)
    binmaps,bintags = mp.get_binmaplist(maptypes)
    return binmaps

#given surveytype tag or binmap tag, extract the XXX part of the divXXX label
def bintest_divstr_from_maptag(maptag):
    isbin='_bin' in maptag
    startind=maptag.rfind('div')+3
    endind=maptag.rfind('_bin')*isbin + (not isbin)*len(maptag)
    return maptag[startind:endind]

#given maptypetag with divXXX, return indices which tell you which of initial
# bins to combine
def bintest_combinewhich(divstr,baseN=6):
    outlists=[]
    innum=np.array([int(char) for char in divstr])
    if np.sum(innum)!=baseN or np.all(innum==1):
        return outlists
    else:
        bincounter=0
        for n in innum:
            combo=range(bincounter,bincounter+n)
            outlists.append(combo)
            bincounter+=n
    return outlists

# Get/generate Cl for 6 (or largest number of) bins
def bintest_get_baseClvals(finestN=6,z0=0.7,sigz=0.05,justread=True):
    binmaps=bintest_get_binmaps(finestN,z0=z0,sigz=sigz,justfinest=True)
    zmax=max(m.zmax for m in binmaps)
    rundat = clu.ClRunData(tag='eucbintest{0:d}s{1:03d}'.format(finestN,int(1000*sigz)),iswilktag='eucbintest',rundir='output/eucbintest/',lmax=95,zmax=zmax)
    return gcc.getCl(binmaps,rundat,dopairs=['all'],DoNotOverwrite=justread)

# if base cls are calcualted (the ones for finest bin divisions)
#      iterate through divisions, combining bins and save results
# if base cls not calcualated, computes them, then does bin combos
# if both already done, or justread, just reads in existing data 
def bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.05,justread=True):
    #get info for isw and finest binned division, calculate cls
    if not justread:
        basecl=bintest_get_baseClvals(finestN,z0,sigz,justread)
        basemaptype=bintest_get_maptypelist(finestN,['1'*finestN],z0,sigz,includeisw=False)[0]
        basemaptag=basemaptype.tag
        maptypes=bintest_get_maptypelist(finestN,['all'],z0,sigz,includeisw=False)    
        # combine bins to get Cl for other divisions
        FIRST=True
        for mt in maptypes:
            t=mt.tag
            print 'on maptype',t,'------'
            if t!=basemaptag: #if it's not the base map type
                divstr=bintest_divstr_from_maptag(t)
                combinebins=bintest_combinewhich(divstr,finestN)
                print ' combinebins=',combinebins
                #print 'combinebins',combinebins
                for i in xrange(len(combinebins)):
                    combolist=combinebins[i]
                    intags=[''.join([basemaptag,'_bin',str(x)]) for x in combolist]
                    outtag=''.join([t,'_bin',str(i)])
                    print '   combining',intags
                    print '      to get',outtag
                    if FIRST:
                        nextcl=gcc.combineCl_binlist(basecl,intags,combotag=outtag,newruntag=basecl.rundat.tag+'all',renamesingle=True)
                        FIRST=False
                    else:
                        nextcl=gcc.combineCl_binlist(nextcl,intags,combotag=outtag,renamesingle=True)
                    print '   nextcl.Nmap',nextcl.Nmap,'nextcl.Ncross',nextcl.Ncross
                    print '   len(nextcl.docross)',len(nextcl.docross)
        #write to file
        gcc.writeCl_file(nextcl)
    else:
        #read in data that has already been combined
        binmaps=bintest_get_binmaps(finestN,z0=z0,sigz=sigz)
        zmax=max(m.zmax for m in binmaps)
        rundat = clu.ClRunData(tag='eucbintest{0:d}s{1:03d}all'.format(finestN,int(1000*sigz)),iswilktag='eucbintest',rundir='output/eucbintest/',lmax=95,zmax=zmax)
        nextcl= gcc.getCl(binmaps,rundat,dopairs=['all'],DoNotOverwrite=True)
        print 'nextcl.Ncross',nextcl.Ncross
        print 'nextcl.docross.size',len(nextcl.docross)
    return nextcl
#----------------------------------------------------------------
# Reconstruction utilities
#----------------------------------------------------------------
#Return list of RecData objects, one per binning scheme
def bintest_get_reclist(finestN=6,z0=0.7,sigz=0.05,getdivs=['all']):
    #get binmaps
    maptypes= bintest_get_maptypelist(finestN,getdivs,z0,sigz,includeisw=False)
    Nrec=len(maptypes)
    reclist=[]
    for i in xrange(Nrec):
        mtype=maptypes[i]
        inmaptag=mtype.tag #label in output glmdat
        includeglm=[b.tag for b in mtype.binmaps]
        recdat = au.RecData(includeglm=includeglm,inmaptag=inmaptag)
        reclist.append(recdat)
    return reclist

#get cl for all bin combos (assumes they're already computed)
# and computes expectation values for rho, saving that data to file
# if overwrite==False and that file exists, just read it in
#also works for s; switch is in varname 
def bintest_get_rhoexp(finestN=6,z0=0.7,sigz=0.05,overwrite=False,doplot=True,getdivs=['all'],saverho=True,varname='rho'):
    if saverho:
        outdir = 'output/eucbintest/plots/'
        datfile='eucbintest{0:03d}_{1:s}exp.dat'.format(int(1000*sigz),varname)
    
        print datfile
        if not overwrite and os.path.isfile(outdir+datfile): #file exists #[added "import os" statement NJW 160606]
            x=np.loadtxt(outdir+datfile)
            divstr=[str(int(x[i,0])) for i in xrange(x.shape[0])]
            rhoarray=x[:,1]
        else:
            #get cl
            cldat=bintest_get_Clvals(finestN,z0,sigz,justread=True)
            #set up recdata objects for each bin combo
            reclist=bintest_get_reclist(finestN,z0,sigz,getdivs) 
            Nrec=len(reclist)
            rhoarray=np.zeros(Nrec)
            if getdivs==['all']:
                divstr=bintest_get_divstr_all(finestN) #string div labels
            elif getdivs==['equal']:
                divstr=bintest_get_divstr_equal(finestN)
            else:
                divstr=getdivs
            for r in xrange(Nrec):
                rhoarray[r]=au.compute_rho_fromcl(cldat,reclist[r],varname=varname)
            #write rhoarray to file
            f=open(outdir+datfile,'w')
            f.write(''.join(['{0:8s} {1:8.3f}\n'.format(divstr[i],rhoarray[i]) for i in xrange(Nrec)]))
            f.close()
    else: #don't interact with files, just compute
        #get cl
        cldat=bintest_get_Clvals(finestN,z0,sigz,justread=True)
        #set up recdata objects for each bin combo
        if getdivs==['all']:
            divstr=bintest_get_divstr_all(finestN) #string div labels
        elif getdivs==['equal']:
            divstr=bintest_get_divstr_equal(finestN)
        else:
            divstr=getdivs
        reclist=bintest_get_reclist(finestN,z0,sigz,getdivs) 
        Nrec=len(reclist)
        rhoarray=np.zeros(Nrec)
        for r in xrange(Nrec):
            rhoarray[r]=au.compute_rho_fromcl(cldat,reclist[r],varname=varname)

    if doplot:
        zedges0=bintest_get_finest_zedges(finestN,z0)
        allzedges=bintest_get_zedgeslist(zedges0,['all'],False)
        bintest_rhoexpplot(allzedges,divstr,rhoarray,varname)
    return divstr,rhoarray

#if we've computed Cl stuff for multiple values of sigz0, compare them
def bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.03,0.05],checkautoonly=False,varname='rho',plotdir='output/eucbintest/plots/',markerlist=[],colorlist=[],datsigs=[],datdivs=[],overwrite=False):
    rholist=[]
    for s in sigzlist:
        divstr,rho=bintest_get_rhoexp(finestN,z0,s,overwrite=overwrite,doplot=False,varname=varname)
        rholist.append(rho)
    legtitle='$\\sigma_{z0}$'
    labellist=['${0:0.3f}$'.format(s) for s in sigzlist]
    zedges0=bintest_get_finest_zedges(finestN,z0)
    allzedges=bintest_get_zedgeslist(zedges0,['all'],False)
    outtag=''
    outname='eucbintest_'+varname+'exp'+outtag+'.pdf'

    datplot=[]
    if len(datsigs) and len(datdivs): #if we want to get datapoints
        datrhos=[]
        datstds=[]
        datdivlist=[]
        datlabels=[]
        datcols=[]
        for i in xrange(len(datsigs)):
            datlabels.append('')
            datdivlist.append(datdivs)
            #color index gotten by comparing with sigzlist
            for j in xrange(len(sigzlist)):
                if sigzlist[j]==datsigs[i]:
                    datcols.append(j)
                    break
            #read in appropraite rho data, assuming already generated
            # get means and standard deviations
            rhogrid=bintest_read_rho_wfiles(datdivs,datsigs[i],varname=varname)
            print 'rhogrid.shape',rhogrid.shape
            datrhos.append([np.mean(rhogrid[j,:]) for j in xrange(len(datdivs))])
            datstds.append([np.std(rhogrid[j,:]) for j in xrange(len(datdivs))])

        #bundle into datplot array
        datplot=[datrhos,datstds,datdivlist,datlabels,datcols]
        print datplot
    bintest_rhoexpplot(allzedges,divstr,rholist,labellist,outname,legtitle,markerlist,colorlist,outtag,varname=varname,plotdir=plotdir,datplot=datplot)

#--------------------
def bintest_test_rhoexp():
    finestN=6
    z0=0.7
    sigzlist=[0.1]
    Nsig=len(sigzlist)
    divstrlist=['6','111111']
    Ndiv=len(divstrlist)
    #get cl
    cldatlist=[]
    recgrid=[]#sigz,divstr
    rhogrid=np.zeros((Nsig,Ndiv))#sigz,divstr
    for s in xrange(Nsig):
        cldat=bintest_get_Clvals(finestN,z0,sigzlist[s],justread=True)
        cldatlist.append(cldat)
    for s in xrange(Nsig):
        print 'sigma=',sigzlist[s],'============'
        cldat=cldatlist[s]
        reclist=bintest_get_reclist(finestN,z0,sigzlist[s],divstrlist) 
        recgrid.append(reclist)
        for r in xrange(Ndiv):
            print '  div:',divstrlist[r],'----------'
            rhogrid[s,r]=au.compute_rho_fromcl(cldat,reclist[r])
            print '    rho=',rhogrid[s,r]

def bintest_get_expected_rell(divstr,varname='rell'):
    cldat=bintest_get_Clvals()
    reclist=bintest_get_reclist(getdivs=divstr)
    rellpred=[]
    for i in xrange(len(reclist)):
        if varname=='rell':
            rellpred.append(au.compute_rell_fromcl(cldat,reclist[i],varname=varname))
    rellpred=np.array(rellpred)#[Nrec,Nell]
    return rellpred
    
#----------------------------------------------------------------
# Make maps and run reconstructions
#----------------------------------------------------------------

#note that we can really just save maps for the finest division
# and then do some adding to get stats on combined bins
#  will need to add glm to do reconstructions though

#use cldat to generate glm, alm, and maps; saves maps but not alm
# does isw reconstruction automatically
def bintest_get_glm_and_rec(Nreal=1,divlist=['6','222','111111'],minreal=0,justgetrho=0,dorell=0):
    t0=time.time()
    allcldat=bintest_get_Clvals(justread=True) #default finestN,z0,sigz
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=bintest_get_reclist(getdivs=divlist)
    #get reduced cl with only desired divlist maps in it; 
    maptypes=bintest_get_maptypelist(finestN=6,getdivs=divlist,z0=0.7,sigz=0.05,includeisw=True)
    mapsfor=[mt.tag for mt in maptypes] #tags for maps we want to make
    cldat=gcc.get_reduced_cldata(allcldat,dothesemaps=mapsfor)
    
    au.getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorell=dorell)
    t1=time.time()
    print "total time for Nreal",Nreal,": ",t1-t0,'sec'


# reconstruct the isw based on existing galaxy maps, without regenerating those maps
# if domaps=False, just computes rho, s, etc without generating maps
def bintest_iswrec(Nreal,divlist=['6','222','111111'],minreal=0,dorell=0,dorho=1,dos=1,domaps=True):
    allcldat=bintest_get_Clvals()
    #get reduced cl with only desired divlist maps in it; 
    maptypes=bintest_get_maptypelist(finestN=6,getdivs=divlist,z0=0.7,sigz=0.05,includeisw=True)
    mapsfor=[mt.tag for mt in maptypes] #tags for maps we want to make
    cldat=gcc.get_reduced_cldata(allcldat,dothesemaps=mapsfor)
    
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=bintest_get_reclist(getdivs=divlist)
    dummyglm = gmc.get_glm(cldat,Nreal=0,matchClruntag=True)
    au.doiswrec_formaps(dummyglm,cldat,rlzns=rlzns,reclist=reclist,domaps=domaps,dorell=dorell,dos=dos)

#get arrays of rho saved in .rho.dat files, or .s.dat files
def bintest_read_rho_wfiles(divlist=['6','222','111111'],sigz=0.05,varname='rho'):
    #print 'in READFILES, divlist=',divlist
    mapdir='output/eucbintest/map_output/'
    files=['iswREC.euc6bins{0:03d}div{1:s}.fid.fullsky-lmin02.eucbintest6s{0:03d}all.{2:s}.dat'.format(int(1000*sigz),d,varname) for d in divlist]
    #print files
    rhogrid=np.array([au.read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

#get arrays of rho saved in .rell.dat files
def bintest_read_rell_wfiles(divlist=['6','222','111111'],sigz=0.05,varname='rell'):
    mapdir='output/eucbintest/map_output/'
    files=['iswREC.euc6bins{0:03d}div{1:s}.fid.fullsky-lmin02.eucbintest6s{0:03d}all.{2:s}.dat'.format(int(1000*sigz),d,varname) for d in divlist]
    rellgrid=np.array([au.read_relldat_wfile(mapdir+f) for f in files])
    return rellgrid

#----------------------------------------------------------------
# Analysis: make plots
#----------------------------------------------------------------

#plot expectation value of rho for different binning strategy
# with illustrative y axis
# also works for s, switch in variable varname
#   dataplot=[(datrho,datstd,datdiv,datlabel,datcol),...] #if you want to plot
#       some data points, pass their plotting info here. datsig!=0 adds error bars
def bintest_rhoexpplot(allzedges,labels,rhoarraylist,labellist=[],outname='',legtitle='',markerlist=[],colorlist=[],outtag='',
                       varname='rho',dotitle=False,plotdir='output/eucbintest/plots/',datplot=[], saveplot=True):
    if type(rhoarraylist[0])!=np.ndarray: #just one array passed,not list of arr
        rhoarraylist=[rhoarraylist]
    if not outname:
        outname='eucbintest_'+varname+'exp'+outtag+'.pdf'
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    scattercolors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    Npoints=len(labels)
    yvals=np.arange(Npoints)
    zvals=allzedges[-1] #should be base value
    Nz=zvals.size
    zinddict={zvals[i]:i for i in xrange(Nz)}

    #divide fiture up into two vertical pieces
    fig=plt.figure(0,figsize=(7,6))
    plt.subplots_adjust(left=0.1, bottom=.17, right=.95, top=.95, wspace=0, hspace=0)
    if dotitle:
        if varname=='rho':
            plt.suptitle(r'Expected correlation coef. between $T^{{\rm ISW}}$ and $T^{{\rm rec}}$', size=18)
        elif varname=='s':
            plt.suptitle(r'Expected ratio between RMS of  $T^{{\rm rec}}-T^{{\rm ISW}}$ and $\sigma_{{T}}^{{\rm ISW}}$', size=18)
        elif varname=='chisq':
            plt.suptitle(r'Expected $\chi^2=\sum_{{\ell}}|a_{{\ell m})^{{\rm ISW}} - a_{{\ell m})^{{\rm rec}}|^2/C_{{\ell}}^{{\rm ISW}}$', size=18)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,(2,3),sharey=ax1)
    fig.subplots_adjust(hspace=0, wspace=0) #put them right next to eachother

    #left side has illustration of binning strategy
    plt.sca(ax1)
    plt.ylim((-1,Npoints))
    plt.xlim((0,6.1))#at 6, plots butt up against each other
    plt.xlabel(r'Redshift bin edges $z$',fontsize=14)
    ax1.xaxis.set_ticks_position('bottom')
    plt.tick_params(axis='x', which='major', labelsize=11)
    plt.yticks(yvals, labels)
    plt.xticks(np.arange(Nz),['{0:0.1f}'.format(z) for z in zvals])
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the left edge are off
        right='off')         # ticks along the right edge are off

    #hide border
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    #ax1.yaxis.ticks.set_visible(False)
    for i in xrange(Npoints):
        Nbins=len(allzedges[i])-1
        xvalsi=[zinddict[allzedges[i][j]] for j in xrange(Nbins+1)]
        yvalsi=i*np.ones(len(allzedges[i]))
        ax1.barh(yvalsi-.5,xvalsi[::-1],color=colors[Nbins-1],edgecolor='white',linewidth=2)

    #right side has the expectation values for rho plotted
    plt.sca(ax2)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    plt.tick_params(axis='x', which='major', labelsize=14)
    if varname=='rho':
        ax2.set_xlabel(r'$\langle \rho \rangle$',fontsize=20)
        ax2.set_xlim((.92,.98))
        #legloc='upper left'
        legloc='lower right'
    elif varname=='s':
        ax2.set_xlabel(r'$\langle s \rangle$',fontsize=14)
        #ax2.set_xlim((.87,.92))
        legloc='upper right'
    elif varname=='chisq':
        ax2.set_xlabel(r'$\langle \chi^2\rangle$',fontsize=14)
        #ax2.set_xlim((.87,.92))
        legloc='upper right'
    ax2.grid(True)
    if not markerlist:
        markerlist=['D']*len(rhoarraylist)
    if not colorlist:
        colorlist=scattercolors
    #plot datapoints if we have them
    DODATA=False
    if len(datplot):
        for i in xrange(len(datplot[0])):
            DODATA=True
            datrho=datplot[0][i]#array
            datstd=datplot[1][i]#array #of standard dev
            datdiv=datplot[2][i]#list of strings, translate into arry of y vals
            daty=[]
            for d in datdiv:
                for j in xrange(len(labels)):
                    if labels[j]==d:
                        daty.append(yvals[j])
            datlabel=datplot[3][i] #dummy empty string for now
            datcol=''
            datcol=datplot[4][i]
            if type(datcol)==int: #is index rather tha color string
                datcol=colorlist[datcol]
            #uniform error bars if datsig is a number, nonuni if array

            print datrho
            print daty
            print np.array(datstd)
            ax2.errorbar(datrho,daty,xerr=np.array(datstd),label='',color=datcol,linestyle='None',marker='x',markersize=15,markeredgewidth=2,zorder=-32)
        
    for i in xrange(len(rhoarraylist)):
        rhoarray=rhoarraylist[i]
        m=markerlist[i]
        print
        print(len(yvals),yvals)
        print(len(rhoarray),rhoarray)
        if labellist:
            ax2.scatter(rhoarray,yvals,label=labellist[i],color=colorlist[i],marker=m,s=50)
        else:
            ax2.scatter(rhoarray,yvals,color=colorlist[i],marker=m)

    if labellist:
        if DODATA:
            ax2.set_xlim((.89,1.))
            plt.legend(bbox_to_anchor=(1,.8),fontsize=16,title=legtitle)
        else:
            plt.legend(loc=legloc,fontsize=16,title=legtitle)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels()[0], visible=False)#don't show number at first label
    if saveplot:
        print 'Saving plot to ',plotdir+outname
        plt.savefig(plotdir+outname)
        plt.close()

#----------------
def bintest_plot_rhohist(divstr=['6','222','111111'],getrhopred=True,reclabels=['1 bin','3 bins','6 bins'],varname='rho',firstNreal=-1):
    plotdir='output/eucbintest/plots/'
    Nrecs=len(divstr)
    rhogrid=bintest_read_rho_wfiles(divstr,varname=varname)
    Nreal=rhogrid.shape[1]
    if firstNreal>0 and firstNreal<Nreal:
        Nreal=firstNreal
        rhogrid=rhogrid[:,:Nreal]
    plotname='eucbintest_{1:s}hist_r{0:05d}'.format(Nreal,varname)
    testname='Bin test'
    if not reclabels:
        reclabels=divstr
    if getrhopred:
        divstrx,rhopred=bintest_get_rhoexp(getdivs=divstr,saverho=False,doplot=False,varname=varname)
    else:
        rhopred=[]
    if varname=='rho':
        au.plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='s':
        au.plot_shist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='chisq':
        au.plot_chisqhist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
        
        
#--------------------------------
# plot r_ell values
def bintest_plot_relldat(divstr=['6','222','111111'],getpred=True,reclabels=['1 bin','3 bins','6 bins'],varname='rell'):
    plotdir='output/eucbintest/plots/'
    testname="Bin test"
    rellgrid=bintest_read_rell_wfiles(divstr,varname=varname)
    Nreal=rellgrid.shape[1]
    Nell=rellgrid.shape[2]
    if getpred:
        rellpred=bintest_get_expected_rell(divstr,varname)
    else:
        rellpred=[]
    plotname ='eucbintest_{1:s}dat_r{0:05d}'.format(Nreal,varname)

    if varname=='rell':
        au.plot_relldat(reclabels,testname,plotdir,plotname,rellgrid,rellpred)   
#----------------
def bintest_plot_zwindowfuncs(finestN=6,z0=0.7,sigz=0.05,doiswkernel=True,plotdir='output/eucbintest/plots/'):
    bins=bintest_get_binmaps(finestN,z0=0.7,sigz=sigz,includeisw=False,justfinest=True)#just gal maps
    sigz0=sigz
    plotname='eucbintest_zbins_s{0:03d}'.format(int(1000*sigz))
    
    Nbins=len(bins)
    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    if doiswkernel:
        cosmparamfile='testparam.cosm'
        plotname=plotname+'_iswkernel'
        cosm=cp.Cosmology(cosmparamfile)
        cosm.tabulateZdep(zmax,nperz=nperz)
        cosmz=cosm.z_array
        kernel=(1.-cosm.f_array)*cosm.g_array #isw kernel, stripped of prefactor
        
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    plt.figure(0,figsize=(8,4))
    plt.subplots_adjust(bottom=.2)
    plt.subplots_adjust(left=.1)
    plt.subplots_adjust(right=.85)
    #plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for item in ([ax.xaxis.label, ax.yaxis.label] ):
        item.set_fontsize(20)
    ax.set_yticklabels([])
    #plt.title(r'Bin test: redshift distributions',fontsize=16)
    plt.xlabel('Redshift z')
    plt.ylabel(r'$dn/dz$ (arb. units)')
    ymax=0.3
    plt.ylim(0,ymax)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x')
    nbartot=0
    for n in xrange(Nbins):
        m=bins[n]
        wgrid=m.window(zgrid)*m.nbar/1.e9
        nbartot+=m.nbar
        colstr=colors[n%len(colors)]
        #plt.fill_between(zgrid,0,wgrid, facecolor=colstr,edgecolor='none',linewidth=2, alpha=0.3)
        plt.plot(zgrid,wgrid,color=colstr,linestyle='-',linewidth=2)

    


    eqstr='$dn/dz$ binned with\n $\\sigma(z)={0:0.3f}(1+z)$'.format(sigz0)

    plt.plot(np.array([]),np.array([]),linestyle='-',color='black',linewidth=2,label=eqstr)
    #textbox=ax.text(1.7, .25, eqstr,fontsize=16,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))
    if doiswkernel:
        kernelmax=np.max(kernel)
        wantmax=.8*ymax
        scaleby=wantmax/kernelmax
        plt.plot(cosmz,kernel*scaleby,color='grey',label='ISW kernel',linewidth=2,linestyle='--')
    plt.legend(loc='upper right',fancybox=False, framealpha=0.,prop={'size':16},handlelength=3.5)

    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#----------------
#look at how correlated differnt base-bins are with one another and ISW
def bintest_plot_cl_vals(finestN=6,z0=0.7,sigz=0.05):
    outdir='output/eucbintest/plots/'
    outname='clvals_eucbintest{0:d}s{1:03d}.pdf'.format(finestN,int(1000*sigz))
    title='$C_{{\\ell}}^{{XY}}$ for Euclid-like survey with {0:d} bins, $\\sigma_z={1:0.3f}$'.format(finestN,sigz)
    cldat=bintest_get_baseClvals(finestN,z0,sigz,justread=True)
    ell=cldat.rundat.lvals
    
    #get shortened label names
    labellist=[]
    counter=0
    for b in cldat.bintaglist:
        if 'isw' in b:
            labellist.append('ISW')
            iswind=counter
        else:
            labellist.append(b[b.rfind('bin'):])#just 'bin0' 'bin1',etc
        counter+=1
    Nbins=cldat.Nmap
        
    #set up plots
    plt.figure(0,figsize=(7.5,10))#letter size paper?
    plt.suptitle(title,size=14)
    colorlist=['#1f78b4', '#33a02c','#e31a1c', '#ff7f00','#cab2d6', '#6a3d9a', '#b15928','#a6cee3','#b2df8a','#fb9a99','#fdbf6f']
    
    #plot auto power vs ell
    autocrosslist=[cldat.crossinds[i,i] for i in xrange(Nbins)]#crossinds to include
    ax1=plt.subplot(311)
    #plt.title('Auto-power, X=Y')
    plt.grid(True)
    plt.ylabel(r'$\ell(\ell+1)|C_{\ell}^{XX}|/(2\pi)$')
    plt.ylim(1.e-12,1.e-2)
    for n in xrange(Nbins):
        if n==iswind: col='grey'
        else: col=colorlist[(n-(n>iswind))%len(colorlist)]
        plt.loglog(ell,ell*(ell+1)*np.fabs(cldat.cl[autocrosslist[n],:])/(2*np.pi),color=col,label=labellist[n])
    plt.legend(title='Auto-power, X=Y',prop={'size':8},ncol=2,loc='center right')
 
    #plot x-corr between maps and ISW
    iswcrosslist=[]
    for i in xrange(Nbins):
        if i!=iswind:
            iswcrosslist.append(cldat.crossinds[i,iswind])
    ax2=plt.subplot(312, sharex=ax1)
    #plt.title('Cross power with ISW')
    plt.grid(True)
    plt.ylabel(r'$\ell(\ell+1)|C_{\ell}^{X-{\rm ISW}}|/(2\pi)$')
    plt.ylim(1.e-10,1.e-6)
    for n in xrange(Nbins):
        if n==iswind: continue
        else:
            #print 'colind',(n-(n>iswind))%len(colorlist)
            col=colorlist[(n-(n>iswind))%len(colorlist)]
            plt.loglog(ell,ell*(ell+1)*np.fabs(cldat.cl[iswcrosslist[n-(n>iswind)],:])/(2*np.pi),color=col,label=labellist[n])
    plt.legend(title='Cross-power w ISW',prop={'size':8},ncol=2,loc='lower right')
    
    #plot x-corr between neighboring bins
    Nnei=2 #number of neibhors to include
    n1crosslist=[]
    n1labellist=[]
    for nn in xrange(1,Nnei+1):
        for n in xrange(Nbins-1-nn):
            if i!=iswind and (i+nn)!=iswind: #assumes isw is at ind 0 or -1
                n1crosslist.append(cldat.crossinds[n,n+nn])
                n1labellist.append(labellist[n]+'-'+labellist[n+nn])
    ax2=plt.subplot(313, sharex=ax1)
    #plt.title('Cross power with neighboring bins')
    plt.grid(True)
    plt.ylabel(r'$\ell(\ell+1)|C_{\ell}^{XY}|/(2\pi)$')
    plt.ylim(1.e-9,1.e-2)
    plt.xlabel(r'Multipole $\ell$')
    for n in xrange(len(n1crosslist)):
        if n==iswind: continue
        else:
            col=colorlist[n%len(colorlist)]
            plt.loglog(ell,ell*(ell+1)*np.fabs(cldat.cl[n1crosslist[n],:])/(2*np.pi),color=col,label=n1labellist[n])
    plt.legend(title='Cross-power w neighbors',prop={'size':8},ncol=2,loc='upper left')
    
    #save output
    print "Saving plot to ",outdir+outname
    plt.savefig(outdir+outname)
    plt.close()



#================================================================
# caltest - Overlay calibration error maps on depthtest fiducial map
#   varying variance of calibration error and lmin used in reconstruction 
#================================================================
def caltest_get_fidbins(z0=0.7):
#    z0=0.7 #moved to input NJW 160630
    fidbins=depthtest_get_binmaps(z0vals=np.array([z0])) #isw+one galbin
    if len(fidbins)!=2:
        print 'more items in fidbins than expected:',fidbins
    return fidbins

def caltest_get_clfid(z0=0.7, DoNotOverwrite=False): #input added 160630 NJW
    bins=caltest_get_fidbins(z0)
    zmax=max(m.zmax for m in bins)
    #use depthtest tags since we'll read in those (already calculated) Cl
    rundat = clu.ClRunData(tag='depthtest',rundir='output/depthtest/',lmax=95,zmax=zmax)
    fidcl=gcc.getCl(bins,rundat,dopairs=['all'],DoNotOverwrite=DoNotOverwrite)
    return fidcl

# get glmdata object with fiducial (no calib error) bin and isw info
#   is a dummy glmdata object; no data or realizations, just map names
def caltest_get_fidglm(fidcl=0):
    #get fiducial cl, with no calibration error
    if not fidcl:
        fidcl=caltest_get_clfid()

    #get fid glmdat; no data needed, just mapnames, etc
    glmdat=gmc.get_glm(fidcl,Nreal=0,matchClruntag=True)
    return glmdat

#return Cl^cal list; return array of shape Nvariance x Nell, NOT ClData object
#  assumes all cl_cal have same shape, just different map variances
def caltest_get_clcallist(varlist=[1.e-1,1.e-2,1.e-3,1.e-4],lmax=30,lmin=0,shape='g',width=10.):
    Nvar=len(varlist)
    maxvar=max(varlist)
    maxind=np.where(varlist==maxvar)[0][0]
    if shape=='g':
        maxcl= gmc.gen_error_cl_fixedvar_gauss(maxvar,lmax,lmin,width)
    elif shape=='l2':
        maxcl = gmc.gen_error_cl_fixedvar_l2(maxvar,lmax,lmin)

    clgrid=np.zeros((Nvar,lmax+1))
    #since Clcal\propto variance, can just scale
    for v in xrange(Nvar):
        clgrid[v,:]=maxcl*varlist[v]/maxvar
    return clgrid
    
#---------------------------------------------------------------
# generate maps, do reconstructions
#---------------------------------------------------------------
# generate reference maps with variance of say, 1.e-2
# rescale approrpiately when doing recs
# If Nreal==0, does no map making, just returns dummyglm containing mapnames
def caltest_apply_caliberrors(varlist,Nreal=0,shape='g',width=10.,lmin=0,lmax=30,overwritecalibmap=False,scaletovar=False,redofits=True):
    print 'IN caltest_apply_caliberrors with NREAL=',Nreal
    #print 'varlist',varlist
    refvar,refind=caltest_get_scaleinfo(varlist,scaletovar)
    #print 'refvar,refind',refvar,refind
    #get fid glmdat; no data needed, just mapnames, et
    fidbins=caltest_get_fidbins()
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    #print 'lssbin',lssbin
    glmdat=caltest_get_fidglm()

    #set up calibration error maps
    calinfolist=[(lssbin,refvar,lmax,shape,width,lmin)] #only for max var
    if Nreal and overwritecalibmap:
        print 'Generating calibration error maps.'
    dothesemods=gmc.get_fixedvar_errors_formaps(glmdat,calinfolist,overwrite=overwritecalibmap,Nreal=Nreal) #generates calibration error maps, returns [(maptag,modtag,masktag)]list

    outglmdatlist=[]
    #apply calibration errors
    for v in varlist:
        scaling=np.sqrt(v/refvar)
        print 'scaling=',scaling
        newcaltag=gmc.getmodtag_fixedvar(v,shape,lmin,lmax)
        print 'Applying calibration errors, newcaltag',newcaltag
        outglmdatlist.append(gmc.apply_caliberror_to_manymaps(glmdat,dothesemods,Nreal=Nreal,calmap_scaling=scaling,newmodtags=[newcaltag],overwritefits=redofits)) #returns dummyglmdat

    outglmdat=glmdat.copy(Nreal=0)
    for n in xrange(len(outglmdatlist)):
        outglmdat=outglmdat+outglmdatlist[n]
    #print 'outglmdat.modtaglist',outglmdat.modtaglist
    return outglmdat #includes, isw, fiduical, and cal error map names

#we only want to generate c(nhat) maps for one var(c); get others by scaling
#   this function picks out the reference variance for which maps will be made
#   and the index where it appears in varlist
def caltest_get_scaleinfo(varlist,scaletovar=False):
    #scaletovar=refvar is the variance for which cl maps are generated
    varlist=np.array(varlist)
    if scaletovar and np.where(varlist==scaletovar):
        #print 'finding scaletovar',scaletovar
        refvar=scaletovar
        refind=np.where(varlist==refvar)[0][0]
    else:
        #print 'finding max'
        refvar=max(varlist)
        refind=np.where(varlist==refvar)[0][0] #scale all calib maps from largest
    print refvar,refind
    return refvar,refind

def caltest_get_reclist(varlist,shape='g',width=10.,lmin=0,lmax=30,recminelllist=np.array([2])):
    reclist=[]
    fidbins=caltest_get_fidbins()
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    lsstype=fidbins[1].typetag
    calinfolist=[(lssbin,v,lmax,shape,width,lmin) for v in varlist]
    fidglm=caltest_get_fidglm()
    dothesemods=gmc.get_fixedvar_errors_formaps(fidglm,calinfolist,overwrite=False,Nreal=0) #returns [(maptag,modtag,masktag)]list
    #put fidicual in
    includecl=[lssbin]
    inmaptype=lsstype
    for recminell in recminelllist:
        reclist.append(au.RecData(includecl,includecl,inmaptype,'unmod',recminell))
    for m in dothesemods:
        includeglm=[m]
        rectag=m[1]#modtag
        for recminell in recminelllist:
            reclist.append(au.RecData(includeglm,includecl,inmaptype,rectag,recminell))
    return reclist #includes fiducial case as first entry

#having already generated maps for reconstructions with calib errors,
# do isw reconstructions from the maps, computing rho and s stats
# set domaps to False if you just want to recalculate states like rho,s 
def caltest_iswrec(Nreal,varlist,shape='g',width=10.,callmin=0,callmax=30,overwritecalibmap=False,scaletovar=False,recminelllist=np.array([2]),domaps=True):
    fidcl=caltest_get_clfid()
    dummyglm=caltest_apply_caliberrors(varlist,0,shape,width,callmin,callmax,overwritecalibmap,scaletovar,redofits=False)#includes fidicual case
    reclist=caltest_get_reclist(varlist,shape,width,callmin,callmax,recminelllist=recminelllist)
    au.doiswrec_formaps(dummyglm,fidcl,Nreal,reclist=reclist,domaps=domaps)

#---------------------------------------------------------------
# rhocalc utils
#---------------------------------------------------------------
#not really specific to caltest: just gives array of vals evenly spaced on logscale
def caltest_get_logspaced_varlist(minvar=1.e-9,maxvar=.1,Nperlog=10):
    logmin=np.log10(minvar)
    logmax=np.log10(maxvar)
    Npoints=int(logmax-logmin)*Nperlog+1
    dlog=1./Nperlog
    logout=logmin+np.arange(Npoints)*dlog
    varout=10**logout
    return varout

#assuming many realizations of maps have been run, read in the rho data
# and return an array which can be used to add those points to a plot
#for compatibility with pre-lmintest data, if recminelllist.size==1, no lmin tag
def caltest_getrhodat_fromfiles(varlist,shape='g',width=10.,lmin=0,lmax=30,recminelllist=np.array([2]),varname='rho',getunmod=True):
    Nvar=len(varlist)
    #read in rho values
    modnames=[gmc.getmodtag_fixedvar(v,shape,lmin,lmax,width) for v in varlist]
    mapdir='output/depthtest/map_output/'
    outrho=[]
    counter=0
    for recminell in recminelllist:
        reclminstr="-lmin{0:02d}".format(recminell)
        files=['iswREC.eucz07.{0:s}.fullsky{2:s}.depthtest.{1:s}.dat'.format(modname,varname,reclminstr) for modname in modnames]
        #append fiducial case (no cal error) for that lmin
        if getunmod:
            files.append('iswREC.eucz07.unmod.fullsky{1:s}.depthtest.{0:s}.dat'.format(varname,reclminstr))
        rhogrid=np.array([au.read_rhodat_wfile(mapdir+f) for f in files])#filesxrho
        outrho.append(rhogrid)
    outrho=np.array(outrho)
    return outrho #[lmin,calerror,realization]

# will return rho data needed for plotting datapoints on caltest plots
# will return 1d list of tuples, in order first of shape, then lminlist
#   note that caltest_compare_clcal_shapes functions assumes just one
#     or the other of shape and reclmin are varied
def caltest_getdataplot_forshapecompare(varname='rho',varlist=[],shapelist=['g'],widthlist=[10.],lminlist=[0],lmaxlist=[30],labellist=[''],cleanplot=False,recminelllist=np.array([2]),colorlist=['#e41a1c'],getlabels=False,getunmod=True):
    print "Reading in rho data"
    #just hard coding these in, since they depend on what realizations
    # I've run, so I don't expect a ton of variation here
    if not varlist:
        varlist=[1.e-6,1.e-5,1.e-4,1.e-3] #for testing datapoints
    Nvar=len(varlist)
    plotdatalist=[]
    #loop through lists, for each set up a tuple, add to list
    for i in xrange(len(shapelist)):
        shape=shapelist[i]
        caliblmin=lminlist[i]
        caliblmax=lmaxlist[i]
        col=colorlist[i]
        if shape=='g':
            #shapetag=r'Gaussian $C_{{\ell}}^{{\rm cal}}$' 
            shapetag=r'$C_{{\ell}}^{{\rm cal}}\propto e^{{-(\ell/{0:.0f})^2}}$'.format(widthlist[i],caliblmin,caliblmax)
            #shapetag='g{0:d}_{2:d}l{1:d}'.format(int(widthlist[i]),caliblmax,caliblmin)
        elif shape=='l2':
            #shapetag=r'Gaussian $C_{{\ell}}^{{\rm cal}}$' 
            shapetag=r'$C_{{\ell}}^{{\rm cal}}\propto e^{{-(\ell/{0:.0f})^2}}$'.format(widthlist[i],caliblmin,caliblmax)
            #shapetag='l2_{1:d}l{0:d}'.format(caliblmax,caliblmin)
        rhogrid=caltest_getrhodat_fromfiles(varlist,shape,widthlist[i],caliblmin,caliblmax,recminelllist,varname,getunmod=getunmod)#3D: [lmin,calerror,realization]
        #if getunmod last entry in calerror dimension is value for no calibration error
        
        #print rhogrid.shape
        Nreal=rhogrid.shape[2]
        Nlmin=rhogrid.shape[0]
        if getlabels and (not labellist[i] and len(recminelllist)==1):
            if cleanplot:
                label='Simulation'
            else:
                label=shapetag+' Nreal={0:d}'.format(Nreal)
        elif len(recminelllist)==1:
            label=labellist[i]
        else:
            label=''
        #find mean, sigmas for each lmin
        means=np.zeros((Nlmin,Nvar))
        refmeans=np.zeros(Nlmin)
        sigs=np.zeros((Nlmin,Nvar))
        for k in xrange(Nlmin):
            if Nlmin!=1:
                label=labellist[k]
            if len(shapelist)==1: #colors hot being used for shapes, just lmin
                col=colorlist[k%len(colorlist)]
            means[k,:]=np.array([np.mean(rhogrid[k,j,:]) for j in xrange(Nvar)])
            if getunmod:
                refmeans[k]=np.mean(rhogrid[k,-1,:])
            else:
                refmeans[k]=-100 #nonsense value so we don't use it by mistake
            sigs[k,:]=np.array([np.std(rhogrid[k,j,:]) for j in xrange(Nvar)])
            plotdatalist.append((varlist,means[k,:],label,col,sigs[k,:],refmeans[k],Nreal))
        #(datvar,datrho,datlabel,datcol,datsig),...]

    return plotdatalist

#if cleanplot, no title, sparser legend.
def caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='rho',lmaxlist=[],lminlist=[],widthlist=[],dodataplot=True,shortvarlist=[],outtag='',cleanplot=False,reclmin=2,plotdir='output/caltest_plots/',justdat=False):
    Nshapes=len(shapelist)
    if not outtag: outtag='shapecompare'
    colorlist=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']#qualitative

    if not lmaxlist:
        lmaxlist=[30]*Nshapes
    if not lminlist:
        lminlist=[]
        for s in shapelist:
            if s=='g':
                lminlist.append(0)
            elif s=='l2':
                lminlist.append(1)
    if not widthlist:
        widthlist=[10.]*Nshapes
    rhoexplist=[] #will be 2D; [shape,variance ] 
    labels=[]
    if not justdat:
        for s in xrange(Nshapes):
            rhoexplist.append(caltest_get_rhoexp(varlist=varlist,lmax=lmaxlist[s],lmin=lminlist[s],shape=shapelist[s],width=widthlist[s],overwrite=False,doplot=False,saverho=True,varname=varname,reclmin=reclmin)) 
            
            if shapelist[s]=='g':
                shapestr=r'$C_{{\ell}}^{{\rm cal}}\propto e^{{-(\ell/{0:.0f})^2}}$'.format(widthlist[s],lminlist[s],lmaxlist[s])
                #shapestr=r'$C_{{\ell}}^{{\rm cal}}\propto e^{{-(\ell/{0:.0f})^2}}$ for ${1:d}\leq \ell \leq {2:d}$'.format(widthlist[s],lminlist[s],lmaxlist[s])
                #shapestr='g{2:d}_{0:d}l{1:d}'.format(lminlist[s],lmaxlist[s],int(widthlist[s]))
            elif shapelist[s]=='l2':
                shapestr=r'$C_{{\ell}}^{{\rm cal}}\propto \ell^{{-2}}$'.format(lminlist[s],lmaxlist[s])
                #shapestr=r'$C_{{\ell}}^{{\rm cal}}\propto \ell^{{-2}}$ for ${0:d}\leq \ell \leq {1:d}$'.format(lminlist[s],lmaxlist[s])
                #shapestr='l2_{0:d}l{1:d}'.format(lminlist[s],lmaxlist[s])
            labels.append(shapestr)
    if dodataplot:
        dataplot=caltest_getdataplot_forshapecompare(varname,shortvarlist,shapelist,widthlist,lminlist,lmaxlist,cleanplot=cleanplot,recminelllist=np.array([reclmin]),colorlist=colorlist,getlabels=justdat) #don't get labels
    else:
        dataplot=[]

    caltest_rhoexpplot(varlist,rhoexplist,labels,outtag=outtag,varname=varname,datplot=dataplot,cleanplot=cleanplot,colorlist=colorlist,plotdir=plotdir)

#one shape, several lmin
def caltest_compare_lmin(varlist,shapecal='g',varname='rho',lmaxcal=30,lmincal=0,widthcal=10.,dodataplot=True,shortvarlist=[],outtag='',cleanplot=False,recminelllist=np.array([2]),shortrecminelllist=np.array([]),plotdir='output/caltest_plots/',justdat=False):
    Nlmin=len(recminelllist)   
    if not outtag: outtag='lmincompare'
    #colorlist=['black','#54278f','#756bb1','#9e9ac8','#bcbddc','#dadaeb']#sequential
    #colorlist=['black','#018571','#80cdc1','#dfc27d','#a6611a']
    colorlist=['black','#018571','#80cdc1','#a6611a']

    if not shortrecminelllist.size:
        shortrecminelllist=recminelllist
        shortcolorlist=colorlist
    else:
        counter=0
        shortcolorlist=[]
        for i in xrange(Nlmin):
            if counter>=len(shortrecminelllist):
                break
            if recminelllist[i]==shortrecminelllist[counter]:
                shortcolorlist.append(colorlist[i])
                counter+=1

    rhoexplist=[] #will be 2D; [reclmin,variance ] 
    labels=[]
    if not justdat:
        for i in xrange(Nlmin):
            rhoexplist.append(caltest_get_rhoexp(varlist=varlist,lmax=lmaxcal,lmin=lmincal,shape=shapecal,width=widthcal,overwrite=False,doplot=False,saverho=True,varname=varname,reclmin=recminelllist[i]))
            labels.append(r'$\ell_{{\rm min}}={0:d}$'.format(recminelllist[i]))
            
    if dodataplot:
        print 'shortrecminelllist',shortrecminelllist
        datlabellist=[r'$\ell_{{\rm min}}={0:d}$'.format(l) for l in shortrecminelllist]
        dataplot=caltest_getdataplot_forshapecompare(varname,shortvarlist,cleanplot=cleanplot,recminelllist=shortrecminelllist,labellist=datlabellist,colorlist=shortcolorlist,getlabels=justdat)
    else:
        dataplot=[]

    caltest_rhoexpplot_wratio(varlist,rhoexplist,labels,outtag=outtag,varname=varname,datplot=dataplot,cleanplot=cleanplot,colorlist=colorlist,plotdir=plotdir,plotlines=not justdat)

    
#caltest_get_rhoexp - approximating calib errors as additive only,
#                     compute expectation value of rho for number of calibration
#                     error field variances, assuming reconstruction is done
#                     assuming no calib error
#         inserts fiducial (no calib error) as last entry
#         returns grid of rho or s values of size Nrec=Nvar
def caltest_get_rhoexp(z0=0.7,varlist=[1.e-4],lmax=30,lmin=1,shape='g',width=10.,overwrite=False,doplot=True,saverho=True,
                       varname='rho',filetag='',reclmin=1,plotdir='output/caltest_plots/',nolmintag=False,dofidrec=True):
    print 'varlist'
    if shape=='g':
        shapestr='g{2:d}_{0:d}l{1:d}'.format(lmin,lmax,int(width))
    elif shape=='l2':
        shapestr='l2_{0:d}l{1:d}'.format(lmin,lmax)
    if saverho:
        outdir = 'output/caltest_plots/'
        if filetag:
            filetagstr='_'+filetag
        else:
            filetagstr=''
        if not nolmintag: #for compatibility with older, pre-lmintest data
            reclminstr="-lmin{0:02d}".format(reclmin)
            filetagstr=filetagstr+reclminstr
        datfile='caltest_{0:s}_{1:s}exp{2:s}.dat'.format(shapestr,varname,filetagstr)

        if not overwrite and os.path.isfile(outdir+datfile): #file exists
            print 'Reading data file:',datfile
            x=np.loadtxt(outdir+datfile,skiprows=2)
            invarlist=x[:,0]
            if not np.all(invarlist==np.array(varlist)):
                print "WARNING, invarlist!=varlist, overwriting"
            else:
                rhoarray=x[:,1]
                return rhoarray
        else:
            print 'Writing to data file:',datfile

    #get fiducial cl, with no calibration error
    fidbins=caltest_get_fidbins(z0=z0)
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    fidcl=caltest_get_clfid(z0=z0)

    #construct map-mod combos for the variances given
    mapmods=caltest_getmapmods_onebin(lssbin,varlist,lmax,lmin,shape,width)
    
    #generate calibration errors with fixed variance, spread through Cl lmin-lmax
    #for each variance, get calib error Cl
    clmodlist=[]
    for mm in mapmods:
        #print 'INRHOEXP---modding Cl---',mm
        clmod=gmc.apply_additive_caliberror_tocl(fidcl,[mm])
        clmodlist.append(clmod)
    # include fidicual cl as last entry
    if varlist[-1]!=0. and dofidrec:
        print "  appending fiducial case, no calib error"
        varlist.append(0.)
        clmodlist.append(fidcl)

    #get recdat; only need to pass maptags, not modtags, since cldata objects
    #  note that his means we can use the same recdat for all variances
    recdat=au.RecData(includeglm=[lssbin],inmaptag=lssbin[:lssbin.rfind('_bin0')],minl_forrec=reclmin)
    
    # return array of shape [Nvar,Nell]
    Nrec=len(varlist)
#    rhoarray=np.zeros(Nrec)
    rholist=[0]*Nrec
    for r in xrange(Nrec):
        #print '--ON VAR=',varlist[r],'---------'
        rholist[r]=au.compute_rho_fromcl(clmodlist[r],recdat,reccldat=fidcl,varname=varname)
    rhoarray=np.array(rholist)
    #if save, write to file
    if saverho:
        f=open(outdir+datfile,'w')
        f.write('Calib error test: Clcal shape={0:s}, ell={1:d}-{2:d}\n'.format(shape,lmin,lmax))
        f.write('var(c(nhat))   <{0:s}>\n'.format(varname))
        f.write(''.join(['{0:.2e} {1:8.3f}\n'.format(varlist[i],rhoarray[i]) for i in xrange(Nrec)]))
        f.close()

    if doplot:
        caltest_rhoexpplot(varlist,rhoarray,varname=varname,outtag=shapestr,plotdir=plotdir)
    return rholist#rhoarray

#-------
def caltest_getmapmods_onebin(lssbintag,varlist=[1.e-1,1.e-2,1.e-3,1.e-4],lmax=30,lmin=0,shape='g',width=10.):
    #construct map-mod combos for the variances given
    if shape=='g':
        mapmods=[(lssbintag,gmc.getmodtag_fixedvar_gauss(v,width,lmax,lmin)) for v in varlist]
    elif shape=='l2':
        mapmods=[(lssbintag,gmc.getmodtag_fixedvar_l2(v,lmax,lmin)) for v in varlist]
    return mapmods

def caltest_getmapmods_multibin(lssbintag_list,varlist=[1.e-1,1.e-2,1.e-3,1.e-4],lmax=30,lmin=0,shape='g',width=10., diffvar=False):
    #construct map-mod combos for the lss bins and variances given. Return as
    #[[(bintagA, modtag0),(bintagB,modtag0)], [(bintagA,modtag1),(bintagB,modtag1),...]]
    #[160815 NW]  add diffvar to allow passing varlist as a list of sublists of length lssbintag_list.
    if diffvar:
        assert type(varlist[0])==list
        if shape=='g':
            mapmods=[[(lssbintag,gmc.getmodtag_fixedvar_gauss(v,width,lmax,lmin)) for (lssbintag,v) in zip(lssbintag_list,vl)] for vl in varlist]
        elif shape=='l2':
            mapmods=[[(lssbintag,gmc.getmodtag_fixedvar_l2(v,lmax,lmin)) for (lssbintag,v) in zip(lssbintag_list,vl)] for vl in varlist]
    else:
        assert type(varlist[0])!=list
        if shape=='g':
            mapmods=[[(lssbintag,gmc.getmodtag_fixedvar_gauss(v,width,lmax,lmin)) for lssbintag in lssbintag_list] for v in varlist]
        elif shape=='l2':
            mapmods=[[(lssbintag,gmc.getmodtag_fixedvar_l2(v,lmax,lmin)) for lssbintag in lssbintag_list] for v in varlist]
    return mapmods

#---------------------------------------------------------------
# caltest_rhoexpplot and caltest_rhoexpplot_wratio both do this:
# make plots of expected rho (or s) vs calibration error variance
# input:
#   varlist: list of variances (x data)
#   rhoarraylist - list of arrays of rho data, each of which w size==varlist.size
#   labellist - list of labels for legend, corresponding to rhodat
#   outname - string name for file, if you want to override default formatting
#   legtitle - if passed, sets title of legend
#   colorlist - sets colors of each dataset, if not passed, uses default
#   outtag - if using default filename formatting, this is added to it
#   varname - what variable are we plotting? 'rho' or 's'
#   dataplot=[(datvar,datrho,datlabel,datcol,datsig),...] #if you want to plot
#       some data points, pass their plotting info here. datsig!=0 adds error bars
#   cleanplot - if True, no plot title, sparser legend
def caltest_rhoexpplot(varlist,rhoarraylist,labellist=[],outname='',legtitle='',colorlist=[],outtag='',varname='rho',datplot=[],cleanplot=False,plotdir='output/caltest_plots/'):
    #assuming last entry in varlist, rhoarray is fiducial (var=0)
    if type(rhoarraylist[0])!=np.ndarray: #just one array passed,not list of arr
        rhoarraylist=[rhoarraylist]
    scattercolors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    if not outname:
        outname='caltest_'+varname+'_exp'+outtag+'.pdf'

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.15)
    fig.subplots_adjust(right=.95)
    ax1 = plt.subplot() #top part has rho
    
    for item in ([ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels()
                  + ax1.get_yticklabels()):
        item.set_fontsize(18)
    if not labellist:
        labellist=['']
    if not colorlist:
        colorlist=scattercolors

    plt.sca(ax1)
    ax1.grid(True)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Variance of calib. error field ${\rm var}[c]$')
    ax1.axhline(0,color='grey',linestyle='-')
    if varname=='rho':
        ax1.set_ylim(-.3,1.2)
        ax1.set_ylabel(r'$\langle \rho \rangle$')
    elif varname=='s':
        ax1.set_ylabel(r'$\langle s \rangle$')
        ax1.set_ylim(.1,1.e4)
        #ax1.set_yscale('symlog',linthreshy=.1)
        ax1.set_yscale('log')
    elif varname=='chisq':
        ax1.set_ylabel(r'$\langle \chi^2 \rangle$')

    for i in xrange(len(rhoarraylist)):
        print len(varlist[:-1]),rhoarraylist[i][:-1].shape
        ax1.plot(varlist[:-1],rhoarraylist[i][:-1],label=labellist[i],color=colorlist[i%len(colorlist)])

    for i in xrange(len(datplot)):
        datvar=datplot[i][0]#array
        datrho=datplot[i][1]#array
        datlabel=datplot[i][2] #string
        datsig=0
        datcol=''
        datrefmean=0
        datNreal=0
        if len(datplot[i])>3:
            datcol=datplot[i][3]
            if len(datplot[i])>4:
                datsig=datplot[i][4] #if one value, same for all points
                #   if array, sets different
                if len(datplot[i])>5:
                    datrefmean=datplot[i][5]
                    if len(datplot[i])>5:
                        datNreal=datplot[i][6]
        if not datcol:
            datcol=colorlist[(i+len(rhoarraylist))%len(colorlist)]
        if type(datsig)==0:
            ax1.plot(datvar,datrho,label=datlabel,color=datcol,linestyle='None',marker='o')
        else: #uniform error bars if datsig is a number, nonuni if array
            ax1.errorbar(datvar,datrho,yerr=datsig,label=datlabel,color=datcol,linestyle='None',marker='o')
    #plot dummy datapoint for legend
    if len(datplot):
        xmin,xmax=ax1.get_xlim()
        ymin,ymax=ax1.get_ylim()
        datlabel='Mean from sim.'
        #datlabel='Mean from {0:g} sim.'.format(datNreal)#assumes all for same #
        plt.errorbar([-1],[.9],yerr=[.01],linestyle='None',marker='o',color='black',label=datlabel)
        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))
    
            
    plt.sca(ax1)
    if varname=='rho':
        plt.legend(fontsize=18,title=legtitle,loc='upper right',numpoints=1)
    elif varname=='s':
        plt.legend(fontsize=18,title=legtitle,loc='upper left',numpoints=1)
    elif varname=='chisq':
        plt.legend(fontsize=18,title=legtitle,loc='upper left',numpoints=1)

    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

def caltest_rhoexpplot_wratio(varlist,rhoarraylist,labellist=[],outname='',legtitle='',colorlist=[],outtag='',varname='rho',datplot=[],cleanplot=False,plotdir='output/caltest_plots/',plotlines=True):
    #assuming last entry in varlist, rhoarray is fiducial (var=0)
    if len(rhoarraylist) and type(rhoarraylist[0])!=np.ndarray: #just one array passed,not list of arr
        rhoarraylist=[rhoarraylist]
    scattercolors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    if not outname:
        outname='caltest_'+varname+'_exp'+outtag+'.pdf'

    fig=plt.figure(figsize=(7,6))
    fig.subplots_adjust(bottom=.15)
    fig.subplots_adjust(left=.175)
    fig.subplots_adjust(right=.95)
    fig.subplots_adjust(hspace=.2)
    ax1 = plt.subplot(3,1,(1,2)) #top part has rho
    ax2 = plt.subplot(3,1,3,sharex=ax1) #bottom has rho/rhofid
    
    for item in ([ax1.xaxis.label, ax1.yaxis.label,ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()+ ax1.get_yticklabels()+ax2.get_yticklabels()):
        item.set_fontsize(20)
    for item in ([ax1.yaxis.label]):
        item.set_fontsize(22)
    #for item in (ax2.get_yticklabels()):
    #    item.set_fontsize(14)
    if not labellist:
        labellist=['']
    if not colorlist:
        colorlist=scattercolors

    plt.sca(ax1)
    ax1.grid(True)
    ax1.set_xscale('log')
    ax1.set_xlim((10**-7.5,10**-1.5))
    if varname=='rho':
        ax1.axhline(0,color='grey',linestyle='-')
        ax1.axhline(1,color='grey',linestyle='-')
    ax2.axhline(0,color='grey',linestyle='-')
    ax2.axhline(-1,color='grey',linestyle='-')
    plt.setp(ax1.get_xticklabels(), visible=False)
    #ax1.ticklabel_format(axis='y',style='')
    ax2.grid(True)
    if varname=='rho':
        ax1.set_ylim(-.3,1.3)
        varstr=r'\rho'
    elif varname=='s':
        #ax2.set_yscale('symlog',linthreshy=.01)
        ax2.set_ylim(-1,5)
        varstr='s'
    elif varname=='chisq':
        varstr=r'\chi^2'
    
    ax2.set_xlabel(r'Variance of calib. error field ${\rm var}[c]$')
    if plotlines:
        ax1.set_ylabel(r'$\langle {0:s} \rangle$'.format(varstr))
        ax2.set_ylabel(r'$\langle {0:s} \rangle /\langle {0:s}_{{c=0}} \rangle -1$'.format(varstr))
        for i in xrange(len(rhoarraylist)):
            print len(varlist[:-1]),rhoarraylist[i][:-1].shape
            ax1.semilogx(varlist[:-1],rhoarraylist[i][:-1],label=labellist[i],color=colorlist[i%len(colorlist)])
            ax2.semilogx(varlist[:-1], rhoarraylist[i][:-1]/rhoarraylist[i][-1]-1.,label=labellist[i],color=colorlist[i%len(colorlist)])
    else:
        ax1.set_ylabel(r'${0:s}$'.format(varstr))
        ax2.set_ylabel(r'${0:s}/{0:s}_{{[c=0]}} -1$'.format(varstr))
    Ndat=len(datplot) #offset points if Ndat>1
    logoffsetunit=0
    leftoffset=0
    if Ndat>1:
        logoffsetunit=.1
        leftoffset=-1*(Ndat-1)/2.*logoffsetunit
    for i in xrange(Ndat):
        xoffset=leftoffset+i*logoffsetunit
        #print 'datplot[i]:[LEN=',len(datplot[i]),'] ',datplot[i]
        datvar=datplot[i][0]#array
        datvar=[d*10**xoffset for d in datvar]
        datrho=datplot[i][1]#array
        datlabel=datplot[i][2] #string
        datsig=0
        datcol=''
        datrefmean=0
        datNreal=0
        if len(datplot[i])>3:
            datcol=datplot[i][3]
            if len(datplot[i])>4:
                datsig=datplot[i][4] #if one value, same for all points
                #   if array, sets different
                if len(datplot[i])>5:
                    datrefmean=datplot[i][5]
                    if len(datplot[i])>6:
                        datNreal=datplot[i][6]
        if not datcol:
            datcol=colorlist[(i+len(rhoarraylist))%len(colorlist)]
        if type(datsig)==0:
            ax1.plot(datvar,datrho,label=datlabel,color=datcol,linestyle='None',marker='o')
            if datrefmean:
                ax2.plot(datvar,datrho/datrefmean -1.,label=datlabel,color=datcol,linestyle='None',marker='o')
        else: #uniform error bars if datsig is a number, nonuni if array
            ax1.errorbar(datvar,datrho,yerr=datsig,label=datlabel,color=datcol,linestyle='None',marker='o')#,capsize=5)
            if datrefmean:
                ax2.errorbar(datvar,datrho/datrefmean-1.,yerr=datsig/datrefmean,label=datlabel,color=datcol,linestyle='None',marker='o')#,capsize=5)

    #plot dummy datapoint for legend
    if len(datplot) and plotlines:
        xmin,xmax=ax1.get_xlim()
        ymin,ymax=ax1.get_ylim()
        datlabel='Mean from sim.'
        #datlabel='Mean from {0:g} sim.'.format(datNreal)#assumes all for same #
        plt.errorbar([-1],[.9],yerr=[.01],linestyle='None',marker='o',color='black',label=datlabel)
        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))

    #reduce number of ticklabels for ratio plot
    if varname=='rho':
        ax2.set_yticks([-1.5+i*.5 for i in xrange(4)])
        ax2.set_ylim((-1.25,.25))
    elif varname=='s':
        #ax2.set_yticks([-1,,0 ,1,2,3,4,5])
        pass
    plt.sca(ax1)
    if varname=='rho':
        plt.legend(fontsize=20,title=legtitle,loc='upper right',numpoints=1)
    elif varname=='s':
        plt.legend(fontsize=20,title=legtitle,loc='upper left',numpoints=1)
    elif varname=='chisq':
        plt.legend(fontsize=20,title=legtitle,loc='upper left',numpoints=1)

    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

    
#---------------------------------------------------------------
# plot comparison bewteen Cl cal and Cl gal or Cl gal-ISW to understand
# why the transition in rho happens where it is
def caltest_Clcomp(varlist=[1.e-7,1.e-6,1.e-5,1.e-4],shape='g',callmin=0,callmax=30,width=10.,plotdir='output/caltest_plots/'):
    Nvar=len(varlist)
    #get Clcal and Cl
    fidbins=caltest_get_fidbins()
    iswbin=fidbins[0].tag #will be the isw map
    iswind=0
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    lssind=1
    fidcl=caltest_get_clfid()
    l=np.arange(fidcl.Nell)
    
    #construct map-mod combos for the variances given
    mapmods=caltest_getmapmods_onebin(lssbin,varlist,callmax,callmin,shape,width)
    clcal=np.zeros((Nvar,fidcl.Nell))
    for i in xrange(Nvar):
        ctag=mapmods[i][1]
        if ctag[:2]=='l2':#power law
            var,maxl,minl=gmc.parsemodtag_fixedvar_l2(ctag)
            thiscalcl=gmc.gen_error_cl_fixedvar_l2(var,maxl,minl)
        elif ctag[:1]=='g':#gaussian
            var,maxl,minl,width=gmc.parsemodtag_fixedvar_gauss(ctag)
            thiscalcl=gmc.gen_error_cl_fixedvar_gauss(var,maxl,minl,width=width)
        clcal[i,:callmax+1]=thiscalcl
        
    #plot Cl cal for various variance values in color gradient
    # http://matplotlib.org/xkcd/examples/color/colormaps_reference.html
    cm=plt.get_cmap('Spectral_r')
    cNorm=colors.LogNorm()#max and min numbers colors need to span
    scalarMap=cmx.ScalarMappable(norm=cNorm,cmap=cm)
    varcols=scalarMap.to_rgba(varlist)
    clscaling=l*(l+1.)/(2*np.pi)
    #to get colorbar key, need ot set up a throw-away map
    dummyz=[[0,0],[0,0]]
    dummylevels=varlist
    dummyplot=plt.contourf(dummyz,dummylevels,cmap=cm,norm=colors.LogNorm())
    plt.clf()
    fig=plt.gcf()
    
    fig=plt.figure(0)
    ax=plt.subplot()
    plt.title(r'Comparing $C_{{\ell}}$ of galaxies, ISW, and calib. errors')
    plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}/2\pi$')
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((0,30))
    plt.ylim((1.e-11,1.))
    plt.yscale('log')
    for i in xrange(Nvar):
        plt.plot(l,np.fabs(clcal[i,:])*clscaling,color=varcols[i])
        
    #plto fid cl
    #overlay Clgal, Clgal-isw in bolded lines
    lssauto=fidcl.crossinds[lssind,lssind]
    lssisw=fidcl.crossinds[iswind,lssind]
    iswauto=fidcl.crossinds[iswind,iswind]

    line1,=plt.plot(l,np.fabs(fidcl.cl[lssauto,:])*clscaling,color='black',linestyle='-',linewidth=2,label='gal-gal')
    line2,=plt.plot(l,np.fabs(fidcl.cl[lssisw,:])*clscaling,color='black',linestyle='--',linewidth=2,label='ISW-gal')
    line3,=plt.plot(l,np.fabs(fidcl.cl[iswauto,:])*clscaling,color='black',linestyle=':',linewidth=2,label='ISW-ISW')

    
    #set up colorbar
    logminvar=int(np.log10(min(varlist)))
    logmaxvar=int(np.log10(max(varlist)))+1
    Nlog=logmaxvar-logminvar
    varticks=[10**(logminvar+n) for n in xrange(Nlog)]
    #cbaxes=fig.add_axes([.8,.1,.03,.8])#controls location of colorbar
    colbar=fig.colorbar(dummyplot,ticks=varticks)
    colbar.set_label(r'Variance of error field $\langle c^2(\hat{{n}})\rangle$')
    
    #get legend entry for Clcal
    #cal_patch = mpatches.Patch( color='red',label='cal. error')
    plt.legend(handles=[line1,line2,line3])

    plt.show()
    plotname='caltest_cl_compare'
    outname=plotdir+plotname+'.pdf'
#    print 'saving',outname
#    plt.savefig(outname)
    plt.close()
    



#---------------------------------------------------------------
# do TT scatter plot for a few realizations

# do TT scatter plot for one realization
# r=realization number
# varlist=list of nonzero calib error variances to plot
#         case with no calib error is automatically included
def caltest_TTscatter(r=0,varlist=[1.e-7,1.e-6,1.e-5,1.e-4],savepngmaps=False):
    plotdir = 'output/caltest_plots/'
    Nvar=len(varlist)
    varlist=np.sort(varlist)#smallest to largest
    recmapfiles=[]
    reclabels=[]
    reclist=caltest_get_reclist(varlist) #includes fiducial case
    dummyglm=caltest_apply_caliberrors(varlist)#includes true isw, fiducial case
    dummyalm=au.get_dummy_recalmdat(dummyglm,reclist,outruntag=dummyglm.runtag)

    #all realizations have the same truemapf
    truemapf=dummyglm.get_mapfile_fortags(r,reclist[0].zerotagstr)
    iswmapfiles=[truemapf]*(Nvar+1)
    if savepngmaps:
        tmax=7.e-5
        lssmax=.04
        truemap=hp.read_map(truemapf,verbose=False)
        dotrueiswpng=True
        lsspngf=[]
        recpngf=[]
    #get rec filenames, go from largest to smallest variance
    for i in reversed(xrange(Nvar)):
        modtag=gmc.getmodtag_fixedvar(varlist[i],'g',lmin=0,lmax=30,width=10)
        masktag='fullsky-lmin02'
        recmapf=dummyalm.get_mapfile_fortags(r,maptag='iswREC.eucz07',modtag=modtag,masktag=masktag)
        recmapfiles.append(recmapf)
        reclabels.append(r'$\langle c^2(\hat{{n}})\rangle$={0:.0e}'.format(varlist[i]))
        if savepngmaps:
            recmap=hp.read_map(recmapf,verbose=False)
            #set up color scheme for lss map
            mono_cm=cmx.Greys_r
            mono_cm.set_under("w") #set background to white
            lssf=dummyglm.get_mapfile_fortags(r,'eucz07_bin0',modtag,masktag)
            lssm=hp.read_map(lssf,verbose=False)
            plotmax=lssmax#0.7*np.max(np.fabs(lssm))
            lssfbase=lssf[lssf.rfind('/')+1:lssf.rfind('.fits')]
            hp.mollview(lssm,title=lssfbase,unit=r' $\delta\rho/\rho$',max=plotmax,min=-1*plotmax,cmap=mono_cm)
            plt.savefig(plotdir+'mapplot_'+lssfbase+'.pdf')
            lsspngf.append(plotdir+'mapplot_'+lssfbase+'.pdf')
            maxtemp=np.max(truemap)
            maxtemp=max(maxtemp,np.max(recmap))
            plotmax=tmax#0.7*maxtemp
            #true isw
            if dotrueiswpng:
                truefbase=truemapf[truemapf.rfind('/')+1:truemapf.rfind('.fits')]
                hp.mollview(truemap,title=truefbase,unit='K',max=plotmax,min=-1*plotmax)
                plt.savefig(plotdir+'mapplot_'+truefbase+'.pdf')
                iswpngf=plotdir+'mapplot_'+truefbase+'.pdf'
                trueiswpng=False #only do this once
            #reconstructed isw
            recfbase=recmapf[recmapf.rfind('/')+1:recmapf.rfind('.fits')]
            hp.mollview(recmap,title=recfbase,unit='K',max=plotmax,min=-1*plotmax)
            recpngf.append(plotdir+'mapplot_'+recfbase+'.pdf')
            plt.savefig(plotdir+'mapplot_'+recfbase+'.pdf')
    #then do fiducial case
    fidrecf=dummyalm.get_mapfile_fortags(r,'iswREC.eucz07','unmod',masktag)
    recmapfiles.append(fidrecf)
    reclabels.append('No calib. error')
    if savepngmaps:
        lssf=dummyglm.get_mapfile_fortags(r,'eucz07_bin0')
        lssm=hp.read_map(lssf,verbose=False)
        plotmax=lssmax#0.7*np.max(np.fabs(lssm))
        lssfbase=lssf[lssf.rfind('/')+1:lssf.rfind('.fits')]
        hp.mollview(lssm,title=lssfbase,unit=r' $\delta\rho/\rho$',max=plotmax,min=-1*plotmax,cmap=mono_cm)
        plt.savefig(plotdir+'mapplot_'+lssfbase+'.pdf')
        lsspngf.append(plotdir+'mapplot_'+lssfbase+'.pdf')
        maxtemp=np.max(truemap)
        maxtemp=max(maxtemp,np.max(recmap))
        plotmax=tmax#0.7*maxtemp
        recmapf=fidrecf
        recfbase=recmapf[recmapf.rfind('/')+1:recmapf.rfind('.fits')]
        hp.mollview(recmap,title=recfbase,unit='K',max=plotmax,min=-1*plotmax)
        recpngf.append(plotdir+'mapplot_'+recfbase+'.pdf')
        plt.savefig(plotdir+'mapplot_'+recfbase+'.pdf')

    #colors=['#253494','#2c7fb8','#41b6c4','#a1dab4','#ffffcc']
    colors=['#a6611a','#08519c','#41b6c4','#78c679','#ffffb2']

    plotname='TrecTisw_scatter_caltest.r{0:05d}'.format(r)
    au.plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir,plotname,colors)


#================================================================
# zdisttest - vary shape of b(z)*dn/dz and see what happens
#  z0test - look at different values of z0, mismatch sim and rec cl
#  bztest - look at effect of introducing quadratic z dep in bias
#================================================================

#-----------------------------
# zdisttest_rhoexplot
#  makes a colorblock plot showing how a stat (rho, s, etc) responds to changing
#  a variable like z0, b2, etc, between simulation and reconstructuion.
#  Prints <stat>_best in corresponding square
def zdisttest_rhoexpplot(rhogrid,simx,recx=np.array([]),varname='z0',statname='rho',outtag='',outname='',plotdir='output/zdisttest/plots/'):
    #varname can be z0, b2, catzfrac
    #these are latex expressions for variables, to be used in plot lables
    if varname=='z0':
        vartex=r'$z_0$'
    elif varname=='b2':
        vartex=r'$b_2$'
    elif varname=='catzfrac':
        vartex=r'$f_{{\rm cat}}$'
        
    #normalize numbers to plot
    if not recx.size:
        recx=simx
    Nsim=simx.size
    Nrec=recx.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname=varname+'test_'+statname+'_exp'+outtag+'.pdf'
    plotnums=np.zeros((Nrec,Nsim)) #column index=xcoord,row=ycoord
    bestforsim=np.zeros(Nsim)
    wherebestforsim=np.zeros(Nsim) #index of rec where best occus
    for i in xrange(Nsim):
        if statname=='rho':
            normi=max(rhogrid[i,:])
            plotnums[:,i] = 1-rhogrid[i,:]/normi 
        elif statname=='s':
            normi=min(rhogrid[i,:])
            plotnums[:,i] = rhogrid[i,:]/normi-1
        wheremax=np.argwhere(rhogrid[i,:]==normi)[0]
        #plotnums[wheremax,i]=0 #this will make the square white
        wherebestforsim[i]=wheremax
        bestforsim[i]=normi

    #have simx and recx be labels for rows and columns respectively
    # but have each one be equal coord spacing
    simmincoord=0
    simmaxcoord=Nsim
    simcoords=np.arange(Nsim+1)
    recmincoord=0
    recmaxcoord=Nrec
    reccoords=np.arange(Nrec+1)

    #make heatmap of rho vs rhobest
    
    plt.figure(0)
    plt.pcolor(simcoords, reccoords, plotnums, norm=colors.LogNorm(), cmap=cmx.jet)
    #have simx and recx be labels for rows and columns respectively
    # and space them out equally
    simmincoord=0
    simmaxcoord=Nsim
    simcoords=np.arange(Nsim+1) #coords are edges of squares
    recmincoord=0
    recmaxcoord=Nrec
    reccoords=np.arange(Nrec+1)
    plt.xlim((simmincoord,simmaxcoord))
    plt.ylim((recmincoord,recmaxcoord))
    plt.xticks(simcoords[:-1]+.5,simx)
    plt.yticks(reccoords[:-1]+.5,recx)

    #print rho_best in diagonal squares
    for i in xrange(Nsim):
        j=wherebestforsim[i]
        bestval=bestforsim[i]
        plt.text(i+.5,j+0.5,'{0:0.3f}'.format(bestval),horizontalalignment='center',verticalalignment='center')
        

    #set up colorbar and labels
    cb=plt.colorbar()
    if statname=='rho':
        cb.set_label(r'$1-\langle \rho \rangle/\langle \rho \rangle_{{\rm best}} $',fontsize=20)
        plt.annotate(r'$\langle \rho \rangle_{{\rm best}}$ for sim.',xy=(.97,1),horizontalalignment='right',verticalalignment='bottom',fontsize=14,xycoords='axes fraction')
    elif statname=='s':
        cb.set_label(r'$\langle s \rangle/\langle s \rangle_{{\rm best}} -1$',fontsize=20)
        plt.annotate(r'$\langle s\rangle_{{\rm best}}$ for sim.',xy=(.97,1),horizontalalignment='right',verticalalignment='bottom',fontsize=14,xycoords='axes fraction')
    plt.annotate('',xy=(.98,1),xytext=(1.02,1.05),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='k'))
    
        
    #plt.title('Varying '+vartex,fontsize=26)
    plt.ylabel(vartex+" used for ISW rec.",fontsize=20)
    plt.xlabel(vartex+" used for sim.",fontsize=20)
    
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()
        
#----------------------------------------------------------
# z0test_getz0vals - return list of z0 values, varying by percent errors
#     above and below the fiducial value. includes fid value
def z0test_getz0vals(percenterrors=np.array([1,10]),fid=0.7):
    Nerror=len(percenterrors)
    Nz0=2*Nerror+1
    z0=np.zeros(Nz0)
    percenterrors.sort() #smallest to largest
    z0[Nerror]=fid
    for n in xrange(Nerror):
        z0[n]=fid*(1.-.01*percenterrors[-1-n])
        z0[-1-n]=fid*(1+.01*percenterrors[-1-n])
    return z0

def z0test_get_binmaps(perrors=np.array([1,10]),fid=0.7,includeisw=True):
    z0=z0test_getz0vals(perrors,fid)
    maptags=['eucz{0:04d}_b2{1:04d}'.format(int(np.rint(z*10000)),0) for z in z0]
    surveys=[mdu.get_Euclidlike_SurveyType(z0=z0[i],onebin=True,tag=maptags[i]) for i in xrange(z0.size)]
    bins=[s.binmaps[0] for s in surveys] #surveys all just have one bin
    if includeisw:
        iswmaptype=mdu.get_fullISW_MapType(zmax=15)
        iswbins=iswmaptype.binmaps
        bins=iswbins+bins
    return bins
    
def z0test_get_Cl(justread=True,perrors=np.array([1,10]),fid=0.7):
    bins=z0test_get_binmaps(perrors,fid)
    pairs=[] #set up pairs for cross corr (autocorr automatic)
    #pair up isw and each LSS maps, but not lss maps together 
    for b in bins:
        if b.isGal:
            pairs.append((b.typetag,'isw'))
            
    zmax=max(m.zmax for m in bins)
    rundat = clu.ClRunData(tag='z0test',rundir='output/zdisttest/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    return gcc.getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)
#--------------------------------------------------------------------
# simz0 are z0 values for which simulated maps were generated
# recz0 are z0 values used in the Cl for estimator
# returns 2d Nsimz0xNrecz0 list of RecData objects
def z0test_get_recgrid(simz0=np.array([]),recz0=np.array([]),perrors=np.array([1,10]),fidz0=.7):
    if not simz0.size:
        simz0=z0test_getz0vals(perrors,fidz0)
    if not recz0.size:
        recz0=z0test_getz0vals(perrors,fidz0)
    simmaptags=['eucz{0:04d}_b2{1:04d}'.format(int(np.rint(z*10000)),0) for z in simz0]
    recmaptags=['eucz{0:04d}_b2{1:04d}'.format(int(np.rint(z*10000)),0) for z in recz0]
    recgrid=[]
    Nsim=simz0.size
    Nrec=recz0.size
    for ns in xrange(Nsim):
        reclist=[]
        for nr in xrange(Nrec):
            #print 'SIM: '+simmaptags[ns]+'_bin0  REC: '+recmaptags[nr]+'_bin0'
            recdat=au.RecData(includeglm=[simmaptags[ns]+'_bin0'],includecl=[recmaptags[nr]+'_bin0'],inmaptag=simmaptags[ns],rectag=recmaptags[nr])
            reclist.append(recdat)
        recgrid.append(reclist)
    return recgrid
    
# will output rhodat in len(simz0)xlen(rez0) array
# if simz0 and recz0 are passed as arrays, use those as the z0 vals
# if either are passed as an empty array, replace it with all vals indicated
#    by the perrors, fidz0 parameters
# will use perrors and fidz0 to get Cl data, so they should match in either case
def z0test_get_rhoexp(simz0=np.array([]),recz0=np.array([]),perrors=np.array([1,10,20,50]),fidz0=.7,overwrite=False,saverho=True,doplot=False,varname='rho',filetag='',plotdir='output/zdisttest/plots/',fitbias=True):
    if not simz0.size:
        simz0=z0test_getz0vals(perrors,fidz0)
    if not recz0.size:
        recz0=z0test_getz0vals(perrors,fidz0)

    if saverho:
        outdir=plotdir
        if filetag:
            filetagstr='_'+filetag
        else:
            filetagstr=filetag
        datfile='z0test_{0:s}exp{1:s}.dat'.format(varname,filetagstr)
        if not overwrite and os.path.isfile(outdir+datfile):#file exists
            print "Reading data file:",datfile
            x=np.loadtxt(outdir+datfile)
            insimz0=x[1:,0] #row labels
            inrecz0=x[0,1:] #column labels
            if not np.all(insimz0==simz0) or not np.all(inrecz0==recz0):
                print "WARNING, input z0 lists don't match requested."
            else:
                rhoarray=x[1:,1:] #rho data
                return rhoarray
        else:
            print "Writing to data file:",datfile
    
    Nsim=simz0.size
    Nrec=recz0.size
    recgrid=z0test_get_recgrid(simz0,recz0,perrors,fidz0) #Nsim x Nrec
    cldat=z0test_get_Cl(True,perrors,fidz0) #read cl already containing all z0's

    rhoarray=np.zeros((Nsim,Nrec))
    #print '***cldat.bintaglist',cldat.bintaglist
    for ns in xrange(Nsim):
        for nr in xrange(Nrec):
            rhoarray[ns,nr]=au.compute_rho_fromcl(cldat,recgrid[ns][nr],reccldat=cldat,varname=varname,fitbias=fitbias)

    if saverho:
        #write to file, 
        f=open(outdir+datfile,'w')
        f.write('{0:9.6f} '.format(0.)+''.join(['{0:9.6f} '.format(z0r) for z0r in recz0])+'\n')
        for ns in xrange(Nsim):
            f.write('{0:9.6f} '.format(simz0[ns])+''.join(['{0:9.6f} '.format(rhoarray[ns,nr]) for nr in xrange(Nrec)])+'\n')
        f.close()
    if doplot:
        z0test_rhoexpplot(simz0,recz0,rhoarray,varname,plotdir=plotdir) 
        
    return rhoarray

#make colorblock plot: useful for looking at a bunch of data at once
def z0test_rhoexpplot(simz0,recz0,rhogrid,varname='rho',outtag='',outname='',legtitle='',colorlist=[],plotdir='output/zdisttest/plots/'): 
    zdisttest_rhoexpplot(rhogrid,simx=simz0,recx=recz0,varname='z0',statname=varname,outtag=outtag,outname=outname,plotdir=plotdir)

#simz0 fixed at fidz0, varying recz0.
    #if dohatch==True, parts of plot with linear scaling will be filled with hatched shading
    #if fitbias=False, doesn't fit for b0
    # if biascomp, sets fitbias=True, will plot greyed out line for no bias fitting version
    #        for comparison
def z0test_onesim_plot(fidz0=0.7,perrors=np.array([1,10,20,50]),varname='rho',colorlist=[],plotdir='output/zdisttest/plots/',outtag='onesim',outname='',dohatch=False,fitbias=True,biascomp=True,overwritedat=True):
    simz0=np.array([fidz0])
    recz0=z0test_getz0vals(perrors,fidz0)
    if not fitbias:
        outtag=outtag+'_nob0fit'
    rhogrid=z0test_get_rhoexp(simz0=simz0,recz0=recz0,overwrite=overwritedat,saverho=True,doplot=False,varname=varname,filetag=outtag,fitbias=fitbias) #should be 1xNrec
    if biascomp:
        nofitrhogrid=z0test_get_rhoexp(simz0=simz0,recz0=recz0,overwrite=overwritedat,saverho=True,doplot=False,varname=varname,filetag=outtag+'_nob0fit',fitbias=False)
        outtag=outtag+'_biascomp'
    
    Nsim=simz0.size
    Nrec=recz0.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='z0test_'+varname+'_exp'+outtag+'.pdf'

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.15)
    fig.subplots_adjust(right=.95)
    ax1=plt.subplot(1,1,1)

    ax1.axhline(0,color='grey',linestyle=':')
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)
    ax1.set_xlabel(r'$z_0$ used in ISW reconstruction')
    if varname=='rho':
        varstr=r'\rho'
        if dohatch:
            aloc=(.23,.65)
        else:
            aloc=(.25,.75)
        linthreshy=1.e-7
        ax1.set_ylim((-4.e-3,linthreshy))
        bestval=np.max(rhogrid)
        arrowystart= .55*linthreshy
        arrowstr=r'$\downarrow$'
        # arrowystart= .75*linthreshy
        # arrowdy=-.2*linthreshy
        # arrowheadlen=.25*linthreshy
        # arrowlabely=arrowystart+arrowdy-arrowheadlen
    elif varname=='s':
        varstr='s'
        if dohatch:
            aloc=(.23,.55)
        else:
            aloc=(.25,.4)
        linthreshy=1.e-5
        ax1.set_ylim((-1*linthreshy,10.))
        bestval=np.min(rhogrid)
        arrowystart= -.55*linthreshy
        arrowstr=r'$\uparrow$'
        # arrowystart= -.75*linthreshy
        # arrowdy=.2*linthreshy
        # arrowheadlen=.25*linthreshy
        # arrowlabely=arrowystart
    ax1.set_ylabel(r'$\left[\langle {0:s} \rangle -\langle {0:s} \rangle_{{\rm match}} \right]/\langle {0:s} \rangle_{{\rm match}}$'.format(varstr))
    
    if biascomp:
        ax1.plot(recz0,nofitrhogrid[0,:]/bestval-1,marker='d',color='#969696')
        if varname=='s': #skipping b0 fitting has no impact on rho
            plt.annotate(r'no bias fitting',color='#969696',xy=(.85,.95),horizontalalignment='right',verticalalignment='top',fontsize=12,xycoords='axes fraction')
    ax1.plot(recz0,rhogrid[0,:]/bestval-1,marker='o',color='#2c7bb6')

    plt.annotate(r'True (sim.) $z_0={0:0.1f}$'.format(fidz0)+'\n'+r'$\langle {1:s}\rangle_{{\rm match}}={2:0.3f}$'.format(fidz0,varstr,bestval),xy=aloc,horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    

    #put arrow above or below fiducial point
    #ax1.arrow(fidz0,arrowystart,0,arrowdy,color='black',head_width=.015,head_length=arrowheadlen)
    plt.plot(fidz0,arrowystart,color='black',marker=arrowstr,markersize=16)
    #plt.annotate(r'match',xy=(fidz0+.02,arrowlabely),horizontalalignment='left',verticalalignment='bottom',fontsize=12)
    plt.annotate(r'match',xy=(fidz0+.02,arrowystart),horizontalalignment='left',verticalalignment='center',fontsize=12)
    
    ax1.set_yscale('symlog',linthreshy=linthreshy)
    if dohatch:
        xmin,xmax=ax1.get_xlim()
        x=np.arange(xmin,xmax,.01)
        ax1.fill_between(x,-linthreshy,linthreshy,color='none',edgecolor='grey',hatch='/',linewidth=0)
    if outtag:
        outtag='_'+outtag
    if not outname: #[this will never occur unless outnames is specifically passed as "False", since default set to ''. NJW 160606]
        outname=varname+'test_'+statname+'_exp'+outtag+'.pdf' #[statname undefined NJW 160606]
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()
 
def z0test_rhoexpplot_lines(simz0,recz0,rhogrid,varname='rho',outtag='',outname='',legtitle='',colorlist=[],plotdir='output/zdisttest/plots/'):
    scattercolors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    if not colorlist:
        colorlist=scattercolors
    
    Nsim=simz0.size
    Nrec=recz0.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='z0test_'+varname+'_exp'+outtag+'.pdf'

    fig=plt.figure(0)
    plt.suptitle('Effect of incorrectly modeled median redshift')
    #plot a line for each simz0, data points for each recz0
    ax1=plt.subplot(1,1,1)
    ax1.grid(True)
    if varname=='rho':
        ax1.set_ylabel(r'$\langle \rho \rangle/\langle \rho \rangle_{{\rm best}} -1$')
        ax1.set_yscale('symlog',linthreshy=1.e-7)
        ax1.set_ylim((-1.,1.e-8))
    elif varname=='s':
        ax1.set_ylabel(r'$\langle s \rangle/\langle s \rangle_{{\rm best}}$')
        #ax1.set_yscale('symlog',linthreshy=1.e-4)
        #ax1.set_yscale('log')
        ax1.set_ylim((.5,8.))
    elif varname=='chisq':
        ax1.set_ylabel(r'$\langle \chi^2 \rangle/\langle \chi^2 \rangle_{{\rm best}}$')
        #ax1.set_yscale('symlog',linthreshy=1.e-4)
        #ax1.set_yscale('log')
        #ax1.set_ylim((.5,8.))
    ax1.set_xlabel(r"$z_0$ used for ISW reconstruction")
    xvals=np.arange(Nrec)
    ax1.set_xlim((-.5,Nrec-.5))
    plt.xticks(xvals,recz0)
    for i in xrange(Nsim):
        if varname=='rho':
            maxrho=max(rhogrid[i,:])
            varstr=r'\rho'
            ax1.plot(xvals,rhogrid[i,:]/maxrho -1 ,label=r'{0:0.3f}; $\langle {2:s}\rangle_{{\rm best}}=${1:0.3f}'.format(simz0[i],maxrho,varstr),color=colorlist[i%len(colorlist)],marker='d')
        elif varname=='s':
            maxrho=min(rhogrid[i,:])#best value for s is smallest
            varstr='s'
            ax1.plot(xvals,rhogrid[i,:]/maxrho ,label=r'{0:0.3f}; $\langle {2:s}\rangle_{{\rm best}}=${1:0.3f}'.format(simz0[i],maxrho,varstr),color=colorlist[i%len(colorlist)],marker='d')
        elif varname=='chisq':
            maxrho=min(rhogrid[i,:])#best value for s is smallest
            varstr=r'\chi^2'
            ax1.plot(xvals,rhogrid[i,:]/maxrho ,label=r'{0:0.3f}; $\langle {2:s}\rangle_{{\rm best}}=${1:0.3f}'.format(simz0[i],maxrho,varstr),color=colorlist[i%len(colorlist)],marker='d')
    if not legtitle:
        legtitle=r'$z_0$ used for simulations'
    if varname=='rho':
        plt.legend(title=legtitle,loc='lower center')
    elif varname=='s':
        plt.legend(title=legtitle,loc='upper left')
    elif varname=='chisq':
        plt.legend(title=legtitle,loc='upper center')
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

#--------------------------
def z0test_Clcomp(perrors=np.array([1,10,20,30,50]),fidz0=.7,plotdir='output/zdisttest/plots/',plotISWgalratio=True,refval=0.7):
    z0vals=z0test_getz0vals(perrors,fidz0)#for labels
    iswind=0
    bins=z0test_get_binmaps(perrors,fidz0)#includes ISW at 0, others in same order as z0
    cldat=z0test_get_Cl(perrors=perrors,fid=fidz0)
    l=np.arange(cldat.Nell)
    z0cols=['#d7191c',
            '#fdae61',
            '#636363',##ffffbf',
            '#abd9e9',
            '#2c7bb6']
    
    clscaling=1#2*l+1.#l*(l+1.)/(2*np.pi)
    
    #set up actual plot
    fig=plt.figure(0)
    fig.subplots_adjust(bottom=.15)
    fig.subplots_adjust(left=.15)
    ax=plt.subplot()
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                  ax.get_yticklabels()+ax.get_xticklabels()):
        item.set_fontsize(22)

    #plt.title(r'Comparing $C_{{\ell}}$ of galaxies, ISW')
    #plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}/2\pi$')
    plt.ylabel(r'$C_{{\ell}}$')
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((1,30))
    if plotISWgalratio:
        plt.ylim((1.e-11,1.))
    else:
        plt.ylim((1.e-10,1.e-5))
    plt.yscale('log')


    for i in xrange(z0vals.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            xiswind=cldat.crossinds[i+1,0]
            plt.plot(l,np.fabs(cldat.cl[autoind,:])*clscaling,color=z0cols[i%len(z0cols)],linestyle='-',label=r'$z_0={0:0.2f}$'.format(z0vals[i]),linewidth=2)
            plt.plot(l,np.fabs(cldat.cl[xiswind,:])*clscaling,color=z0cols[i%len(z0cols)],linestyle='--',linewidth=2)
            if plotISWgalratio:
                plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color=z0cols[i%len(z0cols)],linestyle=':')
    #dummy lines for legend
    line1,=plt.plot(np.array([]),np.array([]),color='black',linestyle='-',label='gal-gal',linewidth=2)
    line2,=plt.plot(np.array([]),np.array([]),color='black',linestyle='--',label='ISW-gal',linewidth=2)
    if plotISWgalratio:
        line3,=plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color='black',linestyle=':',label='ISW-gal/gal-gal')
    plt.legend(loc='center right',fontsize=16,ncol=2)

    plotname='z0test_cl_compare'
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

    #make ratio plot
    fig=plt.figure(1)
    fig.subplots_adjust(bottom=.16)
    fig.subplots_adjust(left=.13)
    fig.subplots_adjust(right=.95)
    ax=plt.subplot()
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                  ax.get_yticklabels()+ax.get_xticklabels()):
        item.set_fontsize(30)
    plt.ylabel(r'$C_{{\ell}}(z_0)/C_{{\ell}}(z_0={0:0.1f})$'.format(refval))
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((1,30))
    plt.ylim((0,8))
    refind=np.argwhere(z0vals==refval)[0][0]#where in z0vals is refval?
    refautoind=cldat.crossinds[refind+1,refind+1]
    for i in xrange(z0vals.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            plt.plot(l,np.fabs(cldat.cl[autoind,:]/cldat.cl[refautoind,:]),color=z0cols[i%len(z0cols)],linestyle='-',label='$z_0={0:0.2f}$'.format(z0vals[i]),linewidth=2)
    plt.legend(loc='center right',fontsize=24,ncol=2)
    plotname=plotname+'_ratio'
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
    
#--------------------------------------------------------------------
# bztest funcs
#--------------------------------------------------------------------
def bztest_get_binmaps(b2vals=np.array([0.,.01,.1,.5]),fid=0,z0=0.7,includeisw=True):
    addfid=fid not in b2vals
    maptags=['eucz{0:04d}_b2{1:04d}'.format(int(z0*10000),int(b2*1000)) for b2 in b2vals]
    surveys=[mdu.get_Euclidlike_SurveyType(z0=z0,onebin=True,tag=maptags[i],b2=b2vals[i]) for i in xrange(b2vals.size)]
    if addfid:
        maptags.append('eucz{0:04d}_b2{1:04d}'.format(int(z0*10000),int(fid*1000)))
        surveys.append(mdu.get_Euclidlike_SurveyType(z0=z0,onebin=True,tag=maptags[-1],b2=fid) )
    bins=[s.binmaps[0] for s in surveys] #surveys all just have one bin
    if includeisw:
        iswmaptype=mdu.get_fullISW_MapType(zmax=15)
        iswbins=iswmaptype.binmaps
        bins=iswbins+bins
    return bins

def bztest_get_Cl(justread=True,b2vals=np.array([0.,.01,.1,.5]),fid=0,z0=0.7):
    bins=bztest_get_binmaps(b2vals,fid,z0)
    pairs=[] #set up pairs for cross corr (autocorr automatic)
    #pair up isw and each LSS maps, but not lss maps together 
    for b in bins:
        if b.isGal:
            pairs.append((b.typetag,'isw'))
            
    zmax=max(m.zmax for m in bins)
    rundat = clu.ClRunData(tag='bztest',rundir='output/zdisttest/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    return gcc.getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)

#--------------------------------------------------------------------
# simz0 are z0 values for which simulated maps were generated
# recz0 are z0 values used in the Cl for estimator
# returns 2d Nsimz0xNrecz0 list of RecData objects
def bztest_get_recgrid(simb2=np.array([0.,.01,.1,.5]),recb2=np.array([0.,.01,.1,.5])):
    z0=.7
    simmaptags=['eucz{0:04d}_b2{1:04d}'.format(int(z0*10000),int(b2*1000)) for b2 in simb2]
    recmaptags=['eucz{0:04d}_b2{1:04d}'.format(int(z0*10000),int(b2*1000)) for b2 in recb2]
    recgrid=[]
    Nsim=simb2.size
    Nrec=recb2.size
    for ns in xrange(Nsim):
        reclist=[]
        for nr in xrange(Nrec):
            #print 'SIM: '+simmaptags[ns]+'_bin0  REC: '+recmaptags[nr]+'_bin0'
            recdat=au.RecData(includeglm=[simmaptags[ns]+'_bin0'],includecl=[recmaptags[nr]+'_bin0'],inmaptag=simmaptags[ns],rectag=recmaptags[nr])
            reclist.append(recdat)
        recgrid.append(reclist)
    return recgrid

# will output rhodat in len(simb2)xlen(recb2) array
# if simb2 and recb2 are passed as arrays, use those as the b2 vals
def bztest_get_rhoexp(simb2=np.array([0.,.01,.1,.5]),recb2=np.array([0.,.01,.1,.5,1.,2.,5.,10.]),overwrite=False,saverho=True,doplot=False,varname='rho',filetag='',fitbias=True):
    if saverho:
        outdir='output/zdisttest/plots/'
        if filetag:
            filetagstr='_'+filetag
        else:
            filetagstr=filetag
        datfile='b2test_{0:s}exp{1:s}.dat'.format(varname,filetagstr)
        if not overwrite and os.path.isfile(outdir+datfile):#file exists
            print "Reading data file:",datfile
            x=np.loadtxt(outdir+datfile)
            insimb2=x[1:,0] #row labels
            inrecb2=x[0,1:] #column labels
            if not np.all(insimb2==simb2) or not np.all(inrecb2==recb2):
                print "WARNING, input b2 lists don't match requested."
            else:
                rhoarray=x[1:,1:] #rho data
                return rhoarray
        else:
            print "Writing to data file:",datfile
    
    Nsim=simb2.size
    Nrec=recb2.size
    recgrid=bztest_get_recgrid(simb2,recb2) #Nsim x Nrec WRITE THIS
    simcldat=bztest_get_Cl(True,simb2) 
    reccldat=bztest_get_Cl(True,recb2)

    rhoarray=np.zeros((Nsim,Nrec))
    for ns in xrange(Nsim):
        for nr in xrange(Nrec):
            rhoarray[ns,nr]=au.compute_rho_fromcl(simcldat,recgrid[ns][nr],reccldat=reccldat,varname=varname,fitbias=fitbias)

    #print rhoarray
    if saverho:
        #write to file, 
        f=open(outdir+datfile,'w')
        f.write('{0:9.6f} '.format(0.)+''.join(['{0:9.6f} '.format(b2r) for b2r in recb2])+'\n')
        for ns in xrange(Nsim):
            f.write('{0:9.6f} '.format(simb2[ns])+''.join(['{0:9.6f} '.format(rhoarray[ns,nr]) for nr in xrange(Nrec)])+'\n')
        f.close()
    if doplot:
        bztest_rhoexpplot(simb2,recb2,rhoarray,varname,filetag) 
        
    return rhoarray

def bztest_rhoexpplot(simb2,recb2,rhogrid,varname='rho',outtag='',outname='',plotdir='output/zdisttest/plots/'):
    zdisttest_rhoexpplot(rhogrid,simb2,recb2,varname='b2',statname=varname,outtag=outtag,outname=outname,plotdir=plotdir)

    
#simb2 fixed at fidb2, varying recb2.
def bztest_onesim_plot(fidb2=0.5,recb2=np.array([0.,.01,.1,.5,1.,2.,5.,10.]),varname='rho',plotdir='output/zdisttest/plots/',outtag='onesim',outname='',dohatch=False,fitbias=True,biascomp=True,overwritedat=True):
    #recb2=np.array([0,.5,1.])#revert
    simb2=np.array([fidb2])
    if not fitbias:
        outtag=outtag+'_nob0fit'
    rhogrid=bztest_get_rhoexp(simb2=simb2,recb2=recb2,overwrite=overwritedat,saverho=True,doplot=False,varname=varname,filetag=outtag,fitbias=fitbias) #should be 1xNrec
    if biascomp:
        nofitrhogrid=bztest_get_rhoexp(simb2=simb2,recb2=recb2,overwrite=overwritedat,saverho=True,doplot=False,varname=varname,filetag=outtag+'_nob0fit',fitbias=False)
        outtag=outtag+'_biascomp'
    
    Nsim=simb2.size
    Nrec=recb2.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='bztest_'+varname+'_exp'+outtag+'.pdf'

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.15)
    fig.subplots_adjust(right=.95)
    ax1=plt.subplot(1,1,1)

    ax1.axhline(0,color='grey',linestyle=':')
    ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    ax1.set_xlabel(r"$b_2$ used in ISW reconstruction")
    if varname=='rho':
        bestval=np.max(rhogrid)
        varstr=r'\rho'
        if dohatch:
            aloc=(.25,.7)
        else:
            aloc=(.25,.7)
        linthreshy=1.e-6
        ax1.set_ylim((-1.e-3,linthreshy))
        bestval=np.max(rhogrid)
        arrowystart= .5*linthreshy
        arrowstr=r'$\downarrow$'
        # arrowdy=-.2*linthreshy
        # arrowheadlen=.25*linthreshy
        # arrowlabely=arrowystart+arrowdy-arrowheadlen
    elif varname=='s':
        bestval=np.min(rhogrid)
        varstr='s'
        if dohatch:
            aloc=(.25,.4)
        else:
            aloc=(.25,.4)
        linthreshy=1.e-4
        ax1.set_ylim((-1*linthreshy,10.))
        bestval=np.min(rhogrid)
        arrowystart= -.5*linthreshy
        arrowstr=r'$\uparrow$'
        # arrowdy=.2*linthreshy
        # arrowheadlen=.25*linthreshy
        # arrowlabely=arrowystart
    ax1.set_ylabel(r'$\left[\langle {0:s} \rangle -\langle {0:s} \rangle_{{\rm match}} \right]/\langle {0:s} \rangle_{{\rm match}}$'.format(varstr))

    if biascomp:
        ax1.plot(recb2,nofitrhogrid[0,:]/bestval-1,marker='d',color='#969696')
        if varname=='s': #skipping b0 fitting has no impact on rho
            plt.annotate(r'no bias fitting',color='#969696',xy=(.95,.97),horizontalalignment='right',verticalalignment='top',fontsize=12,xycoords='axes fraction')
    ax1.plot(recb2,rhogrid[0,:]/bestval-1,marker='o',color='#2c7bb6')

    plt.annotate(r'True (sim.) $b_2={0:0.1f}$'.format(fidb2)+'\n'+r'$\langle {0:s}\rangle_{{\rm match}}={1:0.3f}$'.format(varstr,bestval),xy=aloc,horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction')

    #put arrow above or below fiducial point
    #ax1.arrow(fidb2,arrowystart,0,arrowdy,color='black',head_width=.07,head_length=arrowheadlen)
    #plt.annotate(r'match',xy=(fidb2+.1,arrowlabely),horizontalalignment='left',verticalalignment='bottom',fontsize=12)
    plt.plot(fidb2,arrowystart,color='black',marker=arrowstr,markersize=16)
    plt.annotate(r'match',xy=(fidb2+.2,arrowystart),horizontalalignment='left',verticalalignment='center',fontsize=12)
    
    ax1.set_yscale('symlog',linthreshy=linthreshy)
    linthreshx=.01
    ax1.set_xscale('symlog',linthreshx=linthreshx)
    ax1.set_xlim((-.001,13))

    if dohatch:    #hash fills in where there is a linear scale
        ymin,ymax=ax1.get_ylim()
        y=np.arange(-linthreshx,linthreshx,.1*linthreshx)
        ax1.fill_between(y,ymin,ymax,color='none',edgecolor='grey',hatch='/',linewidth=0)
        xmin,xmax=ax1.get_xlim()
        x=np.arange(xmin,xmax,.01)
        ax1.fill_between(x,-linthreshy,linthreshy,color='none',edgecolor='grey',hatch='/',linewidth=0)
    
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname=varname+'test_'+statname+'_exp'+outtag+'.pdf' #[statname undefined; Though outname won't be "False" unless specifically passed as such. NJW 160606]
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

#simb2 fixed at fidb2, varying recb2.
def bztest_onerec_plot(fidrecb2=0.,simb2=np.array([0.,.01,.1,.5,1.,2.,5.,10.]),varname='rho',plotdir='output/zdisttest/plots/',outtag='onerec',outname=''):
    recb2=np.array([fidrecb2])
    rhogrid=bztest_get_rhoexp(simb2=simb2,recb2=recb2,overwrite=True,saverho=True,doplot=False,varname=varname,filetag=outtag) #should be NsimxNrec
    #print rhogrid
    
    Nsim=simb2.size
    Nrec=recb2.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='bztest_'+varname+'_exp'+outtag+'.pdf'

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.2)
    ax1=plt.subplot(1,1,1)
    #ax1.yaxis.grid(True)
    ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    ax1.set_xlabel(r"True (simulation) $b_2$ ")
    ax1.axhline(0,color='grey',linestyle=':')
        
    wherematch=np.argwhere(simb2==fidrecb2)[0][0]
    matchval=rhogrid[wherematch,0]
    if varname=='rho':
        ax1.set_ylabel(r'$\left[\langle \rho \rangle-\langle \rho \rangle_{{\rm match}}\right]/\langle \rho \rangle_{{\rm match}}$')
        #ax1.set_yscale('linear')
        ax1.set_ylim((-2.5e-2,2.e-2))
    elif varname=='s':
        ax1.set_ylabel(r'$\left[\langle s \rangle - \langle s \rangle_{{\rm match}}\right] /\langle s \rangle_{{\rm match}}$')
        ax1.set_ylim((-.04,.12))

    if varname=='rho':
        varstr=r'\rho'
        ax1.plot(simb2,rhogrid[:,0]/matchval -1,marker='d')
        annoteloc=(.55,.4)
    elif varname=='s':
        varstr='s'
        ax1.plot(simb2,rhogrid[:,0]/matchval -1. ,marker='d')
        annoteloc=(.55,.9)
    plt.annotate(r'$b_2={0:0.0f}$ used ISW rec.'.format(fidrecb2)+'\n'+r'$\langle {0:s}\rangle_{{\rm match}}={1:0.3f}$'.format(varstr,matchval),xy=annoteloc,horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction')

    #put arrow above or below fiducial point
    ax1.arrow(fidz0,arrowystart,0,arrowdy,color='black',head_width=.015,head_length=arrowheadlen)
    plt.annotate(r'match',xy=(fidz0+.02,arrowlabely),horizontalalignment='left',verticalalignment='bottom',fontsize=12)

    linthreshx=.01
    ax1.set_xscale('symlog',linthreshx=linthreshx)
    ax1.set_xlim((-.001,13))
    #hash fills in where there is a linear scale
    ymin,ymax=ax1.get_ylim()
    y=np.arange(-linthreshx,linthreshx,.01*linthreshx)
    ax1.fill_between(y,ymin,ymax,color='none',edgecolor='grey',hatch='/',linewidth=0)
    
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname=varname+'test_'+statname+'_exp'+outtag+'.pdf'
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()    
#--------------------------
def bztest_Clcomp(b2vals=np.array([0.,.01,.1,.5,1.,2.,5.,10.]),plotdir='output/zdisttest/plots/',plotISWgalratio=True,refval=0.5):
    iswind=0
    bins=bztest_get_binmaps(b2vals)#includes ISW at 0, others in same order as z0
    cldat=bztest_get_Cl(b2vals=b2vals)
    l=np.arange(cldat.Nell)
    b2cols=['#d7191c',
            '#fdae61',
            '#636363',##ffffbf',
            '#abd9e9',
            '#2c7bb6']
    
    clscaling=1#2*l+1 #l*(l+1.)/(2*np.pi)
    
    fig=plt.figure(0)
    fig.subplots_adjust(bottom=.15)
    fig.subplots_adjust(left=.15)
    ax=plt.subplot()
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                  ax.get_yticklabels()+ax.get_xticklabels()):
        item.set_fontsize(22)
    
    #plt.title(r'Comparing $C_{{\ell}}$ of galaxies, ISW')
    plt.ylabel(r'$C_{{\ell}}$')
    #plt.ylabel(r'$(2\ell+1)C_{{\ell}}$')
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((1,30))
    if plotISWgalratio:
        plt.ylim((1.e-11,1.))
    else:
        plt.ylim((1.e-10,5.e-5))
    plt.yscale('log')

    for i in xrange(b2vals.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            xiswind=cldat.crossinds[i+1,0]
            plt.plot(l,np.fabs(cldat.cl[autoind,:])*clscaling,color=b2cols[i%len(b2cols)],linestyle='-',label='$b_2={0:0.2f}$'.format(b2vals[i]),linewidth=2)
            plt.plot(l,np.fabs(cldat.cl[xiswind,:])*clscaling,color=b2cols[i%len(b2cols)],linestyle='--',linewidth=2)
            if plotISWgalratio:
                plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color=b2cols[i%len(b2cols)],linestyle=':',linewidth=2)
    #dummy lines for legend
    line1,=plt.plot(np.array([]),np.array([]),color='black',linestyle='-',label='gal-gal',linewidth=2)
    line2,=plt.plot(np.array([]),np.array([]),color='black',linestyle='--',label='ISW-gal',linewidth=2)
    if plotISWgalratio:
        line3,=plt.plot(np.array([]),np.array([]),color='black',linestyle=':',label='ISW-gal/gal-gal',linewidth=2)
    plt.legend(loc='center right',fontsize=16,ncol=2)
    plotname='bztest_cl_compare'
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

    #make ratio plot
    fig=plt.figure(1)
    fig.subplots_adjust(bottom=.16)
    fig.subplots_adjust(left=.17)
    fig.subplots_adjust(right=.95)
    ax=plt.subplot()
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                  ax.get_yticklabels()+ax.get_xticklabels()):
        item.set_fontsize(30)
    plt.ylabel(r'$C_{{\ell}}(b_2)/C_{{\ell}}(b_2={0:0.1f})$ '.format(refval))
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((1,30))
    #plt.ylim((0,10))
    refind=np.argwhere(b2vals==refval)[0][0]#where in b2vals is refval?
    refautoind=cldat.crossinds[refind+1,refind+1]
    for i in xrange(b2vals.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            plt.plot(l,np.fabs(cldat.cl[autoind,:]/cldat.cl[refautoind,:]),color=b2cols[i%len(b2cols)],linestyle='-',label='$b_2={0:0.1f}$'.format(b2vals[i]),linewidth=2)
    plt.legend(loc='center right',fontsize=24,ncol=2)
    plotname=plotname+'_ratio'
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
    
#================================================================
# catztest - test how knowledge of fractin of catastrophic photo-z errors
#             impact recosntruction
#================================================================
def catz_get_maptypes(badfracs=np.array([1.e-3,1.e-2,.1]),Nbins=3,z0=.7,sigz=.05,includeISW=True):
    maintag='euc{0:d}bincatz'.format(Nbins)
    zedges=bintest_get_finest_zedges(finestN=Nbins,z0=z0)
    maptypes=[]
    if includeISW:
        iswmaptype=mdu.get_fullISW_MapType(zmax=15)
        maptypes.append(iswmaptype)
    for x in badfracs:
        tag=maintag+'{0:.0e}'.format(x)
        eucmapx=mdu.get_Euclidlike_SurveyType(sigz=sigz,z0=z0,tag=tag,zedges=zedges,b0=1.,b2=0,fracbadz=x)
        maptypes.append(eucmapx)
    return maptypes

def catz_get_binmaps(badfracs=np.array([1.e-3,1.e-2,.1]),Nbins=3,z0=.7,sigz=.05,includeISW=True):
    maptypes=catz_get_maptypes(badfracs,Nbins,z0,sigz,includeISW)
    binmaps,bintags=mp.get_binmaplist(maptypes)
    return binmaps

def catz_get_Cl(badfracs=np.array([1.e-3,1.e-2,.1]),Nbins=3,z0=.7,sigz=.05,justread=True):
    maptypes=catz_get_maptypes(badfracs,Nbins,z0,sigz,includeISW=True)
    binmaps,bintags=mp.get_binmaplist(maptypes)
    pairs=[]
    for mt in maptypes:
        if mt.isGal:
            pairs.append((mt.tag,'isw'))
    zmax=max(m.zmax for m in binmaps)
    rundat = clu.ClRunData(tag='catztest_{0:d}bins'.format(Nbins),rundir='output/zdisttest/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    return gcc.getCl(binmaps,rundat,dopairs=pairs,DoNotOverwrite=justread)
#------------------------------------------------
def catz_get_recgrid(simfracs,recfracs,Nbins=3,z0=.7,sigz=.05):
    maintag='euc{0:d}bincatz'.format(Nbins)
    simtypetags=[maintag+'{0:.0e}'.format(x) for x in simfracs]
    rectypetags=[maintag+'{0:.0e}'.format(x) for x in recfracs]
    recgrid=[]
    Nsim=simfracs.size
    Nrec=recfracs.size
    for ns in xrange(Nsim):
        #print 'ns=',ns
        reclist=[]
        for nr in xrange(Nrec):
            #print 'nr=',nr
            #print 'SIM: '+simmaptags[ns]+'_bin0  REC: '+recmaptags[nr]+'_bin0'
            inglm=[simtypetags[ns]+'_bin{0:d}'.format(i) for i in xrange(Nbins)]
            incl=[rectypetags[nr]+'_bin{0:d}'.format(i) for i in xrange(Nbins)]
            #print 'inglm',inglm
            #print 'incl',incl
            recdat=au.RecData(includeglm=inglm,includecl=incl,inmaptag=simtypetags[ns],rectag=rectypetags[nr])
            #print 'recdat.Nmap',recdat.Nmap
            reclist.append(recdat)
        recgrid.append(reclist)
    return recgrid
#------------------------------------------------
def catz_rhoexpplot(simfracs,recfracs,rhogrid,varname='rho',Nbins=3,outtag='',outname='',plotdir='output/zdisttest/plots/'):
    if not outtag:
        outtag='{0:d}bins'.format(Nbins)
    zdisttest_rhoexpplot(rhogrid,simfracs,recfracs,varname='catzfrac',statname=varname,outtag=outtag,outname=outname,plotdir='output/zdisttest/plots/')
#--------------------------------------------------------------------    
def catz_windowtest(badfracs,Nbins=3): #check that my modeling of catastrophic photo-zs works
    maptypes=catz_get_maptypes(badfracs=badfracs,Nbins=Nbins,includeISW=False)
    plotdir='output/zdisttest/plots/'
    Nfrac=len(badfracs)
    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    for i in xrange(Nfrac):
        plt.figure(i)
        plt.rcParams['axes.linewidth'] =2
        ax=plt.subplot()
        ax.set_yticklabels([])
        plt.title(r'Cat. z frac {0:.0e}: z distributions'.format(badfracs[i]))
        plt.xlabel('Redshift z',fontsize=20)
        plt.ylabel('Source distribution (arbitrary units)',fontsize=20)
        #plt.ylim(0,.7)
        plt.xlim(0,zmax)
        ax.tick_params(axis='x', labelsize=18)
        bins=maptypes[i].binmaps
        Nbins=len(bins)
        for n in xrange(Nbins):
            m=bins[n]
            #wgrid=m.window(zgrid)*m.nbar/1.e9 #unnorm
            wgrid=m.window(zgrid)/1.e9 #normalized
            colstr=colors[n%len(colors)]
            plt.plot(zgrid,wgrid,color=colstr,linestyle='-',linewidth=2)
        plotname='catztestwindow_{1:d}bins_catz{0:.0e}'.format(badfracs[i],Nbins)
        outname=plotdir+plotname+'.pdf'
        print 'saving',outname
        plt.savefig(outname)
        plt.close()
#--------------------------------------------------------------------     
def catz_get_rhoexp(simfracs=np.array([]),recfracs=np.array([]),badfracs=np.array([]),Nbins=3,z0=.7,sigz=.05,overwrite=False,saverho=True,doplot=False,varname='rho',filetag='',plotdir='output/zdisttest/plots/',fitbias=True):
    #print '******in get rhoexp, Nbins=',Nbins
    if not simfracs.size:
        simfracs=badfracs
    if not recfracs.size:
        recfracs=badfracs
    if not badfracs.size:
        badfracs=np.union1d(simfracs,recfracs)
    if saverho:
        outdir=plotdir
        if filetag:
            filetagstr='_'+filetag
        else:
            filetagstr=filetag
        datfile='catztest{2:d}bin_{0:s}exp{1:s}.dat'.format(varname,filetagstr,Nbins)
        if not overwrite and os.path.isfile(outdir+datfile):#file exists
            print "Reading data file:",datfile
            x=np.loadtxt(outdir+datfile)
            insimfracs=x[1:,0] #row labels
            inrecfracs=x[0,1:] #column labels
            if not np.all(insimfracs==simfracs) or not np.all(inrecfracs==recfracs):
                print "WARNING, input frac lists don't match requested."
            else:
                rhoarray=x[1:,1:] #rho data
                return rhoarray
        else:
            print "Writing to data file:",datfile
    
    Nsim=simfracs.size
    Nrec=recfracs.size
    recgrid=catz_get_recgrid(simfracs,recfracs,Nbins) #Nsim x Nrec
    cldat=catz_get_Cl(badfracs,Nbins,z0,sigz,True) #read cl already containing all fracs's

    rhoarray=np.zeros((Nsim,Nrec))
    for ns in xrange(Nsim):
        for nr in xrange(Nrec):
            #print 'simfrac ',simfracs[ns],'; recfrac ',recfracs[nr]
            rhoarray[ns,nr]=au.compute_rho_fromcl(cldat,recgrid[ns][nr],reccldat=cldat,varname=varname,fitbias=fitbias)

    if saverho:
        #write to file, 
        f=open(outdir+datfile,'w')
        f.write('{0:15.6f} '.format(0.)+''.join(['{0:15.2e} '.format(fr) for fr in recfracs])+'\n')
        for ns in xrange(Nsim):
            f.write('{0:15.2e} '.format(simfracs[ns])+''.join(['{0:15.12f} '.format(rhoarray[ns,nr]) for nr in xrange(Nrec)])+'\n')
        f.close()
    if doplot:
        catz_rhoexpplot(simfracs,recfracs,rhoarray,varname,plotdir=plotdir,Nbins=Nbins) 
    return rhoarray

#simf fixed at fidf, varying recf. where f=fraction of catastrophic photo-z's
def catztest_onerec_plot(fidrecf=0.,simf=np.array([0.,5.e-4,1.e-3,2.e-3,.01,.02,.1,.2]),varname='rho',plotdir='output/zdisttest/plots/',outtag='onerec',outname='',Nbins=1):
    dolog=True
    recf=np.array([fidrecf])
    rhogrid=catz_get_rhoexp(simfracs=simf,recfracs=recf,overwrite=True,saverho=True,doplot=False,varname=varname,filetag=outtag,Nbins=Nbins) #should be NsimxNrec
    #print rhogrid
    
    Nsim=simf.size
    Nrec=recf.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='catztest{0:d}bin_{1:s}_exp{2:s}.pdf'.format(Nbins,varname,outtag)

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.2)
    ax1=plt.subplot(1,1,1)

    xmax=.5
    linthreshx=1.e-4
    xmin=-.1*linthreshx
    ax1.set_xscale('symlog',linthreshx=linthreshx)
    ax1.set_xlim((xmin,xmax))  
    #ax1.yaxis.grid(True)
    ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
    if dolog:
        if varname=='rho':
            linthreshy=1.e-4
        elif varname=='s':
            linthreshy=1.e-3
        ax1.set_yscale('symlog',linthreshy=linthreshy)
        x=np.arange(xmin,xmax,.01)
        ax1.fill_between(x,-linthreshy,linthreshy,color='none',edgecolor='grey',hatch='/',linewidth=0)
    ax1.axhline(0,color='grey',linestyle=':')

    
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    ax1.set_xlabel(r"True (simulation) $f_{\rm cat}$ ")
        
    wherematch=np.argwhere(simf==fidrecf)[0][0]
    matchval=rhogrid[wherematch,0]
    if varname=='rho':
        ax1.set_ylabel(r'$\left[\langle \rho \rangle-\langle \rho \rangle_{{\rm match}}\right]/\langle \rho \rangle_{{\rm match}}$')
        varstr=r'\rho'
        ax1.plot(simf,rhogrid[:,0]/matchval -1,marker='d')
        annoteloc=(.47,.3)
        #ax1.set_ylim((1.e-6,1.e-3))
    elif varname=='s':
        ax1.set_ylabel(r'$\left[\langle s \rangle - \langle s \rangle_{{\rm match}}\right] /\langle s \rangle_{{\rm match}}$')
        #ax1.set_yscale('log')
        varstr='s'
        ax1.plot(simf,rhogrid[:,0]/matchval -1. ,marker='d')
        annoteloc=(.5,.9)
    #hash fills in where there is a linear scale
    ymin,ymax=ax1.get_ylim()
    y=np.arange(-linthreshx,linthreshx,.01*linthreshx)
    ax1.fill_between(y,ymin,ymax,color='none',edgecolor='grey',hatch='/',linewidth=0)
        
    plt.annotate(r'$f_{{\rm cat}}={0:0.0f}$ used ISW rec.'.format(fidrecf)+'\n'+r'$\langle {0:s}\rangle_{{\rm match}}={1:0.3f}$'.format(varstr,matchval),xy=annoteloc,horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname=varname+'test_'+statname+'_exp'+outtag+'.pdf'
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

def catztest_onesim_plot(fidf=0.02,recf=np.array([0.,5.e-4,1.e-3,2.e-3,.01,.02,.1,.2]),varname='rho',plotdir='output/zdisttest/plots/',outtag='onesim',outname='',Nbins=1,secondfidf=-1,dohatch=False,biascomp=True,overwritedat=True): #working here
    print '****Nbins',Nbins
    #if secondfidf>0, will plot a second line
    colors=['#a6611a','#2c7bb6']
    if secondfidf>0:
        simf=np.array([fidf,secondfidf])
        
    else:
        simf=np.array([fidf])
    Nsim=simf.size
    Nrec=recf.size
    wherematch=[np.argwhere(recf==sf)[0][0] for sf in simf] #Nsim len list
        
    rhogrid=catz_get_rhoexp(simfracs=simf,recfracs=recf,overwrite=True,saverho=True,doplot=False,varname=varname,filetag=outtag,Nbins=Nbins,fitbias=True) #should be NsimxNrec
    matchval=[rhogrid[i,wherematch[i]] for i in xrange(Nsim)]#Nsim len list
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='catztest{0:d}bin_{1:s}_exp{2:s}.pdf'.format(Nbins,varname,outtag)

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.15)
    fig.subplots_adjust(right=.95)
    
    ax1=plt.subplot(1,1,1)
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)
    xmax=.5
    linthreshx=1.e-3
    xmin=-.1*linthreshx
    ax1.set_xscale('symlog',linthreshx=linthreshx)
    ax1.set_xlim((xmin,xmax))
    ax1.set_xlabel(r"$x$ used in ISW rec.")
    #ax1.yaxis.grid(True)
    ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
    ax1.axhline(0,color='grey',linestyle=':')
    if varname=='rho':
        linthreshy=1.e-6
        varstr=r'\rho'
        if dohatch:
            annoteloc=[(.55,.3)]
        else:
            annoteloc=[(.25,.83),(.3,.45)]
        ax1.set_ylim((-1,linthreshy))
        arrowystart= .55*linthreshy
        arrowstr=r'$\downarrow$'
    elif varname=='s':
        linthreshy=1.e-4
        varstr='s'
        if dohatch:
            annoteloc=[(.55,.9)]
        else:
            annoteloc=[(.25,.6),(.3,.95)]
        ax1.set_ylim((-1*linthreshy,100.))
        arrowystart= -.55*linthreshy
        arrowstr=r'$\uparrow$'
    ax1.set_yscale('symlog',linthreshy=linthreshy)
    ax1.set_ylabel(r'$\left[\langle {0:s} \rangle -\langle {0:s} \rangle_{{\rm match}} \right]/\langle {0:s} \rangle_{{\rm match}}$'.format(varstr))
    
    if dohatch: #hatched shading where axis has linear scaling
        x=np.arange(xmin,xmax,.01)
        ax1.fill_between(x,-linthreshy,linthreshy,color='none',edgecolor='grey',hatch='/',linewidth=0)
        ax1.axhline(0,color='grey',linestyle=':')
        ymin,ymax=ax1.get_ylim()
        y=np.arange(-linthreshx,linthreshx,.01*linthreshx)
        ax1.fill_between(y,ymin,ymax,color='none',edgecolor='grey',hatch='/',linewidth=0)
    if biascomp:
        nofitrhogrid=catz_get_rhoexp(simfracs=simf,recfracs=recf,overwrite=overwritedat,saverho=True,doplot=False,varname=varname,filetag=outtag[1:]+'_nob0fit',fitbias=False,Nbins=Nbins)
        outtag=outtag+'_biascomp'
        for i in xrange(Nsim):
            ax1.plot(recf,nofitrhogrid[i,:]/matchval[i]-1,marker='d',color='#969696')
        if varname=='s': #skipping b0 fitting has no impact on rho
            plt.annotate(r'no bias fitting',color='#969696',xy=(.35,.3),horizontalalignment='right',verticalalignment='top',fontsize=12,xycoords='axes fraction')
    #plot data
    for i in xrange(Nsim):
        ax1.plot(recf,rhogrid[i,:]/matchval[i] -1,marker='o',color=colors[i])
        #put arrow above or below fiducial point
        plt.plot(simf[i],arrowystart,color='black',marker=arrowstr,markersize=16)
        plt.annotate(r'match',xy=(simf[i]*1.3,arrowystart),horizontalalignment='left',verticalalignment='center',fontsize=12)
        
        plt.annotate(r'True (sim.) $x={0:0.2f}$'.format(simf[i])+'\n'+r'$\langle {0:s}\rangle_{{\rm match}}={1:0.3f}$'.format(varstr,matchval[i]),xy=annoteloc[i],horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction',color=colors[i])
    
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()   
#--------------------------------------------------------------------     
def catz_Clcomp(badfracs=np.array([0.,1.e-3,5.e-3,1.e-2,2.e-2,5.e-2,.1,.2]),Nbins=1,plotdir='output/zdisttest/plots/',plotISWgalratio=True,refval=0.01):
    iswind=0
    bins=catz_get_binmaps(badfracs,Nbins)#includes ISW at 0, others in same order as z0
    cldat=catz_get_Cl(badfracs,Nbins)
    l=np.arange(cldat.Nell)
    clscaling=1#2*l+1.#l*(l+1.)/(2*np.pi)
    # fcols=['#d53e4f',
    #        '#fc8d59',
    #        '#fee08b',
    #        '#e6f598',
    #        '#99d594',
    #        '#3288bd']
    fcols=['#d7191c',
            '#fdae61',
            '#636363',##ffffbf',
            '#abd9e9',
            '#2c7bb6']
    
    #set up actual plot
    fig=plt.figure(0)
    fig.subplots_adjust(bottom=.15)
    fig.subplots_adjust(left=.15)
    ax=plt.subplot()
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                  ax.get_yticklabels()+ax.get_xticklabels()):
        item.set_fontsize(22)
    #plt.title(r'Comparing $C_{{\ell}}$ of galaxies, ISW')
    #plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}/2\pi$')
    #plt.ylabel(r'$(2\ell+1)C_{{\ell}}$')
    plt.ylabel(r'$C_{{\ell}}$')
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((1,30))
    if plotISWgalratio:
        plt.ylim((1.e-11,1.))
    else:
        plt.ylim((1.e-10,5.e-6))
    plt.yscale('log')
    
    for i in xrange(badfracs.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            xiswind=cldat.crossinds[i+1,0]
            plt.plot(l,np.fabs(cldat.cl[autoind,:])*clscaling,color=fcols[i%len(fcols)],linestyle='-',label='$x={0:0.2f}$'.format(badfracs[i]),linewidth=2)
            plt.plot(l,np.fabs(cldat.cl[xiswind,:])*clscaling,color=fcols[i%len(fcols)],linestyle='--',linewidth=2)
            if plotISWgalratio:
                plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color=fcols[i%len(fcols)],linestyle=':',linewidth=2)
    #dummy lines for legend
    line1,=plt.plot(np.array([]),np.array([]),color='black',linestyle='-',label='gal-gal',linewidth=2)
    line2,=plt.plot(np.array([]),np.array([]),color='black',linestyle='--',label='ISW-gal',linewidth=2)
    if plotISWgalratio:
        line3,=plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color='black',linestyle=':',label='ISW-gal/gal-gal',linewidth=2)
    plt.legend(loc='center right',fontsize=16,ncol=2)
    plotname='catztest_cl_compare'
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

    #make ratio plot
    fig=plt.figure(1)
    fig.subplots_adjust(bottom=.16)
    fig.subplots_adjust(left=.13)
    fig.subplots_adjust(right=.95)
    ax=plt.subplot()
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                  ax.get_yticklabels()+ax.get_xticklabels()):
        item.set_fontsize(30)
    plt.ylabel(r'$C_{{\ell}}(x)/C_{{\ell}}(x={0:0.2f})$'.format(refval))
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((1,30))
    #plt.ylim((0,20))
    refind=np.argwhere(badfracs==refval)[0][0]#where in badfracs is refval?
    refautoind=cldat.crossinds[refind+1,refind+1]
    for i in xrange(badfracs.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            plt.plot(l,np.fabs(cldat.cl[autoind,:]/cldat.cl[refautoind,:]),color=fcols[i%len(fcols)],linestyle='-',label='$x={0:0.2f}$'.format(badfracs[i]),linewidth=2)
    plt.legend(loc='center right',fontsize=24,ncol=2)
    plotname=plotname+'_ratio'
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#================================================================
# lmintest - vary lmin used for reconstruction to study fsky effects
#================================================================
# use depthtest fiducial map for the Cl's and maps
# vary lmin from 1-20. start with just <rho> calc but also run on maps
#maybe make a similar plot as the caltest ones

# get bin maps: these are just the same as depthtest with one value of z0
# can specify via includeisw bool whether you just want LSS maps or not
# returns list of binmaps (len 1 or 2, depending on includeisw)
def lmintest_get_binmaps(z0=0.7,includeisw=True):
    return depthtest_get_binmaps(z0vals=np.array([z0]),includeisw=includeisw)

# get ClData object, same as for depthtest but getting a single LSS map
def lmintest_get_cl(includeISW=True,z0=.7):
    return depthtest_get_Cl(justread=True,z0vals=np.array([z0]))

# for the purposes of ISW reconstruction, get glmData object containing input
# map names and other info but no actual data for realizations
def lmintest_get_dummyglm(cldat=0,z0=.7):
    if not cldat:
        cldat=lmintest_get_cl(z0=z0)
    glmdat=gmc.get_glm(cldat,Nreal=0,matchClruntag=True)
    return glmdat

#given two arrays, one of unique lmin values, one of lmax,
# returns a pair of arrays set up so each index corresponds to a
# unique lmin/max combo,  containing all possible combos
def lmintest_get_lminmaxcombos(lminlist,lmaxlist):
    Nlmin=lminlist.size
    Nlmax=lmaxlist.size
    Ncombo=Nlmin*Nlmax
    outlmin=np.ones(Ncombo,dtype=int)
    outlmax=np.ones(Ncombo,dtype=int)
    counter=0
    for i in xrange(Nlmax):
        for j in xrange(Nlmin):
            outlmax[counter]=lmaxlist[i]
            outlmin[counter]=lminlist[j]
            #print counter,i,j,lmaxlist[i],lminlist[j]
            counter+=1
    return outlmin,outlmax
    
# get list of RecData objects, one for each set of lmin,lmax choices
#  lmaxlist should be either a single integer or the same size as lminlist
#  lmaxlist=int means use that val w all lmin, 
#  lmaxlist.size==lminlist.size means each index is a pair to use
#  if individual lmax==-1, that rec will use max available ell
def lmintest_get_reclist(lminlist=np.arange(1,20),lmaxlist=-1,z0=.7):
    galbin=lmintest_get_binmaps(z0=z0,includeisw=False)[0]#only one bin in list
    if type(lmaxlist)!=np.ndarray:
        lmaxlist=lmaxlist*np.ones(lminlist.size,dtype=int)
    elif lmaxlist.size!=lminlist.size:
        print 'WARNING: lmaxlist not the same size as lminlist! Setting =-1 (reclist)'
        lmaxlist=-1*np.ones(lminlist.size,dtype=int)
    reclist=[]
    for l in xrange(lminlist.size):
        bintag=galbin.tag
        includeglm=[bintag]
        inmaptag=bintag[:bintag.rfind('_bin0')]
        recdat=au.RecData(includeglm=includeglm,inmaptag=inmaptag,minl_forrec=lminlist[l],maxl_forrec=lmaxlist[l])
        reclist.append(recdat)
    return reclist

# gets theoretical calcs of <rho> (or <s>, depending on varname).
#   computes expectation value given lminlist and lmaxlist (see comments above
#       'lmintest_get_reclist' for info about these), writes to file if saverho
#   if saverho and overwrite, just writes to file no matter what
#              but not overwrite, if file exists w right lmin/max, reads
#                                 otherwise overwrites
def lmintest_get_rhoexp(lminlist=np.arange(1,20),lmaxlist=-1,z0=.7,overwrite=False,saverho=True,varname='rho',filetag='',plotdir='output/lmintest_plots/'):
    if type(lmaxlist)!=np.ndarray:
        lmaxlist=lmaxlist*np.ones(lminlist.size,dtype=int)
    elif lmaxlist.size!=lminlist.size:
        print 'WARNING: lmaxlist not the same size as lminlist! Setting =-1 (rhoexp)'
        lmaxlist=-1*np.ones(lminlist.size,dtype=int)
    if saverho:
        outdir=plotdir
        if filetag:
            filetagstr='_'+filetag
        else:
            filetagstr=filetag
        datfile='lmintest_{0:s}exp{1:s}.dat'.format(varname,filetagstr)
        if not overwrite and os.path.isfile(outdir+datfile):#file exists
            print "Reading data file:",datfile
            x=np.loadtxt(outdir+datfile,skiprows=1)
            #print x
            inlmin=x[:,0] #rows
            inlmax=x[:,1]
            if not np.all(inlmin==lminlist) or not np.all(inlmax==lmaxlist):
                print "WARNING, input lmin/max lists don't match requested."
            else:
                rhoarray=x[:,2] #rho data
                return rhoarray
        else:
            print "Writing to data file:",datfile

    Nlmin=lminlist.size
    reclist=lmintest_get_reclist(lminlist,lmaxlist,z0)
    cldat=lmintest_get_cl(z0=z0)
    rhoarray=np.zeros(Nlmin)
    for l in xrange(Nlmin):
        rhoarray[l]=au.compute_rho_fromcl(cldat,reclist[l],reccldat=cldat,varname=varname)
    if saverho:
        #write to file, 
        f=open(outdir+datfile,'w')
        f.write('{0:9s} {1:9s} {2:9s}\n'.format('reclmin','reclmax','rho'))
        for l in xrange(Nlmin):
            f.write('{0:9d} {1:9d} {2:9.6f}\n'.format(lminlist[l],lmaxlist[l],rhoarray[l]))
        f.close()
    return rhoarray

# do ISW reconstructions from existing galaxy maps for various lmin values
#  can pass lmaxlist of same length as lminlist if you want to look at
#  different lmin,lmax combos. If lmaxlist=int, fill in all of array w that val
#    individual lmax = -1 means use max available ell for rec
#  if domaps=False, wont redo isw rec, but stats like rho, s will be recalculated
def lmintest_iswrec(Nreal,lminlist=np.arange(1,20),lmaxlist=-1,domaps=True,z0=0.7):
    cldat=lmintest_get_cl(z0=z0)
    reclist=lmintest_get_reclist(lminlist,lmaxlist,z0)
    dummyglm=lmintest_get_dummyglm(cldat=cldat,z0=z0)
    au.doiswrec_formaps(dummyglm,cldat,Nreal,reclist=reclist,domaps=domaps)

# assuming reconstructions have already been done, along with rho or s calc,
# and assuming all lmin,lmax combos have the same number of realizations
#  read in the statistic (rho or s) info from files
#  returns a Nlmin x Nreal array of rho values
def lmintest_getrhodat_fromfiles(lminlist=np.arange(1,20),lmaxlist=-1,varname='rho'):
    if type(lmaxlist)!=np.ndarray:
        lmaxlist=lmaxlist*np.ones(lminlist.size,dtype=int)
    elif lmaxlist.size!=lminlist.size:
        print 'WARNING: lmaxlist not the same size as lminlist! Setting =-1 (fromfiles)'
        lmaxlist=-1*np.ones(lminlist.size,dtype=int)
    Nlmin=lminlist.size
    mapdir='output/depthtest/map_output/'
    files=[]
    for i in xrange(Nlmin):
        lminstr="-lmin{0:02d}".format(lminlist[i])
        lmaxstr=''
        if lmaxlist[i]>0:
            lmaxstr="-lmax{0:02d}".format(lmaxlist[i])
        files.append('iswREC.eucz07.fid.fullsky{0:s}{1:s}.depthtest.{2:s}.dat'.format(lminstr,lmaxstr,varname))
    rhogrid=np.array([au.read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

# get rho data for many lmin values
# assumes all lmin,lmax pairs have same number of realizations
# returns arrays of same size as lminlist of mean(rho) and std(rho) and int Nreal
def lmintest_get_rhodat(lminlist=np.arange(1,20),lmaxlist=-1):
    # Get 2d Nlmin x Nreal array of rho data, assuming it has already been calc'd
    rhodat=lmintest_getrhodat_fromfiles(lminlist,lmaxlist)
    Nlmin=rhodat.shape[0]
    Nreal=rhodat.shape[1]
    rhomean=np.zeros(Nlmin)
    rhostd=np.zeros(Nlmin)
    for i in xrange(Nlmin):
        rho=rhodat[i,:]
        rhomean[i]=np.mean(rho)
        rhostd[i]=np.std(rho)
    return rhomean,rhostd,Nreal

#plot <rho> (or <s>) for different lmin
#  if dodata==True, also get stat data from realizations, assuming they've
#                   already had maps generated and stats computed
#for now, set up assuming lmax=-1 (i.e., max possible l used in rec)
#  maybe in future could set up optionf or diff lmax in diff colors
# if lmaxlist is passed, each one gets all lmin vals, plot separate line
#  !!!note that this is different how lmaxlist is treated in other functions
# datlmin and datlmax are for if data is run on different set of lmin/max
#        but have the same form as is specific to this function

def lmintest_plot_rhoexp(lminlist=np.arange(1,30),lmaxlist=-1,z0=.7,overwrite=False,saverho=True,varname='rho',filetag='',plotdir='output/lmintest_plots/',dodata=False,datlmin=np.array([]),datlmax=np.array([]),dummylegpt=False):

    Nlmax=1
    if type(lmaxlist)!=np.ndarray:
        lmaxlist=np.array([lmaxlist])
    Nlmax=lmaxlist.size
    
    rhogrid=[]#to become 2d array [lmax,lmin]
    for i in xrange(Nlmax):
        if lmaxlist[i]==-1:
            filetagi=filetag
        else:
            filetagi=filetag+'lmax{0:d}'.format(lmaxlist[i])
        rhogrid.append(lmintest_get_rhoexp(lminlist=lminlist,lmaxlist=lmaxlist[i],z0=z0,overwrite=overwrite,saverho=saverho,varname=varname,filetag=filetagi,plotdir=plotdir))
    rhogrid=np.array(rhogrid)
    np.array(rhogrid)
    print 'rhogrid.shape',rhogrid.shape
    plt.figure(0)
    plt.subplots_adjust(bottom=.2)
    plt.subplots_adjust(left=.2)
    plt.subplots_adjust(right=.95)
    ax=plt.subplot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)
    for item in ([ax.xaxis.label, ax.yaxis.label] ):
        item.set_fontsize(32)
    
    colors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    if varname=='rho':
        varstr=r'\rho'
    elif varname=='s':
        varstr='s'
    for i in xrange(Nlmax):
        if Nlmax>1:
            linelabel=r'$\ell_{{\rm max}}={0:d}$'.format(lmaxlist[i])
        else:
            linelabel=r'$\langle {0:s} \rangle_{{[\ell\geq \ell_{{\rm min}}]}}$ from theory'.format(varstr)
        Nrho=rhogrid[i,:].size
        plt.plot(lminlist[:Nrho],rhogrid[i,:],color=colors[i%len(colors)],label=linelabel)
    plt.xlabel(r'$\ell_{\rm min}$')
    plt.ylabel(r'${0:s}_{{[\ell\geq \ell_{{\rm min}}]}}$'.format(varstr))
    plt.xlim((0,15))
    plt.ylim((.75,.965))

    if dodata:
        rhomean=[]
        rhostd=[]
        Nreal=[]
        if not datlmin.size:
            datlmin=lminlist
        if not datlmax.size:
            datlmax=lmaxlist
        for i in xrange(Nlmax):
            rhomeani,rhostdi,Nreali=lmintest_get_rhodat(datlmin,datlmax[i])
            rhomean.append(rhomeani)
            rhostd.append(rhostdi)
            Nreal.append(Nreali)
            #colors will match lines
            if dummylegpt:
                label=''
            else:
                label='Results from sim.'
            plt.errorbar(datlmin,rhomeani,yerr=rhostdi,linestyle='None',marker='o',color=colors[i%len(colors)],label=label)
        datlabel='Results from sim.'
        #plot dummy point for legend
        if dummylegpt:
            plt.errorbar([-1],[.9],yerr=[.01],linestyle='None',marker='o',color='black',label=datlabel)

    if varname=='rho':
        plt.legend(fontsize=20,loc='lower right',numpoints=1)
    elif varname=='s':
        plt.legend(fontsize=20,loc='lower left',numpoints=1)

    if filetag:
        outtag='_'+filetag
    else:
        outtag=filetag
    outname='lmintest_'+varname+'_exp'+outtag+'.pdf'
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

#===============================================================
# shot noise tests
#===============================================================
# How does increasing the amount of shot noise affect <rho> or <s>?
#------------------------
def shottest_getrhoexp(nbarlist=np.array([1.e5,1.e7,1.e9]),varname='rho'): #assumes nbarlist in units of 1/sr
    mapname='eucz07'
    cldat=depthtest_get_Cl(z0vals=np.array([0.7]))
    rhovals=[]
    for n in nbarlist:
        #print 'n=',n
        cldat.changenbar(mapname,n)
        #print '  cldat.nbar[1]=',cldat.nbar[1]*(np.pi/180./60.)**2,cldat.noisecl[cldat.crossinds[1,1],4]
        includeglm=[mapname+'_bin0']
        recdat=au.RecData(includeglm=includeglm,inmaptag=mapname)
        rhopred=au.compute_rho_fromcl(cldat,recdat,varname=varname)
        rhovals.append(rhopred)
    rhovals=np.array(rhovals)
    return rhovals

def shottest_get_fidCl(): #with no shot noise
    mapname='eucz07'
    cldat=depthtest_get_Cl(z0vals=np.array([0.7]))
    cldat.changenbar(mapname,-1)
    return cldat

#when generating data maps, basically treating shot noise as modeled calib error
# asumes nbarlist in inverse steradians
def shottest_apply_noisetomap(nbarlist=[1.e4],Nreal=0,overwritecalibmap=False,scaletovar=False,redofits=True):
    #print '====applying shot noise to map'
    cldat=shottest_get_fidCl()
    refvar,refind=caltest_get_scaleinfo(nbarlist,scaletovar)
    #print '>refvar,refind',refvar,refind
    fidbins=caltest_get_fidbins()
    lssbin=fidbins[1].tag #just the fiducial depthtest binmap
    glmdat=caltest_get_fidglm()

    #set up noise maps
    noiseinfolist=[(lssbin,refvar)] #only for max var
    print 'noiseinfolist',noiseinfolist
    if Nreal and overwritecalibmap:
        print 'Generating shot noise maps.'
    dothesemods=gmc.get_shotnoise_formaps(glmdat,noiseinfolist,overwrite=overwritecalibmap,Nreal=Nreal) #generates calibration error maps, returns [(maptag,modtag,masktag)]list
    #print 'dothesemods',dothesemods

    #apply calibration errors
    outglmdatlist=[]
    for n in nbarlist:
        scaling =np.sqrt(refvar/n) #to multiply noise maps
        newcaltag=get_modtag_shotnbar(n)
        print 'Applying shot noise to map, newcaltag',newcaltag
        outglmdatlist.append(gmc.apply_caliberror_to_manymaps(glmdat,dothesemods,Nreal=Nreal,calmap_scaling=scaling,newmodtags=[newcaltag],overwritefits=redofits,justaddnoise=True))

    outglmdat=glmdat.copy(Nreal=0)
    for n in xrange(len(outglmdatlist)):
        outglmdat=outglmdat+outglmdatlist[n]
    #print 'outglmdat.modtaglist',outglmdat.modtaglist
    return outglmdat #includes, isw, fiduical, and cal error map names

def shottest_get_reclist(nbarlist):
    reclist=[]
    fidbins=caltest_get_fidbins()
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    lsstype=fidbins[1].typetag
    noiseinfolist=[(lssbin,nbar) for nbar in nbarlist]
    fidglm=caltest_get_fidglm()
    dothesemods=gmc.get_shotnoise_formaps(fidglm,noiseinfolist,overwrite=False,Nreal=0)
    print '****',dothesemods
    #put fidicual in
    includecl=[lssbin]
    inmaptype=lsstype
    #reclist.append(au.RecData(includecl,includecl,inmaptype,'unmod'))
    for m in dothesemods:
        includeglm=[m]
        rectag=m[1]#modtag
        reclist.append(au.RecData(includeglm,includecl,inmaptype,rectag))
    return reclist #includes fiducial case as first entry
    
def shottest_iswrec(Nreal,nbarlist=[1.e4],overwritecalibmap=False,scaletovar=1.e4,domaps=True):
    fidcl=shottest_get_fidCl()#don't use fidcl here!
    dummyglm=shottest_apply_noisetomap(nbarlist,0,overwritecalibmap,scaletovar,redofits=False)#includes fidicual case
    #do one rec at a time to be able to properly handle shot noise in cldat
    for n in nbarlist:
        reclist=shottest_get_reclist([n])
        fidcl.changenbar('eucz07',n)
        au.doiswrec_formaps(dummyglm,fidcl,Nreal,reclist=reclist,domaps=domaps)

# [work in progress, no code yet to do these reconstructions]
# assuming reconstructions have already been done, along with rho or s calc,
# and assuming all nbar have the same number of realizations
#  read in the statistic (rho or s) info from files
#  returns a Nbar x Nreal array of rho values
def shottest_getrhodat_fromfiles(nbarlist,varname='rho'):
    Nnbar=nbarlist.size
    mapdir='output/depthtest/map_output/'
    files=[]
    for n in nbarlist: #nbar in steradiancs
        files.append('iswREC.eucz07.shotnbar{0:0.1e}.fullsky-lmin02.depthtest.{1:s}.dat'.format(n,varname))
    rhogrid=np.array([au.read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

def shottest_get_rhodat(datnbar,varname='rho'):
    #get 2d Nnbar x Nreal array of rho data, assumign it has already been calc'd
    print 'datnbar =',datnbar*(np.pi/180./60.)**2
    rhodat=shottest_getrhodat_fromfiles(datnbar,varname) #nbar in steradians
    print 'rhodat.shape',rhodat.shape
    Nnbar=rhodat.shape[0]
    Nreal=rhodat.shape[1]
    rhomean=np.zeros(Nnbar)
    rhostd=np.zeros(Nnbar)
    for i in xrange(Nnbar):
        rho=rhodat[i,:]
        rhomean[i]=np.mean(rho)
        rhostd[i]=np.std(rho)
    return rhomean,rhostd,Nreal


# plot expectation values
def shottest_plot_rhoexp(nbarlist=np.array([1.e5,1.e6,1.e7,1.e8,1.e9]),varname='rho',overwrite=True,saverho=True,filetag='',plotdir='output/shottest_plots/',passnbarunit='sr',plotnbarunit='amin2',dodata=False,datnbar=np.array([]),dummylegpt=False):
    #passednbarunit - what units are nbarlist?per... sr deg2 or amin2
    # assumes nbarlist adn datnbar are in same units
    #plotnbarunit - sr deg2 or amin2, with implied ^-1
    fidnbar=1.e9#in sr^-1, will be noted on plot
    fidlabel=r'${\rm fiducial }\quad \bar{n}=10^9\,{\rm sr}^{-1}$'
    #datnbar=np.array([.1])#revert
    if passnbarunit=='sr':
        tosr=1.
        if plotnbarunit=='sr':
            toplot=1.
        elif plotnbarunit=='deg2':
            toplot=(np.pi/180.)**2
        elif plotnbarunit=='amin2':
            toplot=(np.pi/180./60.)**2
        fidnbar*=toplot
    elif passnbarunit=='deg2':
        tosr=(180./np.pi)**2
        if plotnbarunit=='sr':
            toplot=tosr
        elif plotnbarunit=='deg2':
            toplot=1.
            fidnbar*=(np.pi/180.)**2
        elif plotnbarunit=='amin2':
            toplot=(1/60.)**2
            fidnbar*=(np.pi/180./60.)**2
    elif passnbarunit=='amin2':
        tosr=(180.*60./np.pi)**2
        if plotnbarunit=='sr':
            toplot=tosr
        elif plotnbarunit=='deg2':
            toplot=(60.)**2
            fidnbar*=(np.pi/180.)**2
        elif plotnbarunit=='amin2':
            toplot=1.
            fidnbar*=(np.pi/180./60.)**2

    #print 'to plot=',toplot
    #print 'to sr = ',tosr
    rhogrid=shottest_getrhoexp(nbarlist*tosr,varname)#assumes steradian
    plt.figure(0)
    plt.subplots_adjust(bottom=.23)
    plt.subplots_adjust(left=.2)
    plt.subplots_adjust(right=.95)
    ax=plt.subplot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22)
    for item in ([ax.xaxis.label, ax.yaxis.label] ):
        item.set_fontsize(32)
    

    #plot data/theory
    colors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    if varname=='rho':
        varstr=r'\rho'
    elif varname=='s':
        varstr='s'
    plt.plot(nbarlist*toplot,rhogrid,color=colors[0],label=r'$\langle {0:s}\rangle$ from theory'.format(varstr))
    if plotnbarunit=='sr':
        plt.xlabel(r'$\bar{{n}}$ [${\rm sr}^{-1}$]')
    elif plotnbarunit=='deg2':
        plt.xlabel(r'$\bar{{n}}$ [${\rm deg}^{-2}$]')
    elif plotnbarunit=='amin2':
        plt.xlabel(r'$\bar{{n}}$ $[{\rm arcmin}^{-2}]$')
        
    plt.ylabel(r'${0:s}$'.format(varstr))
    plt.xscale('log')
    #plt.xlim((0,15))
    #plt.ylim((.75,.965))

    if dodata:
        if not datnbar.size:
            datnbar=nbarlist
        rhomean,rhostd,Nreal=shottest_get_rhodat(datnbar*tosr,varname)
        #colors will match lines
        if dummylegpt:
            label=''
        else:
            label='Results from sim.'
        plt.errorbar(datnbar*toplot,rhomean,yerr=rhostd,linestyle='None',marker='o',color=colors[0],label=label)
        datlabel='Results from sim.'
        #plot dummy point for legend
        if dummylegpt:
            plt.errorbar([-1],[.9],yerr=[.01],linestyle='None',marker='o',color='black',label=datlabel)
    plt.xlim((1.e-5,1.e3))
    if varname=='rho':
        plt.axhline(0,color='grey',linestyle=':')
        plt.axhline(1,color='grey',linestyle=':')
        plt.legend(fontsize=20,loc='lower right',numpoints=1)
        aloc=(fidnbar,0.175)
    elif varname=='s':
        plt.legend(fontsize=20,loc='lower left',numpoints=1)
        #plt.yscale('log')
        aloc=(fidnbar,.5)
    #plot reference liens
    plt.axvline(fidnbar,linestyle='-',color='grey')
    plt.annotate(fidlabel,xy=aloc,horizontalalignment='right',verticalalignment='bottom',fontsize=18,color='grey',rotation=90)

    if filetag:
        outtag='_'+filetag
    else:
        outtag=filetag
    outname='shottest_'+varname+'_exp'+outtag+'.pdf'
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()
    

#===============================================================
# anglular momentum tests
#===============================================================
# These functions are for making plots based on some angular
# momentum data dragan produced. He did so based on the fiducial, euc07
# depthtest maps: true ISW gives Ltrue, iswREC give Lrec
#-------------------------
def linefit_residuals(params,xdat,ydat):
    m,b=params
    ytheory=b+m*xdat
    err=ydat-ytheory
    return err

def angmomtest_Lvsrho_plot(plotdat='change',Lfile='Lmax_true_Lmax_rec.dat',plotdir='output/angmom_study/',dofit=True,note=r'$\ell=2,3$',fileprefix='',shuffleLrec=False,ellfilter=()):
    z0fid=0.7
    lminstr=''
    lmaxstr=''
    if ellfilter:
        lmin=ellfilter[0]
        lmax=ellfilter[1]
        lminstr="-lmin{0:02d}".format(lmin)
        if lmax>0:
            lmaxstr="-lmax{0:02d}".format(lmax)
    rhofile='iswREC.eucz07.fid.fullsky'+lminstr+lmaxstr+'.depthtest.rho.dat'
    if fileprefix:
        randstr=''
        if shuffleLrec: randstr='shuffled'
        fileprefix=fileprefix+randstr+'_'
    outfile='{0:s}L_vs_rho{2:s}{3:s}_{1:s}.pdf'.format(fileprefix,plotdat,lminstr,lmaxstr)
    scatcolor='#92c5de'
    meancolor='#ca0020'
    #read in data

    Ldat=np.loadtxt(plotdir+Lfile)
    Ltrue=Ldat[:,0]
    Lrec=Ldat[:,1]
    print 'preshuffle', type(Lrec)
    if shuffleLrec:
        np.random.shuffle(Lrec)
        note+='\n'+r'$L_{\rm rec}$ randomized'
    
    NL=Ltrue.size #assume first NL of the rho match w provided L vals
    rhodat=au.read_rhodat_wfile(plotdir+rhofile)[:NL]
    
    plt.figure(0)
    ax=plt.subplot()
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    plt.xlabel(r'$\rho$'+lminstr+lmaxstr)
    #plt.xlim((0.,1.))

    #plotdat specific info
    hline=True
    onetooneline=False
    if plotdat=='change':
        plt.ylabel(r'$L_{\rm rec} - L_{\rm true}$')
        ydat=Lrec-Ltrue
        whereyline=0
    elif plotdat=='abschange':
        plt.ylabel(r'$|L_{\rm rec} - L_{\rm true}|$')
        ydat=np.fabs(Lrec-Ltrue)
        whereyline=0
    elif plotdat=='fracchange':
        plt.ylabel(r'$\left[L_{\rm rec} - L_{\rm true}\right]/L_{\rm true}$')
        ydat=(Lrec-Ltrue)/Ltrue
        whereyline=0
    elif plotdat=='ratio':
        plt.ylabel(r'$L_{\rm rec}/L_{\rm true}$')
        ydat=(Lrec)/Ltrue
        whereyline=1
        
    #binned points setup
    nbins=7
    histrange=(.3,1.)
    n,hedges=np.histogram(rhodat,bins=nbins,range=histrange)
    sy,_=np.histogram(rhodat,bins=nbins,weights=ydat,range=histrange)
    sy2,_=np.histogram(rhodat,bins=nbins,weights=ydat*ydat,range=histrange)
    mean=sy/n
    std=np.sqrt(sy2/n-mean*mean)

    #plot point and ref line
    plt.plot(rhodat,ydat,linestyle='None',marker='.',color=scatcolor,zorder=-32)
    # zorder arg in there to try to make points drawn behind error bars
    if hline: ax.axhline(whereyline,color='grey',linestyle='-')
    
    #fit line to points
    if dofit:
        if hline: startm,startb=0,0
        if onetooneline: startm,startb=1,0
        params,_=leastsq(linefit_residuals,(startm,startb),args=(rhodat,ydat))
        m,b=params
        print 'fitted slope and intercept:',m,b
        lineends=np.array([np.min(rhodat),np.max(rhodat)])
        plt.plot(lineends,b+m*lineends,color='black',linestyle='--',label='fit to scatter')
        
    #plot hist points
    plt.errorbar((hedges[1:]+hedges[:-1])/2,mean,yerr=std,color=meancolor,marker='x',markersize=8,markeredgewidth=1,linestyle='None',linewidth=1,capsize=5,label='binned mean')#binned points
    plt.legend(loc='lower left',numpoints=1)
    plt.annotate(note,xy=(.1,.9),horizontalalignment='left',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    outname=plotdir+outfile
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
    
def angmomtest_LvsLtrue_plot(plotdat='change',Lfile='Lmax_true_Lmax_rec.dat',plotdir='output/angmom_study/',dofit=True,note=r'$\ell=2,3$',fileprefix='',shuffleLrec=False):
    z0fid=0.7
    rhofile='iswREC.eucz07.fid.fullsky-lmin02.depthtest.rho.dat'
    if fileprefix:
        randstr=''
        if shuffleLrec: randstr='shuffled'
        fileprefix=fileprefix+randstr+'_'
    outfile='{0:s}dL_vs_Ltrue_{1:s}.pdf'.format(fileprefix,plotdat)
    scatcolor='#92c5de'
    meancolor='#ca0020'

    #read in data
    Ldat=np.loadtxt(plotdir+Lfile)
    Ltrue=Ldat[:,0]
    Lrec=Ldat[:,1]
    if shuffleLrec:
        np.random.shuffle(Lrec)
        note+='\n'+r'$L_{\rm rec}$ randomized'
    plt.figure(0)
    ax=plt.subplot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    plt.xlabel(r'$L_{\rm true}$')
    #plt.xlim((0.,1.))
    hline=True
    onetooneline=False
    if plotdat=='change':
        plt.ylabel(r'$L_{\rm rec} - L_{\rm true}$')
        ydat=Lrec-Ltrue
        whereyline=0
    elif plotdat=='abschange':
        plt.ylabel(r'$|L_{\rm rec} - L_{\rm true}|$')
        ydat=np.fabs(Lrec-Ltrue)
        whereyline=0
    elif plotdat=='fracchange':
        plt.ylabel(r'$\left[L_{\rm rec} - L_{\rm true}\right]/L_{\rm true}$')
        ydat=(Lrec-Ltrue)/Ltrue
        whereyline=0
    elif plotdat=='ratio':
        plt.ylabel(r'$L_{\rm rec}/L_{\rm true}$')
        ydat=(Lrec)/Ltrue
        whereyline=1
    elif plotdat=='Lrec':
        plt.ylabel(r'$L_{\rm rec}$')
        ydat=(Lrec)
        hline=False
        onetooneline=True
        whereyline=1
        
    #binned points
    nbins=10
    histrange=(np.min(Ltrue),np.max(Ltrue))
    n,hedges=np.histogram(Ltrue,bins=nbins,range=histrange)
    sy,_=np.histogram(Ltrue,bins=nbins,weights=ydat,range=histrange)
    sy2,_=np.histogram(Ltrue,bins=nbins,weights=ydat*ydat,range=histrange)
    mean=sy/n
    std=np.sqrt(sy2/n-mean*mean)
    plt.plot(Ltrue,ydat,linestyle='None',marker='.',color=scatcolor,zorder=-32)
    # zorder arg in there to try to make points drawn behind error bars)
    if hline:
        ax.axhline(whereyline,color='grey',linestyle='-')
    elif onetooneline:
        xmax=np.max(np.fabs(Ldat))
        x0,x1=plt.xlim()
        y0,y1=plt.ylim()
        plt.plot(10*np.array([-xmax,xmax]),10*np.array([-xmax,xmax]),linestyle='-',color='grey')
        plt.xlim((x0,x1))
        plt.ylim((y0,y1))

    #fit line to points
    if dofit:
        if hline: startm,startb=0,0
        if onetooneline: startm,startb=1,0
        params,_=leastsq(linefit_residuals,(startm,startb),args=(Ltrue,ydat))
        m,b=params
        print 'fitted slope and intercept:',m,b
        lineends=np.array([np.min(Ltrue),np.max(Ltrue)])
        plt.plot(lineends,b+m*lineends,color='black',linestyle='--',label='fit to scatter')
    
    plt.errorbar((hedges[1:]+hedges[:-1])/2,mean,yerr=std,color=meancolor,marker='x',markersize=8,markeredgewidth=1,linestyle='None',linewidth=1,capsize=5,label='binned mean')#binned points
    plt.legend(loc='lower left',numpoints=1)
    plt.annotate(note,xy=(0.1,.9),horizontalalignment='left',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    outname=plotdir+outfile
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#plot rho for filtered ISW rec vs rho for all multipoles
def angmomtest_rhovsrho_plot(plotdir='output/angmom_study/',dofit=True,note=r'$\ell=2,3$',fileprefix='',shufflerhofilt=False,ellfilter=()):
    z0fid=0.7
    lminstr=''
    lmaxstr=''
    if ellfilter:
        lmin=ellfilter[0]
        lmax=ellfilter[1]
        lminstr="-lmin{0:02d}".format(lmin)
        if lmax>0:
            lmaxstr="-lmax{0:02d}".format(lmax)
    nofilterfile='iswREC.eucz07.fid.fullsky-lmin02.depthtest.rho.dat'
    filterfile='iswREC.eucz07.fid.fullsky'+lminstr+lmaxstr+'.depthtest.rho.dat'
    if fileprefix:
        randstr=''
        if shufflerhofilt: randstr='shuffled'
        fileprefix=fileprefix+randstr+'_'
    outfile='{0:s}rho{1:s}{2:s}_vs_rho.pdf'.format(fileprefix,lminstr,lmaxstr)
    scatcolor='#92c5de'
    meancolor='#ca0020'
    #read in data

    rhodat=au.read_rhodat_wfile(plotdir+nofilterfile)
    rhofiltdat=au.read_rhodat_wfile(plotdir+filterfile)
    if shufflerhofilt:
        np.random.shuffle(rhofiltdat)
        note+='\n'+r'$rho_{\rm filtered}$ randomized'
    
    plt.figure(0)
    ax=plt.subplot()
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    plt.ylabel(r'$\rho$'+lminstr+lmaxstr)
    plt.xlabel(r'$\rho$ (all $\ell$)')

    #plotdat specific info
    ydat=rhofiltdat
        
    #binned points setup
    nbins=7
    histrange=(.3,1.)
    n,hedges=np.histogram(rhodat,bins=nbins,range=histrange)
    sy,_=np.histogram(rhodat,bins=nbins,weights=ydat,range=histrange)
    sy2,_=np.histogram(rhodat,bins=nbins,weights=ydat*ydat,range=histrange)
    mean=sy/n
    std=np.sqrt(sy2/n-mean*mean)

    #plot point and ref line
    plt.plot(rhodat,ydat,linestyle='None',marker='.',color=scatcolor,zorder=-32)
    # zorder arg in there to try to make points drawn behind error bars

    # plot reference line
    xmax=np.max(np.fabs(rhodat))
    x0,x1=plt.xlim()
    y0,y1=plt.ylim()
    plt.plot(10*np.array([-xmax,xmax]),10*np.array([-xmax,xmax]),linestyle='-',color='grey')
    plt.xlim((x0,x1))
    plt.ylim((y0,y1))
    
    #fit line to points
    if dofit:
        startm,startb=1,0
        params,_=leastsq(linefit_residuals,(startm,startb),args=(rhodat,ydat))
        m,b=params
        print 'fitted slope and intercept:',m,b
        lineends=np.array([np.min(rhodat),np.max(rhodat)])
        plt.plot(lineends,b+m*lineends,color='black',linestyle='--',label='fit to scatter')
        
    #plot hist points
    plt.errorbar((hedges[1:]+hedges[:-1])/2,mean,yerr=std,color=meancolor,marker='x',markersize=8,markeredgewidth=1,linestyle='None',linewidth=1,capsize=5,label='binned mean')#binned points
    plt.legend(loc='lower left',numpoints=1)
    plt.annotate(note,xy=(.1,.9),horizontalalignment='left',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    outname=plotdir+outfile
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
    
    
def angmomtest_checkrho_plot(plotdir='output/angmom_study/',dofit=True,fileprefix='',note='',shuffle=False):
    z0fid=0.7
    rhofile='iswREC.eucz07.fid.fullsky-lmin02.depthtest.rho.dat'
    draganrhofile='fromdragan_l5_rho_10000.dat'
    randstr=''
    if shuffle:
        randstr='shuffled'
        fileprefix=fileprefix+randstr+'_'
    outfile='{0:s}rhocomp.pdf'.format(fileprefix)
    scatcolor='#92c5de'
    meancolor='#ca0020'

    #read in data
    myrhodat=au.read_rhodat_wfile(plotdir+rhofile)
    dhrhodat=np.loadtxt(plotdir+draganrhofile)
    if shuffle:
        np.random.shuffle(myrhodat)
        note+='\n'+r'my $\rho$ randomized'
    plt.figure(0)
    ax=plt.subplot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    plt.xlabel(r"Jessie's $\rho$")
    plt.ylabel(r"Dragan's $\rho$")
    ydat=dhrhodat        

    #binned points
    nbins=10
    histrange=(np.min(myrhodat),np.max(myrhodat))
    n,hedges=np.histogram(myrhodat,bins=nbins,range=histrange)
    sy,_=np.histogram(myrhodat,bins=nbins,weights=ydat,range=histrange)
    sy2,_=np.histogram(myrhodat,bins=nbins,weights=ydat*ydat,range=histrange)
    mean=sy/n
    std=np.sqrt(sy2/n-mean*mean)
    plt.plot(myrhodat,ydat,linestyle='None',marker='.',color=scatcolor)
    #put line thorugh dhrhodat=myrhodat
    xmax=np.max(np.fabs(myrhodat))
    x0,x1=plt.xlim()
    y0,y1=plt.ylim()
    plt.plot(10*np.array([-xmax,xmax]),10*np.array([-xmax,xmax]),linestyle='-',color='grey')
    plt.xlim((x0,x1))
    plt.ylim((y0,y1))

    #fit line to points
    note=''
    if dofit:
        startm,startb=1,0
        params,_=leastsq(linefit_residuals,(startm,startb),args=(myrhodat,ydat))
        m,b=params
        note+='\n fitted slope and intercept: \n{0:0.3g}, {1:0.3g}'.format(m,b)
        print 'fitted slope and intercept:',m,b
        lineends=np.array([np.min(myrhodat),np.max(myrhodat)])
        plt.plot(lineends,b+m*lineends,color='black',linestyle='--',label='fit to scatter')
    
    plt.errorbar((hedges[1:]+hedges[:-1])/2,mean,yerr=std,color=meancolor,marker='x',markersize=8,markeredgewidth=1,linestyle='None',linewidth=1,capsize=5,label='binned mean')#binned points
    plt.legend(loc='lower left',numpoints=1)
    plt.annotate(note,xy=(0.1,.9),horizontalalignment='left',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    outname=plotdir+outfile
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

# compute rho for reconstructions using limited ell values
# llmins is a list of tuples containing sets of (lmin,lmax)
# domaps - bool, if true redoes isw rec maps, if false, just recomputes stats
def anglmomtest_get_filteredrho(llims=[(2,3),(5,5)],Nreal=10000,varname='rho',domaps=True):
    Nlim=len(llims)
    lminlist=np.ones(Nlim,dtype=int)
    lmaxlist=np.ones(Nlim,dtype=int)
    for i in xrange(Nlim):
        lminlist[i]=llims[i][0]
        lmaxlist[i]=llims[i][1]
    lmintest_iswrec(Nreal,lminlist,lmaxlist,domaps)


def angmomtest_LrecLtrue_corrcoef(datdir='output/angmom_study/',plotdir='output/angmom_study/'):
    labels=['z0=0.3','z0=0.7']
    files=['Lmax_true_Lmax_rec_z0_03.dat','Lmax_true_Lmax_rec_z0_07.dat']
    N = len(files)
    for n in xrange(N):
        dat=np.loadtxt(datdir+files[n])
        Ltrue=dat[:,0]
        Lrec=dat[:,1]
        Rmatrix=np.corrcoef(dat,rowvar=0) #matix correletion coefficents
        r=Rmatrix[0,1] #off diag tells us corr between maps
        print labels[n],', Pearson corr between Lrec-Ltrue = ',r


#################################################################
if __name__=="__main__":
    #Below are a number of function calls used in our analysis
    # the "if 0"s serve as switches, to only run one calculation at a time
    # They are organized roughly in the order of the sections in our paper
    #   but note that they may not generate the same plots. for some plots
    #   i rewrote the functions in 'genplots_forpaper.py' so taht I could adjust
    #   formatting in more detail. 
    st = time.time()
    depthtestz0=np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.3,1.5]) #np.array([.3,.5,.6,.7,.8])
    if 0: #compute Cl # this takes a while;
        depthtest_get_Cl(justread=False,z0vals=depthtestz0)

    if 0: #for testing
        depthtest_plot_zwindowfuncs(depthtestz0)
#---SINGLE---
    if 0: #generate depthhtest maps
        Nreal=1000
        simmaps=False #do you want to simulate maps, or just do reconstructions (on maps that already exist?
        z0vals=depthtestz0
        #z0vals=np.array([0.7])
        if simmaps: #generate maps and do reconstructions
            depthtest_get_glm_and_rec(Nreal=Nreal,z0vals=z0vals,justgetrho=False,minreal=0,dorho=1,dos=True,dochisq=False,dorell=0,dochisqell=False)
        else: #do recs based on existing galaxy maps
            depthtest_iswrec(Nreal,z0vals=z0vals,minreal=0,dorho=1,dos=1,domaps=True)
            #note, if you just want t compute rho but don't want to redo isw recs
            # change domaps to False
#----- MULTI ------        
    outtag = '_multi_same_cltimed_full'
    if 1: #compute Cl # this takes a while;
        depthtest_get_Cl(justread=False,z0vals=depthtestz0, outtag=outtag)
    if 0: #generate MULTI depthhtest maps (don't want togr)
        Nreal=100
        simmaps=True #do you want to simulate maps, or just do reconstructions (on maps that already exist?
        z0vals=depthtestz0
        
        doMulti = range(len(depthtestz0)) + [(0,1),(0,2),(1,2),(0,1,2)] #list of z0val maps to combine for reconstruction.
        
        #z0vals=np.array([0.7])
        if simmaps: #generate maps and do reconstructions
            depthtest_get_glm_and_rec(Nreal=Nreal,z0vals=z0vals,justgetrho=False,minreal=0,dorho=1,dos=True,dochisq=False,dorell=0,dochisqell=False, multi=doMulti, outtag=outtag)
        else: #do recs based on existing galaxy maps
            depthtest_iswrec(Nreal,z0vals=z0vals,minreal=0,dorho=1,dos=1,domaps=True, multi=doMulti, outtag=outtag)
            #note, if you just want t compute rho but don't want to redo isw recs
            # change domaps to False

    if 0: #plot info about depthtest maps, assumes you've already done isw recs
        for r in xrange(10,20):
             #depthtest_TTscatter(r,depthtestz0,savepngmaps=False)
             pass
        for N in 1000*np.arange(1,10): #how much do results change with Nreal
            #depthtest_plot_rhohist(depthtestz0,varname='rho',firstNreal=1000,startrealat=N)
            pass
        depthtest_plot_rhohist(depthtestz0,varname='rho')
        depthtest_plot_rhohist(depthtestz0,varname='s')
        depthtest_TTscatter(13,depthtestz0,savepngmaps=True)

    if 0: #bin test rho expectation value calculations, can be done w/out maps
        if 0:         #compute cl, can take a while
            # generally, i will have computed the 'base Cl' for this on
            # a computing cluster or something, then have these funcs do bin combos
            cldat05=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.05,justread=0)
            cldat03=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.03,justread=0)
            cldat001=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.001,justread=0)
            cldat100=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.1,justread=0)
        
        #compute and save expectation values for rho[0.001,0.03,0.05,0.1]
        bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.001,0.03,0.05,.1])
        #bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.001,0.03,0.05,.1],varname='s')

        for s in [0.001,0.03,0.05,0.1]: #some tests for z distributions
            #bintest_plot_cl_vals(finestN=6,z0=0.7,sigz=s)
            #bintest_plot_zwindowfuncs(sigz=s)
            pass
        
    if 0: #bin test with many realizations, generate maps
        Nreal=10000
        simmaps=False #if true, simulates galaxy maps, if false, just do isw recs
        if simmaps: #simulate galaxy maps and do isw rec
            bintest_get_glm_and_rec(Nreal=Nreal,divlist=['6','222','111111'],minreal=0,justgetrho=False,dorell=0) #set justgetrho=true if you just want to compute rho etc
        else: #do isw rec from existing gal maps. set domaps=False if you 
            bintest_iswrec(Nreal,divlist=['6','222','111111'],minreal=0,dorho=1,dos=1,domaps=True) #set domaps=False if you just want rho and s
    if 0: #bin test with many realizations, make plots after having already done recs
        bintest_plot_rhohist(getrhopred=True,varname='rho')
        bintest_plot_rhohist(getrhopred=True,varname='s')

    #lmin tests; assuming depthtest maps already have been generated
    #    look at how changing lmin affects reconstruction
    if 0: 
        Nreal=1
        inlminlist=np.array([1,2,3,4,5])
        #inlminlist=np.array([10])
        inlmaxlist=np.array([-1])#3,5,10,20,-1]) #-1 will use largest available lmax
        lminlist,lmaxlist=lmintest_get_lminmaxcombos(inlminlist,inlmaxlist)
        if 0: #do reconstructions for Nreal for combos of inlminlist and inlmax
            domaps=True
            Nreal=10000
            lmintest_iswrec(Nreal=Nreal,lminlist=lminlist,lmaxlist=lmaxlist,domaps=domaps)
        #if dodata=True, assumes you've done recs for many realizations and
        # adds datapoints to plot. can plot without simulations if you set dodata=0
        lmintest_plot_rhoexp(overwrite=0,lminlist=np.arange(1,20),lmaxlist=inlmaxlist,varname='rho',dodata=True,datlmin=inlminlist)
        
    # shot noise tests; assumes depthtest maps have already been generated
    if 0:
        shortnbarlist=np.array([1.e-4,1.e-3,1.,10.,100.])#np.array([1.e-4,1.e-3,.01,.1,1.,10.,100.])#in arcmin^-2
        shortnbarsr=shortnbarlist*((180.*60./np.pi)**2)
        scaletovar=shortnbarsr[0]
        nbarlist=caltest_get_logspaced_varlist(1.e-6,1.e3)
        if 0: #gen many maps
            Nreal=10000
            shottest_apply_noisetomap(nbarlist=shortnbarsr,Nreal=Nreal,overwritecalibmap=1,scaletovar=scaletovar) #only need to do this once
            shottest_iswrec(Nreal,nbarlist=shortnbarsr,scaletovar=scaletovar,domaps=True)
        shortnbarlist=np.array([1.e-4,1.e-3,.01,.1,1,10.,100.])
        shottest_plot_rhoexp(nbarlist=nbarlist,varname='rho',passnbarunit='amin2',overwrite=0,dodata=True,datnbar=shortnbarlist)
        shottest_plot_rhoexp(nbarlist=nbarlist,varname='s',passnbarunit='amin2',overwrite=0,dodata=True,datnbar=shortnbarlist)

    if 0: #z0test and b2test theory calcs; used these mostly for testing
        #  if doplot=True, will genearate colorblock plots that we ended up not using
        #  since the plots were hard to read.
        # the plotting functions haven't been tested after some edits, may not work
        simz0=np.array([.35,.56,.63,.693,.7,.707,.7700,.84,1.05])
        perrors=[1,10,20,50]
        z0test_get_rhoexp(overwrite=True,doplot=True,varname='rho',perrors=perrors)
        z0test_get_rhoexp(overwrite=True,doplot=True,varname='s',perrors=perrors)
        simb2=np.array([0.,.01,.1,.5,1.,2.,5.,10.])
        recb2=simb2
        bztest_get_rhoexp(simb2,recb2,overwrite=True,doplot=True,varname='rho')
        bztest_get_rhoexp(simb2,recb2,overwrite=True,doplot=True,varname='s')

    if 0: #catztest theory calcs, makes colorblock plots; see note above
        for Nbins in [1,3]: #numbers in badfracs reflect what i computed Cl for
            if Nbins==1:
                badfracs=np.array([0.,5.e-4,1.e-3,2.e-3,5.e-3,1.e-2,2.e-2,.1,.2])
            elif Nbins==3:
                badfracs=np.array([1.e-3,1.e-2,.1,.2])                                
            #catz_windowtest(badfracs,Nbins=Nbins) #plot window functions
            catz_get_rhoexp(overwrite=True,doplot=True,varname='rho',badfracs=badfracs,Nbins=Nbins)
            catz_get_rhoexp(overwrite=True,doplot=True,varname='s',badfracs=badfracs,Nbins=Nbins)

    if 0: #plot how these variables change Cl
        z0test_Clcomp()
        bztest_Clcomp()
        catz_Clcomp()

    if 0: #zdist tests with less info; these are what we ended up using
        overwritedat=1 #if true, overwrites rho and s .dat files, otherwise readonly
        z0test_onesim_plot(varname='rho',overwritedat=overwritedat)
        z0test_onesim_plot(varname='s',overwritedat=overwritedat)
        bztest_onesim_plot(varname='rho',overwritedat=overwritedat) 
        bztest_onesim_plot(varname='s',overwritedat=overwritedat)
        
        badfracs=np.array([0.,1.e-3,5.e-3,1.e-2,2.e-2,5.e-2,.1,.2])
        catztest_onesim_plot(varname='rho',Nbins=1,recf=badfracs,fidf=.01,secondfidf=.1,overwritedat=overwritedat)
        catztest_onesim_plot(varname='s',Nbins=1,recf=badfracs,fidf=.01,secondfidf=.1,overwritedat=overwritedat)
        badfracs=np.array([0.,1.e-3,1.e-2,.1,.2])
        catztest_onesim_plot(varname='rho',Nbins=3,recf=badfracs,fidf=.01,secondfidf=.1,overwritedat=overwritedat)
        catztest_onesim_plot(varname='s',Nbins=3,recf=badfracs,fidf=.01,secondfidf=.1,overwritedat=overwritedat)
   
        
    #caltest: look at the impact of calibration errors!
    if 0: #plotting cl for caltest to look at where cl^cal crosses cl^gal
        varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))    
        caltest_Clcomp(varlist)
    

    #lmin caltests; vary both the amount of calibration error and lmin
    #  assumes depthtest galaxy maps already exist, generates maps and data needed
    #  for both the basic calibration error test and the one where lmin is varied
    if 0:
        shortvarlist=[1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,1.e-2] #var[c] value to simulate
        shortreclminlist=np.array([2,3,5])#1,3,10])
        if 0: #do recs for many realizations
            Nreal=10000
            #caltest_apply_caliberrors(Nreal=Nreal,varlist=shortvarlist,overwritecalibmap=False,scaletovar=1.e-3,shape='g',width=10.,lmin=0,lmax=30) #only do once
            caltest_iswrec(Nreal,shortvarlist,scaletovar=1.e-3,recminelllist=shortreclminlist,domaps=True) #set domaps=False if you just want to calculate rho, s

        varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))#for theory calcs
        reclminlist=np.array([2,3,5])
        
        caltest_compare_lmin(varlist,varname='rho',dodataplot=True,recminelllist=reclminlist,shortrecminelllist=shortreclminlist,shortvarlist=shortvarlist,justdat=True)
        caltest_compare_lmin(varlist,varname='s',dodataplot=True,recminelllist=reclminlist,shortrecminelllist=shortreclminlist,shortvarlist=shortvarlist,justdat=True)
    
    
    if 0: #scatter plots for calib test, do be done after maps and rec generation
        for r in xrange(5): #adjust range to generate plots for diff realizaitons
            caltest_TTscatter(r)
            pass
        #caltest_TTscatter(4,savepngmaps=True)#actuallys saves pdf
        
    #angular momentum tests; requires external data about Ltrue, Lrec
    #  has not been edited since some other changes in code, so may not work 100%
    if 0:
        shuffle=0
        angmomfiles=['Lmax_true_Lmax_rec.dat','Lmax_true_Lmax_rec_5_to_5.dat']
        notes=[r'$\ell=2,3$',r'$\ell=5$']
        fprefix=['l23','l5']
        #angmomtest_checkrho_plot(shuffle=shuffle) #check that Dragan and i get same rho
        for i in xrange(len(angmomfiles)):
            f=angmomfiles[i]
            n=notes[i]
            p=fprefix[i]
            for dattype in ['change','abschange']:
                angmomtest_Lvsrho_plot(plotdat=dattype,Lfile=f,note=n,fileprefix=p,shuffleLrec=shuffle)
                pass
            for dattype in ['change','abschange','Lrec']:
                angmomtest_LvsLtrue_plot(plotdat=dattype,Lfile=f,note=n,fileprefix=p,shuffleLrec=shuffle)
                pass
    if 0: #more angular momentum tests, using filtered rho
        llims=[(2,3),(5,5)]
        angmomfiles=['Lmax_true_Lmax_rec.dat','Lmax_true_Lmax_rec_5_to_5.dat']
        notes=[r'$\ell=2,3$',r'$\ell=5$']
        fprefix=['l23','l5']
        shuffle=0
        #compute rho_filtered
        #anglmomtest_get_filteredrho() #do rec for (lmin,lmax)

        #plot rho_filtered vs rho and dL vs rho_filtered
        for i in xrange(len(angmomfiles)):
            f=angmomfiles[i]
            n=notes[i]
            p=fprefix[i]
            notes=[r'$\ell=2,3$',r'$\ell=5$']
            fprefix=['l23','l5']
            for dattype in ['change','abschange']:
                angmomtest_Lvsrho_plot(plotdat=dattype,Lfile=f,note=n,fileprefix=p,shuffleLrec=shuffle,ellfilter=llims[i])
                pass
            angmomtest_rhovsrho_plot(note=n,fileprefix=p,shufflerhofilt=shuffle,ellfilter=llims[i])
    if 0: #correlation coefs for Lmax numbers taht dragan emailed on 3/10/16
        angmomtest_LrecLtrue_corrcoef()
    
    print "Time to Run: {0:0f} mins".format((time.time() - st)/60.,)
