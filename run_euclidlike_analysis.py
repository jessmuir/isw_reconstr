import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
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
def depthtest_get_reclist(z0vals=np.array([.3,.6,.7,.8])):
    bins=depthtest_get_binmaps(z0vals)
    reclist=[]
    for i in xrange(len(bins)):
        bintag=bins[i].tag
        if bintag!='isw_bin0':
            includeglm=[bintag]
            inmaptag=bintag[:bintag.rfind('_bin0')]
            recdat=RecData(includeglm=includeglm,inmaptag=inmaptag)
            reclist.append(recdat)
    return reclist

#use cldat to generate glm, alm, and maps; saves maps but not alm
def depthtest_get_glm_and_rec(Nreal=1,z0vals=np.array([.3,.6,.7,.8]),minreal=0,justgetrho=0):
    t0=time.time()
    cldat=depthtest_get_Cl(justread=True,z0vals=z0vals)
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

def depthtest_plot_rhohist(z0vals=np.array([.3,.6,.7,.8]),plotsampledist=True,getrhopred=True):
    #if plotsampledist, plot curve of expected sample distribution
    plotdir='output/depthtest/plots/'
    Nrecs=z0vals.size
    Nbins=100
    rhogrid=depthtest_read_rho_wfiles(z0vals)
    Nreal=rhogrid.shape[1]
    maxrho=np.max(rhogrid)
    minrho=np.min(rhogrid)
    rholim=(minrho,maxrho)
    #rholim=(0.,maxrho)

    if getrhopred:
        rhopred=depthtest_get_expected_rho(z0vals)
    
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    plt.figure(0)
    plt.title(r'Depth test: correlation coef. $\rho$ for {0:g} rlzns'.format(Nreal))
    plt.xlabel(r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$',fontsize=16)
    plt.ylabel('realizations',fontsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        colstr=colors[i%len(colors)]
        if getrhopred:
            rhoval=rhopred[i]
            plt.axvline(rhoval,linestyle='-',color=colstr)
            label=r'$z_0={0:0.1f}$: $\langle\rho\rangle={3:0.3f}$; $\bar{{\rho}}={1:0.3f}$, $\sigma={2:0.3f}$'.format(z0vals[i],mean,sigma,rhoval)
        else:
            label=r'$z_0={0:0.1f}$: $\bar{{\rho}}={1:0.3f}$, $\sigma={2:0.3f}$'.format(z0vals[i],mean,sigma)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=rholim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.6)

    if getrhopred:
        plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from sample')
        plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='expectation value')

    plt.legend(loc='upper left')
    plotname='depthtest_rhohist_r{0:05d}'.format(Nreal)
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
#rho_sampledist(r,rho,NSIDE=32)

def depthtest_get_expected_rho(z0vals=np.array([0.3,0.6,0.7,0.8])):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rhopred=np.zeros_like(z0vals)
    for i in xrange(z0vals.size):
        rhopred[i]=compute_rho_fromcl(cldat,reclist[i])
    return rhopred
    
#testing or knowledge of predicted rho and distribution
def depthtest_rho_tests(z0vals=np.array([0.7])):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rhopred=compute_rho_fromcl(cldat,reclist[0])
    print "predicted rho:",rhopred
    rhogrid=depthtest_read_rho_wfiles(z0vals)
    Nrho=rhogrid.shape[1]
    rhomean=np.mean(rhogrid[0,:])
    print "mean rho:",rhomean
    #get histogram data
    nvals,evals,patches=plt.hist(rhogrid[0,:],bins=100,range=(-.4,1),histtype='stepfilled')
    bincenters=np.array([.5*(evals[i]+evals[i+1]) for i in xrange(len(nvals))])
    binwidth=evals[1]-evals[0]
    plt.figure()
    
    from scipy.optimize import leastsq
    distfit=lambda x,n:rho_sampledist(x,rhopred,Nsample=n)*Nrho*binwidth
    p0=np.array([12])
    errfunc=lambda n,x,y:distfit(x,n)-y
    n0=100 #initial guess
    n1,success=leastsq(errfunc,n0,args=(bincenters,nvals))
    
    print 'best fit N:',n1
    rvals=-1+np.arange(201)*.01
    distvals=distfit(rvals,n1)
    plt.plot(rvals,distvals,label='fitted function')
    plt.plot(bincenters,nvals,label='histogram points',linestyle='None',marker='o')
    plt.xlim(-.4,1)
    plt.title("Pearson sampling distribution fit to hist for depth z0=0.7")
    plt.xlabel('r')
    plt.ylabel('counts')
    plt.legend(loc='upper left')
    plt.show()

#do Tisw-Trec scatter plot for a given realization r
def depthtest_TTscatter(r=0, z0vals=np.array([0.3,0.6,0.7,0.8]),savepngmaps=True):
    plotdir='output/depthtest/plots/'
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    Nrec=z0vals.size
    #get map names and read in maps
    recmapfiles=[]
    recmaps=[]
    iswmapfiles=[]
    iswmaps=[]
    #get dummy glm and alm for filenames
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    glmdat=get_glm(cldat,Nreal=0,runtag=cldat.rundat.tag)
    almdat=get_dummy_recalmdat(glmdat,reclist,outruntag=glmdat.runtag)
    for i in xrange(Nrec):
        truemapf=glmdat.get_mapfile_fortags(r,reclist[i].zerotagstr)
        truemap=hp.read_map(truemapf,verbose=False)
        iswmapfiles.append(truemapf)
        iswmaps.append(truemap)
        recmapf=almdat.get_mapfile(r,i,'fits')
        recmap=hp.read_map(recmapf,verbose=False)
        recmapfiles.append(recmapf)
        recmaps.append(recmap)
        if savepngmaps:
            #set up color scheme for lss map
            mono_cm=matplotlib.cm.Greys_r
            mono_cm.set_under("w") #set background to white
            lssmapfs=[]
            for glmstr in reclist[i].includeglm: #assumes default mod and mask
                lssmapfs.append(glmdat.get_mapfile_fortags(r,glmstr))
            for lssf in lssmapfs:
                lssm=hp.read_map(lssf,verbose=False)
                plotmax=0.7*np.max(np.fabs(lssm))
                lssfbase=lssf[lssf.rfind('/')+1:lssf.rfind('.fits')]
                hp.mollview(lssm,title=lssfbase,unit=r' $\delta\rho/\rho$',max=plotmax,min=-1*plotmax,cmap=mono_cm)
                plt.savefig(plotdir+'mapplot_'+lssfbase+'.png')
            maxtemp=np.max(truemap)
            maxtemp=max(maxtemp,np.max(recmap))
            plotmax=0.7*maxtemp
            truefbase=truemapf[truemapf.rfind('/')+1:truemapf.rfind('.fits')]
            hp.mollview(truemap,title=truefbase,unit='K',max=plotmax,min=-1*plotmax)
            plt.savefig(plotdir+'mapplot_'+truefbase+'.png')
            recfbase=recmapf[recmapf.rfind('/')+1:recmapf.rfind('.fits')]
            hp.mollview(recmap,title=recfbase,unit='K',max=plotmax,min=-1*plotmax)
            plt.savefig(plotdir+'mapplot_'+recfbase+'.png')

            
    #compute rho (could also read from file but this seams simplest)
    rhovals=[rho_onereal(iswmaps[n],recmaps[n]) for n in xrange(Nrec)]
    reclabels=['z0={0:0.1f}'.format(z0) for z0 in z0vals]

    #set up plot
    plotname='TrecTisw_scatter_depthtest.r{0:05d}'.format(r)
    plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir,plotname,colors)
    
    
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
    zedges[-2]=2.
    dz=2./(finestN-1)
    for n in xrange(finestN-1):
        zedges[n]=dz*n
    zedges[0]=.01 #don't take integrals all the way to zero
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
    #print 'called: n,N=',n,N
    outlist=[]
    maxval=N-n+1
    #print '1 maxval',maxval
    nlist=np.ones(n,dtype=int)
    nlist[0]=maxval
    outlist.append(np.array(nlist))
    while maxval>1 and n>1:
        maxval-=1
        #print '2 maxval',maxval
        sublists=get_sumperm(n-1,N-maxval)
        #print 'sublists',sublists
        for s in sublists:
            newlist=np.array([maxval]+list(s))
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

def bintest_get_zedgeslist(zedges,getdivs=['equal']):
    Nmax=zedges.size-1
    slist=getdivs
    if len(getdivs)==1:
        if getdivs[0]=='equal':
            slist=bintest_get_divstr_equal(Nmax)
        elif getdivs[0]=='all':
            slist=bintest_get_divstr_all(Nmax)
    zedgelist=[]#will be list of arrays, one for each binning strategy
    for s in slist:
        nlist=[int(x) for x in s]
        zlist=[zedges[0]]
        zind=0
        for n in nlist:
            zind+=n
            zlist.append(zedges[zind])
        zedgelist.append(np.array(zlist))
    return zedgelist

#----------------------------------------------------------------    
# Generate Cl
#----------------------------------------------------------------
def bintest_get_maptypelist(finestN=6,getdivs=['equal'],z0=0.7):
    #get zedges
    zedges=bintest_get_finest_zedges(finestN,z0)
    maptypes=[] #list of maptype objects, put finest div first
    #WORKING HERE
    return maptypes

# Generate Cl for 6 bins
def bintest_get_Clvals(justread=True):
    pass
    #just do integrals for finest division,
    # combine bins to get Cl for other divisions
    #WORKING HERE


#----------------------------------------------------------------
# Make maps and run reconstructions
#----------------------------------------------------------------

#note that we can really just save maps for the finest division
# and then do some adding to get stats on combined bins
#  will need to add glm to do reconstructions though

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
        depthtest_TTscatter(0,depthtestz0,False)
        #depthtest_plot_zwindowfuncs(depthtestz0)
        #depthtest_plot_rhohist(depthtestz0,True)
        #depthtest_rho_tests()
