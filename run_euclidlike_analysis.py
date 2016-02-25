import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import host_subplot #to have two axis labels
import mpl_toolkits.axisartist as AA #to have two axis labels
from itertools import permutations
from scipy.optimize import leastsq
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
    surveys=[get_Euclidlike_SurveyType(z0=z0,onebin=True,tag='eucz{0:02d}'.format(int(10*z0))) for z0 in z0vals]
    bins=[s.binmaps[0] for s in surveys] #all surveys have only one bin
    if includeisw:
        iswmaptype=get_fullISW_MapType(zmax=15)
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
def depthtest_get_glm_and_rec(Nreal=1,z0vals=np.array([.3,.6,.7,.8]),minreal=0,justgetrho=0,dorell=0,dorho=1,dos=1,dochisq=1,dochisqell=0):
    t0=time.time()
    cldat=depthtest_get_Cl(justread=True,z0vals=z0vals)
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=depthtest_get_reclist(z0vals)
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorell=dorell,dorho=dorho,dos=dos,dochisq=dochisq,dochisqell=dochisqell)
    t1=time.time()
    print "total time for Nreal",Nreal,": ",t1-t0,'sec'

#get arrays of rho saved in .rho.dat files or .s.dat
def depthtest_read_rho_wfiles(z0vals=np.array([.3,.6,.7,.8]),varname='rho'):
    mapdir='output/depthtest/map_output/'
    files=['iswREC.eucz{0:02d}.fid.fullsky.depthtest.{1:s}.dat'.format(int(10*z0),varname) for z0 in z0vals]
    rhogrid=np.array([read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

#get arrays of rho saved in .rell.dat files
def depthtest_read_rell_wfiles(z0vals=np.array([.3,.6,.7,.8]),varname='rell'):
    mapdir='output/depthtest/map_output/'
    files=['iswREC.eucz{0:02d}.fid.fullsky.depthtest.{1:s}.dat'.format(int(10*z0),varname) for z0 in z0vals]
    rellgrid=np.array([read_relldat_wfile(mapdir+f) for f in files])
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
def depthtest_plot_rhohist(z0vals=np.array([.3,.6,.7,.8]),getrhopred=True,varname='rho',firstNreal=-1):
    plotdir='output/depthtest/plots/'
    testname="Depth test"
    rhogrid=depthtest_read_rho_wfiles(z0vals,varname)
    Nreal=rhogrid.shape[1]
    if firstNreal>0 and firstNreal<Nreal:
        Nreal=firstNreal
        rhogrid=rhogrid[:,:Nreal]
    if getrhopred:
        rhopred=depthtest_get_expected_rho(z0vals,varname)
    else:
        rhopred=[]
    plotname ='depthtest_{1:s}hist_r{0:05d}'.format(Nreal,varname)
    reclabels=['$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]

    if varname=='rho':
        plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='s':
        plot_shist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='chisq':
        plot_chisqhist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)

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

    plot_relldat(reclabels,testname,plotdir,plotname,rellgrid,rellpred,varname=varname)
        
#--------------------------------
# get expectation values of rho or s, choose variable via varname
def depthtest_get_expected_rho(z0vals=np.array([0.3,0.6,0.7,0.8]),varname='rho'):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rhopred=np.zeros_like(z0vals)
    for i in xrange(len(z0vals)):
        rhopred[i]=compute_rho_fromcl(cldat,reclist[i],varname=varname)
    return rhopred

def depthtest_get_expected_rell(z0vals=np.array([0.3,0.6,0.7,0.8]),varname='rell'):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rellpred=[]
    for i in xrange(z0vals.size):
        rellpred.append(compute_rell_fromcl(cldat,reclist[i],varname=varname))
    rellpred=np.array(rellpred)#[Nrec,Nell]
    #print rellpred
    return rellpred

#--------------------------------
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
    if finestN>1:
        zedges[-2]=2.
        dz=2./(finestN-1)
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
def bintest_get_maptypelist(finestN=6,getdivs=['all'],z0=0.7,sigz=0.05,includeisw=True):
    #get zedges
    zedges0=bintest_get_finest_zedges(finestN,z0) #for finest division
    zedges,divstr=bintest_get_zedgeslist(zedges0,getdivs,True) 
    Ntypes=len(zedges)
    maptypes=[] #list of maptype objects, put finest div first
    maintag='euc{0:d}bins{1:03d}div'.format(finestN,int(1000*sigz))
    if includeisw:
        iswmaptype=get_fullISW_MapType(zmax=15)
        maptypes.append(iswmaptype)
    for i in xrange(Ntypes):
        #print 'getting survey for zedges=',zedges[i]
        tag=maintag+divstr[i]
        survey=get_Euclidlike_SurveyType(sigz=sigz,z0=0.7,tag=tag,zedges=zedges[i])
        maptypes.append(survey)
    return maptypes

def bintest_get_binmaps(finestN=6,getdivs=['all'],z0=0.7,sigz=0.05,includeisw=True,justfinest=False):
    if justfinest:
        getdivs=['1'*finestN]
    maptypes=bintest_get_maptypelist(finestN,getdivs,z0,sigz,includeisw)
    binmaps,bintags=get_binmaplist(maptypes)
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
    rundat = ClRunData(tag='eucbintest{0:d}s{1:03d}'.format(finestN,int(1000*sigz)),iswilktag='eucbintest',rundir='output/eucbintest/',lmax=95,zmax=zmax)
    return getCl(binmaps,rundat,dopairs=['all'],DoNotOverwrite=justread)
# 
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
                        nextcl=combineCl_binlist(basecl,intags,combotag=outtag,newruntag=basecl.rundat.tag+'all',renamesingle=True)
                        FIRST=False
                    else:
                        nextcl=combineCl_binlist(nextcl,intags,combotag=outtag,renamesingle=True)
                    print '   nextcl.Nmap',nextcl.Nmap,'nextcl.Ncross',nextcl.Ncross
                    print '   len(nextcl.docross)',len(nextcl.docross)
        #write to file
        writeCl_file(nextcl)
    else:
        #read in data that has already been combined
        binmaps=bintest_get_binmaps(finestN,z0=z0,sigz=sigz)
        zmax=max(m.zmax for m in binmaps)
        rundat = ClRunData(tag='eucbintest{0:d}s{1:03d}all'.format(finestN,int(1000*sigz)),iswilktag='eucbintest',rundir='output/eucbintest/',lmax=95,zmax=zmax)
        nextcl= getCl(binmaps,rundat,dopairs=['all'],DoNotOverwrite=True)
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
        recdat=RecData(includeglm=includeglm,inmaptag=inmaptag)
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
        if not overwrite and os.path.isfile(outdir+datfile): #file exists
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
                rhoarray[r]=compute_rho_fromcl(cldat,reclist[r],varname=varname)
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
            rhoarray[r]=compute_rho_fromcl(cldat,reclist[r],varname=varname)

    if doplot:
        zedges0=bintest_get_finest_zedges(finestN,z0)
        allzedges=bintest_get_zedgeslist(zedges0,['all'],False)
        bintest_rhoexpplot(allzedges,divstr,rhoarray,varname)
    return divstr,rhoarray

#if we've computed Cl stuff for multiple values of sigz0, compare them
def bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.03,0.05],checkautoonly=False,varname='rho',plotdir='output/eucbintest/plots/',markerlist=[],colorlist=[],datsigs=[],datdivs=[]):
    rholist=[]
    for s in sigzlist:
        divstr,rho=bintest_get_rhoexp(finestN,z0,s,overwrite=False,doplot=False,varname=varname)
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
            rhogrid[s,r]=compute_rho_fromcl(cldat,reclist[r])
            print '    rho=',rhogrid[s,r]

def bintest_get_expected_rell(divstr,varname='rell'):
    cldat=bintest_get_Clvals()
    reclist=bintest_get_reclist(getdivs=divstr)
    rellpred=[]
    for i in xrange(len(reclist)):
        if varname=='rell':
            rellpred.append(compute_rell_fromcl(cldat,reclist[i],varname=varname))
    rellpred=np.array(rellpred)#[Nrec,Nell]
    return rellpred
    
#----------------------------------------------------------------
# Make maps and run reconstructions
#----------------------------------------------------------------

#note that we can really just save maps for the finest division
# and then do some adding to get stats on combined bins
#  will need to add glm to do reconstructions though

#use cldat to generate glm, alm, and maps; saves maps but not alm
def bintest_get_glm_and_rec(Nreal=1,divlist=['6','222','111111'],minreal=0,justgetrho=0,dorell=0):
    t0=time.time()
    allcldat=bintest_get_Clvals(justread=True) #default finestN,z0,sigz
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=bintest_get_reclist(getdivs=divlist)
    #get reduced cl with only desired divlist maps in it; 
    maptypes=bintest_get_maptypelist(finestN=6,getdivs=divlist,z0=0.7,sigz=0.05,includeisw=True)
    mapsfor=[mt.tag for mt in maptypes] #tags for maps we want to make
    cldat=get_reduced_cldata(allcldat,dothesemaps=mapsfor)
    
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorell=dorell)
    t1=time.time()
    print "total time for Nreal",Nreal,": ",t1-t0,'sec'

#get arrays of rho saved in .rho.dat files, or .s.dat files
def bintest_read_rho_wfiles(divlist=['6','222','111111'],sigz=0.05,varname='rho'):
    #print 'in READFILES, divlist=',divlist
    mapdir='output/eucbintest/map_output/'
    files=['iswREC.euc6bins{0:03d}div{1:s}.fid.fullsky.eucbintest6s{0:03d}all.{2:s}.dat'.format(int(1000*sigz),d,varname) for d in divlist]
    #print files
    rhogrid=np.array([read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

#get arrays of rho saved in .rell.dat files
def bintest_read_rell_wfiles(divlist=['6','222','111111'],sigz=0.05,varname='rell'):
    mapdir='output/eucbintest/map_output/'
    files=['iswREC.euc6bins{0:03d}div{1:s}.fid.fullsky.eucbintest6s{0:03d}all.{2:s}.dat'.format(int(1000*sigz),d,varname) for d in divlist]
    rellgrid=np.array([read_relldat_wfile(mapdir+f) for f in files])
    return rellgrid

#----------------------------------------------------------------
# Analysis: make plots
#----------------------------------------------------------------

#plot expectation value of rho for different binning strategy
# with illustrative y axis
# also works for s, switch in variable varname
#   dataplot=[(datrho,datstd,datdiv,datlabel,datcol),...] #if you want to plot
#       some data points, pass their plotting info here. datsig!=0 adds error bars
def bintest_rhoexpplot(allzedges,labels,rhoarraylist,labellist=[],outname='',legtitle='',markerlist=[],colorlist=[],outtag='',varname='rho',dotitle=False,plotdir='output/eucbintest/plots/',datplot=[]):
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
        ax2.set_xlim((.87,.92))
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
        if labellist:
            ax2.scatter(rhoarray,yvals,label=labellist[i],color=colorlist[i],marker=m,s=50)
        else:
            ax2.scatter(rhoarray,yvals,color=colorlist[i],marker=m)

    if labellist:
        if DODATA:
            ax2.set_xlim((.78,.99))
            plt.legend(bbox_to_anchor=(1,.8),fontsize=16,title=legtitle)
        else:
            plt.legend(loc=legloc,fontsize=16,title=legtitle)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels()[0], visible=False)#don't show number at first label
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
        plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='s':
        plot_shist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='chisq':
        plot_chisqhist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
        
        
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
        plot_relldat(reclabels,testname,plotdir,plotname,rellgrid,rellpred)   
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
        cosm=Cosmology(cosmparamfile)
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
def caltest_get_fidbins():
    z0=0.7
    fidbins=depthtest_get_binmaps(z0vals=np.array([z0])) #isw+one galbin
    if len(fidbins)!=2:
        print 'more items in fidbins than expected:',fidbins
    return fidbins

def caltest_get_clfid():
    bins=caltest_get_fidbins()
    zmax=max(m.zmax for m in bins)
    #use depthtest tags since we'll read in those (already calculated) Cl
    rundat = ClRunData(tag='depthtest',rundir='output/depthtest/',lmax=95,zmax=zmax)
    fidcl=getCl(bins,rundat,dopairs=['all'],DoNotOverwrite=True)
    return fidcl

# get glmdata object with fiducial (no calib error) bin and isw info
#   is a dummy glmdata object; no data or realizations, just map names
def caltest_get_fidglm(fidcl=0):
    #get fiducial cl, with no calibration error
    if not fidcl:
        fidcl=caltest_get_clfid()

    #get fid glmdat; no data needed, just mapnames, etc
    glmdat=get_glm(fidcl,Nreal=0,matchClruntag=True)
    return glmdat

#return Cl^cal list; return array of shape Nvariance x Nell, NOT ClData object
#  assumes all cl_cal have same shape, just different map variances
def caltest_get_clcallist(varlist=[1.e-1,1.e-2,1.e-3,1.e-4],lmax=30,lmin=0,shape='g',width=10.):
    Nvar=len(varlist)
    maxvar=max(varlist)
    maxind=np.where(varlist==maxvar)[0][0]
    if shape=='g':
        maxcl= gen_error_cl_fixedvar_gauss(maxvar,lmax,lmin,width)
    elif shape=='l2':
        maxcl = gen_error_cl_fixedvar_l2(maxvar,lmax,lmin)

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
def caltest_apply_caliberrors(varlist,Nreal=0,shape='g',width=10.,lmin=0,lmax=30,overwritecalibmap=False,scaletovar=False):
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
    if Nreal:
        print 'Generating calibraiton error maps.'
    dothesemods=get_fixedvar_errors_formaps(glmdat,calinfolist,overwrite=overwritecalibmap,Nreal=Nreal) #generates calibration error maps, returns [(maptag,modtag,masktag)]list

    outglmdatlist=[]
    #apply calibration errors
    for v in varlist:
        scaling=np.sqrt(v/refvar)
        print 'scaling=',scaling
        newcaltag=getmodtag_fixedvar(v,shape,lmin,lmax)
        print 'Applying calibration errors, newcaltag',newcaltag
        outglmdatlist.append(apply_caliberror_to_manymaps(glmdat,dothesemods,Nreal=Nreal,calmap_scaling=scaling,newmodtags=[newcaltag])) #returns dummyglmdat

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

def caltest_get_reclist(varlist,shape='g',width=10.,lmin=0,lmax=30,recminelllist=np.array([1])):
    reclist=[]
    fidbins=caltest_get_fidbins()
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    lsstype=fidbins[1].typetag
    calinfolist=[(lssbin,v,lmax,shape,width,lmin) for v in varlist]
    fidglm=caltest_get_fidglm()
    dothesemods=get_fixedvar_errors_formaps(fidglm,calinfolist,overwrite=False,Nreal=0) #returns [(maptag,modtag,masktag)]list
    #put fidicual in
    includecl=[lssbin]
    inmaptype=lsstype
    if recminelllist.size==1: #should eventually remove this, but old data has no tag
        NOLMINTAG=False#True
    else:
        NOLMINTAG=False
    for recminell in recminelllist:
        reclist.append(RecData(includecl,includecl,inmaptype,'unmod',recminell,nolmintag=NOLMINTAG))
    for m in dothesemods:
        includeglm=[m]
        rectag=m[1]#modtag
        for recminell in recminelllist:
            reclist.append(RecData(includeglm,includecl,inmaptype,rectag,recminell,nolmintag=NOLMINTAG))
    return reclist #includes fiducial case as first entry

#having already generated maps for reconstructions with calib errors,
# do isw reconstructions from the maps, computing rho and s stats
# set domaps to False if you just want to recalculate states like rho,s 
def caltest_iswrec(Nreal,varlist,shape='g',width=10.,callmin=0,callmax=30,overwritecalibmap=False,scaletovar=False,recminelllist=np.array([1]),domaps=True):
    fidcl=caltest_get_clfid()
    dummyglm=caltest_apply_caliberrors(varlist,0,shape,width,callmin,callmax,overwritecalibmap,scaletovar)#includes fidicual case
    reclist=caltest_get_reclist(varlist,shape,width,callmin,callmax,recminelllist=recminelllist)
    doiswrec_formaps(dummyglm,fidcl,Nreal,reclist=reclist,domaps=domaps)

#---------------------------------------------------------------
# rhocalc utils
#---------------------------------------------------------------

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
def caltest_getrhodat_fromfiles(varlist,shape='g',width=10.,lmin=0,lmax=30,recminelllist=np.array([1]),varname='rho'):
    NOLMINTAG=False#recminelllist.size==1
    Nvar=len(varlist)
    #read in rho values
    modnames=[getmodtag_fixedvar(v,shape,lmin,lmax,width) for v in varlist]
    mapdir='output/depthtest/map_output/'
    outrho=[]
    counter=0
    for recminell in recminelllist:
        if NOLMINTAG:
            reclminstr=''
        else:
            reclminstr="-lmin{0:02d}".format(recminell)
        files=['iswREC.eucz07.{0:s}.fullsky{2:s}.depthtest.{1:s}.dat'.format(modname,varname,reclminstr) for modname in modnames]
        #append fiducial case (no cal error) for that lmin
        files.append('iswREC.eucz07.unmod.fullsky{1:s}.depthtest.{0:s}.dat'.format(varname,reclminstr))
        rhogrid=np.array([read_rhodat_wfile(mapdir+f) for f in files])#filesxrho
        outrho.append(rhogrid)
    outrho=np.array(outrho)
    return outrho #[lmin,calerror,realization]

# will return rho data needed for plotting datapoints on caltest plots
# will return 1d list of tuples, in order first of shape, then lminlist
#   note that caltest_compare_clcal_shapes functions assumes just one
#     or the other of shape and reclmin are varied
def caltest_getdataplot_forshapecompare(varname='rho',varlist=[],shapelist=['g'],widthlist=[10.],lminlist=[0],lmaxlist=[30],labellist=[''],cleanplot=False,recminelllist=np.array([1]),colorlist=['#e41a1c'],getlabels=False):
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
        rhogrid=caltest_getrhodat_fromfiles(varlist,shape,widthlist[i],caliblmin,caliblmax,recminelllist,varname)#3D: [lmin,calerror,realization]
        #  last entry in calerror dimension is value for no calibration error
        
        #print rhogrid.shape
        Nreal=rhogrid.shape[2]
        Nlmin=rhogrid.shape[0]
        if not labellist[i] and len(recminelllist)==1:
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
            refmeans[k]=np.mean(rhogrid[k,-1,:])
            sigs[k,:]=np.array([np.std(rhogrid[k,j,:]) for j in xrange(Nvar)])
            plotdatalist.append((varlist,means[k,:],label,col,sigs[k,:],refmeans[k],Nreal))
        #(datvar,datrho,datlabel,datcol,datsig),...]

    return plotdatalist

#if cleanplot, no title, sparser legend.
def caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='rho',lmaxlist=[],lminlist=[],widthlist=[],dodataplot=True,shortvarlist=[],outtag='',cleanplot=False,reclmin=1,plotdir='output/caltest_plots/'):
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
    for s in xrange(Nshapes):
        nolmintag=False
        rhoexplist.append(caltest_get_rhoexp(varlist,lmax=lmaxlist[s],lmin=lminlist[s],shape=shapelist[s],width=widthlist[s],overwrite=False,doplot=False,saverho=True,varname=varname,reclmin=reclmin,nolmintag=nolmintag)) #nolmintag for compatibility with pre-lmintest data
            
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
        dataplot=caltest_getdataplot_forshapecompare(varname,shortvarlist,shapelist,widthlist,lminlist,lmaxlist,cleanplot=cleanplot,recminelllist=np.array([reclmin]),colorlist=colorlist,getlabels=False) #don't get labels
    else:
        dataplot=[]

    caltest_rhoexpplot(varlist,rhoexplist,labels,outtag=outtag,varname=varname,datplot=dataplot,cleanplot=cleanplot,colorlist=colorlist,plotdir=plotdir)

#one shape, several lmin
def caltest_compare_lmin(varlist,shapecal='g',varname='rho',lmaxcal=30,lmincal=0,widthcal=10.,dodataplot=True,shortvarlist=[],outtag='',cleanplot=False,recminelllist=np.array([1]),shortrecminelllist=np.array([]),plotdir='output/caltest_plots/',justdat=False):
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
            rhoexplist.append(caltest_get_rhoexp(varlist,lmax=lmaxcal,lmin=lmincal,shape=shapecal,width=widthcal,overwrite=False,doplot=False,saverho=True,varname=varname,reclmin=recminelllist[i]))
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
def caltest_get_rhoexp(varlist=[1.e-4],lmax=30,lmin=1,shape='g',width=10.,overwrite=False,doplot=True,saverho=True,varname='rho',filetag='',reclmin=1,plotdir='output/caltest_plots/',nolmintag=False):
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
    fidbins=caltest_get_fidbins()
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    fidcl=caltest_get_clfid()

    #construct map-mod combos for the variances given
    mapmods=caltest_getmapmods_onebin(lssbin,varlist,lmax,lmin,shape,width)
    
    #generate calibration errors with fixed variance, spread through Cl lmin-lmax
    #for each variance, get calib error Cl
    clmodlist=[]
    for mm in mapmods:
        clmodlist.append(apply_additive_caliberror_tocl(fidcl,[mm]))
    # include fidicual cl as last entry
    if varlist[-1]!=0.:
        print "  appending fiducial case, no calib error"
        varlist.append(0.)
        clmodlist.append(fidcl)

    #get recdat; only need to pass maptags, not modtags, since cldata objects
    #  note that his means we can use the same recdat for all variances
    recdat=RecData(includeglm=[lssbin],inmaptag=lssbin[:lssbin.rfind('_bin0')],minl_forrec=reclmin)
    
    # return array of shape [Nvar,Nell]
    Nrec=len(varlist)
    rhoarray=np.zeros(Nrec)
    for r in xrange(Nrec):
        rhoarray[r]=compute_rho_fromcl(fidcl,recdat,reccldat=clmodlist[r],varname=varname)

    #if save, write to file
    if saverho:
        f=open(outdir+datfile,'w')
        f.write('Calib error test: Clcal shape={0:s}, ell={1:d}-{2:d}\n'.format(shape,lmin,lmax))
        f.write('var(c(nhat))   <{0:s}>\n'.format(varname))
        f.write(''.join(['{0:.2e} {1:8.3f}\n'.format(varlist[i],rhoarray[i]) for i in xrange(Nrec)]))
        f.close()

    if doplot:
        caltest_rhoexpplot(varlist,rhoarray,varname=varname,outtag=shapestr,plotdir=plotdir)
    return rhoarray

#-------
def caltest_getmapmods_onebin(lssbintag,varlist=[1.e-1,1.e-2,1.e-3,1.e-4],lmax=30,lmin=0,shape='g',width=10.):
    #construct map-mod combos for the variances given
    if shape=='g':
        mapmods=[(lssbintag,getmodtag_fixedvar_gauss(v,width,lmax,lmin)) for v in varlist]
    elif shape=='l2':
        mapmods=[(lssbintag,getmodtag_fixedvar_l2(v,lmax,lmin)) for v in varlist]
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
        outname='caltest_'+varname+'_exp'+outtag+'.png'

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.2)
    fig.subplots_adjust(hspace=.3)
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
    ax1.set_xlabel(r'Variance of calib. error field ${\rm var}[c(\hat{{n}})]$')
    ax1.axhline(0,color='grey',linestyle='-')
    if varname=='rho':
        ax1.set_ylim(-.4,1.)
        ax1.set_ylabel(r'$\langle \rho \rangle$')
    elif varname=='s':
        ax1.set_ylabel(r'$\langle s \rangle$')
    elif varname=='chisq':
        ax1.set_ylabel(r'$\langle \chi^2 \rangle$')

    for i in xrange(len(rhoarraylist)):
        print len(varlist[:-1]),rhoarraylist[i][:-1].shape
        ax1.semilogx(varlist[:-1],rhoarraylist[i][:-1],label=labellist[i],color=colorlist[i%len(colorlist)])

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
            ax1.errorbar(datvar,datrho,yerr=datsig/datrefmean,label=datlabel,color=datcol,linestyle='None',marker='o')
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
        outname='caltest_'+varname+'_exp'+outtag+'.png'

    fig=plt.figure(figsize=(7,7))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.2)
    fig.subplots_adjust(hspace=.3)
    ax1 = plt.subplot(3,1,(1,2)) #top part has rho
    ax2 = plt.subplot(3,1,3,sharex=ax1) #bottom has rho/rhofid
    
    for item in ([ax1.xaxis.label, ax1.yaxis.label,ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax1.get_yticklabels()+ax2.get_yticklabels()):
        item.set_fontsize(18)
    for item in (ax1.get_yticklabels()+ax2.get_yticklabels()):
        item.set_fontsize(16)
    if not labellist:
        labellist=['']
    if not colorlist:
        colorlist=scattercolors

    plt.sca(ax1)
    ax1.grid(True)
    ax1.set_xscale('log')
    ax1.axhline(0,color='grey',linestyle='-')
    ax1.axhline(1,color='grey',linestyle='-')
    ax2.axhline(0,color='grey',linestyle='-')
    ax2.axhline(-1,color='grey',linestyle='-')
    plt.setp(ax1.get_xticklabels(), visible=False)
    #ax1.ticklabel_format(axis='y',style='')
    ax2.grid(True)
    if varname=='rho':
        ax1.set_ylim(-.4,1.)
        ax1.set_ylabel(r'$\langle \rho \rangle$')
        ax2.set_ylabel(r'$\langle \rho \rangle /\langle \rho_{{c=0}} \rangle -1$')
    elif varname=='s':
        ax1.set_ylabel(r'$\langle s \rangle$')
        ax2.set_ylabel(r'$\langle s \rangle /\langle s_{{c=0}} \rangle - 1$')
    elif varname=='chisq':
        ax1.set_ylabel(r'$\langle \chi^2 \rangle$')
        ax2.set_ylabel(r'$\langle \chi^2 \rangle /\langle \chi^2_{{c=0}} \rangle - 1$')
    
    ax2.set_xlabel(r'Variance of calib. error field ${\rm var}[c(\hat{{n}})]$')
    if plotlines:
        for i in xrange(len(rhoarraylist)):
            print len(varlist[:-1]),rhoarraylist[i][:-1].shape
            ax1.semilogx(varlist[:-1],rhoarraylist[i][:-1],label=labellist[i],color=colorlist[i%len(colorlist)])
            ax2.semilogx(varlist[:-1], rhoarraylist[i][:-1]/rhoarraylist[i][-1]-1.,label=labellist[i],color=colorlist[i%len(colorlist)])

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
            ax1.errorbar(datvar,datrho,yerr=datsig/datrefmean,label=datlabel,color=datcol,linestyle='None',marker='o')#,capsize=5)
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
    ax2.set_yticks([-1.5+i*.25 for i in xrange(8)])
    
    plt.sca(ax1)
    if varname=='rho':
        plt.legend(fontsize=18,title=legtitle,loc='lower left',numpoints=1)
    elif varname=='s':
        plt.legend(fontsize=18,title=legtitle,loc='upper left',numpoints=1)
    elif varname=='chisq':
        plt.legend(fontsize=18,title=legtitle,loc='upper left',numpoints=1)

    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()
#---------------------------------------------------------------
# plot comparison bewteen Cl cal and Cl gal or Cl gal-ISW to understand
# why the transition in rho happens where it is
def caltest_Clcomp(varlist=[1.e-7,1.e-6,1.e-5,1.e-4],shape='g',callmin=0,callmax=30,width=10.):
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
            var,maxl,minl=parsemodtag_fixedvar_l2(ctag)
            thiscalcl=gen_error_cl_fixedvar_l2(var,maxl,minl)
        elif ctag[:1]=='g':#gaussian
            var,maxl,minl,width=parsemodtag_fixedvar_gauss(ctag)
            thiscalcl=gen_error_cl_fixedvar_gauss(var,maxl,minl,width=width)
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

    plotdir='output/caltest_plots/'
    plotname='caltest_cl_compare'
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
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
    dummyalm=get_dummy_recalmdat(dummyglm,reclist,outruntag=dummyglm.runtag)
    
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
        modtag=getmodtag_fixedvar(varlist[i],'g',lmin=0,lmax=30,width=10)
        recmapf=dummyalm.get_mapfile_fortags(r,maptag='iswREC.eucz07',modtag=modtag)
        recmapfiles.append(recmapf)
        reclabels.append(r'$\langle c^2(\hat{{n}})\rangle$={0:.0e}'.format(varlist[i]))
        if savepngmaps:
            recmap=hp.read_map(recmapf,verbose=False)
            #set up color scheme for lss map
            mono_cm=matplotlib.cm.Greys_r
            mono_cm.set_under("w") #set background to white
            lssf=dummyglm.get_mapfile_fortags(r,'eucz07_bin0',modtag)
            lssm=hp.read_map(lssf,verbose=False)
            plotmax=lssmax#0.7*np.max(np.fabs(lssm))
            lssfbase=lssf[lssf.rfind('/')+1:lssf.rfind('.fits')]
            hp.mollview(lssm,title=lssfbase,unit=r' $\delta\rho/\rho$',max=plotmax,min=-1*plotmax,cmap=mono_cm)
            plt.savefig(plotdir+'mapplot_'+lssfbase+'.png')
            lsspngf.append(plotdir+'mapplot_'+lssfbase+'.png')
            maxtemp=np.max(truemap)
            maxtemp=max(maxtemp,np.max(recmap))
            plotmax=tmax#0.7*maxtemp
            #true isw
            if dotrueiswpng:
                truefbase=truemapf[truemapf.rfind('/')+1:truemapf.rfind('.fits')]
                hp.mollview(truemap,title=truefbase,unit='K',max=plotmax,min=-1*plotmax)
                plt.savefig(plotdir+'mapplot_'+truefbase+'.png')
                iswpngf=plotdir+'mapplot_'+truefbase+'.png'
                trueiswpng=False #only do this once
            #reconstructed isw
            recfbase=recmapf[recmapf.rfind('/')+1:recmapf.rfind('.fits')]
            hp.mollview(recmap,title=recfbase,unit='K',max=plotmax,min=-1*plotmax)
            recpngf.append(plotdir+'mapplot_'+recfbase+'.png')
            plt.savefig(plotdir+'mapplot_'+recfbase+'.png')
    #then do fiducial case
    fidrecf=dummyalm.get_mapfile_fortags(r,'iswREC.eucz07','unmod')
    recmapfiles.append(fidrecf)
    reclabels.append('No calib. error')
    if savepngmaps:
        lssf=dummyglm.get_mapfile_fortags(r,'eucz07_bin0')
        lssm=hp.read_map(lssf,verbose=False)
        plotmax=lssmax#0.7*np.max(np.fabs(lssm))
        lssfbase=lssf[lssf.rfind('/')+1:lssf.rfind('.fits')]
        hp.mollview(lssm,title=lssfbase,unit=r' $\delta\rho/\rho$',max=plotmax,min=-1*plotmax,cmap=mono_cm)
        plt.savefig(plotdir+'mapplot_'+lssfbase+'.png')
        lsspngf.append(plotdir+'mapplot_'+lssfbase+'.png')
        maxtemp=np.max(truemap)
        maxtemp=max(maxtemp,np.max(recmap))
        plotmax=tmax#0.7*maxtemp
        recmapf=fidrecf
        recfbase=recmapf[recmapf.rfind('/')+1:recmapf.rfind('.fits')]
        hp.mollview(recmap,title=recfbase,unit='K',max=plotmax,min=-1*plotmax)
        recpngf.append(plotdir+'mapplot_'+recfbase+'.png')
        plt.savefig(plotdir+'mapplot_'+recfbase+'.png')

    #colors=['#253494','#2c7fb8','#41b6c4','#a1dab4','#ffffcc']
    colors=['#a6611a','#08519c','#41b6c4','#78c679','#ffffb2']

    plotname='TrecTisw_scatter_caltest.r{0:05d}'.format(r)
    plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir,plotname,colors)


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
        outname=varname+'test_'+statname+'_exp'+outtag+'.png'
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
    surveys=[get_Euclidlike_SurveyType(z0=z0[i],onebin=True,tag=maptags[i]) for i in xrange(z0.size)]
    bins=[s.binmaps[0] for s in surveys] #surveys all just have one bin
    if includeisw:
        iswmaptype=get_fullISW_MapType(zmax=15)
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
    rundat = ClRunData(tag='z0test',rundir='output/zdisttest/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    return getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)
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
            recdat=RecData(includeglm=[simmaptags[ns]+'_bin0'],includecl=[recmaptags[nr]+'_bin0'],inmaptag=simmaptags[ns],rectag=recmaptags[nr])
            reclist.append(recdat)
        recgrid.append(reclist)
    return recgrid
    
# will output rhodat in len(simz0)xlen(rez0) array
# if simz0 and recz0 are passed as arrays, use those as the z0 vals
# if either are passed as an empty array, replace it with all vals indicated
#    by the perrors, fidz0 parameters
# will use perrors and fidz0 to get Cl data, so they should match in either case
def z0test_get_rhoexp(simz0=np.array([]),recz0=np.array([]),perrors=np.array([1,10,20,50]),fidz0=.7,overwrite=False,saverho=True,doplot=False,varname='rho',filetag='',plotdir='output/zdisttest/plots/'):
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
            rhoarray[ns,nr]=compute_rho_fromcl(cldat,recgrid[ns][nr],reccldat=cldat,varname=varname)

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
def z0test_onesim_plot(fidz0=0.7,perrors=np.array([1,10,20,50]),varname='rho',colorlist=[],plotdir='output/zdisttest/plots/',outtag='onesim',outname=''):
    logplot=True
    simz0=np.array([fidz0])
    recz0=z0test_getz0vals(perrors,fidz0)
    rhogrid=z0test_get_rhoexp(simz0=simz0,recz0=recz0,overwrite=False,saverho=True,doplot=False,varname=varname,filetag=outtag) #should be 1xNrec
    
    Nsim=simz0.size
    Nrec=recz0.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='z0test_'+varname+'_exp'+outtag+'.png'

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.2)
    ax1=plt.subplot(1,1,1)
    
    ax1.axhline(0,color='grey',linestyle=':')
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)
    ax1.set_xlabel(r'$z_0$ used in ISW reconstruction')
    if varname=='rho':
        ax1.set_ylabel(r'$\left[\langle \rho \rangle -\langle \rho \rangle_{{\rm match}} \right]/\langle \rho \rangle_{{\rm match}}$')
        if logplot:
            ax1.set_ylim((-4.e-3,5.e-8))
            aloc=(.2,.7)
            linthreshy=1.e-7
        else:
            aloc=(.6,.5)
            ax1.set_ylim((-1.3e-3,1.e-4))

        bestval=np.max(rhogrid)
    elif varname=='s':
        ax1.set_ylabel(r'$\left[\langle s \rangle - \langle s \rangle_{{\rm match}}\right] /\langle s \rangle_{{\rm match}}$')
        if logplot:
            aloc=(.25,.6)
            linthreshy=1.e-5
        else:
            aloc=(.6,.9)
            ax1.set_ylim((-1.e-2,.3))
        bestval=np.min(rhogrid)

    if varname=='rho':
        varstr=r'\rho'
        ax1.plot(recz0,rhogrid[0,:]/bestval-1,marker='d')
    elif varname=='s':
        varstr='s'
        ax1.plot(recz0,rhogrid[0,:]/bestval -1. ,marker='d')
    plt.annotate(r'True (sim.) $z_0={0:0.1f}$'.format(fidz0)+'\n'+r'$\langle {1:s}\rangle_{{\rm match}}={2:0.3f}$'.format(fidz0,varstr,bestval),xy=aloc,horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    xmin,xmax=ax1.get_xlim()
    xwide=xmax-xmin
    ax2=ax1.twiny() #top border with percent error units
    ax2.set_xticks([(z-xmin)/xwide for z in recz0]) #put them in right places
    perrorlabels=[]
    count=1
    for z in recz0:
        if z<fidz0:
            if perrors[-1*count]==1:
                pstr=''
            else:
                pstr='-{0:d}%'.format(perrors[-1*count])                
            count+=1
        elif z==fidz0:
            pstr=r'$\pm$1%'#r'$z_0^{\rm sim}$'
            count=0
        elif z>fidz0:
            if perrors[count]==1:
                pstr=''
            else:
                pstr='+{0:d}%'.format(perrors[count])
            count+=1
        perrorlabels.append(pstr)
    ax2.set_xticklabels(perrorlabels)
    
    if logplot:
        #ax1.yaxis.grid(True)
        ax1.set_yscale('symlog',linthreshy=linthreshy)
        x=np.arange(xmin,xmax,.01)
        ax1.fill_between(x,-linthreshy,linthreshy,color='none',edgecolor='grey',hatch='/',linewidth=0)
    else:
        ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
        ax1.set_xlim((.24,1.1))
    
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname=varname+'test_'+statname+'_exp'+outtag+'.png'
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
        outname='z0test_'+varname+'_exp'+outtag+'.png'

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
def z0test_Clcomp(perrors=np.array([1,10,20,30,50]),fidz0=.7):
    z0vals=z0test_getz0vals(perrors,fidz0)#for labels
    iswind=0
    bins=z0test_get_binmaps(perrors,fidz0)#includes ISW at 0, others in same order as z0
    cldat=z0test_get_Cl(perrors=perrors,fid=fidz0)
    l=np.arange(cldat.Nell)
    #set up color range
    cm=plt.get_cmap('Spectral_r')
    cNorm=colors.LogNorm()#max and min numbers colors need to span
    scalarMap=cmx.ScalarMappable(norm=cNorm,cmap=cm)
    z0cols=scalarMap.to_rgba(z0vals)
    clscaling=2*l+1.#l*(l+1.)/(2*np.pi)
    
    #to get colorbar key, need ot set up a throw-away map
    dummyz=[[0,0],[0,0]]
    dummylevels=z0vals
    dummyplot=plt.contourf(dummyz,dummylevels,cmap=cm,norm=colors.LogNorm())
    plt.clf()
    fig=plt.gcf()
    
    #set up actual plot
    fig=plt.figure(0)
    ax=plt.subplot()
    plt.title(r'Comparing $C_{{\ell}}$ of galaxies, ISW')
    #plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}/2\pi$')
    plt.ylabel(r'$(2\ell+1)C_{{\ell}}$')
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((0,30))
    plt.ylim((1.e-11,1.))
    plt.yscale('log')
    for i in xrange(z0vals.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            xiswind=cldat.crossinds[i+1,0]
            plt.plot(l,np.fabs(cldat.cl[autoind,:])*clscaling,color=z0cols[i],linestyle='-')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:])*clscaling,color=z0cols[i],linestyle='--')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color=z0cols[i],linestyle=':')
    #dummy lines for legend
    line1,=plt.plot(np.array([]),np.array([]),color='black',linestyle='-',label='gal-gal')
    line2,=plt.plot(np.array([]),np.array([]),color='black',linestyle='--',label='ISW-gal')
    line3,=plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color='black',linestyle=':',label='ISW-gal/gal-gal')

    #set up colorbar
    logminvar=int(np.log10(min(z0vals)))
    logmaxvar=int(np.log10(max(z0vals)))+1
    Nlog=logmaxvar-logminvar
    z0ticks=z0vals#[.1*10**(logminvar+n) for n in xrange(Nlog)]
    #cbaxes=fig.add_axes([.8,.1,.03,.8])#controls location of colorbar
    colbar=fig.colorbar(dummyplot,ticks=z0ticks)
    colbar.ax.set_yticklabels(['{0:0.3f}'.format(z0) for z0 in z0vals])
    colbar.set_label(r'$z_0$')

    plt.legend(handles=[line1,line2,line3],loc='lower right')
    plotdir='output/zdisttest/plots/'
    plotname='z0test_cl_compare'
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
    
#--------------------------------------------------------------------
# bztest funcs
#--------------------------------------------------------------------
def bztest_get_binmaps(b2vals=np.array([0.,.01,.1,.5]),fid=0,z0=0.7,includeisw=True):
    addfid=fid not in b2vals
    maptags=['eucz{0:04d}_b2{1:04d}'.format(int(z0*10000),int(b2*1000)) for b2 in b2vals]
    surveys=[get_Euclidlike_SurveyType(z0=z0,onebin=True,tag=maptags[i],b2=b2vals[i]) for i in xrange(b2vals.size)]
    if addfid:
        maptags.append('eucz{0:04d}_b2{1:04d}'.format(int(z0*10000),int(fid*1000)))
        surveys.append(get_Euclidlike_SurveyType(z0=z0,onebin=True,tag=maptags[-1],b2=fid) )
    bins=[s.binmaps[0] for s in surveys] #surveys all just have one bin
    if includeisw:
        iswmaptype=get_fullISW_MapType(zmax=15)
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
    rundat = ClRunData(tag='bztest',rundir='output/zdisttest/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    return getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)

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
            recdat=RecData(includeglm=[simmaptags[ns]+'_bin0'],includecl=[recmaptags[nr]+'_bin0'],inmaptag=simmaptags[ns],rectag=recmaptags[nr])
            reclist.append(recdat)
        recgrid.append(reclist)
    return recgrid

# will output rhodat in len(simb2)xlen(recb2) array
# if simb2 and recb2 are passed as arrays, use those as the b2 vals
def bztest_get_rhoexp(simb2=np.array([0.,.01,.1,.5]),recb2=np.array([0.,.01,.1,.5,1.,2.,5.,10.]),overwrite=False,saverho=True,doplot=False,varname='rho',filetag=''):
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
            rhoarray[ns,nr]=compute_rho_fromcl(simcldat,recgrid[ns][nr],reccldat=reccldat,varname=varname)

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
def bztest_onesim_plot(fidb2=0.5,recb2=np.array([0.,.01,.1,.5,1.,2.,5.,10.]),varname='rho',plotdir='output/zdisttest/plots/',outtag='onesim',outname=''):
    simb2=np.array([fidb2])
    rhogrid=bztest_get_rhoexp(simb2=simb2,recb2=recb2,overwrite=False,saverho=True,doplot=False,varname=varname,filetag=outtag) #should be 1xNrec
    
    Nsim=simb2.size
    Nrec=recb2.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='bztest_'+varname+'_exp'+outtag+'.png'

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.2)
    ax1=plt.subplot(1,1,1)
    #ax1.yaxis.grid(True)
    ax1.axhline(0,color='grey',linestyle=':')
    ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    ax1.set_xlabel(r"$b_2$ used in ISW reconstruction")
    if varname=='rho':
        ax1.set_ylabel(r'$\left[ \langle \rho \rangle-\langle \rho \rangle_{{\rm match}} \right]/\langle \rho \rangle_{{\rm match}}$')
        #ax1.set_yscale('log')
        ax1.set_ylim((-7.e-5,1.e-5))
        bestval=np.max(rhogrid)
    elif varname=='s':
        ax1.set_ylabel(r'$\left[\langle s \rangle - \langle s \rangle_{{\rm match}}\right] /\langle s \rangle_{{\rm match}}$')
        #ax1.set_yscale('log')
        ax1.set_ylim((-2.e-3,.025))
        bestval=np.min(rhogrid)

    if varname=='rho':
        varstr=r'\rho'
        ax1.plot(recb2,rhogrid[0,:]/bestval-1 ,marker='d')
        aloc=(.7,.5)
    elif varname=='s':
        varstr='s'
        aloc=(.7,.9)
        ax1.plot(recb2,rhogrid[0,:]/bestval -1. ,marker='d')
    plt.annotate(r'True (sim.) $b_2={0:0.1f}$'.format(fidb2)+'\n'+r'$\langle {0:s}\rangle_{{\rm match}}={1:0.3f}$'.format(varstr,bestval),xy=aloc,horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction')


    linthreshx=.01
    ax1.set_xscale('symlog',linthreshx=linthreshx)
    ax1.set_xlim((-.001,13))
    #hash fills in where there is a linear scale
    ymin,ymax=ax1.get_ylim()
    y=np.arange(-linthreshx,linthreshx,.1*linthreshx)
    ax1.fill_between(y,ymin,ymax,color='none',edgecolor='grey',hatch='/',linewidth=0)
    
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname=varname+'test_'+statname+'_exp'+outtag+'.png'
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
        outname='bztest_'+varname+'_exp'+outtag+'.png'

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
        outname=varname+'test_'+statname+'_exp'+outtag+'.png'
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()    
#--------------------------
def bztest_Clcomp(b2vals=np.array([0.,.01,.1,.5,1.,2.,5.,10.])):
    iswind=0
    bins=bztest_get_binmaps(b2vals)#includes ISW at 0, others in same order as z0
    cldat=bztest_get_Cl(b2vals=b2vals)
    l=np.arange(cldat.Nell)
    
    
    #to get colorbar key, need ot set up a throw-away map
    dummyz=[[0,0],[0,0]]
    dummylevels=[]
    for b in b2vals:
        if b: dummylevels.append(b)
    dummylevels=np.array(dummylevels)
    #set up color range
    cm=plt.get_cmap('Spectral_r')
    cNorm=colors.LogNorm()#max and min numbers colors need to span
    scalarMap=cmx.ScalarMappable(norm=cNorm,cmap=cm)
    b2cols=scalarMap.to_rgba(dummylevels)
    clscaling=2*l+1 #l*(l+1.)/(2*np.pi)
    #print dummylevels
    dummyplot=plt.contourf(dummyz,dummylevels,cmap=cm,norm=colors.LogNorm())
    plt.clf()
    fig=plt.gcf()
    
    #set up actual plot
    fig=plt.figure(0)
    ax=plt.subplot()
    plt.title(r'Comparing $C_{{\ell}}$ of galaxies, ISW')
    #plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}/2\pi$')
    plt.ylabel(r'$(2\ell+1)C_{{\ell}}$')
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((0,30))
    plt.ylim((1.e-11,1.))
    plt.yscale('log')
    for i in xrange(b2vals.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        elif b2vals[i]==0.:
            autoind=cldat.crossinds[i+1,i+1]
            xiswind=cldat.crossinds[i+1,0]
            plt.plot(l,np.fabs(cldat.cl[autoind,:])*clscaling,color='grey',linestyle='-')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:])*clscaling,color='grey',linestyle='--')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color='grey',linestyle=':')
        else:
            autoind=cldat.crossinds[i+1,i+1]
            xiswind=cldat.crossinds[i+1,0]
            plt.plot(l,np.fabs(cldat.cl[autoind,:])*clscaling,color=b2cols[i-1],linestyle='-')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:])*clscaling,color=b2cols[i-1],linestyle='--')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color=b2cols[i-1],linestyle=':')
    #dummy lines for legend
    line1,=plt.plot(np.array([]),np.array([]),color='black',linestyle='-',label='gal-gal')
    line2,=plt.plot(np.array([]),np.array([]),color='black',linestyle='--',label='ISW-gal')
    line3,=plt.plot(np.array([]),np.array([]),color='black',linestyle=':',label='ISW-gal/gal-gal')

    #set up colorbar
    b2ticks=dummylevels#[.1*10**(logminvar+n) for n in xrange(Nlog)]
    #cbaxes=fig.add_axes([.8,.1,.03,.8])#controls location of colorbar
    colbar=fig.colorbar(dummyplot,ticks=b2ticks)
    colbar.ax.set_yticklabels(['{0:0.3f}'.format(b2) for b2 in b2ticks])
    colbar.set_label(r'$b_2$')

    plt.legend(handles=[line1,line2,line3],loc='lower right')
    plotdir='output/zdisttest/plots/'
    plotname='bztest_cl_compare'
    outname=plotdir+plotname+'.png'
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
        iswmaptype=get_fullISW_MapType(zmax=15)
        maptypes.append(iswmaptype)
    for x in badfracs:
        tag=maintag+'{0:.0e}'.format(x)
        eucmapx=get_Euclidlike_SurveyType(sigz=sigz,z0=z0,tag=tag,zedges=zedges,b0=1.,b2=0,fracbadz=x)
        maptypes.append(eucmapx)
    return maptypes

def catz_get_binmaps(badfracs=np.array([1.e-3,1.e-2,.1]),Nbins=3,z0=.7,sigz=.05,includeISW=True):
    maptypes=catz_get_maptypes(badfracs,Nbins,z0,sigz,includeISW)
    binmaps,bintags=get_binmaplist(maptypes)
    return binmaps

def catz_get_Cl(badfracs=np.array([1.e-3,1.e-2,.1]),Nbins=3,z0=.7,sigz=.05,justread=True):
    maptypes=catz_get_maptypes(badfracs,Nbins,z0,sigz,includeISW=True)
    binmaps,bintags=get_binmaplist(maptypes)
    pairs=[]
    for mt in maptypes:
        if mt.isGal:
            pairs.append((mt.tag,'isw'))
    zmax=max(m.zmax for m in binmaps)
    rundat = ClRunData(tag='catztest_{0:d}bins'.format(Nbins),rundir='output/zdisttest/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    return getCl(binmaps,rundat,dopairs=pairs,DoNotOverwrite=justread)
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
            recdat=RecData(includeglm=inglm,includecl=incl,inmaptag=simtypetags[ns],rectag=rectypetags[nr])
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
        outname=plotdir+plotname+'.png'
        print 'saving',outname
        plt.savefig(outname)
        plt.close()
#--------------------------------------------------------------------     
def catz_get_rhoexp(simfracs=np.array([]),recfracs=np.array([]),badfracs=np.array([]),Nbins=3,z0=.7,sigz=.05,overwrite=False,saverho=True,doplot=False,varname='rho',filetag='',plotdir='output/zdisttest/plots/'):
    print '******in get rhoexp, Nbins=',Nbins
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
            rhoarray[ns,nr]=compute_rho_fromcl(cldat,recgrid[ns][nr],reccldat=cldat,varname=varname)

    if saverho:
        #write to file, 
        f=open(outdir+datfile,'w')
        f.write('{0:9.6f} '.format(0.)+''.join(['{0:9.2e} '.format(fr) for fr in recfracs])+'\n')
        for ns in xrange(Nsim):
            f.write('{0:9.2e} '.format(simfracs[ns])+''.join(['{0:9.6f} '.format(rhoarray[ns,nr]) for nr in xrange(Nrec)])+'\n')
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
        outname='catztest{0:d}bin_{1:s}_exp{2:s}.png'.format(Nbins,varname,outtag)

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
        outname=varname+'test_'+statname+'_exp'+outtag+'.png'
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

def catztest_onesim_plot(fidf=0.02,recf=np.array([0.,5.e-4,1.e-3,2.e-3,.01,.02,.1,.2]),varname='rho',plotdir='output/zdisttest/plots/',outtag='onesim',outname='',Nbins=1):
    dolog=True
    simf=np.array([fidf])
    rhogrid=catz_get_rhoexp(simfracs=simf,recfracs=recf,overwrite=True,saverho=True,doplot=False,varname=varname,filetag=outtag,Nbins=Nbins) #should be NsimxNrec
    #print rhogrid
    Nsim=simf.size
    Nrec=recf.size
    if outtag:
        outtag='_'+outtag
    if not outname:
        outname='catztest{0:d}bin_{1:s}_exp{2:s}.png'.format(Nbins,varname,outtag)

    fig=plt.figure(figsize=(8,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.2)
    ax1=plt.subplot(1,1,1)

    xmax=.5
    linthreshx=1.e-3
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

    ax1.set_xlabel(r"$f_{\rm cat}$ used in ISW rec.")
        
    wherematch=np.argwhere(recf==fidf)[0][0]
    matchval=rhogrid[0,wherematch]
    ax1.plot(recf,rhogrid[0,:]/matchval -1,marker='d')
    if varname=='rho':
        ax1.set_ylabel(r'$\left[\langle \rho \rangle-\langle \rho \rangle_{{\rm match}}\right]/\langle \rho \rangle_{{\rm match}}$')
        varstr=r'\rho'
        annoteloc=(.55,.3)
        ax1.set_ylim((-1,.1*linthreshy))
    elif varname=='s':
        ax1.set_ylabel(r'$\left[\langle s \rangle - \langle s \rangle_{{\rm match}}\right] /\langle s \rangle_{{\rm match}}$')
        #ax1.set_yscale('log')
        varstr='s'
        annoteloc=(.55,.9)
        ax1.set_ylim((-.1*linthreshy,1.))

    #hash fills in where there is a linear scale
    ymin,ymax=ax1.get_ylim()
    y=np.arange(-linthreshx,linthreshx,.01*linthreshx)
    ax1.fill_between(y,ymin,ymax,color='none',edgecolor='grey',hatch='/',linewidth=0)
    
    plt.annotate(r'True (sim.) $f_{{\rm cat}}={0:0.2f}$'.format(fidf)+'\n'+r'$\langle {0:s}\rangle_{{\rm match}}={1:0.3f}$'.format(varstr,matchval),xy=annoteloc,horizontalalignment='center',verticalalignment='top',fontsize=18,xycoords='axes fraction')
    
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()   
#--------------------------------------------------------------------     
def catz_Clcomp(badfracs=np.array([0.,5.e-4,1.e-3,2.e-3,1.e-2,2.e-2,.1,.2]),Nbins=1):
    iswind=0
    firstiszero=badfracs[0]==0. #assumes 0 nowhere else
    bins=catz_get_binmaps(badfracs,Nbins)#includes ISW at 0, others in same order as z0
    cldat=catz_get_Cl(badfracs,Nbins)
    l=np.arange(cldat.Nell)
    #set up color range
    cm=plt.get_cmap('Spectral_r')
    cNorm=colors.LogNorm()#max and min numbers colors need to span
    scalarMap=cmx.ScalarMappable(norm=cNorm,cmap=cm)
    fcols=scalarMap.to_rgba(badfracs[firstiszero:])
    if firstiszero:
        newfcols=['black']
        for c in fcols:
            newfcols.append(c)
        fcols=newfcols
    clscaling=2*l+1.#l*(l+1.)/(2*np.pi)
    
    #to get colorbar key, need ot set up a throw-away map
    dummyz=[[0,0],[0,0]]
    dummylevels=badfracs
    dummyplot=plt.contourf(dummyz,dummylevels,cmap=cm,norm=colors.LogNorm())
    plt.clf()
    fig=plt.gcf()
    
    #set up actual plot
    fig=plt.figure(0)
    ax=plt.subplot()
    plt.title(r'Comparing $C_{{\ell}}$ of galaxies, ISW')
    #plt.ylabel(r'$\ell(\ell+1)C_{{\ell}}/2\pi$')
    plt.ylabel(r'$(2\ell+1)C_{{\ell}}$')
    plt.xlabel(r'Multipole $\ell$')
    plt.xlim((0,30))
    plt.ylim((1.e-11,1.))
    plt.yscale('log')
    for i in xrange(badfracs.size):
        bmap=bins[i+1]
        if bmap.isISW:
            continue
        else:
            autoind=cldat.crossinds[i+1,i+1]
            xiswind=cldat.crossinds[i+1,0]
            plt.plot(l,np.fabs(cldat.cl[autoind,:])*clscaling,color=fcols[i],linestyle='-')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:])*clscaling,color=fcols[i],linestyle='--')
            plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color=fcols[i],linestyle=':')
    #dummy lines for legend
    line1,=plt.plot(np.array([]),np.array([]),color='black',linestyle='-',label='gal-gal')
    line2,=plt.plot(np.array([]),np.array([]),color='black',linestyle='--',label='ISW-gal')
    line3,=plt.plot(l,np.fabs(cldat.cl[xiswind,:]/cldat.cl[autoind,:])*clscaling,color='black',linestyle=':',label='ISW-gal/gal-gal')

    #set up colorbar
    logminvar=int(np.log10(min(badfracs[firstiszero:])))
    logmaxvar=int(np.log10(max(badfracs[firstiszero:])))+1
    print 'LOGMINMAX',logminvar,logmaxvar
    Nlog=logmaxvar-logminvar
    fticks=badfracs[firstiszero:]#[.1*10**(logminvar+n) for n in xrange(Nlog)]
    #cbaxes=fig.add_axes([.8,.1,.03,.8])#controls location of colorbar
    colbar=fig.colorbar(dummyplot,ticks=fticks)
    colbar.ax.set_yticklabels(['{0:0.0e}'.format(f) for f in badfracs[firstiszero:]])
    colbar.set_label(r'$f_{\rm cat}$')

    plt.legend(handles=[line1,line2,line3],loc='lower right')
    plotdir='output/zdisttest/plots/'
    plotname='catztest_cl_compare'
    outname=plotdir+plotname+'.png'
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
    glmdat=get_glm(cldat,Nreal=0,matchClruntag=True)
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
        recdat=RecData(includeglm=includeglm,inmaptag=inmaptag,minl_forrec=lminlist[l],maxl_forrec=lmaxlist[l])
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
        rhoarray[l]=compute_rho_fromcl(cldat,reclist[l],reccldat=cldat,varname=varname)
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
    doiswrec_formaps(dummyglm,cldat,Nreal,reclist=reclist,domaps=domaps)

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
    rhogrid=np.array([read_rhodat_wfile(mapdir+f) for f in files])
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

def lmintest_plot_rhoexp(lminlist=np.arange(1,30),lmaxlist=-1,z0=.7,overwrite=False,saverho=True,varname='rho',filetag='',plotdir='output/lmintest_plots/',dodata=False,datlmin=np.array([]),datlmax=np.array([])):

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
    ax=plt.subplot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for item in ([ax.xaxis.label, ax.yaxis.label] ):
        item.set_fontsize(24)
    
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
            plt.errorbar(datlmin,rhomeani,yerr=rhostdi,linestyle='None',marker='o',color=colors[i%len(colors)])
        datlabel='Results from {0:d} sim.'.format(Nreal[0])#assumes all for same #
        #plot dummy point for legend
        plt.errorbar([-1],[.9],yerr=[.01],linestyle='None',marker='o',color='black',label=datlabel)

    if varname=='rho':
        plt.legend(fontsize=18,loc='lower right',numpoints=1)
    elif varname=='s':
        plt.legend(fontsize=18,loc='lower left',numpoints=1)

    if filetag:
        outtag='_'+filetag
    else:
        outtag=filetag
    outname='lmintest_'+varname+'_exp'+outtag+'.png'
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
    outfile='{0:s}L_vs_rho{2:s}{3:s}_{1:s}.png'.format(fileprefix,plotdat,lminstr,lmaxstr)
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
    rhodat=read_rhodat_wfile(plotdir+rhofile)[:NL]
    
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
    rhofile='iswREC.eucz07.fid.fullsky.depthtest.rho.dat'
    if fileprefix:
        randstr=''
        if shuffleLrec: randstr='shuffled'
        fileprefix=fileprefix+randstr+'_'
    outfile='{0:s}dL_vs_Ltrue_{1:s}.png'.format(fileprefix,plotdat)
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
    nofilterfile='iswREC.eucz07.fid.fullsky.depthtest.rho.dat'
    filterfile='iswREC.eucz07.fid.fullsky'+lminstr+lmaxstr+'.depthtest.rho.dat'
    if fileprefix:
        randstr=''
        if shufflerhofilt: randstr='shuffled'
        fileprefix=fileprefix+randstr+'_'
    outfile='{0:s}rho{1:s}{2:s}_vs_rho.png'.format(fileprefix,lminstr,lmaxstr)
    scatcolor='#92c5de'
    meancolor='#ca0020'
    #read in data

    rhodat=read_rhodat_wfile(plotdir+nofilterfile)
    rhofiltdat=read_rhodat_wfile(plotdir+filterfile)
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
    rhofile='iswREC.eucz07.fid.fullsky.depthtest.rho.dat'
    draganrhofile='fromdragan_l5_rho_10000.dat'
    randstr=''
    if shuffle:
        randstr='shuffled'
        fileprefix=fileprefix+randstr+'_'
    outfile='{0:s}rhocomp.png'.format(fileprefix)
    scatcolor='#92c5de'
    meancolor='#ca0020'

    #read in data
    myrhodat=read_rhodat_wfile(plotdir+rhofile)
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



#################################################################
if __name__=="__main__":
    #plot_isw_kernel()
    depthtestz0=np.array([.3,.5,.6,.7,.8])
    #depthtestz0=np.array([.5])
    if 0: #compute Cl
        t0=time.time()
        depthtest_get_Cl(justread=False,z0vals=depthtestz0)
        t1=time.time()
        print "time:",str(t1-t0),"sec"
    if 0: #generate depthhtest maps
        nomaps=False#True
        Nreal=10000
        depthtest_get_glm_and_rec(Nreal=Nreal,z0vals=depthtestz0,justgetrho=nomaps,minreal=0,dorho=1,dos=True,dochisq=False,dorell=0,dochisqell=False)
    if 0: #plot info about depthtest maps
        depthtest_TTscatter(0,depthtestz0,savepngmaps=False)
        #depthtest_TTscatter(0,np.array([.3,.6,.8]),colors=['#1b9e77','#7570b3','#66a61e'],savepngmaps=False)
        #depthtest_plot_zwindowfuncs(depthtestz0)
        #for N in 1000*np.arange(1,11):
        #    depthtest_plot_rhohist(depthtestz0,varname='rho',firstNreal=N)
        depthtest_plot_rhohist(depthtestz0,varname='rho')
        depthtest_plot_rhohist(depthtestz0,varname='s')
        #depthtest_plot_rhohist(depthtestz0,varname='chisq')
        #depthtest_plot_relldat(depthtestz0,getpred=True,varname='rell')
        #depthtest_plot_relldat(depthtestz0,getpred=True,varname='chisqell')
        #depthtest_rho_tests()

    if 0: #bin test rho expectation value calculations
        #compute cl
        #cldat05=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.05,justread=0)
        #cldat03=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.03,justread=0)
        #cldat001=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.001,justread=0)
        #cldat100=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.1,justread=0)
        
        #compute and save expectation values for rho[0.001,0.03,0.05,0.1]
        bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.001,0.03,0.05,.1])
        bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.001,0.03,0.05,.1],varname='s')
        #bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.001,0.03,0.05],checkautoonly=0,varname='chisq')

        for s in [0.001,0.03,0.05,0.1]:
            #bintest_plot_cl_vals(finestN=6,z0=0.7,sigz=s)
            bintest_plot_zwindowfuncs(sigz=s)
            pass
        
    if 0: #bin test with many realizations, generate maps
        nomaps=False
        bintest_get_glm_and_rec(Nreal=10000,divlist=['6','222','111111'],minreal=0,justgetrho=nomaps,dorell=0)
    if 0: #bin test with many realizations, make plots
        #for N in 1000*np.arange(1,11):
        #    bintest_plot_rhohist(getrhopred=True,varname='rho',firstNreal=N)
        bintest_plot_rhohist(getrhopred=True,varname='rho')
        bintest_plot_rhohist(getrhopred=True,varname='s')
        #bintest_plot_rhohist(getrhopred=True,varname='chisq')
        #bintest_plot_relldat()

    #shortvarlist=[1.e-6,1.e-5,1.e-4,1.e-3] #for testing datapoints
    #
    if 0: #cal test, rho expectation value calcs
        shortvarlist=[1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,1.e-2]
        #shortvarlist=[1.e-6,1.e-5,1.e-4,1.e-3]
        varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))    
        #caltest_get_rhoexp(varlist,overwrite=1,doplot=1,saverho=1,varname='rho')
        #caltest_get_rhoexp(varlist,overwrite=1,doplot=1,saverho=1,varname='s')
        #caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='rho',lmaxlist=[],lminlist=[],widthlist=[],dodataplot=True)
        #caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='rho',shortvarlist=shortvarlist)
        caltest_compare_clcal_shapes(varlist,shapelist=['g'],varname='rho',shortvarlist=shortvarlist)
        #caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='s',shortvarlist=shortvarlist)

    if 0: #plotting cl for caltest
        varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))    
        caltest_Clcomp(varlist)
        
    if 0: #caltest, rho for many realizations
        shortvarlist=[1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,1.e-2]
        nomaps=False
        #caltest_get_scaleinfo(shortvarlist,scaletovar=False)
        Nreal=10000
        #caltest_apply_caliberrors(Nreal=Nreal,varlist=shortvarlist,overwritecalibmap=False,scaletovar=1.e-3,shape='g',width=10.,lmin=0,lmax=30)
        #caltest_apply_caliberrors(Nreal=Nreal,varlist=shortvarlist,overwritecalibmap=False,scaletovar=1.e-3,shape='l2',lmin=1,lmax=30) #working here, run l2 datapoints
        
        #domaps= do we redo reconstructions?
        caltest_iswrec(Nreal=Nreal,varlist=shortvarlist,shape='g',width=10.,callmin=0,callmax=30,scaletovar=1.e-3,domaps=False)
        #caltest_iswrec(Nreal=Nreal,varlist=shortvarlist,shape='l2',callmin=1,callmax=30,scaletovar=1.e-3,domaps=False) #working here; run this

    
    if 0: #scatter plots for calib test
        for r in xrange(5):
            caltest_TTscatter(r)
            pass
        #caltest_TTscatter(4,savepngmaps=True)'

    if 0: #z0test and b2test theory calcs
        simz0=np.array([.35,.56,.63,.693,.7,.707,.7700,.84,1.05])
        perrors=[1,10,20,50]
        #z0test_get_rhoexp(overwrite=True,doplot=True,varname='rho',perrors=perrors)
        z0test_get_rhoexp(overwrite=True,doplot=True,varname='s',perrors=perrors)
        #z0test_get_rhoexp(overwrite=True,doplot=True,varname='chisq',simz0=simz0)
        #recb2=np.array([0.,.5,1.,10.])
        simb2=np.array([0.,.01,.1,.5,1.,2.,5.,10.])
        recb2=simb2
        #bztest_get_rhoexp(simb2,recb2,overwrite=True,doplot=True,varname='rho')
        #bztest_get_rhoexp(simb2,recb2,overwrite=True,doplot=True,varname='s')
        #z0test_Clcomp()
        #bztest_Clcomp()

    if 0: #catztest theory calcs, makes colorblock plots
        for Nbins in [1,3]:
            if Nbins==1:
                badfracs=np.array([0.,5.e-4,1.e-3,2.e-3,5.e-3,1.e-2,2.e-2,.1,.2])
            elif Nbins==3:
                badfracs=np.array([1.e-3,1.e-2,.1,.2])                                
            catz_windowtest(badfracs,Nbins=Nbins)
            catz_get_rhoexp(overwrite=True,doplot=True,varname='rho',badfracs=badfracs,Nbins=Nbins)
            catz_get_rhoexp(overwrite=True,doplot=True,varname='s',badfracs=badfracs,Nbins=Nbins)
        #catz_Clcomp()


    if 0: #zdist tests with less info
        # z0test_onesim_plot(varname='rho')
        # z0test_onesim_plot(varname='s')
        #bztest_onesim_plot(varname='rho')
        #bztest_onesim_plot(varname='s')
        #bztest_onerec_plot(varname='rho')
        #bztest_onerec_plot(varname='s')
        
        Nbins=1
        if Nbins==1:
            badfracs=np.array([0.,1.e-3,5.e-3,1.e-2,2.e-2,5.e-2,.1,.2])#1 bin
            badfracs1sim=badfracs#1 bin
            #catz_Clcomp(badfracs=badfracs)
        elif Nbins==3:
            badfracs=np.array([0.,1.e-3,1.e-2,.1,.2])#3 bin
            badfracs1sim=badfracs
        catztest_onerec_plot(varname='rho',Nbins=Nbins,simf=badfracs)
        catztest_onerec_plot(varname='s',Nbins=Nbins,simf=badfracs)
        catztest_onesim_plot(varname='rho',Nbins=Nbins,recf=badfracs1sim,fidf=.01)
        catztest_onesim_plot(varname='s',Nbins=Nbins,recf=badfracs1sim,fidf=.01)
        catz_Clcomp()
        catz_windowtest(badfracs,Nbins=Nbins)
        #z0test_Clcomp()
        #bztest_Clcomp()
    #angular momentum tests
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


            
    #lmin tests
    if 0: #generate rho data from many realizations
        Nreal=1
        inlminlist=np.array([2,3,4,5])
        #inlminlist=np.array([10])
        inlmaxlist=np.array([-1])#3,5,10,20,-1])
        lminlist,lmaxlist=lmintest_get_lminmaxcombos(inlminlist,inlmaxlist)
        if 0: #do reconstructions for Nreal for combos of inlminlist and inlmax
            domaps=True
            Nreal=10000
            lmintest_iswrec(Nreal=Nreal,lminlist=lminlist,lmaxlist=lmaxlist,domaps=domaps)
        lmintest_plot_rhoexp(overwrite=0,lminlist=np.arange(1,20),lmaxlist=inlmaxlist,varname='rho',dodata=True,datlmin=inlminlist)
        #lmintest_plot_rhoexp(overwrite=0,lmaxlist=np.array([-1]),varname='rho',dodata=False)

    #lmin caltests
    if 0:
        shortvarlist=[1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,1.e-2]
        shortreclminlist=np.array([1,3,5])#1,3,10])
        if 0: #do recs for many realizations
            Nreal=10000
            caltest_iswrec(Nreal,shortvarlist,scaletovar=1.e-3,recminelllist=shortreclminlist,domaps=True)#working here

        #shortvarlist=[1.e-6,1.e-5,1.e-4,1.e-3]
        varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))
        reclminlist=np.array([1,2,3,5])
        
        caltest_compare_lmin(varlist,varname='rho',dodataplot=True,recminelllist=reclminlist,shortrecminelllist=shortreclminlist,shortvarlist=shortvarlist,justdat=True)
        
