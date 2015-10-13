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
def depthtest_get_glm_and_rec(Nreal=1,z0vals=np.array([.3,.6,.7,.8]),minreal=0,justgetrho=0,dorell=0):
    t0=time.time()
    cldat=depthtest_get_Cl(justread=True,z0vals=z0vals)
    makeplots=Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=depthtest_get_reclist(z0vals)
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorell=dorell)
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
    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
    
#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def depthtest_plot_rhohist(z0vals=np.array([.3,.6,.7,.8]),getrhopred=True,varname='rho'):
    plotdir='output/depthtest/plots/'
    testname="Depth test"
    rhogrid=depthtest_read_rho_wfiles(z0vals,varname)
    Nreal=rhogrid.shape[1]
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

#--------------------------------
# plot r_ell values
def depthtest_plot_relldat(z0vals=np.array([.3,.6,.7,.8]),getpred=True,varname='rell'):
    plotdir='output/depthtest/plots/'
    testname="Depth test"
    rellgrid=depthtest_read_rell_wfiles(z0vals,varname)
    Nreal=rellgrid.shape[1]
    Nell=rellgrid.shape[2]
    if getpred:
        rellpred=depthtest_get_expected_rell(z0vals,varname)
    else:
        rellpred=[]
    plotname ='depthtest_{1:s}dat_r{0:05d}'.format(Nreal,varname)
    reclabels=['$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]

    if varname=='rell':
        plot_relldat(reclabels,testname,plotdir,plotname,rellgrid,rellpred)
        
#--------------------------------
# get expectation values of rho or s, choose variable via varname
def depthtest_get_expected_rho(z0vals=np.array([0.3,0.6,0.7,0.8]),varname='rho'):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rhopred=np.zeros_like(z0vals)
    for i in xrange(z0vals.size):
        if varname=='rho':
            rhopred[i]=compute_rho_fromcl(cldat,reclist[i])
        elif varname=='s':
            rhopred[i]=compute_s_fromcl(cldat,reclist[i])
    return rhopred

def depthtest_get_expected_rell(z0vals=np.array([0.3,0.6,0.7,0.8]),varname='rell'):
    cldat=depthtest_get_Cl(z0vals=z0vals)
    reclist=depthtest_get_reclist(z0vals)
    rellpred=[]
    for i in xrange(z0vals.size):
        if varname=='rell':
            rellpred.append(compute_rell_fromcl(cldat,reclist[i]))
    rellpred=np.array(rellpred)#[Nrec,Nell]
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
        iswmaptype=get_fullISW_MapType(zmax=10)
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
def bintest_get_rhoexp(finestN=6,z0=0.7,sigz=0.05,overwrite=False,doplot=True,Nneighb=-1,getdivs=['all'],saverho=True,varname='rho'):
    if saverho:
        outdir = 'output/eucbintest/plots/'
        if Nneighb>-1:
            if varname!='rho':
                print "**WARNING: only rho stat set up to handle Neighb>-1."
            datfile='eucbintest{0:03d}_{2:s}exp_neighb{1:1d}.dat'.format(int(1000*sigz),Nneighb,varname)
        else:
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
                if varname=='rho':
                    rhoarray[r]=compute_rho_fromcl(cldat,reclist[r],Nneighb)
                elif varname=='s':
                    rhoarray[r]=compute_s_fromcl(cldat,reclist[r])
            #write rhoarray to file
            f=open(outdir+datfile,'w')
            f.write(''.join(['{0:8s} {1:8.3f}\n'.format(divstr[i],rhoarray[i]) for i in xrange(Nrec)]))
            f.close()
    else: #don't interact with files, just compute
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
            if varname=='rho':
                rhoarray[r]=compute_rho_fromcl(cldat,reclist[r],Nneighb)
            elif varname=='s':
                rhoarray[r]=compute_s_fromcl(cldat,reclist[r])

    if doplot:
        zedges0=bintest_get_finest_zedges(finestN,z0)
        allzedges=bintest_get_zedgeslist(zedges0,['all'],False)
        bintest_rhoexpplot(allzedges,divstr,rhoarray,varname)
    return divstr,rhoarray

#if we've computed Cl stuff for multiple values of sigz0, compare them
def bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.03,0.05],checkautoonly=True):
    rholist=[]
    for s in sigzlist:
        divstr,rho=bintest_get_rhoexp(finestN,z0,s,overwrite=False,doplot=False)
        rholist.append(rho)
    legtitle='$\\sigma_z(z)*(1+z)^{{-1}}$'
    labellist=['${0:0.3f}$'.format(s) for s in sigzlist]
    zedges0=bintest_get_finest_zedges(finestN,z0)
    allzedges=bintest_get_zedgeslist(zedges0,['all'],False)
    markerlist=[]
    colorlist=[]
    outtag=''
    if checkautoonly: #see what happens if cross bin power info not included
        outtag='_varyneighb'
        scattercolors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
        colorlist=scattercolors[:len(labellist)]
        markerlist=['D']*len(labellist)
        marks=['x','*']
        for n in [1,0]:
            i=0
            for s in sigzlist:
                divstr,rho=bintest_get_rhoexp(finestN,z0,s,overwrite=False,doplot=False,Nneighb=n)
                #print divstr,rho
                rholist.append(rho)
                markerlist.append(marks[n])
                labellist.append('${0:0.3f}$,{1:1d}nb'.format(s,n))
                colorlist.append(scattercolors[i])
                i+=1
    outname='eucbintest_rhoexp'+outtag+'.png'
    bintest_rhoexpplot(allzedges,divstr,rholist,labellist,outname,legtitle,markerlist,colorlist,outtag)

#--------------------
def bintest_test_rhoexp():
    Nneighb=0
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
            rhogrid[s,r]=compute_rho_fromcl(cldat,reclist[r],Nneighb)
            print '    rho=',rhogrid[s,r]

def bintest_get_expected_rell(divstr,varname='rell'):
    cldat=bintest_get_Clvals()
    reclist=bintest_get_reclist(getdivs=divstr)
    rellpred=[]
    for i in xrange(len(reclist)):
        if varname=='rell':
            rellpred.append(compute_rell_fromcl(cldat,reclist[i]))
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
    print '--------cldat info-----------------'
    print 'nmaps',cldat.Nmap
    print 'ncross',cldat.Ncross
    print cldat.bintaglist
    print 'shape',cldat.cl.shape
    print '-----------------------------------'
    
    getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorell=dorell)
    t1=time.time()
    print "total time for Nreal",Nreal,": ",t1-t0,'sec'

#get arrays of rho saved in .rho.dat files, or .s.dat files
def bintest_read_rho_wfiles(divlist=['6','222','111111'],sigz=0.05,varname='s'):
    mapdir='output/eucbintest/map_output/'
    files=['iswREC.euc6bins{0:03d}div{1:s}.fid.fullsky.eucbintest6s{0:03d}all.{2:s}.dat'.format(int(1000*sigz),d,varname) for d in divlist]
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
def bintest_rhoexpplot(allzedges,labels,rhoarraylist,labellist=[],outname='',legtitle='',markerlist=[],colorlist=[],outtag='',varname='rho'):
    if type(rhoarraylist[0])!=np.ndarray: #just one array passed,not list of arr
        rhoarraylist=[rhoarraylist]
    plotdir='output/eucbintest/plots/'
    if not outname:
        outname='eucbintest_'+varname+'exp'+outtag+'.png'
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    scattercolors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    Npoints=len(labels)
    yvals=np.arange(Npoints)
    zvals=allzedges[-1] #should be base value
    Nz=zvals.size
    zinddict={zvals[i]:i for i in xrange(Nz)}

    #divide fiture up into two vertical pieces
    fig=plt.figure(0)
    if varname=='rho':
        plt.suptitle(r'Expected correlation coef. between $T^{{\rm ISW}}$ and $T^{{\rm rec}}$', size=18)
    elif varname=='s':
        plt.suptitle(r'Expected ratio between RMS of  $T^{{\rm rec}}-T^{{\rm ISW}}$ and $\sigma_{{T}}^{{\rm ISW}}$', size=18)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,(2,3),sharey=ax1)
    fig.subplots_adjust(hspace=0, wspace=0) #put them right next to eachother

    #left side has illustration of binning strategy
    plt.sca(ax1)
    plt.ylim((-1,Npoints))
    plt.xlim((0,6.1))#at 6, plots butt up against each other
    plt.xlabel(r'Redshift bin edges $z$')
    ax1.xaxis.set_ticks_position('bottom')
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
    if varname=='rho':
        ax2.set_xlabel(r'$\langle \rho \rangle$')
    elif varname=='s':
        ax2.set_xlabel(r'$\langle s \rangle$')
    ax2.grid(True)
    if not markerlist:
        markerlist=['D']*len(labellist)
    if not colorlist:
        colorlist=scattercolors
    for i in xrange(len(rhoarraylist)):
        rhoarray=rhoarraylist[i]
        m=markerlist[i]
        if labellist:
            if m=='D':
                ax2.scatter(rhoarray,yvals,label=labellist[i],color=colorlist[i],marker=m)#,edgecolor='black')
            else:
                ax2.scatter(rhoarray,yvals,label=labellist[i],color=colorlist[i],marker=m)
        else:
            ax2.scatter(rhoarray,yvals,color=colorlist[i],marker=m)

    if labellist:
        plt.legend(loc='upper left',fontsize=12,title=legtitle)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels()[0], visible=False)#don't show number at first label
    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()

#----------------
def bintest_plot_rhohist(divstr=['6','222','111111'],getrhopred=True,reclabels=['1 bin','3 bins','6 bins'],varname='rho'):
    plotdir='output/eucbintest/plots/'
    Nrecs=len(divstr)
    rhogrid=bintest_read_rho_wfiles(divstr,varname=varname)
    Nreal=rhogrid.shape[1]
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
def bintest_plot_zwindowfuncs(finestN=6,z0=0.7,sigz=0.05,doiswkernel=True):
    bins=bintest_get_binmaps(finestN,z0=0.7,sigz=sigz,includeisw=False,justfinest=True)#just gal maps
    sigz0=sigz
    plotdir='output/eucbintest/plots/'
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
    plt.figure(0)
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    ax.set_yticklabels([])
    plt.title(r'Bin test: redshift distributions',fontsize=16)
    plt.xlabel('Redshift z',fontsize=16)
    plt.ylabel('Source distribution (arbitrary units)',fontsize=16)
    ymax=0.3
    plt.ylim(0,ymax)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x', labelsize=18)
    nbartot=0
    for n in xrange(Nbins):
        m=bins[n]
        wgrid=m.window(zgrid)*m.nbar/1.e9
        nbartot+=m.nbar
        colstr=colors[n%len(colors)]
        #plt.fill_between(zgrid,0,wgrid, facecolor=colstr,edgecolor='none',linewidth=2, alpha=0.3)
        plt.plot(zgrid,wgrid,color=colstr,linestyle='-',linewidth=2)

    if doiswkernel:
        kernelmax=np.max(kernel)
        wantmax=.8*ymax
        scaleby=wantmax/kernelmax
        plt.plot(cosmz,kernel*scaleby,color='grey',label='ISW kernel',linewidth=2,linestyle='--')
        plt.legend(loc='upper right',fancybox=False, framealpha=0.,prop={'size':16},handlelength=3.5)
        
    eqstr='$\\frac{{dn}}{{dz}} \\propto \\,z^2 e^{{-\\left(z/z_0\\right)^{{1.5}}}}$\n $z_0={0:0.1f}$, $\\sigma_z={1:0.3f}(1+z)$\n $\\bar{{n}}_{{\\rm tot}}={2:g}$'.format(z0,sigz0,nbartot)
    #eqstr='$\frac{{dn}}{{dz}} \propto \,z^2 e^{{-\left(z/z_0\right)^{{1.5}}}}$\n $z_0={0:0.1f}$, $\sigma_z={1:0.2f}(1+z)$'.format(z0,sigz0)

    textbox=ax.text(1.7, .25, eqstr,fontsize=16,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))

    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#----------------
#look at how correlated differnt base-bins are with one another and ISW
def bintest_plot_cl_vals(finestN=6,z0=0.7,sigz=0.05):
    outdir='output/eucbintest/plots/'
    outname='clvals_eucbintest{0:d}s{1:03d}.png'.format(finestN,int(1000*sigz))
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
    fidbins=caltest_get_fidbins()
    lssbin=fidbins[1].tag #will just be the depthtest bin map
    if not fidcl:
        fidcl=caltest_get_clfid()

    #get fid glmdat; no data needed, just mapnames, et
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

def caltest_get_reclist(varlist,shape='g',width=10.,lmin=0,lmax=30,recminell=1):
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
    reclist.append(RecData(includecl,includecl,inmaptype,'unmod',recminell))
    for m in dothesemods:
        includeglm=[m]
        rectag=m[1]#modtag
        reclist.append(RecData(includeglm,includecl,inmaptype,rectag,recminell))
    return reclist

#having already generated maps for reconstructions with calib errors,
# do isw reconstructions from the maps, computing rho and s stats
def caltest_iswrec(Nreal,varlist,shape='g',width=10.,lmin=0,lmax=30,overwritecalibmap=False,scaletovar=False,recminell=1):
    fidcl=caltest_get_clfid()
    dummyglm=caltest_apply_caliberrors(varlist,0,shape,width,lmin,lmax,overwritecalibmap,scaletovar)
    reclist=caltest_get_reclist(varlist,shape,width,lmin,lmax,recminell=1)
    doiswrec_formaps(dummyglm,fidcl,Nreal,reclist=reclist,domaps=True)

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

def caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='rho',lmaxlist=[],lminlist=[],widthlist=[]):
    Nshapes=len(shapelist)
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
    rhoexplist=[]
    labels=[]
    for s in xrange(Nshapes):
        rhoexplist.append(caltest_get_rhoexp(varlist,lmax=lmaxlist[s],lmin=lminlist[s],shape=shapelist[s],width=widthlist[s],overwrite=False,doplot=False,saverho=True,varname=varname))
        if shapelist[s]=='g':
            shapestr='g{2:d}_{0:d}l{1:d}'.format(lminlist[s],lmaxlist[s],int(widthlist[s]))
        elif shapelist[s]=='l2':
            shapestr='l2_{0:d}l{1:d}'.format(lminlist[s],lmaxlist[s])
        labels.append(shapestr)
    print 'rhoexplist.shape',np.array(rhoexplist).shape
    print 'Nshapes',Nshapes
    print "Nvar",len(varlist)
    caltest_rhoexpplot(varlist,rhoexplist,labels,legtitle=r'$C_{{\ell}}^{{\rm cal}}$ shape',outtag='shapecompare',varname=varname)

#caltest_get_rhoexp - approximating calib errors as additive only,
#                     compute expectation value of rho for number of calibration
#                     error field variances, assuming reconstruction is done
#                     assuming no calib error
#         inserts fiducial (no calib error) as last entry
#         returns grid of rho or s values of size Nrec=Nvar
def caltest_get_rhoexp(varlist=[1.e-4],lmax=30,lmin=1,shape='g',width=10.,overwrite=False,doplot=True,saverho=True,varname='rho',filetag=''):
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
    recdat=RecData(includeglm=[lssbin],inmaptag=lssbin[:lssbin.rfind('_bin0')])
    
    # return array of shape [Nvar,Nell]
    Nrec=len(varlist)
    rhoarray=np.zeros(Nrec)
    for r in xrange(Nrec):
        if varname=='rho':
            rhoarray[r]=compute_rho_fromcl(fidcl,recdat,reccldat=clmodlist[r])
        elif varname=='s':
            rhoarray[r]=compute_s_fromcl(fidcl,recdat,reccldat=clmodlist[r])

    #if save, write to file
    if saverho:
        f=open(outdir+datfile,'w')
        f.write('Calib error test: Clcal shape={0:s}, ell={1:d}-{2:d}\n'.format(shape,lmin,lmax))
        f.write('var(c(nhat))   <rho>\n')
        f.write(''.join(['{0:.2e} {1:8.3f}\n'.format(varlist[i],rhoarray[i]) for i in xrange(Nrec)]))
        f.close()

    if doplot:
        caltest_rhoexpplot(varlist,rhoarray,varname=varname,outtag=shapestr)
    return rhoarray

#-------
def caltest_getmapmods_onebin(lssbintag,varlist=[1.e-1,1.e-2,1.e-3,1.e-4],lmax=30,lmin=0,shape='g',width=10.):
    #construct map-mod combos for the variances given
    if shape=='g':
        mapmods=[(lssbintag,getmodtag_fixedvar_gauss(v,width,lmax)) for v in varlist]
    elif shape=='l2':
        mapmods=[(lssbintag,getmodtag_fixedvar_l2(v,lmax)) for v in varlist]
    return mapmods

#---------------------------------------------------------------
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
def caltest_rhoexpplot(varlist,rhoarraylist,labellist=[],outname='',legtitle='',colorlist=[],outtag='',varname='rho',datplot=[]):
    #assuming last entry in varlist, rhoarray is fiducial (var=0)
    if type(rhoarraylist[0])!=np.ndarray: #just one array passed,not list of arr
        rhoarraylist=[rhoarraylist]
    scattercolors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    plotdir = 'output/caltest_plots/'
    if not outname:
        outname='caltest_'+varname+'_exp'+outtag+'.png'

    fig=plt.figure(0)
    if varname=='rho':
        plt.suptitle(r'Calibration error test: Expected correlation coef. between $T^{{\rm ISW}}$ and $T^{{\rm rec}}$', size=14)
    elif varname=='s':
        plt.suptitle(r'Calibration error test: Expected [RMS of  $T^{{\rm rec}}-T^{{\rm ISW}}$]/$\sigma_{{T}}^{{\rm ISW}}$', size=14)
    ax1 = plt.subplot(3,1,(1,2)) #top part has rho
    ax2 = plt.subplot(3,1,3,sharex=ax1) #bottom has rho/rhofid
    if not labellist:
        labellist=['']
    if not colorlist:
        colorlist=scattercolors

    plt.sca(ax1)
    ax1.grid(True)
    ax1.set_xscale('log')
    #ax1.ticklabel_format(axis='y',style='')
    ax2.grid(True)
    if varname=='rho':
        ax1.set_ylabel(r'$\langle \rho \rangle$')
        ax2.set_ylabel(r'$\langle \rho \rangle /\langle \rho_{{c=0}} \rangle -1$')
    elif varname=='s':
        ax1.set_ylabel(r'$\langle s \rangle$')
        ax2.set_ylabel(r'$\langle s \rangle /\langle s_{{c=0}} \rangle - 1$')
    ax2.set_xlabel(r'Variance of calib. error field ${\rm var}[c(\hat{{n}})]$')
    for i in xrange(len(rhoarraylist)):
        print len(varlist[:-1]),rhoarraylist[i][:-1].shape
        ax1.semilogx(varlist[:-1],rhoarraylist[i][:-1],label=labellist[i],color=colorlist[i%len(colorlist)])
        ax2.semilogx(varlist[:-1], rhoarraylist[i][:-1]/rhoarraylist[i][-1]-1.,label=labellist[i],color=colorlist[i%len(colorlist)])

    for i in xrange(len(datplot)):
        datvar=datplot[0]#array
        datrho=datplot[1]#array
        datlabel=datplot[2] #string
        datsig=0
        datcol=''
        if len(datplot)>3:
            datcol=datplot[3]
            if len(datplot)>4:
                datsig=datplot[4] #if one value, same for all points
                #   if array, sets different
        if not datcol:
            datcol=colorlist[(i+len(rhoarraylist))%len(colorlist)]
        if type(datsig)==0:
        else: #uniform error bars if datsig is a number, nonuni if array
            ax1.errorbar(datvar,datrho,yerr=datsig,label=datlabel,color=datcol)
        #working here, need to finish setting up and test data point plotting
    plt.sca(ax1)
    if labellist and labellist[0]:
        plt.legend(fontsize=12,title=legtitle)

    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()
#---------------------------------------------------------------
# rho histogram 


#################################################################
if __name__=="__main__":
    #plot_isw_kernel()
    depthtestz0=np.array([.3,.6,.7,.8])
    if 0: #compute Cl
        t0=time.time()
        depthtest_get_Cl(justread=False,z0vals=depthtestz0)
        t1=time.time()
        print "time:",str(t1-t0),"sec"
    if 0: #generate depthhtest maps
        nomaps=True
        depthtest_get_glm_and_rec(Nreal=10000,z0vals=depthtestz0,justgetrho=nomaps,minreal=0,dorell=1)
    if 0: #plot info about depthtest maps
        #depthtest_TTscatter(0,depthtestz0,False)
        #depthtest_plot_zwindowfuncs(depthtestz0)
        #depthtest_plot_rhohist(depthtestz0,'rho')
        #depthtest_plot_rhohist(depthtestz0,varname='s')
        depthtest_plot_relldat(depthtestz0,getpred=True,varname='rell')
        #depthtest_rho_tests()

    if 0: #bin test rho expectation value calculations
        #compute cl
        #cldat05=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.05,justread=0)
        #cldat03=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.03,justread=0)
        #cldat001=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.001,justread=0)
        #cldat100=bintest_get_Clvals(finestN=6,z0=0.7,sigz=0.1,justread=0)
        
        #compute and save expectation values for rho[0.001,0.03,0.05,0.1]
        bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.001,0.03,0.05],checkautoonly=1)

        for s in [0.001,0.03,0.05,0.1]:
            #bintest_plot_cl_vals(finestN=6,z0=0.7,sigz=s)
            #bintest_plot_zwindowfuncs(sigz=s)
            pass
        
    if 0: #bin test with many realizations, generate maps
        nomaps=True
        bintest_get_glm_and_rec(Nreal=10000,divlist=['6','222','111111'],minreal=0,justgetrho=nomaps,dorell=1)
    if 0: #bin test with many realizations, make plots
        #bintest_plot_rhohist(getrhopred=True,varname='rho')
        #bintest_plot_rhohist(getrhopred=True,varname='s')
        bintest_plot_relldat()

    shortvarlist=[1.e-6,1.e-5,1.e-4,1.e-3] #for testing datapoints

    if 0: #cal test, rho expectation value calcs
        varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))    
        #caltest_get_rhoexp(varlist,overwrite=1,doplot=1,saverho=1,varname='rho')
        #caltest_get_rhoexp(varlist,overwrite=1,doplot=1,saverho=1,varname='s')
        caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='rho')
        caltest_compare_clcal_shapes(varlist,shapelist=['g','l2'],varname='s')

    if 1: #caltest, rho for many realizations
        nomaps=False
        #caltest_get_scaleinfo(shortvarlist,scaletovar=False)
        Nreal=10000
        #caltest_apply_caliberrors(Nreal=Nreal,varlist=shortvarlist,overwritecalibmap=True)
        caltest_iswrec(Nreal=Nreal,varlist=shortvarlist)
