##################################################################
# This script is intended to run simulations similar to what is used in
# Manzotti and Dodelson's ISW rec paper, to be used as a cross check.
# 
##################################################################
from scipy.optimize import leastsq
import MapParams as mp
import ClRunUtils as clu
import genCrossCor as gcc
import genMapsfromCor as gmc
import AnalysisUtils as au
import mapdef_utils as mu
import numpy as np
import run_euclidlike_analysis as euc
import matplotlib.pyplot as plt
import os
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
    bias = mu.nobias
    dndz = mu.dndz_NVSSlike
    longtag='NVSS-like survey with bias=1'
    sigz=.1 #only one bin, so this won't be used
    biasargs=[]
    dndzargs=[.32,0.36]
    return mp.SurveyType(tag,zedges,sigz,nbar,dndz,bias,dndzargs=dndzargs,longtag=longtag)

def get_Planck_NVSSlike_SurveyType(tag='', nbar=1.584e5):
    """From Planck XXI table 1"""
    if not tag:
        tag='nvss'
    zedges=np.array([0.01,6.])
    bias = mu.quadbias
    dndz = mu.dndz_NVSSlike
    longtag='NVSS-like survey with b(z)'
    sigz=.1 #only one bin, so this won't be used
    biasargs=[0.9, 0.54]
    dndzargs=[.33,0.37]
    return mp.SurveyType(tag,zedges,sigz,nbar,dndz=dndz,bias=bias,dndzargs=dndzargs,biasargs=biasargs,longtag=longtag)
    ### see MapParams for SurveyType

def get_Planck_MphG_like_SurveyType(tag='', nbar=9.680e6):
    """From Planck XXI table 1"""
    if not tag:
        tag='MphG'
    zedges=np.array([0.01,2.])
    bias = mu.quadbias
    dndz = mu.dndz_MphG_like
    longtag='SDSS_MphG-like survey with bias=1.2'
    sigz=.1 #only one bin, so this won't be used
    biasargs=[1.2, 0] #const bias of 1.2, per Planck
    dndzargs=[1.5, 2.3, 0.34]
    return mp.SurveyType(tag,zedges,sigz,nbar,dndz=dndz,bias=bias,dndzargs=dndzargs,biasargs=biasargs,longtag=longtag)
    
def get_MDDESlike_SurveyType(tag='',nbins=2):
    if not tag:
        tag='desMD{0:d}bin'.format(nbins)
    nbar=1.e9 #picking a big number, assuming shot noise will be negligible

    if nbins==2:
        zedges=np.array([0.1,.5,1.6])
    elif nbins==3:
        zedges=np.array([0.1,.5,1.,1.6])
    bias=mu.nobias
    dndz=dndz_desMDlike
    longtag='DES-like survey ala M&D with bias=1'
    sigz=.025 #only one bin, so this won't be used
    return mp.SurveyType(tag,zedges,sigz,nbar,dndz,bias,longtag=longtag)

def MDtest_get_maptypelist(includeisw=False,Ndesbins=[2,3],nvss=True):
    surveys=[]
    if nvss:
        surveys.append(get_NVSSlike_SurveyType())
    #print 'dndz nvss args:',surveys[0].dndzargs
    if includeisw:
        surveys.append(mu.get_fullISW_MapType(zmax=15))   
    for n in Ndesbins:
        surveys.append(get_MDDESlike_SurveyType(nbins=n))
    return surveys
    
def MDtest_get_binmaps(includeisw=True,Ndesbins=[2,3],nvss=True):
    surveys=MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)
    bins=[]
    for survey in surveys:
        bins=bins+survey.binmaps
    if includeisw:
        iswmaptype = mu.get_fullISW_MapType(zmax=15)
        iswbins = iswmaptype.binmaps
        bins=iswbins+bins
    return bins

def PlanckTest_get_Cl(justread=True, addCMB=True, limberl=20):
    surveys=[]
    surveys.append(get_Planck_NVSSlike_SurveyType())
    surveys.append(get_Planck_MphG_like_SurveyType())
    bins=[]
    for survey in surveys:
        bins=bins+survey.binmaps
        
    iswmaptype = mu.get_fullISW_MapType(zmax=15)
    iswbins = iswmaptype.binmaps
    bins=iswbins+bins
        
    zmax = max(m.zmax for m in bins)*1.01 #make sure integrate out past highest z of all the maps
    if limberl!=20:
        mytag = 'Plancktest_lim{0}'.format(limberl)
    else: mytag= 'Plancktest'
    rundat = clu.ClRunData(tag=mytag,rundir='output/Planck_checks/',lmax=95,zmax=zmax,limberl=limberl,iswilktag='fidisw',noilktag=True)
    #this object holds a bunch of info needed to do the integrals ^^
    pairs=['all']
    cldat=gcc.getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)
    if addCMB:
        cldat.addCMBtemp()
    return cldat
    
#
def MDtest_get_Cl(justread=True,Ndesbins=[2,3],nvss=True): #do a survey splitting DES into 2 bins and one splitting into 3 bins. NVSS is just one bin
    surveys = MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss) #each 
    bins = MDtest_get_binmaps(Ndesbins=Ndesbins,nvss=nvss) #each redshift bin for each bin is its own map
    zmax = max(m.zmax for m in bins) #make sure integrate out to highest z of all the maps
    rundat = clu.ClRunData(tag='MDtest',rundir='output/MDchecks/',lmax=95,zmax=zmax,iswilktag='fidisw',noilktag=True)
    #this object holds a bunch of info needed to do the integrals ^^
    pairs=['all']
    #pair up isw and each LSS maps, but not lss maps together 
    # for s in surveys:
    #     pairs.append((s.tag,'isw'))
    #     pairs.append((s.tag,s.tag))
    cldat=gcc.getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)
    #pass binobjject, rundata object, computer all crosspairs, if already have Cls, best to have DNO = True. Rundata will tell it where the output file is.
    #can pass specific bins (which must be in the output). So don't need to recompute all the time. So this function mostly gets used to just read them in.
    return cldat

def MDtest_boostNVSSnoise(cldat, newnbar=1.e8):
    """go through cldat and change nbar to newnbar value for any bin that has nvss in name."""
    #smaller than prev used 1.e9
    for i in xrange(cldat.Nmap):
        if 'nvss' in cldat.bintaglist[i]:
            nvssind=i
            break
    cldat.nbar[i]=newnbar
    nvssdiagind=cldat.crossinds[nvssind,nvssind]
    cldat.noisecl[nvssdiagind,:]=1./newnbar
    return cldat

def MDtest_get_reclist(Ndesbins=[2,3],lmin=3,lmax=80,nvss=True):
    maptypes=MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)
    Nrec=len(maptypes)
    reclist=[]
    for i in xrange(Nrec):
        mtype=maptypes[i]
        inmaptag=mtype.tag #label in output glmdat
        includeglm=[b.tag for b in mtype.binmaps]
        recdat = au.RecData(includeglm=includeglm,inmaptag=inmaptag,minl_forrec=lmin,maxl_forrec=lmax)
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
    makeplots = Nreal==1
    rlzns=np.arange(minreal,minreal+Nreal)
    reclist=MDtest_get_reclist(Ndesbins=Ndesbins,lmin=lmin,lmax=lmax,nvss=nvss)
    au.getmaps_fromCl(cldat,rlzns=rlzns,reclist=reclist,justgetrho=justgetrho,dorho=dorho,dos=False,dochisq=False,rhofiletag=rhofiletag)

#use cldat to generate glm, no iswrec
def MDtest_get_glm(Nreal=1,minreal=0,Ndesbins=[2,3],nvss=True):
    cldat=MDtest_get_Cl(justread=True,Ndesbins=Ndesbins,nvss=nvss)
    print 'cldat.bintaglist:',cldat.bintaglist
    
    #cldat seems to be associated with correct crosspairs
    # also, manipulating cl data has expected effects on rho hist
    rlzns=np.arange(minreal,minreal+Nreal)
    #leaving reclist empty means just gen gal and isw maps
    au.getmaps_fromCl(cldat,rlzns=rlzns)
    
#assuming maps already generated, do reconstructions
def MDtest_iswrec(Nreal,minreal=0,justgetrho=0,dorho=1,Ndesbins=[2,3],lmin=3,lmax=80,rhofiletag='',nvss=True,fitbias=True):
    rlzns=np.arange(minreal,minreal+Nreal)
    cldat=MDtest_get_Cl(justread=False,Ndesbins=Ndesbins,nvss=nvss)
    reclist=MDtest_get_reclist(Ndesbins=Ndesbins,lmin=lmin,lmax=lmax,nvss=nvss)
    dummyglm=gmc.get_glm(cldat,Nreal=0,runtag=cldat.rundat.tag)
    au.doiswrec_formaps(dummyglm,cldat,rlzns=rlzns,reclist=reclist,rhofiletag=rhofiletag,dos=False,fitbias=fitbias)
    
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
    rhogrid=np.array([au.read_rhodat_wfile(mapdir+f) for f in files])
    return rhogrid

# get expectation values of rho or s, choose variable via varname
def MDtest_get_expected_rho(varname='rho',Ndesbins=[2,3],lmin=3,lmax=80,nvss=True):
    Nrec=len(Ndesbins)+nvss
    cldat=MDtest_get_Cl(Ndesbins=Ndesbins,nvss=nvss)
    cldat=MDtest_boostNVSSnoise(cldat)
    reclist=MDtest_get_reclist(Ndesbins,lmin,lmax,nvss=nvss)
    rhopred=np.zeros(Nrec)
    for i in xrange(Nrec):
        rhopred[i]=au.compute_rho_fromcl(cldat,reclist[i],varname=varname)
    return rhopred

def checkMD_cl_ordering():
    c23=MDtest_get_Cl(justread=True,Ndesbins=[2,3],nvss=1)
    c32=MDtest_get_Cl(justread=True,Ndesbins=[3,2],nvss=1)
    #set up map between the two mapind lists
    Nmap=c23.Nmap
    map23to32=np.zeros(Nmap)
    for i in xrange(Nmap):
        si=c23.bintaglist[i]
        for j in xrange(Nmap):
            sj=c32.bintaglist[j]
            if sj==si:
                map23to32[i]=j
    for n23 in xrange(c23.Ncross):
        i23,j23=c23.crosspairs[n23,:]
        i32=map23to32[i23]
        j32=map23to32[j23]
        n32=c32.crossinds[i32,j32]
        if not np.all(c23.cl[n23,:]==c32.cl[n32,:]):
            print 'MISMATCH ON CROSSCORR BETWEEN',c23.bintaglist[i23],c23.bintaglist[j23]
        else:
            print '-'
    #conclusion: changing order of [2,3] doesn't mix up cl values; all shift
    # as expected
    
    

######################
def MDtest_plot_zwindowfuncs(desNbins=[3],nvss=True,plotdir='output/MDchecks/plots/'):
    maptypes=MDtest_get_maptypelist(includeisw=False,Ndesbins=desNbins,nvss=True)
    Nrecs=len(maptypes)
    plotname='MDtest_zbins'
    Nrecs=len(maptypes)
    binsetlist=[s.binmaps for s in maptypes]
    labels=[]
    for s in maptypes:
        if 'nvss' in s.tag:
            labels.append('NVSS')
        elif 'desMD2bin' in s.tag:
            labels.append('DES 2 bin')
        elif 'desMD3bin' in s.tag:
            labels.append('DES 3 bin')
    colors=['#ff7f00','#377eb8','#e7298a']#'#d95f02''#1b9e77'
    zmax=3.
    nperz=100 ### resolution for grid
    zgrid=np.arange(nperz*zmax)/float(nperz)
    plt.figure(0)
    ax=plt.subplot()
    ax.set_yticklabels([])
    plt.subplots_adjust(bottom=.2)
    plt.subplots_adjust(left=.1)
    plt.subplots_adjust(right=.85)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)
    plt.xlabel('Redshift z')
    plt.ylabel(r'$dn/dz*b(z)$ (arb. units)')
    #ymax=.33
    #plt.ylim(0,ymax)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x')
    ax.set_yticklabels([])
    for n in xrange(Nrecs): ### for each recon
        colstr=colors[n%len(colors)]
        ntot=0
        for i in xrange(len(binsetlist[n])):
            ntot+=binsetlist[n][i].nbar
        for i in xrange(len(binsetlist[n])):#loop through individual bins
            m=binsetlist[n][i]
            if maptypes[n].tag=='nvss': ### WHY NVSS DIFFERENT? Maybe because it's first?
                wgrid=m.window(zgrid)
            else:
                wgrid=m.window(zgrid)*m.nbar/ntot
            if i==0:
                label=labels[n]
            else:
                label=''
            plt.plot(zgrid,wgrid,color=colstr,label=label,linewidth=2)

    plt.legend(prop={'size':20})
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()
######################
    
def Plancktest_plot_zwindowfuncs(desNbins=[3],nvss=True, nvss_old=True, MphG=True, plotdir='output/MDchecks/plots/', dndz_only=False,show=True):
    maptypes = [get_Planck_NVSSlike_SurveyType(), get_Planck_MphG_like_SurveyType(), get_NVSSlike_SurveyType()] + [get_MDDESlike_SurveyType(nbins=n) for n in desNbins]
        
#    maptypes += MDtest_get_maptypelist(includeisw=False, Ndesbins=desNbins, nvss=True)
    Nrecs=len(maptypes)
    plotname='Plancktest_zbins'
    Nrecs=len(maptypes)
    binsetlist=[s.binmaps for s in maptypes]
    print 'binsetlist = ',binsetlist
    labels=[s.tag for s in maptypes]
#    for s in maptypes:
#        if 'nvss' in s.tag:
#            labels.append('NVSS')
#        elif 'desMD2bin' in s.tag:
#            labels.append('DES 2 bin')
#        elif 'desMD3bin' in s.tag:
#            labels.append('DES 3 bin')
    colors=['#ff7f00','#377eb8','#e7298a','red','blue']#d95f02','#1b9e77']
    zmax=6.
    nperz=100 ### resolution for grid
    zgrid=np.arange(nperz*zmax)/float(nperz)
    plt.figure(0)
    ax=plt.subplot()
    ax.set_yticklabels([])
    plt.subplots_adjust(bottom=.2)
    plt.subplots_adjust(left=.1)
    plt.subplots_adjust(right=.85)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)
    plt.xlabel('Redshift z')
    if dndz_only:#all normalized to 1
        plt.ylabel(r'$dn/dz$ (arb. units)') 
    else: #dndz normed to 1, then multiplied by b(z)
        plt.ylabel(r'$dn/dz*b(z)$ (arb. units)')
    #ymax=.33
    #plt.ylim(0,ymax)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x')
    ax.set_yticklabels([])
    for n in xrange(Nrecs): ### for each recon
        print labels[n]
        colstr=colors[n%len(colors)]
        ntot=0
        
        for i in xrange(len(binsetlist[n])):
            print 'adding nbar= {0:g}'.format(binsetlist[n][i].nbar)
            ntot+=binsetlist[n][i].nbar
        print 'ntot = {0:g}'.format(ntot)
        for i in xrange(len(binsetlist[n])):#loop through individual bins
            m=binsetlist[n][i]
#            if maptypes[n].tag=='nvss': ### WHY NVSS DIFFERENT? Maybe because it's first? --> No, because only one bin
#                wgrid=m.window(zgrid)
#            else:
#            print m.nbar
#            wgrid=m.window(zgrid)*m.nbar/ntot #this doesn't normalize everyhting to same, since still scaled by bias.
            if dndz_only:
                wgrid=m.dndzfull(zgrid)*m.nbar/ntot
            else:
                wgrid=m.window(zgrid)*m.nbar/ntot #m.window(z) = m.dndzfull(z) * m.bias(z)
#            print wgrid.sum()
            if i==0:
                label=labels[n]
            else:
                label=''
            plt.plot(zgrid,wgrid,color=colstr,label=label,linewidth=2)

    plt.legend(prop={'size':20})
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    if show:
        plt.show()
    else:
        plt.close()


def MDtest_plot_rhohist(varname='rho',Ndesbins=[2,3],lmin=3,lmax=80,getrhopred=True,firstNreal=-1,rhofiletag='',nvss=True,plottag=''):
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
    if plottag:
        tag='_'+plottag
    else:
        tag=''
    plotname ='MDtest_{1:s}hist_r{0:05d}{2:s}'.format(Nreal,varname,tag)
    reclabels=[m.tag for m in MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)]
    # if nvss:
    #     reclabels.append('NVSS')
    # for n in Ndesbins:
    #     reclabels.append('DES {0:d} bins'.format(n))


    if varname=='rho':
        au.plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='s':
        au.plot_shist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)

#for testing
def MDtest_plot_clvals(Ndesbins=[2,3],nvss=True,tag='check'):
    cldat=MDtest_get_Cl(Ndesbins=Ndesbins,nvss=nvss)
    ell=cldat.rundat.lvals
    outdir='output/MDchecks/plots/'
    outname='cl_MD_'+tag
    #get shortened label names
    labellist=[]
    counter=0
    for b in cldat.bintaglist:
        if 'isw' in b:
            labellist.append('ISW')
            iswind=counter
        else:
            labellist.append(b.replace('bin_bin',''))#just 'bin0' 'bin1',etc
        counter+=1
    Nbins=cldat.Nmap
    plt.figure(0)
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

    

#################################################################
if __name__=="__main__":
    MDtest_plot_zwindowfuncs([2,3])
    Ndesbins=[2,3]
    lmin=3 #3
    lmax=80
    
    Nreal=10000 #10000
    if 0:
        #rhofiletag='nob0fit'
        rhofiletag=''
        #given Cls, create the true ISW and LSS maps
        MDtest_get_glm(Nreal,Ndesbins=[2,3],nvss=1)
        # do reconsruction for each map
        MDtest_iswrec(Nreal,Ndesbins=[2,3],nvss=1,lmin=lmin,lmax=lmax,rhofiletag=rhofiletag,fitbias=True) 
        #plot the test statistics
        MDtest_plot_rhohist('rho',Ndesbins=[2,3],nvss=1,lmin=lmin,lmax=lmax,firstNreal=Nreal,rhofiletag=rhofiletag,plottag=rhofiletag)
        
        #for debugging
    if 0: #Looking at Cl to test that they look reasonable
        MDtest_plot_clvals(Ndesbins=[2,3],nvss=True,tag='all')
        MDtest_plot_clvals(Ndesbins=[3],nvss=0,tag='just3')
        MDtest_plot_clvals(Ndesbins=[2],nvss=0,tag='just2')
        MDtest_plot_clvals(Ndesbins=[],nvss=1,tag='justnvss')

    if 1:
        cldat=PlanckTest_get_Cl(justread=False, limberl=0)
        Plancktest_plot_zwindowfuncs(dndz_only=False)