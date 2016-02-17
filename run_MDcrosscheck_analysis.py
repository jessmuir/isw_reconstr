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
    cldat=getCl(bins,rundat,dopairs=pairs,DoNotOverwrite=justread)

    #messing with cldata for test case; make des3_bin0 copy of nvss
    #workin ghere
    des20inds=[]
    des30inds=[]
    for m in xrange(cldat.Nmap):
        if 'nvss' in cldat.bintaglist[m]:
            nvssind=m
        elif ('desMD3' in cldat.bintaglist[m]):
            des30inds.append(m)
        elif ('desMD2' in cldat.bintaglist[m]):
            des20inds.append(m)
        elif ('isw' in cldat.bintaglist[m]):
            iswind=m
    nvsscl=cldat.cl[cldat.crossinds[nvssind,nvssind],:]
    nvssiswcl=cldat.cl[cldat.crossinds[nvssind,iswind],:]
    for n in xrange(cldat.Ncross): #basically, make des3 bin 0 a copy of nvss
        p,q=cldat.crosspairs[n,:]
        if (p in des30inds and q in des30inds):
            if p==des30inds[0] and q==des30inds[0]: 
                cldat.cl[cldat.crossinds[des20inds[0],des20inds[0]],:]
            elif  p==des30inds[1] and q==des30inds[1]: 
                cldat.cl[cldat.crossinds[des20inds[1],des20inds[1]],:]
            elif  (p==des30inds[1] and q==des30inds[0]) or (p==des30inds[0] and q==des30inds[1]): 
                cldat.cl[cldat.crossinds[des20inds[0],des20inds[1]],:]
            else:
                cldat.cl[n,:]=np.zeros(cldat.Nell)  
        elif p in des30inds or q in des30inds:
            if p in des30inds:
                y=p #ind assiciated with 3 bin des
                x=q # other map
            else:
                y=q
                x=p
            if x==iswind or (x in des20inds):
                if y==des30inds[0]:
                    cldat.cl[cldat.crossinds[des20inds[0],x],:]
                elif y==des30inds[1]:
                    cldat.cl[cldat.crossinds[des20inds[1],x],:]
                else:
                    cldat.cl[n,:]=np.zeros(cldat.Nell) 
            else:
                cldat.cl[n,:]=np.zeros(cldat.Nell)   
    
    return cldat

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
    print 'cldat.bintaglist:',cldat.bintaglist
    
    #cldat seems to be associated with correct crosspairs
    # also, manipulating cl data has expected effects on rho hist
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
    reclabels=[m.tag for m in MDtest_get_maptypelist(Ndesbins=Ndesbins,nvss=nvss)]
    # if nvss:
    #     reclabels.append('NVSS')
    # for n in Ndesbins:
    #     reclabels.append('DES {0:d} bins'.format(n))


    if varname=='rho':
        plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    elif varname=='s':
        plot_shist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)

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
    lmin=3
    lmax=80
    
    Nreal=10#0000
    #checkMD_cl_ordering() #when order of maps changes, cl vals shift as expected
    if 1:
        rhofiletag=''
        rhofiletag='3only'
        #MDtest_get_glm_and_rec(Nreal,justgetrho=False,dorho=1,Ndesbins=Ndesbins,lmin=lmin,lmax=lmax,rhofiletag=rhofiletag)
        donvss=False
        MDtest_get_glm(Nreal,Ndesbins=[3,2],nvss=1)#this seems to be where mixup is happening; nvss+2 ok, each individually is ok, but if i pass all three maps, only the first one gets a reasonable looking histogram. When I play with the content of the cldat.cl matrix, the results respond in a reasonable way. not sure what's up

        MDtest_iswrec(Nreal,Ndesbins=[2,3],nvss=1,lmin=lmin,lmax=lmax,rhofiletag=rhofiletag) #including more maps here doesn't mess up results

        MDtest_plot_rhohist('rho',Ndesbins=[2,3],nvss=1,lmin=lmin,lmax=lmax,firstNreal=Nreal,rhofiletag=rhofiletag)
    if 0: #these all look reasonable
        MDtest_plot_clvals(Ndesbins=[2,3],nvss=True,tag='all')
        MDtest_plot_clvals(Ndesbins=[3],nvss=0,tag='just3')
        MDtest_plot_clvals(Ndesbins=[2],nvss=0,tag='just2')
        MDtest_plot_clvals(Ndesbins=[],nvss=1,tag='justnvss')
