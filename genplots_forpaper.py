import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from itertools import permutations
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *
from AnalysisUtils import *
from mapdef_utils import *
import time
from run_euclidlike_analysis import *
from run_MDcrosscheck_analysis import *

     #plots for isw paper
#################################################################

#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def depthtest_plot_rhohist_forpaper(z0vals=np.array([.3,.5,.6,.7,.8])):
    varname='rho'
    plotdir='output/plots_forpaper/'
    rhogrid=depthtest_read_rho_wfiles(z0vals,varname)
    Nreal=rhogrid.shape[1]
    rhopred=depthtest_get_expected_rho(z0vals,varname)
    plotname ='depthtest_{1:s}hist_r{0:05d}_forpaper'.format(Nreal,varname)
    reclabels=['$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]
    
    #plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    varstr=r'\rho'
    Nreal=rhogrid.shape[1]
    title=''#r'{0:s}: correlation coef. $\rho$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$'
    #plothist(varstr,rhogrid,reclabels,title,xtitle,plotdir,plotname,rhopred)
    predvals=rhopred
    Nbins=100
    Nrecs=rhogrid.shape[0]
    Nreal=rhogrid.shape[1]
    maxval=np.max(rhogrid)
    minval=np.min(rhogrid)
    vallim=(minval,maxval)
    #rholim=(0.,maxrho)
    #colors=['#1b9e77','#e7298a']
    #colors=['#1b9e77','#d95f02','#e7298a','#7570b3','#66a61e','#e6ab02']
    colors=['#e41a1c','#ff7f00','#984ea3','#377eb8','#4daf4a']
    plt.figure()#figsize=(7,7))
    plt.subplots_adjust(left=0.15, bottom=.2, right=.95, top=.95, wspace=0, hspace=0)
    #plt.title(plotname)
    plt.xlabel(xtitle,fontsize=26)
    plt.ylabel('Realizations',fontsize=26)
    #plt.ylim((0,1300))
    #plt.xlim(-.2,1.)
    plt.tick_params(axis='y', which='both', labelsize=16)
    plt.tick_params(axis='x', which='both', labelsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        colstr=colors[i%len(colors)]
        if len(predvals):
            predval=predvals[i]
            plt.axvline(predval,linestyle='-',color=colstr)
            label=r'{0:s}: $\langle {1:s}\rangle={2:0.3f}$'.format(reclabels[i],varstr,predval)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=vallim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.6)

    plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from sim.')
    plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='mean from theory')
    plt.legend(loc='upper left',frameon=True,fontsize=14,ncol=2)

    #plot window functions as inset
    ax=plt.axes([.2,.4,.35,.2])#lower left
    bins=depthtest_get_binmaps(z0vals,False) #just gal maps
    Nbins=len(bins)
    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title(r'Depth test: redshift distributions')
    plt.xlabel('Redshift z',fontsize=12)
    #plt.ylabel(r'arbitrary units',fontsize=14)
    plt.ylim(0,.7)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x', labelsize=14)
    for n in xrange(Nbins):
        m=bins[n]
        wgrid=m.window(zgrid)*m.nbar/1.e9
        colstr=colors[n%len(colors)]
        plt.plot(zgrid,wgrid,color=colstr,linestyle='-',linewidth=2)
    eqstr=r'$\frac{dn}{dz}\propto \,z^2 e^{-\left(z/z_0\right)^{1.5}}$'
    textbox=ax.text(.73, .7, eqstr,fontsize=20,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def bintest_plot_rhohist_forpaper(divstr=['6','222','111111']):
    varname='rho'
    plotdir='output/plots_forpaper/'
    rhogrid=bintest_read_rho_wfiles(divstr,varname=varname)
    Nreal=rhogrid.shape[1]
    _,rhopred=bintest_get_rhoexp(doplot=False,getdivs=divstr,saverho=False)
    plotname ='bintest_{1:s}hist_r{0:05d}_forpaper'.format(Nreal,varname)
    reclabels=['1 bin','3 bins','6 bins']
    
    #plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    varstr=r'\rho'
    Nreal=rhogrid.shape[1]
    title=''#r'{0:s}: correlation coef. $\rho$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$'
    #plothist(varstr,rhogrid,reclabels,title,xtitle,plotdir,plotname,rhopred)
    predvals=rhopred
    Nbins=100
    Nrecs=rhogrid.shape[0]
    Nreal=rhogrid.shape[1]
    maxval=np.max(rhogrid)
    minval=np.min(rhogrid)
    vallim=(minval,maxval)
    #rholim=(0.,maxrho)
    #colors=['#1b9e77','#d95f02','#e7298a','#7570b3','#66a61e','#e6ab02']
    colors=['#ff7f00','#377eb8','#e7298a']#'#d95f02''#1b9e77'
    #colors=['#e7298a','#e6ab02','#66a61e']
    plt.figure(0)
    plt.subplots_adjust(left=0.15, bottom=.17, right=.95, top=.95, wspace=0, hspace=0)
    #plt.title(plotname)
    plt.xlabel(xtitle,fontsize=26)
    plt.ylabel('Realizations',fontsize=26)
    plt.tick_params(axis='y', which='both', labelsize=16)
    plt.tick_params(axis='x', which='both', labelsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        colstr=colors[i%len(colors)]
        if len(predvals):
            predval=predvals[i]
            plt.axvline(predval,linestyle='-',color=colstr)
            label=r'{0:s}: $\langle {1:s}\rangle={2:0.3f}$'.format(reclabels[i],varstr,predval)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=vallim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.6)

    plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from simulation')
    plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='mean from theory')
    plt.legend(loc='upper left',frameon=True,fontsize=16)

    #plot window functions as inset
    ax=plt.axes([.21,.33,.33,.25])#lower left
    #ax=plt.axes([.2,.72,.33,.85])
    Nrecs=len(divstr)
    binsetlist=[]
    for n in xrange(Nrecs):
        bins=bintest_get_binmaps(getdivs=[divstr[n]],includeisw=False) #just gal maps
        binsetlist.append(bins)

    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title(r'Depth test: redshift distributions')
    plt.xlabel('Redshift z',fontsize=14)
    #plt.ylabel(r'arbitrary units',fontsize=14)
    ymax=.33
    plt.ylim(0,ymax)
    plt.xlim(0,zmax)
    #ax.tick_params(axis='x', labelsize=18)

    #plot isw kernel
    cosmparamfile='testparam.cosm'
    #plotname=plotname+'_iswkernel'
    cosm=Cosmology(cosmparamfile)
    cosm.tabulateZdep(zmax,nperz=nperz)
    cosmz=cosm.z_array
    kernel=(1.-cosm.f_array)*cosm.g_array #isw kernel, stripped of prefactor
    kernelmax=np.max(kernel)
    wantmax=.8*ymax
    scaleby=wantmax/kernelmax

    eqstr=r'$\frac{dn}{dz}$'
    #plt.plot(np.array([]),np.array([]),color='black',label=eqstr,linewidth=2,linestyle='-')#dummy line
    #plt.plot(cosmz,kernel*scaleby,color='grey',label='ISW kernel',linewidth=2,linestyle='--')
    #plt.legend(loc='upper right',fancybox=False, framealpha=0.,prop={'size':16},handlelength=3.5)
    
    #plot bin envelopes
    linewidths=[6,4,2]
    linestyles=['-','-','-']
    for n in xrange(Nrecs):
        for i in xrange(len(binsetlist[n])):#loop through individual bins
            m=binsetlist[n][i]
            wgrid=m.window(zgrid)*m.nbar/1.e9
            colstr=colors[n%len(colors)]
            plt.plot(zgrid,wgrid,color=colstr,linestyle=linestyles[n],linewidth=linewidths[n])

    textbox=ax.text(1.75, .25, eqstr,fontsize=24,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))
    
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#==========================================
#do Tisw-Trec scatter plot for a given realization r
def depthtest_TTscatter_forpaper(r=0, z0vals=np.array([0.3,0.5,0.6,0.7,0.8]),savepngmaps=False):
    plotdir='output/plots_forpaper/'
    #colors=['#1b9e77','#d95f02','#e7298a','#7570b3','#66a61e','#e6ab02']
    colors=['#e41a1c','#ff7f00','#984ea3','#377eb8','#4daf4a']
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
    reclabels=[r'$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]

    #set up plot
    plotname='TrecTisw_scatter_depthtest.r{0:05d}'.format(r)
    plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir,plotname,colors,filesuffix='pdf')
    
#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def MDtest_plot_rhohist_forpaper():
    varname='rho'
    plotdir='output/plots_forpaper/'
    Ndesbins=[2,3]
    lmin=3
    lmax=80
    rhofiletag=''
    nvss=True
    rhogrid=MDtest_read_rho_wfiles(varname,Ndesbins,lmin,lmax,rhofiletag,nvss=nvss)

    Nreal=rhogrid.shape[1]
    rhopred=MDtest_get_expected_rho(varname,Ndesbins,lmin,lmax,nvss=nvss)
    plotname ='MDtest_{1:s}hist_r{0:05d}_forpaper'.format(Nreal,varname)
    reclabels=['NVSS','DES 2 bin','DES 3 bin']
    
    varstr=r'\rho'
    xtitle=r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$'
    predvals=rhopred
    Nbins=100
    Nrecs=rhogrid.shape[0]
    maxval=np.max(rhogrid)
    minval=np.min(rhogrid)
    vallim=(minval,maxval)
    #rholim=(0.,maxrho)
    #colors=['#1b9e77','#d95f02','#e7298a','#7570b3','#66a61e','#e6ab02']
    colors=['#ff7f00','#377eb8','#e7298a']#'#d95f02''#1b9e77'
    #colors=['#e7298a','#e6ab02','#66a61e']
    plt.figure(0)
    plt.subplots_adjust(left=0.15, bottom=.17, right=.95, top=.95, wspace=0, hspace=0)
    #plt.title(plotname)
    plt.xlabel(xtitle,fontsize=26)
    plt.xlim((.0,1.))
    plt.ylabel('Realizations',fontsize=26)
    plt.tick_params(axis='y', which='both', labelsize=16)
    plt.tick_params(axis='x', which='both', labelsize=16)
    MDrhovals=[0.47,0.77,0.84]
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        colstr=colors[i%len(colors)]
        if len(predvals):
            predval=predvals[i]
            plt.axvline(predval,linestyle='-',color=colstr)
            label=r'{0:s}: $\langle {1:s}\rangle={2:0.3f}$'.format(reclabels[i],varstr,predval)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=vallim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.6)
        ax=plt.axes()

    for i in xrange(Nrecs):
        colstr=colors[i%len(colors)]
        ax.arrow(MDrhovals[i],1200,0,-50,color=colstr,head_width=.02,head_length=30,zorder=100)

    plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from simulation')
    plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='mean from theory')
    plt.legend(loc='center left',frameon=True,fontsize=16)


  
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#################################################################
if __name__=="__main__":
    #depthtest_plot_rhohist_forpaper(z0vals=np.array([.3,.5, .6,.7,.8]))
    #depthtest_plot_dndz_forpaper()
    #bintest_plot_rhohist_forpaper()
    #bintest_plot_zwindowfuncs(plotdir='output/plots_forpaper/')

    for r in xrange(5):
        depthtest_TTscatter_forpaper(r)
    #bintest_rhoexp_comparesigs(sigzlist=[0.001,0.03,0.05,.1],markerlist=['d','d','d','d'],plotdir='output/plots_forpaper/',datsigs=[0.05],datdivs=['111111','222','6'])
    #z0test_get_rhoexp(simz0=np.array([]),recz0=np.array([]),perrors=np.array([1,10,20,30,50]),fidz0=.7,doplot=True,varname='rho',plotdir='output/plots_forpaper/')
    
    #MDtest_plot_rhohist_forpaper()
    #MDtest_plot_zwindowfuncs(desNbins=[2,3],plotdir='output/plots_forpaper/')
