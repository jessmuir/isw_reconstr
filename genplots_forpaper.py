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
    print 'rhogrid.shape',rhogrid.shape
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
def depthtest_plot_shist_forpaper(z0vals=np.array([.3,.5,.6,.7,.8])):
    varname='s'
    plotdir='output/plots_forpaper/'
    rhogrid=depthtest_read_rho_wfiles(z0vals,varname)
    Nreal=rhogrid.shape[1]
    rhopred=depthtest_get_expected_rho(z0vals,varname)
    plotname ='depthtest_{1:s}hist_r{0:05d}_forpaper'.format(Nreal,varname)
    reclabels=['$z_0={0:0.1f}$'.format(z0) for z0 in z0vals]
    
    #plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred)
    varstr=r's'
    Nreal=rhogrid.shape[1]
    title=''#r'{0:s}: correlation coef. $\rho$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$s=\langle (T_{{\rm true}}-T_{{\rm rec}})^2\rangle_{{\rm pix}}^{{1/2}}/\sigma_{{T}}^{{\rm true}}$'
    #plothist(varstr,rhogrid,reclabels,title,xtitle,plotdir,plotname,rhopred)
    predvals=rhopred
    Nbins=100
    Nrecs=rhogrid.shape[0]
    Nreal=rhogrid.shape[1]
    maxval=np.max(rhogrid)
    minval=np.min(rhogrid)
    vallim=(minval,maxval)
    colors=['#e41a1c','#ff7f00','#984ea3','#377eb8','#4daf4a']
    plt.figure()#figsize=(7,7))
    plt.subplots_adjust(left=0.15, bottom=.2, right=.95, top=.95, wspace=0, hspace=0)
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
    plt.legend(loc='upper right',frameon=True,fontsize=14,ncol=1)


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
# plots rho exp and data points for one shape and lmin
#combine stuff from caltest_compare_clcal_shapes with rhoexpplot
def caltest_basic_expplot_forpaper(varname='rho',plotdir='output/plots_forpaper/'):
    outname='caltest_'+varname+'_exp_basic.pdf'
    colorlist=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']#qualitative
    reclmin=2
    #info about calibration errors
    shape='g'
    width=10.
    lmincal=0#min ell for calib error
    lmaxcal=30#max ell for caliberror

    #what points to plot
    varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))
    shortvarlist=[1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,1.e-2]#for data points 

    #get data
    rhoexplist=[]#will be 1D; [variance ] 
    rhoexplist=caltest_get_rhoexp(varlist,lmax=lmaxcal,lmin=lmincal,shape=shape,width=width,overwrite=False,doplot=False,saverho=True,varname=varname,reclmin=reclmin)
    #shapestr=r'$C_{{\ell}}^{{\rm cal}}\propto e^{{-(\ell/{0:.0f})^2}}$'.format(width,lmincal,lmaxcal)
    linelabel='Theory (add. only)'
    datplot=caltest_getdataplot_forshapecompare(varname,shortvarlist,[shape],[width],[lmincal],[lmaxcal],recminelllist=np.array([reclmin]),colorlist=colorlist,labellist=['Results from sim.'])

    #do plotting!
    #assuming last entry in varlist, rhoarray is fiducial (var=0)
    fig=plt.figure(figsize=(7,4))
    fig.subplots_adjust(bottom=.2)
    fig.subplots_adjust(left=.175)
    fig.subplots_adjust(right=.95)
    ax1 = plt.subplot() 
    for item in ([ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels()
                  + ax1.get_yticklabels()):
        item.set_fontsize(18)
    for item in ([ax1.yaxis.label]):
        item.set_fontsize(22)
    plt.sca(ax1)
    ax1.grid(True)
    ax1.set_xscale('log')
    ax1.set_xlim((10**-7.5,10**-1.5))
    ax1.set_xlabel(r'Variance of calib. error field ${\rm var}[c]$')
    ax1.axhline(0,color='grey',linestyle='-')
    if varname=='rho':
        ax1.set_ylim(-.3,1.3)
        ax1.axhline(1,color='grey',linestyle='-')
        ax1.set_ylabel(r'$\langle \rho \rangle$')
    elif varname=='s':
        ax1.set_ylabel(r'$\langle s \rangle$')
        ax1.set_ylim(.1,1.e4)
        ax1.set_yscale('log')

    #fill in regions for current/future values
    nowmin=1.e-4
    nowmax=1.e-3
    nowcenter=10**(.5*(np.log10(nowmin)+np.log10(nowmax)))
    nowlabel='current'
    nowcol='#377eb8'
    futuremin=1.e-6
    futuremax=1.e-5
    futurecenter=10**(.5*(np.log10(futuremin)+np.log10(futuremax)))
    futurelabel='future'
    futurecol='#4daf4a'
    ymin,ymax=ax1.get_ylim()
    #plt.fill_between(varlist[:-1],ymin,ymax, facecolor=nowcol,edgecolor='none',linewidth=2, alpha=0.3)
    ax1.axvspan(nowmin,nowmax,alpha=0.1,color=nowcol)
    plt.annotate(nowlabel,xy=(nowcenter,ymin),horizontalalignment='center',verticalalignment='bottom',fontsize=16,color=nowcol)
    ax1.axvspan(futuremin,futuremax,alpha=0.1,color=futurecol)
    plt.annotate(futurelabel,xy=(futurecenter,ymin),horizontalalignment='center',verticalalignment='bottom',fontsize=16,color=futurecol)
        
    #theory line
    ax1.plot(varlist[:-1],rhoexplist[:-1],label=linelabel,color=colorlist[0])

    #data points should just have one entry, but copying whole mess anyway
    for i in xrange(len(datplot)):
        datvar=datplot[i][0]#array
        datrho=datplot[i][1]#array
        datlabel=datplot[i][2] #string
        datNreal=0
        datcol=datplot[i][3]
        datsig=datplot[i][4]         
        datrefmean=datplot[i][5]
        if len(datplot[i])>5:
            datNreal=datplot[i][6]
        if not datcol:
            datcol=colorlist[0]
        ax1.errorbar(datvar,datrho,yerr=datsig/datrefmean,label=datlabel,color=datcol,linestyle='None',marker='o')    
            
    plt.sca(ax1)
    if varname=='rho':
        plt.legend(fontsize=16,loc='upper right',numpoints=1)
    elif varname=='s':
        plt.legend(fontsize=16,loc='upper left',numpoints=1)

    print 'Saving plot to ',plotdir+outname
    plt.savefig(plotdir+outname)
    plt.close()
    
#--------------------------------------
def caltest_lmin_plot_forpaper(varname='rho',plotdir='output/plots_forpaper/'):
    shortvarlist=[1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,1.e-2]
    varlist=list(caltest_get_logspaced_varlist(minvar=1.e-8,maxvar=.1,Nperlog=10))
    reclminlist=np.array([2,3,5])
    caltest_compare_lmin(varlist,varname='rho',dodataplot=True,recminelllist=reclminlist,shortrecminelllist=reclminlist,shortvarlist=shortvarlist,justdat=True,plotdir=plotdir)

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
    plt.close()
            
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
    #depthtest_plot_shist_forpaper(z0vals=np.array([.3,.5, .6,.7,.8]))
    #depthtest_plot_dndz_forpaper()
    #for r in xrange(13,14):
    #    depthtest_TTscatter_forpaper(r,savepngmaps=False)

    #bintest_plot_rhohist_forpaper()
    #bintest_plot_zwindowfuncs(plotdir='output/plots_forpaper/')
    #bintest_rhoexp_comparesigs(sigzlist=[0.001,0.05,.1],markerlist=['v','d','^','d'],plotdir='output/plots_forpaper/',datsigs=[0.05],datdivs=['111111','222','6'],overwrite=0)


    #lmintest_plot_rhoexp(lminlist=np.arange(1,20),varname='rho',dodata=True,datlmin=np.array([1,2,3,4,5]),plotdir='output/plots_forpaper/')

    #shot noise tests
    shortnbarlist=np.array([1.e-4,1.e-3,.01,.1,1.,10.])#in arcmin^-2
    shortnbarsr=shortnbarlist*((180.*60./np.pi)**2)
    scaletovar=shortnbarsr[0]
    nbarlist=caltest_get_logspaced_varlist(1.e-6,1.e3)
    shottest_plot_rhoexp(nbarlist=nbarlist,varname='rho',passnbarunit='amin2',overwrite=0,dodata=True,datnbar=shortnbarlist,plotdir='output/plots_forpaper/')
    shottest_plot_rhoexp(nbarlist=nbarlist,varname='s',passnbarunit='amin2',overwrite=0,dodata=True,datnbar=shortnbarlist,plotdir='output/plots_forpaper/')
    
    #z0test_onesim_plot(varname='rho',dohatch=False,plotdir='output/plots_forpaper/')
    #z0test_onesim_plot(varname='s',dohatch=False,plotdir='output/plots_forpaper/')
    #bztest_onesim_plot(varname='rho',dohatch=False,plotdir='output/plots_forpaper/') 
    #bztest_onesim_plot(varname='s',dohatch=False,plotdir='output/plots_forpaper/')
    badfracs=np.array([0.,1.e-3,5.e-3,1.e-2,2.e-2,5.e-2,.1,.2])
    #catztest_onesim_plot(varname='rho',Nbins=1,recf=badfracs,fidf=.01,secondfidf=.1,dohatch=False,plotdir='output/plots_forpaper/')
    #catztest_onesim_plot(varname='s',Nbins=1,recf=badfracs,fidf=.01,secondfidf=.1,dohatch=False,plotdir='output/plots_forpaper/')

    #z0test_Clcomp(perrors=np.array([10,50]),plotdir='output/plots_forpaper/',plotISWgalratio=False)
    #bztest_Clcomp(b2vals=np.array([0.,.1,.5,1.]),plotdir='output/plots_forpaper/',plotISWgalratio=False)
    #catz_Clcomp(badfracs=np.array([0.,1.e-2,5.e-2,.1,.2]),plotdir='output/plots_forpaper/',plotISWgalratio=False)
        
    #caltest_basic_expplot_forpaper('rho')
    #caltest_basic_expplot_forpaper('s')
    #caltest_lmin_plot_forpaper('rho')
    
    #caltest_Clcomp(varlist,plotdir='output/plots_forpaper/')

    #caltest_compare_lmin(varlist,varname='rho',dodataplot=True,recminelllist=reclminlist,shortrecminelllist=shortreclminlist,shortvarlist=shortvarlist,justdat=True,plotdir='output/plots_forpaper/')
    
    
    #MDtest_plot_rhohist_forpaper()
    #MDtest_plot_zwindowfuncs(desNbins=[2,3],plotdir='output/plots_forpaper/')
