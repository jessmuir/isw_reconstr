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

     #plots for Dragan's grant proposal
        #depth test histogram for only z0=.3,.7; no title, clean up legend
        #  maybe put inset of dndz distributions?

        #scatter plot for one realization for calib errors; clean up legend

        #binning test plot; only one sigma z point, remove title

#################################################################
def caltest_TTscatter_forprop(r=0,varlist=[1.e-7,1.e-6,1.e-5,1.e-4]):
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
    #get rec filenames, go from largest to smallest variance
    for i in reversed(xrange(Nvar)):
        modtag=getmodtag_fixedvar(varlist[i],'g',lmin=0,lmax=30,width=10)
        recmapf=dummyalm.get_mapfile_fortags(r,maptag='iswREC.eucz07',modtag=modtag)
        recmapfiles.append(recmapf)
        reclabels.append(r'$\langle c^2(\hat{{n}})\rangle$={0:.0e}'.format(varlist[i]))
    #then do fiducial case
    fidrecf=dummyalm.get_mapfile_fortags(r,'iswREC.eucz07','unmod')
    recmapfiles.append(fidrecf)
    reclabels.append('No calib. error')
    colors=['#a6611a','#08519c','#41b6c4','#78c679','#ffffb2']

    plotname='TrecTisw_scatter_caltest_forprop.r{0:05d}'.format(r)
    plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir,plotname,colors)

#--------------------------------
# plot histogram of rho or s, switch between variables given by varname
def depthtest_plot_rhohist_forproposal(z0vals=np.array([.3,.7])):
    varname='rho'
    plotdir='output/depthtest/plots/'
    testname="Depth test"
    rhogrid=depthtest_read_rho_wfiles(z0vals,varname)
    Nreal=rhogrid.shape[1]
    rhopred=depthtest_get_expected_rho(z0vals,varname)
    plotname ='depthtest_{1:s}hist_r{0:05d}_forprop'.format(Nreal,varname)
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
    colors=['#1b9e77','#e7298a']
    plt.figure(0)
    #plt.title(plotname)
    plt.xlabel(xtitle,fontsize=14)
    plt.ylabel('Realizations',fontsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(rhogrid[i,:])
        sigma=np.std(rhogrid[i,:])
        colstr=colors[i%len(colors)]
        if len(predvals):
            predval=predvals[i]
            plt.axvline(predval,linestyle='-',color=colstr)
            label=r'{0:s}: $\langle {4:s}\rangle={3:0.3f}$'.format(reclabels[i],mean,sigma,predval,varstr)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(rhogrid[i,:],bins=Nbins,range=vallim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.7)

    plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from sample')
    plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='expectation value')
    plt.legend(loc='best')

    #plot window functions as inset
    ax=plt.axes([.23,.28,.3,.25])
    bins=depthtest_get_binmaps(z0vals,False) #just gal maps
    Nbins=len(bins)
    zmax=3.
    nperz=100
    zgrid=np.arange(nperz*zmax)/float(nperz)
    ax.set_yticklabels([])
    #plt.title(r'Depth test: redshift distributions')
    plt.xlabel('Redshift z')#,fontsize=20)
    plt.ylabel('Arbitrary units')#,fontsize=20)
    plt.ylim(0,.7)
    plt.xlim(0,zmax)
    #ax.tick_params(axis='x', labelsize=18)
    for n in xrange(Nbins):
        m=bins[n]
        wgrid=m.window(zgrid)*m.nbar/1.e9
        colstr=colors[n%len(colors)]
        #plt.fill_between(zgrid,0,wgrid, facecolor=colstr,edgecolor='none',linewidth=2, alpha=0.3)
        plt.plot(zgrid,wgrid,color=colstr,linestyle='-',linewidth=2)#,label='{0:0.1f}'.format(z0vals[n]))
    #plt.legend(title='$z_0$',loc='center right',fancybox=False, framealpha=0.,prop={'size':20},handlelength=3.5)
    eqstr=r'$\frac{dn}{dz}\propto \,z^2 e^{-\left(z/z_0\right)^{1.5}}$'
    textbox=ax.text(.75, .6, eqstr,fontsize=16,verticalalignment='top',ha='left')#, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'facecolor':'none','edgecolor':'none'))



    outname=plotdir+plotname+'.png'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#---------------------------------------------------
def bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.05]):
    rholist=[]
    for s in sigzlist:
        divstr,rho=bintest_get_rhoexp(finestN,z0,s,overwrite=False,doplot=False)
        rholist.append(rho)
    legtitle=''#$\\sigma_z(z)*(1+z)^{{-1}}$'
    labellist=[]#'${0:0.3f}$'.format(s) for s in sigzlist]
    zedges0=bintest_get_finest_zedges(finestN,z0)
    allzedges=bintest_get_zedgeslist(zedges0,['all'],False)
    markerlist=[]
    colorlist=[]
    outtag='_forprop'
    outname='eucbintest_rhoexp'+outtag+'.png'
    bintest_rhoexpplot(allzedges,divstr,rholist,labellist,outname,legtitle,markerlist,colorlist,outtag)


#################################################################
if __name__=="__main__":
    #depthtest_plot_rhohist_forproposal(z0vals=np.array([.3,.7]))
    #caltest_TTscatter_forprop(r=0,varlist=[1.e-7,1.e-6,1.e-5,1.e-4])
    bintest_rhoexp_comparesigs(finestN=6,z0=0.7,sigzlist=[0.05])
