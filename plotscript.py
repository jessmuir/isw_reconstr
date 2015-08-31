###########################################
# staging area for plotting functions which
#  will get moved to larger files once they work
###########################################
import numpy as np
import matplotlib.pyplot as plt
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *
from AnalysisUtils import *
from mapdef_utils import *
from run_euclidlike_analysis import *

###########################################
def dummyvals_for_rhoexplot(): #just return random values for rho
    baseN=6
    z0=0.7
    basezedges=bintest_get_finest_zedges(baseN,z0)
    allzedges,divstr=bintest_get_zedgeslist(basezedges,['all'],True)
    Npoints=len(allzedges)
    rhomin=.4
    rhomax=1.
    drho=(rhomax-rhomin)/Npoints
    rholist=rhomin+np.arange(Npoints)*drho
    return allzedges,divstr,rholist

def bintest_rhoexpplot(allzedges,labels,rhoarray):
    plotdir='output/eucbintest/plots/'
    outname='bintest_rhoexp.png'
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    Npoints=len(labels)
    yvals=np.arange(Npoints)
    zvals=allzedges[-1] #should be base value
    Nz=zvals.size
    zinddict={zvals[i]:i for i in xrange(Nz)}

    #divide fiture up into two vertical pieces
    fig=plt.figure(0)
    plt.suptitle(r'Expected correlation coef. between $T^{{\rm ISW}}$ and $T^{{\rm rec}}$', size=18)
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
    ax2.set_xlabel(r'$\langle \rho \rangle$')
    ax2.grid(True)
    ax2.scatter(rhoarray,yvals)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels()[0], visible=False)#don't show number at first label

    plt.savefig(plotdir+outname)


#################################################################
if __name__=="__main__":
    allzedges,divstrlist,rholist=dummyvals_for_rhoexplot()
    bintest_rhoexpplot(allzedges,divstrlist,rholist)
