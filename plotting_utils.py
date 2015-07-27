##################################################################
# This script is intended to to provide some functions for plotting
#  I_l(k) functions etc, mostly to make sure calculations are working as
# expected
##################################################################
import numpy as np
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import sph_jn
import matplotlib.pyplot as plt
##################################################################
div10colors=['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#c7eae5','#80cdc1', '#35978f','#01665e','#003c30']

def read_Ilkfile(infile):
    print "Reading Ilk from file",infile
    x = np.loadtxt(infile,skiprows=6)
    inkrcut=x[0,0]
    k=x[1:,0]
    l=x[0,1:].astype(int)
    I=np.transpose(x[1:,1:])
    #I has indices corresponding to [ell,k]
    return k,l,I

def plot_Ilk_all_ell(ilkfile,outdir=''):
    filebase=ilkfile[ilkfile.rfind('/')+1:ilkfile.rfind('.dat')]
    if not outdir:
        outdir=ilkfile[:ilkfile.rfind('/')+1]
    outf=outdir+filebase+'.png'
    k,l,I=read_Ilkfile(ilkfile)
    absI=np.fabs(I)#+1.e-10*(I==0)
    plt.figure(0)
    plt.title('I_l(k) from '+filebase)
    plt.ylim((1.e-15,10.))
    for lind in np.arange(l.size)[::-1]:
        if l[lind]<=10:
            usecolor=div10colors[l[lind]-1]
        else:
            usecolor=div10colors[l[lind]/10-1]
        if l[lind]<=10 or l[lind]%10==0:
            #plt.semilogx(k,I[lind,:])
            plt.loglog(k,absI[lind,:],color=usecolor)
    plt.savefig(outf)
    plt.close()

def plot_Ilk_someell(I,l,k,l_to_plot,xbounds=(1.e-5,10.),ybounds=(1.e-15,1.)):
    lvals=l_to_plot
   
    lindlist=[]
    for ell in lvals:
        lind=np.where(l==ell)[0][0]
        lindlist.append(lind)
    absI=np.fabs(I)#+1.e-10*(I==0)
    plt.figure(0)
    plt.title('I_l(k)')
    plt.ylim(ybounds)
    plt.xlim(xbounds)
    i=0
    for lind in lindlist:
        i+=1
        #usecolor=div10colors[(i-1)%10]
        plt.loglog(k,absI[lind,:],label=r'$\ell={0:d}$'.format(l[lind]))
    plt.legend()
    plt.show()
    plt.close()

##################################################################
def plot_Ilk_fordeslike():
    datadir='output/Cl_output/Ilktab/'
    #fnames=['desphoto_bin0_Ilk.dat','desphoto_bin1_Ilk.dat','desphoto_bin2_Ilk.dat','desphoto_bin3_Ilk.dat','desphoto_bin4_Ilk.dat','desphoto_bin5_Ilk.dat','desspec_bin0_Ilk.dat','desspec_bin1_Ilk.dat','desspec_bin2_Ilk.dat','desspec_bin4_Ilk.dat','desspec_bin5_Ilk.dat','isw_bin0_Ilk.dat']
    fnames=['isw_bin0_Ilk.dat']
    for n in xrange(5):
        fnames.append('desspec_bin'+str(n)+'_Ilk.dat')
        fnames.append('desphoto_bin'+str(n)+'_Ilk.dat')
    outdir=datadir+'plots/'
    for f in fnames:
        print 'on',f
        plot_Ilk_all_ell(datadir+f,outdir)

def plot_Ilk_for_krtests():
    datadir='test_output/Ilktests/Cl_output/Ilktab/'
    calist=[10,20,50,100,200]
    cmlist=[00,10,15,20]
    types=['isw','mat']
    bins=['lowz','highz','widez']
    fnames=[]
    for t in types:
        for b in bins:
            for ca in calist:
                for cm in cmlist:
                    #fnames.append(t+'_'+b+'_bin0_Ilk.krcuta{0:03d}_m{1:03d}_edge0.010.dat'.format(ca,cm))
                    pass
    fnames.append('isw_highz_bin0_Ilk.krcutNone.dat')
    fnames.append('isw_widez_bin0_Ilk.krcutNone.dat')
    fnames.append('isw_lowz_bin0_Ilk.krcutNone.dat')
    fnames.append('mat_highz_bin0_Ilk.krcutNone.dat')
    fnames.append('mat_widez_bin0_Ilk.krcutNone.dat')
    fnames.append('mat_lowz_bin0_Ilk.krcutNone.dat')
    outdir=datadir+'plots/'
    for f in fnames:
        print 'on',f
        plot_Ilk_all_ell(datadir+f,outdir)
    
#################################################################
if __name__=="__main__":
    plot_Ilk_fordeslike()
    #plot_Ilk_for_krtests()
