# For running tests associated with adding cmb info to isw rec project
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import host_subplot #to have two axis labels
import mpl_toolkits.axisartist as AA #to have two axis labels
from itertools import permutations
from scipy.optimize import leastsq
import MapParams as mp
import CosmParams as cp 
import ClRunUtils as clu
import genCrossCor as gcc
import genMapsfromCor as gmc
import AnalysisUtils as au
import mapdef_utils as mdu
import healpy as hp
import run_MDcrosscheck_analysis as runMD

#use MD test existing Cl files
cldat=runMD.MDtest_get_Cl(justread=True,Ndesbins=[2],nvss=False)
print "\nBEFORE ADDING CMB"
print cldat.bintaglist
print cldat.Nmap
print cldat.nbar
print cldat.crossinds
print cldat.cl[:,5]

cambfnoisw = 'camb_workspace/noisw_scalCls.dat'
cambfwisw = 'camb_workspace/withisw_scalCls.dat'

fiddat = np.loadtxt(cambfwisw,skiprows=1)
ell = fiddat[:cldat.Nell-2,0]
fidtt = fiddat[:cldat.Nell-2,1]/(ell*(ell+1))*2*np.pi*1.e-6**2
noiswtt = np.loadtxt(cambfnoisw,skiprows=1)[:cldat.Nell-2,1]/(ell*(ell+1))*2*np.pi*1.e-6**2
iswcontrib = fidtt - noiswtt

iswtag = 'isw_bin0'
iswind = cldat.tagdict[iswtag]
iswautocl = cldat.cl[cldat.crossinds[iswind,iswind],2:]*(.3**2)*(2.73**2)

#plt.semilogx(ell,iswcontrib,label='isw from camb')
#plt.semilogx(ell,iswautocl,label='from our code')
# had missed adding units to ISW, but also missed a factor of Omega_m in window
plt.semilogx(ell,iswautocl/iswcontrib)
plt.ylabel('[our ISW Cl]/[CAMB ISW Cl]')
plt.xlabel('ell')
plt.xlim((2,cldat.Nell+1))
plt.legend()
plt.show()


cldat.addCMBtemp(cambfnoisw,hasISWpower=True)

print "\nAFTER ADDING CMB"
print cldat.bintaglist
print cldat.Nmap
print cldat.nbar
print cldat.crossinds
print cldat.cl[:,5]
