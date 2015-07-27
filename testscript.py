#################################################################
#  This script has been used to test that genCrossCor, genMaps and 
#   other helper modules run as expected. 3/29/15
#################################################################
import numpy as np
import matplotlib.pyplot as plt
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *

def biastest(z):
    return 1.

def dndztest(z):
    z0=.32
    alpha=.36
    return (z/z0)**(alpha)*np.exp(-alpha*z/z0)

zbins = np.array([.01,5.,1.])
sig = .05
nbar=1.e8

iswtest = MapType('isw',np.array([.01,3.]),True,False,'A test ISW map')
#mattest = MapType('mat',zbins,False,False,'A test DM map')
galtesta = SurveyType('galA', np.array([.01,5.,1.]),sig,nbar,dndztest,biastest,'A test galaxy map, two bins')
galtestb = SurveyType('galB', np.array([.01,1.]),sig,nbar,dndztest,biastest,'A test galaxy map, one bin')
galtestc=  SurveyType('galC', np.array([.01,1.]),sig,nbar/1.e2,dndztest,biastest,'A test galaxy map, one noisy bin')

if 0: #testing galaxy binning functions
    print 'tag',galtest.tag
    print 'longtag',galtest.longtag
    print 'isISW',galtest.isISW
    print 'isgal',galtest.isGal
    print 'sigz',galtest.sigz(0)
    print 'nbar',galtest.nbar
    print galtest.bintags
    for b in galtest.binmaps:
        plt.plot(b.zgrid,b.wgrid,label=b.tag)
    plt.legend()
    plt.show()

#ggnerate C_l for map-making and recosntruction testing purposes
#these will not be accurate, but will have the right file formats


ktest = KData(kmin=1.e-2,kmax=1,nperlogk=10,krcut=1)
outdir='test_output/test/'
clrun = ClRunData('test',rundir = outdir,kdata=ktest,lmax=30,zmax=5.)

maplist=iswtest.binmaps+galtesta.binmaps+galtestb.binmaps+galtestc.binmaps
maptags=get_bintaglist(maplist)
pairs=['all']
#maplist=maplist+galtestc.binmaps
#pairs.append(('galC','isw'))
cldat=getCl(maplist,clrun,pairs,DoNotOverwrite=1,redoIlk=0)

print 'cldat.nbar',cldat.nbar
glmdat=get_glm(cldat,'nbartest',2,overwrite=1)
print 'glmdat.nbarlist',glmdat.nbarlist
print glmdat.glm.shape
