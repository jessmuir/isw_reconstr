##################################################################
# This script is intended to investigate how binning strategy of
#   a survey with DES-like dndz affects is power for ISW reconstruction.
#
#  It contains functions which will be accessed both by scripts used
#  to generate data on the flux clusters and those which will be used locally
#  to make plots, etc. 
##################################################################
import numpy as np
import MapParams as mp
from ClRunUtils import *
from genCrossCor import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import sph_jn
##################################################################

def nobias(z):
    return 1

def quadbias(z,b0=1.,b2=5.):
    return b0*(1.+b2*(1.+z)**2)

def dndz_DESlike(z):
    z0=.3
    n0=.054
    return n0*z*z*np.exp(-z/z0)

def dndz_Euclidlike(z,z0=0.7):
    #from eq 1 in arxiv:1506.02192
    result=1.5/(z0**3)
    result*=z*z
#    print (z, z0, result)
    exponent=-1.*(z/z0)**1.5
    result*=np.exp(exponent)
    return result
    
def dndz_Spherexlike(z,z0=0.46):
    #from eq 1 in arxiv:1506.02192
    #same as euclid for now, since the more general form isn't a much better fit
    result=1.5/(z0**3)
    result*=z*z
    exponent=-1.*(z/z0)**1.5
    result*=np.exp(exponent)
    return result

def dndz_NVSSlike(z,z0=.32,alpha=0.36):
    exponent=-1.*alpha*z/z0
    result=(z/z0)**alpha
    result*=np.exp(exponent)
    return result

#===============================================================
# This will return a surveyType object with dndz like DES
#     6 redshift bins; 10 of width .2 going from 0-1., then 1 for z>1
#     no shot noise included 
#===============================================================
def get_DESlike_SurveyType(sigz,tag='',nbar=1.e9, zedges=None): #added zedges 170412
    if not tag:
        tag='des_sigz{0:0.3f}'.format(sigz)
    if zedges==None:
        zedges=np.array([.01,.2,.4,.6,.8,1.,5.])
    if nbar=='default':
        nbar=1.e9#From Dragan via email, early June'15
    else:
        nbar=nbar
    dndz=dndz_DESlike
    bias=nobias
    longtag='DES-like survey w bias=1'
    return mp.SurveyType(tag,zedges,sigz,nbar,dndz,bias,longtag,addnoise=False)

def get_Euclidlike_SurveyType(sigz=0.05,z0=0.7,nbar=3.5e8,onebin=False,tag='',zedges=np.array([]),b0=1.,b2=0,fracbadz=0.):
    if not tag:
        tag='euc_z0{0:0.2f}sz{1:0.3f}'.format(z0,sigz)
    bias=quadbias
    biasargs=[b0,b2]
    dndz=dndz_Euclidlike
    dndzargs=[z0]
    if nbar=='default':
        nbar=3.5e8
    else:
        nbar=nbar
    assert type(nbar)==float, nbar
    longtag='Euclid-like survey, z0={0:0.3f}, sigz={1:0.3f}, b0={2:0.3f}, b2={3:0.3f}'.format(z0,sigz,b0,b2)
    if not zedges.size:
        if onebin:
            zedges=np.array([.01,5*z0])
        else:
            zedges=np.array([.01,.4,.8,1.2,1.6,2.,5*z0])
#    print zedges
    return mp.SurveyType(tag,zedges,sigz,nbar,dndz,bias,dndzargs,biasargs,longtag,addnoise=False,fracbadz=fracbadz)
    
def get_Spherexlike_SurveyType(sigz=0.1,z0=0.46,nbar=6.6e7,onebin=False,tag='',zedges=np.array([]),b0=1.,b2=0,fracbadz=0.):
    if not tag:
        tag='spx_z0{0:0.2f}sz{1:0.3f}'.format(z0,sigz)
    bias=quadbias
    biasargs=[b0,b2]
    dndz=dndz_Spherexlike
    dndzargs=[z0]
#    print nbar
    if nbar=='default':
        if sigz==0.1:
            nbar=6.6e7
        elif sigz==0.03:
            nbar=2.4e7
        else: raise TypeError, 'Passed nbar=default but no default survey type with sigz={0}'.format(sigz)
    else:
        nbar=nbar
    assert type(nbar)==float, nbar
    longtag='SphereX-like survey, z0={0:0.3f}, sigz={1:0.3f}, b0={2:0.3f}, b2={3:0.3f}'.format(z0,sigz,b0,b2)
#    print 'spx zedges = ',zedges
    if not zedges.size:
        if onebin:
            zedges=np.array([.01,5*z0])
        else:
            finestN=6
            zedges=np.zeros(finestN+1)
            zedges[-1]=5.*z0 
            zedges[-2]=2./0.7*z0 #2.  #scale by z0
            dz=zedges[-2]/(finestN-1)
            for n in xrange(finestN-1):
                zedges[n]=dz*n
#            zedges=np.array([.01, .35,.7, 1.05, 1.4, 1.75, 5*z0])
    return mp.SurveyType(tag,zedges,sigz,nbar,dndz,bias,dndzargs,biasargs,longtag,addnoise=False,fracbadz=fracbadz)
    
    

#===============================================================
# This will return a mapType object for the full ISW effect
#===============================================================
def get_fullISW_MapType(zmax=15):
    tag='isw'
    zedges=[.01,zmax]
    sharpparam=.01/(zedges[-1]-zedges[0])
    return mp.MapType(tag,zedges,isISW=True,sharpness=sharpparam)

def get_testISW_MapType(zmax=.1):
    tag='iswTEST'
    zedges=[.01,zmax]
    sharpparam=.01/(zedges[-1]-zedges[0])
    return mp.MapType(tag,zedges,isISW=True,sharpness=sharpparam)

#===============================================================
# This will return a mapType object for the full ISW effect
#===============================================================
def get_generic_rundat(outdir='data/',zmax=10.1,tag='',ilktag='',noilktag=False):
    cosmfile='testparam.cosm'
    return ClRunData(tag=tag,rundir=outdir,cosmpfile=cosmfile,lmax=100,zmax=10.1)

def get_test_rundat(outdir='data/',zmax=1.,tag='',ilktag='',noilktag=False):
    kdat = KData(kmin=1.e-3,kmax=.1,nperlogk=1,krcut=1) 
    cosmfile='testparam.cosm'
    return ClRunData(tag=tag,ilktag=ilktag,rundir=outdir,kdata=kdat,cosmpfile=cosmfile,lmax=5,zmax=.5,noilktag=noilktag)
