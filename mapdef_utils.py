##################################################################
# This script is intended to investigate how binning strategy of
#   a survey with DES-like dndz affects is power for ISW reconstruction.
#
#  It contains functions which will be accessed both by scripts used
#  to generate data on the flux clusters and those which will be used locally
#  to make plots, etc. 
##################################################################
import numpy as np
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import sph_jn
##################################################################

def nobias(z):
    return 1

def dndz_DESlike(z):
    z0=.3
    n0=.054
    return n0*z*z*np.exp(-z/z0)

def dndz_Euclidlike(z,z0=0.7):
    #from eq 1 in arxiv:1506.02192
    result=1.5/(z0**3)
    result*=z*z
    exponent=-1.*(z/z0)**1.5
    result*=np.exp(exponent)
    return result

#===============================================================
# This will return a surveyType object with dndz like DES
#     6 redshift bins; 10 of width .2 going from 0-1., then 1 for z>1
#     no shot noise included 
#===============================================================
def get_DESlike_SurveyType(sigz,tag=''):
    if not tag:
        tag='des_sigz{0:0.3f}'.format(sigz)
    zedges=np.array([.01,.2,.4,.6,.8,1.,5.])
    nbar=1.e9 #From Dragan via email, early June'15
    dndz=dndz_DESlike
    bias=nobias
    longtag='DES-like survey w bias=1'
    return SurveyType(tag,zedges,sigz,nbar,dndz,bias,longtag,addnoise=False)

def get_Euclidlike_SurveyType(sigz=0.05,z0=0.7,nbar=3.5e8,onebin=False,tag=''):
    if not tag:
        tag='euc_z0{0:0.2f}sz{1:0.3f}'.format(z0,sigz)
    bias=nobias
    dndz=lambda z: dndz_Euclidlike(z,z0)
    nbar=3.5e8
    longtag='Euclid-like survey w bias=1, z0={0:0.2f}, sigz={1:0.3f}'.format(z0,sigz)
    if onebin:
        zedges=np.array([.01,7*z0])
    else:
        zedges=np.array([.01,.4,.8,1.2,1.6,2.,5.])
    return SurveyType(tag,zedges,sigz,nbar,dndz,bias,longtag,addnoise=False)
    
    

#===============================================================
# This will return a mapType object for the full ISW effect
#===============================================================
def get_fullISW_MapType(zmax=10):
    tag='isw'
    zedges=[.01,zmax]
    sharpparam=.01/(zedges[-1]-zedges[0])
    return MapType(tag,zedges,isISW=True,sharpness=sharpparam)

def get_testISW_MapType(zmax=.1):
    tag='iswTEST'
    zedges=[.01,zmax]
    sharpparam=.01/(zedges[-1]-zedges[0])
    return MapType(tag,zedges,isISW=True,sharpness=sharpparam)

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
