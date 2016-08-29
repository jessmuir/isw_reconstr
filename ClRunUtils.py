import numpy as np
import os,  shutil, copy_reg, types
from CosmParams import Cosmology
#Classes which will be useful for computing Cl's


###########################################################################
#KData - bundleds together info about kmin, kmax, lnperk, holds k array
#        used when tabulating Ilk
###########################################################################
class KData(object):
    def __init__(self,kmin=1.e-5,kmax=10,nperlogk=300,krcutadd=50,krcutmult=20,krcut=-1):
        self.kmin=kmin
        self.kmax=kmax
        self.nperlogk = int(nperlogk)
        self.krcutadd = krcutadd #additive cutoff point for Ilk computation
        self.krcutmult=krcutmult
        if krcut>0: #handles old format (before I split into krcutadd and krcutmult)
            self.krcutadd=krcut
        logkmin = np.log10(self.kmin)
        logkmax = np.log10(self.kmax)
        dlogk = 1./self.nperlogk
        Nk = int(logkmax-logkmin)*self.nperlogk +1
        self.karray = 10**(logkmin+dlogk*np.arange(Nk))
        
        self.infostr= 'kmin={0:0.3e}, kmax={1:0.3e}, kperlog={2:d}, krcut(add,mult)=({3:0.3g},{4:0.3g});'.format(self.kmin,self.kmax,self.nperlogk,self.krcutadd,self.krcutmult)
#        print "Initializing Kdata:"
        print self.infostr

###########################################################################
#RunData - bundleds together info used in computations
#   basically, holds info about cosmology and lvals
###########################################################################
class RunData(object):
    def __init__(self,tag='',rundir='output/',cosmpfile='testparam.cosm',lmax=0,lvals=np.array([]),clrundat=False,mapdat=False):
        self.tag=tag
        #output dirs
        self.rundir = rundir
        self.cambdir = rundir+'camb_output/'
        self.cldir = rundir+'Cl_output/'
        self.ilkdir = self.cldir+'Ilktab/'
        self.glmdir = rundir+'glm_output/'
        self.mapdir = rundir+'map_output/'
        self.plotdir= rundir+'plots/'
        
        #create them if they don't exist
        if not os.path.isdir(self.rundir):
            print "    creating dir",self.rundir
            os.mkdir(self.rundir)
        if not os.path.isdir(self.cambdir):
            print "    creating dir",self.cambdir
            os.mkdir(self.cambdir)
        if not os.path.isdir(self.cldir):
            print "    creating dir",self.cldir
            os.mkdir(self.cldir)
        if not os.path.isdir(self.ilkdir):
            print "    creating dir",self.ilkdir
            os.mkdir(self.ilkdir)
        if not os.path.isdir(self.glmdir):
            print "    creating dir",self.glmdir
            os.mkdir(self.glmdir)
        if not os.path.isdir(self.mapdir):
            print "    creating dir",self.mapdir
            os.mkdir(self.mapdir)
        if not os.path.isdir(self.plotdir):
            print "    creating dir",self.plotdir
            os.mkdir(self.plotdir)
            
        if lmax: #if lmax given, do all values below
            self.lmax=lmax
            self.lvals=np.arange(lmax+1)
        else: #if array given, do just those in list
            self.lvals=lvals
            self.lmax=max(lvals)

        #copy cosmpfile to rundir if not already there
        cosmfilebase = cosmpfile[cosmpfile.rfind('/')+1:]
        if not os.path.isfile(rundir+cosmfilebase):
            #if self.tag:
                #tag=self.tag+'_'
            self.cosmfile=''.join([rundir,cosmfilebase])
            shutil.copyfile(cosmpfile,self.cosmfile)
        else:
            self.cosmfile = cosmpfile
        #generate cosmology class, default just holds parameters, no tabulation
        if not clrundat:
            self.cosm = Cosmology(self.cosmfile)

    def equivRunData(self):
        #returns a MapRunData object with equivalent properties
        return RunData(self.tag,self.rundir,self.cosmfile,lvals=self.lvals)
    
###########################################################################
#ClRunData - bundleds together info used in computing C_l
#          this includes lvals, kdata, outdirs, cosmology
#          does not contain info about maps
# tag: used to label C_l data
# ilktag: used to label Ilk data files. If not passed, set equal to tag
#         unless noilktag=True; then ilktag is an empty string
# iswilktag: used to label isw Ilk data file, if not there, uses ilktag or tag
# rundir: where output will be put
# cosmpfile: file containing cosmological parameters
# kdata: KData object; if left as default zero, will use default init
# lmax, lvals: if lmax given, lvals set to array going from zero to lmax
#              if just lvals given, computes for just those
# zmax: how high in z should the cosmology bkgrd functions be tabulated?
#       however, we override this if we are asked to compute C_l for a
#       map with a higher zmax
# epsilon: sets the relative and absolute tolerance of integrals
# limberl: Use limber approx for ell>= this number. if -1, don't use Limber
# cosm_zrhgf_bkgd: if nonempty, take 2d of tabulated cosmology bkgd functs
#         use those instead of tabulating them ourselves
# pk_ext: if nonempty, is a tabulated matter power spectrum, use instead
#       of reading in from CAMB
# sharkpkcut: if True, instead of switching to bessel fns asymptotic approx
#       at large arg, sets them to zero
# besselxmincut: if True, set the low x part of j_l(x) to zero, for x
#       below where j_l(x) first becomes > epsilon
# noilktag: if True, and if ilktag not passed, doesn't set ilktag=tag,
#       just leaves it as an empty string
###########################################################################
class ClRunData(RunData):
    zintlim=10000
    kintlim=10000
    def __init__(self,tag='',ilktag='',iswilktag='',rundir='output/',cosmpfile='testparam.cosm',kdata=0,lmax=0,lvals=np.array([]),zmax=2.,limberl=20,epsilon=1.e-10,cosm_zrhgf_bkgrd=np.array([]),pk_ext=np.array([]),sharpkcut=False,besselxmincut=True,noilktag=False,nperz=200.):
        RunData.__init__(self,tag,rundir,cosmpfile,lmax,lvals,clrundat=True)
        self.limberl=limberl
        self.epsilon=epsilon #used to set tolerance on integrals
        self.tag=tag
        if ilktag or noilktag: #mostly just used for testing
            self.ilktag=ilktag
        else:
            self.ilktag=tag
        if iswilktag:
            self.iswilktag=iswilktag
        else:
            self.iswilktag=ilktag
        self.zmax=zmax #max z for H(z),D(z) etc tabulation

        # read in data for desired k and l values
        if kdata:
            self.kdata = kdata
        else:
            self.kdata = KData()#default values
        self.sharpkcut=sharpkcut
        self.besselxmincut=besselxmincut
        #print 'in ClRundata, zmax=',self.zmax
        #redo barebones cosm with one containing correct kdata, etc
        self.cosm = Cosmology(self.cosmfile,cambdir=self.cambdir,kmin=self.kdata.kmin,kmax=self.kdata.kmax,epsilon=self.epsilon,bkgd_zrhgf_ext=cosm_zrhgf_bkgrd,pk_ext=pk_ext,nperz=nperz)

        #infostr will hold run data to print in header of data files
        kinfo = self.kdata.infostr
        cosminfo = self.cosm.infostr
        if self.lvals.size==self.lmax+1: #sequential ells
            ellstr= 'lmax={0:d}'.format(self.lmax)
        else: #a few ells sampled for testing
            ellstr = 'lvals={0:s}'.format(self.lvals)
        if self.ilktag != self.tag:
            ilkstr=' (ilk:{0:s})'.format(self.ilktag)
        else:
            ilkstr=''
        if self.iswilktag != self.ilktag:
            iswilkstr=' (iswilk:{0:s})'.format(self.iswilktag)
        else:
            iswilkstr=''
                
        self.infostr='runtag {0:s}{1:s}{8:s}, {2:s}, eps={3:0.1e},besselmincut={4:b}, besselmaxcut={5:b}\n{6:s}\nk-data: {7:s}'.format(self.tag,ilkstr,ellstr,epsilon,besselxmincut,sharpkcut,cosminfo,kinfo,iswilkstr)
        
    def equivRunData(self):
        #returns a MapRunData object with equivalent properties
        return RunData(self.tag,self.rundir,self.cosmfile,lvals=self.lvals)
    
###########################################################################
#MapRunData - bundleds together info used in generating maps from Cl
###########################################################################
class MapRunData(RunData):
    def __init__(self,tag='',rundir='output/',cosmpfile='testparam.cosm',lmax=0,lvals=np.array([])):
        RunData.__init__(self,tag,rundir,cosmpfile,lmax,lvals,mapdat=True)
        
        pass

    #maybe put some info here about which maps have which masks, eg.
    # haven't actually done anything with this; the though was to have some way
    # to bundle info needed for maps but not Cl calculations.


###########################################################################
#helper functions for multiprocessing with class methods
###########################################################################
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
