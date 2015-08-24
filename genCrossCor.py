import numpy as np
from scipy.integrate import quad
from scipy.special import jv
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os, subprocess,copy,copy_reg,types
from multiprocessing import Pool, Manager
import itertools
import matplotlib.pyplot as plt

# This file contains functions used to compute I(k) functions for maps
# and the angular cross correlations between those maps

#will use the classes defined in these files:
from MapParams import *
from CosmParams import Cosmology
from ClRunUtils import *

###########################################################################
# ClData - contains C_l data and info about how it was produced
#          plus indices relevant for 
###########################################################################
class ClData(object):
    def __init__(self,rundata,bintags,dopairs=[],clgrid=np.array([]),addauto=True,docrossind=[],nbarlist=[]):
        if rundata.tag: runtag = '_'+rundata.tag
        else: runtag=''
        self.clfile= ''.join([rundata.cldir,'Cl',runtag,'.dat'])
        self.rundat = rundata #Clrundata instance
        self.bintaglist=bintags #tag, given mapind
        self.Nmap=len(bintags)
        self.tagdict={bintags[m]:m for m in xrange(self.Nmap)} #mapind, given tag
        self.Ncross=self.Nmap*(self.Nmap+1)/2
        crosspairs,crossinds=get_index_pairs(self.Nmap)
        self.crosspairs=crosspairs #[crossind,mapinds] (NCross x2)
        self.crossinds=crossinds #[mapind,mapind] (Nmap x Nmap)

        if len(docrossind): #if list of cross pair indices given, use those
            self.docross = docrossind
            self.pairs=get_pairs_fromcrossind(self.bintaglist,docrossind,self.crosspairs,self.crossinds)
        else: #otherwise uses pairs. if both empty, just does auto correlations
            self.pairs=consolidate_dotags(dopairs,bintags)
            docross=get_docross_ind(self.tagdict,self.pairs,self.crossinds,addauto=addauto)
            self.docross=docross #crossinds which have C_l computed

        self.Nell=rundata.lvals.size
        self.cl=clgrid #[crossind, ell]

        #when adding shot noise and/or applying calibration errors
        # need to know the average number density per steradian per map
        nbarlist=np.array(nbarlist)
        if nbarlist.size==self.Nmap:
            self.nbar =nbarlist#same size as bintags, contains nbar for galaxy maps, -1 for otherrs
        else: #minus one means no nbar given for map at that index
            self.nbar=-1*np.ones(self.Nmap)
        
        #keep noise contrib to C_l in separate array #NEED TO TEST
        self.noisecl = np.zeros((self.Ncross,self.Nell))
        for i in xrange(self.Nmap):
            if self.nbar[i]!=-1: #assumes -1 for no noise or isw
                diagind=self.crossinds[i,i]
                self.noisecl[diagind,:]=1/self.nbar[i]


    def hasClvals(self):
        return bool(self.cl.size)

    def clcomputed_forpair(self,tag1,tag2):
        mapind1=self.tagdict[tag1]
        mapind2=self.tagdict[tag1]
        xind=self.crossinds[mapind1,mapind2]
        return xind in self.docross

    #given binmap tag, remove that map
    def deletemap(self,tag):
        if tag not in self.bintaglist:
            return False
        newNmap=self.Nmap-1
        newNcross=newNmap*(newNmap+1)/2
        oldmapind=self.tagdict[tag]
        newcl=np.zeros((newNcross,self.Nell))
        delxinds=self.crossinds[oldmapind,:]
        #newdocross=np.setdiff1d(self.docross,delxinds)#unique elements of docross not in delxinds


        newi=0        
        for j in xrange(self.Ncross):
            if not j in delxinds: #copy over values we're keeping
                newcl[newi,:]=self.cl[j,:]
                newi+=1
        #set up new values
        self.Nmap=newNmap
        self.bintaglist.remove(tag)
        self.tagdict={self.bintaglist[m]:m for m in xrange(self.Nmap)}
        self.Ncross=newNcross
        crosspairs,crossinds=get_index_pairs(self.Nmap)
        self.crosspairs=crosspairs #[crossind,mapinds] (NCross x2)
        self.crossinds=crossinds #[mapind,mapind] (Nmap x Nmap)
        #THIS IS A TEMPORARY HACK
        self.pairs=consolidate_dotags(['all'],self.bintaglist)
        #self.docross=['all'] 
        #self.pairs=get_pairs_fromcrossind(self.bintaglist,newdocross,self.crosspairs,self.crossinds)
        self.cl=newcl
        self.nbar=np.delete(self.nbar,oldmapind)
        #just set up noisecl again
        self.noisecl = np.zeros((self.Ncross,self.Nell)) 
        for i in xrange(self.Nmap):
            if self.nbar[i]!=-1: #assumes -1 for no noise or isw
                self.noisecl[i,i]=1/self.nbar[i]
        return True


###########################################################################
def sphericalBesselj(n,x):
    return jv(n + 0.5, x) * np.sqrt(np.pi/(2*x))

def findxmin(n,tol=1.e-10):
    #use this to find the min xvalue where we'll call the bessel fn nonzero
    #basically, smallest x where j_l(x)=tol
    return brentq(lambda x:sphericalBesselj(n,x)-tol,tol,n)

###########################################################################
# functions for computing, tabulating,and using I_l(k) functions
###########################################################################
#=========================================================================
#Functions for computing Cl with Limber approx
def LimberCl_intwrapper(argtuple):
    nl,indocross,mappair,cosm,zintlim,epsilon=argtuple
    if not indocross: #don't do anything if we don't want this pair
        return 0.
    n,lval=nl
    if lval==0:
        return 0
    binmap1,binmap2=mappair

    #get cosmological functions
    co_r = cosm.co_r #function with arg z
    z_from_cor = cosm.z_from_cor #function with arg r
    hubble = cosm.hub #functionw ith arg z 
    D = cosm.growth #function with arg z
    f = cosm.growthrate #function with arg z
    c = cosm.c
    
    #limber approx writes k in terms of ell, z; set P(k) up for this
    # clip off first entry to avoid dividing by zero
    kofz_tab=(lval+.5)/cosm.r_array[1:] #the k value corresponding to each z value;
    Pofz_tab=cosm.P(kofz_tab) #tabulated, the P(k) corresponding to k=ell/r(z) for each z value; 
    kofz=interp1d(cosm.z_array[1:],kofz_tab,bounds_error=False,fill_value=0.)
    Pofz=interp1d(cosm.z_array[1:],Pofz_tab,bounds_error=False,fill_value=0.)

    #use info in binmaps to figure out zmin and zmax
    zmin=max(binmap1.zmin,binmap2.zmin)
    zmin=max(0.01,zmin)
    zmax=min(binmap1.zmax,binmap2.zmax)
    if zmax<=zmin:
        return 0.

    #set up the ISW prefactor as a function of z
    Nisw=binmap1.isISW + binmap2.isISW
    #print binmap1.tag,binmap2.tag,Nisw
    if Nisw:
        prefactor= (100.)**2 #H0^2 in units h^2km^2/Mpc^2/s^2 
        prefactor*= 3./cosm.c**2 #h^2/Mpc^2 
        iswpref =lambda z: prefactor*(1.-f(z))/(kofz(z)**2) #unitless function
        if Nisw==1:
            iswprefactor= iswpref
        elif Nisw==2:
            iswprefactor=lambda z: iswpref(z)*iswpref(z)
    else:
        iswprefactor=lambda z:1.

    result=quad(lambda z: LimberCl_integrand(z,hubble,D,co_r,Pofz,iswprefactor,binmap1.window,binmap2.window,c),zmin,zmax,full_output=1,limit=zintlim,epsabs=epsilon,epsrel=epsilon)[0]
    return result

def LimberCl_integrand(z,hubble,growth,cor,Pz_interpfn,iswprefactor,window1,window2,c=299792):
    result=window1(z)*window2(z)    
    #print 'windows;',result
    #print 'pzinterp',Pz_interpfn(z)
    #print 'hubble*growth*r^2',hubble(z)*(growth(z)**2)/(cor(z)**2)
    if result==0 or z==0:
        return 0
    result*=Pz_interpfn(z)*hubble(z)*(growth(z)**2)/(cor(z)**2)/c
    result*=iswprefactor(z)
    return result


#=============================================================
# functions handling Ilk for an individual bin map
#=========================================================================
# getIlk: reads in Ilk file if there, otherwise computes
def getIlk_for_binmap(binmap,rundata,redo=False,DoNotOverwrite=False):
    needIlk=True
    if not redo:
        #check if file w appropriate name exists
        if rundata.ilktag: runtag = '.'+rundata.ilktag
        else: runtag=''
        f = ''.join([rundata.ilkdir,binmap.tag,'_Ilk',runtag,'.dat'])
        if os.path.isfile(f):
            #read it in, check that ell and k vals are good
            Ilk,k_forI=readIlk_file(binmap,rundata)
            if Ilk.size:
                needIlk=False
    if needIlk and (not DoNotOverwrite):
        Ilk=computeIlk(binmap,rundata)
        k_forI=rundata.kdata.karray
    elif DoNotOverwrite:
        print "***in getIlk: DoNotOverwrite=True, but need Ilk values"
    return Ilk,k_forI

#-------------------------------------------------------------------------
def computeIlk(binmap,rundata):
    DOPARALLEL=1
    print "Computing Ilk for ",binmap.tag,'DOPARALLEL=',DOPARALLEL
    #set up arrays
    kvals = rundata.kdata.karray
    Nk = kvals.size
    # just do the ell with no limber approx
    if rundata.limberl>=0 and rundata.limberl<=rundata.lmax:
        lvals = rundata.lvals[:np.where(rundata.lvals<rundata.limberl)[0][-1]+1]#rundata.lvals
    else:
        lvals=rundata.lvals
    Nell = lvals.size
    Ivals = np.zeros((Nell,Nk))
    eps = rundata.epsilon
    zintlim = rundata.zintlim

    #set up labels to help references go faster
    cosm = rundata.cosm
    if not cosm.tabZ or cosm.zmax<binmap.zmax:
        cosm.tabulateZdep(max(rundata.zmax,binmap.zmax),nperz=cosm.nperz)
    co_r = cosm.co_r #function with arg z
    
    krcutadd=rundata.kdata.krcutadd #to make integral well behaved w fast osc
    krcutmult=rundata.kdata.krcutmult
    #bounds for integral in comoving radius
    rmin=co_r(binmap.zmin)
    rmax=co_r(binmap.zmax)
    
    lk= itertools.product(lvals,kvals) #items=[l,k]
    argiter=itertools.izip(lk,itertools.repeat(rmin),itertools.repeat(rmax),itertools.repeat(cosm),itertools.repeat(binmap),itertools.repeat(krcutadd),itertools.repeat(krcutmult),itertools.repeat(zintlim),itertools.repeat(eps),itertools.repeat(rundata.sharpkcut),itertools.repeat(rundata.besselxmincut))
    if DOPARALLEL:
        pool = Pool()
        results=pool.map_async(Iintwrapper,argiter)
        newI=np.array(results.get())
        pool.close()
        pool.join()

        #rearrange into [l,k] shape
        Ivals=newI.reshape(Nell,Nk)

    else:
        argiter=list(argiter)
        for i in xrange(len(argiter)):
            argtuple=argiter[i]
            lk,rmin,rmax,cosm,binmap,krcutadd,krcutmult,zintlim,epsilon,zeropostcut,besselxmincut= argtuple
            l,kval=lk
            lind=np.where(lvals==l)[0][0]
            kind=np.where(kvals==kval)[0][0]
            Ival=Iintwrapper(argtuple)
            Ivals[lind,kind]=Ival
 
    #save result to file
    writeIlk(Ivals,binmap,rundata)
    return Ivals
#--------------------------------------------------
#wrapper function for integral, so multithreading works
def Iintwrapper(argtuple):#(l,kval,rmin,rmax,cosm,binmap,zintlim=10000):
    #print "in Iintwrapper"
    lk,rmin,rmax,cosm,binmap,krcutadd,krcutmult,zintlim,epsilon,zeropostcut,besselxmincut = argtuple
    l,kval=lk
    dr=rmax-rmin
    if l==0: return 0. #don't compute monopole
    
    #bessel function will be effectively zero below some argument; adjust rmin accordingly
    if besselxmincut:
        xmin=findxmin(l,epsilon) #ADDED 5/19; seems to speed things up without chaning Ilk results much
        rmin=max(rmin,xmin/kval) #ADDED 5/19
        if rmin>=rmax:
            return 0.
    #print '   reading binmap info'
    window =binmap.window #function with args i,z
    isISW=binmap.isISW
    #print '   readin cosm info'
    co_r = cosm.co_r #function with arg z
    z_from_cor = cosm.z_from_cor #function with arg r
    hubble = cosm.hub #functionw ith arg z 
    D = cosm.growth #function with arg z
    f = cosm.growthrate #function with arg z
    c = cosm.c
    #print '   computing prefactor'
    #get appropriate prefactors
    prefactor=1.
    if binmap.isISW:
        H02 = (100.)**2 #h^2km^2/Mpc^2/s^2 
        prefactor= 3.*H02/cosm.c**2 #h^2/Mpc^2 
        prefactor=prefactor/(kval**2) #unitless
    #print '   looking at pre/post cut division'
    #find r where we want to switch from full bessel to approx
    ALLPRECUT=False
    ALLPOSTCUT=False
    if krcutmult<0 or krcutadd<0: #set these to negative to turn off approx
        ALLPRECUT=True
        r_atkrcut=rmax
    elif kval*dr>2*np.pi*10.: #only use approx if many oscillations fit inside bin
        r_atkrcut=(l*krcutmult+krcutadd)/kval
        if r_atkrcut<rmin:
            r_atkrcut=rmin
            ALLPOSTCUT=True
        if r_atkrcut>rmax:
            r_atkrcut=rmax
            ALLPRECUT=True
    else:
        r_atkrcut=rmax
        ALLPRECUT=True

    #print '   doing integrals'
    #print 'krcutmult=',krcutmult,'krcutadd',krcutadd
    #print "r-atkrcut=",r_atkrcut,'ALLPRECUT=',ALLPRECUT,"ALLPOSTCUT=",ALLPOSTCUT
    #calculate!
    if ALLPOSTCUT:
        result_precut=0.
    else:
        result_precut=quad(lambda r: Iintegrand(r,l,kval,window,z_from_cor,hubble,D,f,isISW,c,prefactor),rmin,r_atkrcut,full_output=1,limit=zintlim,epsabs=epsilon,epsrel=epsilon)[0]

    if zeropostcut or ALLPRECUT: 
        result_postcut= 0
    elif l%2==0: #after krcut, use quad's ability to weight with sin or cos
        #even l bessels act line sin/x
        result_postcut=quad(lambda r: Iintegrand_postcut(r,l,kval,window,z_from_cor,hubble,D,f,isISW,c,prefactor),r_atkrcut,rmax,full_output=1,limit=zintlim,epsabs=epsilon,epsrel=epsilon,weight='sin',wvar=kval)[0]
    else: #odd bessels act like cos/x
        result_postcut=quad(lambda r: Iintegrand_postcut(r,l,kval,window,z_from_cor,hubble,D,f,isISW,c,prefactor),r_atkrcut,rmax,full_output=1,limit=zintlim,epsabs=epsilon,epsrel=epsilon,weight='cos',wvar=kval)[0]

    return result_precut+result_postcut
#--------------------------------------------------                   
# function which is integrated over to get Ilk
def Iintegrand(r,l,k,window,z_from_cor,hubble,growth,growthrate,isISW=False,c=299792,prefactor=1.):
    z = z_from_cor(r)
    w= window(z)
    if w==0:
        return 0
    else:
        dI = w*growth(z)*hubble(z)/c
        if isISW: #ISW gets f-1 piece
            dI*= (1.-growthrate(z))
            if dI==0: return 0
        bessel = sphericalBesselj(l,k*r)
        dI*=bessel
        return dI*prefactor
    
# function which is integrated over to get Ilk after k past krcut
def Iintegrand_postcut(r,l,k,window,z_from_cor,hubble,growth,growthrate,isISW=False,c=299792,prefactor=1.):
    z = z_from_cor(r)
    w= window(z) 
    if w==0:
        return 0
    dI = w*growth(z)*hubble(z)/c
    if isISW: #ISW gets f-1 piece
        dI*= (1.-growthrate(z))
        if dI==0: return 0
    if l%2==0: #even l, sin weighting; sin(x) handled by quad
        bessel = np.sin(np.pi*(l+1.)/2.)/(k*r)
    else: #odd l, cos weighting
        bessel = np.cos(np.pi*(l+1.)/2.)/(k*r)
    dI*=bessel
    return dI*prefactor
#-------------------------------------------------------------------------
def writeIlk(Ilkarray,binmap,rundata):
    if rundata.ilktag: runtag = '.'+rundata.ilktag
    else: runtag=''
    outfile = ''.join([rundata.ilkdir,binmap.tag,'_Ilk',runtag,'.dat'])

    print 'Writing Ilk data to ',outfile
    k = rundata.kdata.karray
    lvals = rundata.lvals
    Nell = sum(l<rundata.limberl for l in lvals) #number below limber switch
    krcutstr='{0:13g}.{1:<10g}'.format(rundata.kdata.krcutadd,rundata.kdata.krcutmult)
    if rundata.kdata.krcutadd<0 or rundata.kdata.krcutmult<0:
        krcutstr='{0:23g}'.format(-1.)
    
    headerstr = '\n'.join([binmap.infostr,rundata.infostr])
    collabels =''.join([' {0:23s} {1:23s}\n{2:s}'.format('k[h/Mpc] (top=krcutadd.mult)','ell=>',krcutstr),''.join([' {0:23d}'.format(lvals[n]) for n in xrange(Nell)]),'\n'])
    bodystr=''.join([\
                         ''.join([' {0:+23.16e}'.format(k[row]),''.join([' {0:+23.16e}'.format(Ilkarray[lind,row]) for lind in xrange(Nell)]),'\n'])\
                         for row in xrange(k.size)])
    f=open(outfile,'w')
    f.write(headerstr) #5 lines long: bin,map,run,kdata,cosm
    f.write('\n##############################\n') #6th dummy line
    f.write(collabels) #line 7 has row, col labels, line 8 has lvals
    f.write(bodystr)
    f.close()
    
#-------------------------------------------------------------------------
# read in file containing Ilk for given map bin, resturn Ilk array,lvals, kvals
def readIlk_file(binmap,rundata):
    if rundata.ilktag: runtag = '.'+rundata.ilktag
    else: runtag=''
    infile=''.join([rundata.ilkdir,binmap.tag,'_Ilk',runtag,'.dat'])
    print "Reading Ilk from file",infile
    x = np.loadtxt(infile,skiprows=6)
    inkrcut=x[0,0]
    inkrcutadd=int(inkrcut)
    inkrcutmult=int(str(inkrcut)[str(inkrcut).find('.')+1:])
    k=x[1:,0]
    l=x[0,1:].astype(int)
    I=np.transpose(x[1:,1:])
    #read header to get nperlogk info
    f=open(infile,'r')
    f.readline()#binmap infoline
    f.readline()#runtag and lvals
    f.readline()#cosmolog info
    kstr = f.readline() #kdata info
    f.close()
    kstr=kstr[kstr.find('kperlog=')+len('kperlog='):]#cut just before nperlogk
    innperlogk=int(kstr[:kstr.find(',')])
    inkmin=k[0]
    inkmax=k[-1]

    #return ivals if nperlogk and l values match up, otherwise return empty array
    #should have all ell in lvals where ell<limberl, assume ascending order
    limberl=rundata.limberl
    if limberl>=0 and limberl<=rundata.lmax:
        # these are the expected ell values we want out
        checkell=rundata.lvals[:np.where(rundata.lvals<limberl)[0][-1]+1]
    else:
        checkell=rundata.lvals
    if l.size>=checkell.size:
        lind_incheck=[] #index of each checkell element in l
        for lval in checkell:
            where = np.where(l==lval)[0]
            if where.size==1:
                lind_incheck.append(where[0])
            else:
                print " *** unexpected lvals, recompute."
                return np.array([]),np.array([])
        lind_incheck=np.array(lind_incheck)
        if innperlogk>= rundata.kdata.nperlogk and inkmin<=rundata.kdata.kmin and inkmax>=rundata.kdata.kmax:
            return I[lind_incheck,:],k #k_forI can be different than kdata, as long as it samples enough
        else:
            print " *** unexpected kvals, recompute."
            return np.array([]),np.array([])

    else:
        print " *** unexpected number of lvals, recompute."
        return np.array([]),np.array([])

###########################################################################
# functions for computing, tabulating,and using cross corr functions
###########################################################################

#-------------------------------------------------------------------------
# getCl - returns desired cross corr for given list of binmaps
#     Checks for existing Cl file,  checks that all maps wanted are in it
#     Computes necessary cross corr, saves
#     dopairs = list [(maptag1,maptag2)...] for pairs we want Cl for
#               if empty: just get the autocorrelation Cl
#               if contains the string 'all', compute all
#     redoAllCl -> compute requested values, overwrite any existing Cl file
#     redoTheseCl -> don't overwrite old file, but recompute all requested Cl
#                    vals and overwrite existing data for those pairs
#     redoAutoCl -> Like redoTheseCl, but also includes autocorrelations
#     redoIlk - recompute + overwrite existing tabulated Ilk data
#     DoNotOverwrite - "read only" safeguard 
#-------------------------------------------------------------------------
def getCl(binmaplist,rundata,dopairs=[],redoAllCl=False,redoTheseCl=False,redoAutoCl=False,redoIlk=False,DoNotOverwrite=True):
    print "in getCL, DoNotOverwrite=",DoNotOverwrite
    if redoIlk:
        #if we're recomputing Ilk, we need to recompute all the Cl
        redoAllCl=True
    print "Getting C_l for auto-corr and requested pairs:",dopairs
    if 'all' in dopairs:
        dopairs=[p for p in itertools.combinations_with_replacement([m.tag for m in binmaplist],2)]
    
    #oldcl,oldtags,olddo= readCl_file(rundata)
    oldcl=readCl_file(rundata)
    #print 'oldcl.cl.shape',oldcl.cl.shape
    #print 'olcl.hasClvals',oldcl.hasClvals()
    #print "    pairs computed previously:",olddo
    #if redoAllCl or not oldcl.size:
    if redoAllCl or not oldcl.hasClvals():
        if DoNotOverwrite:
            print "***In getCl: DoNotOverwrite=True but need C_l values."
        else:
            print "Computing new C_l for all requested cross corr, overwriting existing data."
            #compute and write new C_l file if one of the redo bool=True
            #  or if clfile doesn't exist, or if lvals are wrong in clfile
            #Clvals=computeCl(binmaplist,rundata,dopairs=dopairs,redoIlk=redoIlk,addauto=True)
            cldat=computeCl(binmaplist,rundata,dopairs=dopairs,redoIlk=redoIlk,addauto=True)
            writeCl_file(cldat)
   
    else: #can potentially use previously computed values
        ANYNEW=False
        if redoAutoCl:
            autoinnew=True
            print "    Will recompute auto-corr for requested maps"
        else:
            print "    Using previously computed auto-corr."
            autoinnew=False
        #indices etc requested in arguments
        taglist=get_bintaglist(binmaplist)
        nbarlist=[m.nbar for m in binmaplist]
        cldat=ClData(rundata,taglist,dopairs=dopairs,addauto=True,nbarlist=nbarlist)
        Nmap=cldat.Nmap
        dopairs=cldat.pairs
        tagdict = cldat.tagdict
        crosspairs=cldat.crosspairs
        crossinds=cldat.crossinds
        Ncross = cldat.Ncross
        docross=cldat.docross
        
        #old = follow indices for maplist in prev existing file
        oldind=-1*np.ones(Nmap) #for each binmap, its index in oldbinmaplist
        
        #get set up to navigate existing (old) Cl data
        oldxinds=oldcl.crossinds
        olddox=oldcl.docross #index of (in oldtag basis) cross corrs to do

        #get indices of tags existing in oldbintags
        oldind=translate_tag_inds(cldat,oldcl)
        for t in xrange(Nmap): #add autocorr for any maps not in oldtags
            if oldind[t]<0 and not redoAutoCl:
                docross.append(crossinds[t,t])

        newdocross = docross[:]
        crossfromold = []#crossinds of x corrs previously computed
        if not (redoTheseCl or redoAutoCl):
            print "  Checking for previously computed C_l values."
            #Remove from newdocross for pairs already computed
            for t in xrange(Nmap):
                if oldind[t]>=0: #tag in oldtags
                    #check which desired pairs are already computed
                    for t2 in xrange(t,Nmap): #loop through possible pairs
                        #if pair not in dopairs, don't compute
                        if crossinds[t,t2] not in newdocross:
                            continue
                        #otherwise, check if second tag in oldtags
                        #if pair in olddo, already computed; don't need it
                        elif (oldind[t2]>=0) and (oldxinds[oldind[t],oldind[t2]] in olddox):
                            newdocross.remove(crossinds[t,t2])
                            crossfromold.append(crossinds[t,t2])
        else:
            print "  Will compute C_l for all requested pairs."
            ANYNEW=True

        #need new values if entries in newdocross, otherwise returns zero array
        if not DoNotOverwrite:
            newcl= computeCl(binmaplist,rundata,docrossind=newdocross,redoIlk=redoIlk)
        else:
            #if we're not saving data, don't bother computing
            # just get dummy ClData object
            if newdocross:
                print "***WARNING. Need new Cl data have set READONLY."
            newcl= computeCl(binmaplist,rundata,docrossind=np.array([]),redoIlk=False)
        
        if np.any(newcl.cl!=0):
            ANYNEW=True
        #Clvals = Clgrid to return, all asked for in this call
        Clvals = np.copy(newcl.cl)
        for n in crossfromold: #get the prev computed values from oldcl
            i0 = crosspairs[n,0]
            i1 = crosspairs[n,1]
            oldn = oldxinds[oldind[i0],oldind[i1]]
            Clvals[n,:] = oldcl.cl[oldn,:]
        #put Clvals data into the relevant ClData instance
        cldat.cl=Clvals
            
        #combine new and old Cl to write everything to file
        if ANYNEW:
            if not DoNotOverwrite:
                print "  Combining new and old C_l for output file."
                overwriteold=(redoTheseCl or redoAutoCl)
                comboCl=combine_old_and_new_Cl(cldat,oldcl)
                writeCl_file(comboCl)
            else:
                print "***In getCl: DoNotOverwrite=True, but computed some new values. Not saving new vals."
    return cldat

#------------------------------------------------------------------------
# Given list of binmaps, computes Cl for each pair, returns Ncross x Nell array
#     dopairs = list [(maptag1,maptag2)...] for pairs we  want Cl for
#     if redoIlk, recomputes even if files exist
#     if addauto and no crossinds given,
#           compute autocorrelations even if not in dopairs
def computeCl(binmaps,rundata,dopairs=[],docrossind=[],redoIlk=False,addauto=False):
    bintags=[m.tag for m in binmaps]
    nbars=[m.nbar for m in binmaps] #will be -1 for e.g. ISW

    cldat=ClData(rundata,bintags,dopairs=dopairs,docrossind=docrossind,addauto=addauto,nbarlist=nbars)
    
    #get list of pairs of indices for all unique cross corrs
    Nmap=cldat.Nmap #len(binmaps)
    Nell = cldat.Nell #rundata.lvals.size
    crosspairs=cldat.crosspairs
    crossinds=cldat.crossinds
    Ncross=cldat.Ncross
    tagdict=cldat.tagdict
    docross=cldat.docross
    
    #print 'in computeCl, dopairs',dopairs
    #if we're not computing anything, just return array ofzeros
    Clvals = np.zeros((Ncross,Nell))

    if not len(docross):
        print "    No new values needed."
        cldat.cl=Clvals
        return cldat
    
    print "  Computing new C_l values."    

    # First sort out when to switch to limber approx
    limberl=rundata.limberl #where to switch to Limber
    print "limberl=",limberl
    if limberl>0 and limberl<=rundata.lmax:
        lvals_preLim=rundata.lvals[:np.where(rundata.lvals<limberl)[0][-1]+1]
        Nell_preLim=lvals_preLim.size
        lvals_postLim=rundata.lvals[np.where(rundata.lvals<limberl)[0][-1]+1:]
        Nell_postLim=Nell-Nell_preLim
    elif limberl==0:
        lvals_preLim=np.array([])
        Nell_preLim=0
        lvals_postLim=rundata.lvals
        Nell_postLim=Nell
    else:
        lvals_preLim=rundata.lvals
        lvals_postLim=np.array([])
        Nell_preLim=Nell
        Nell_postLim=0

    #print 'preLim lvals:',lvals_preLim
    #print 'Nell_preLim',Nell_preLim
    #print 'postLim lvals:',lvals_postLim
    #print 'Nell_postLim',Nell_postLim  

    #get k and power spectrum info for run, need this limber or not
    kdata=rundata.kdata
    cosm = rundata.cosm
    if not cosm.havePk:
        # For Pk, just use camb's default adaptive nperlogk spacing 
        print 'getting CAMB P(k), kmin,kmax=',kdata.kmin,kdata.kmax
        cosm.getPk(kdata.kmin,kdata.kmax)#kperln=kdata.nperlogk*np.log(10))

    if Nell_preLim:
        #get Ilk functions
        print "  Getting Ilk transfer functions.."
        Igrid=[]#map,ell,k; ell indices only for ell<limberl 
        kforIgrid=[]#map,k
        #np.zeros((Nmap,Nell_preLim,rundata.kdata.karray.size))
        for m in xrange(Nmap):
            Igridbit,k_forI=getIlk_for_binmap(binmaps[m],rundata,redoIlk)
            Igrid.append(Igridbit)
            kforIgrid.append(k_forI)
        Igrid=np.array(Igrid)
        kforIgrid = np.array(kforIgrid)
        lnkforIgrid = np.log(kforIgrid)

        #set up P(k) in terms of lnk
        Plnk = interp1d(np.log(cosm.k_forPower),cosm.P_forPower,bounds_error=False,fill_value=0.)
        lnkmin=np.log(kdata.kmin)
        lnkmax=np.log(kdata.kmax)
  
        #Do Cl computations, interating through crosspairs and lvals
        print "  Performing non-Limber C_l integrals."
        nl= itertools.product(xrange(Ncross),xrange(Nell_preLim)) #items=[n,lind]
        Ipair_fornl=[(Igrid[crosspairs[xind,0],lind,:],Igrid[crosspairs[xind,1],lind,:]) for (xind,lind) in itertools.product(xrange(Ncross),xrange(Nell_preLim))]
        lnkforIpair=[(lnkforIgrid[crosspairs[xind,0],:],lnkforIgrid[crosspairs[xind,1],:]) for (xind,lind) in itertools.product(xrange(Ncross),xrange(Nell_preLim))]
        indocross=[xind in docross for (xind,lind) in itertools.product(xrange(Ncross),xrange(Nell_preLim))]
    
        #put everything into a tuple for the integral wrapper
        argiter = itertools.izip(nl,indocross,itertools.repeat(lnkmin),itertools.repeat(lnkmax),itertools.repeat(Plnk),Ipair_fornl,lnkforIpair,itertools.repeat(rundata.kintlim),itertools.repeat(rundata.epsilon)) #for quad
    
        pool = Pool()
        results=pool.map_async(Clintwrapper,argiter)
        newCl=np.array(results.get())
        pool.close()
        pool.join()

        #rearrange into [n,l] shape
        Clvals[:,:Nell_preLim]=newCl.reshape(Ncross,Nell_preLim)

    # Do Limber approx calculations 
    if Nell_postLim:
        print "  Performing Limber approx C_l integrals."
        #make sure z-dep functions have been tabulated
        #print [m.zmax for m in binmaps]
        zmax=max([m.zmax for m in binmaps])
        if not cosm.tabZ or cosm.zmax<zmax:
            cosm.tabulateZdep(zmax,nperz=cosm.nperz)

        nl= itertools.product(xrange(Ncross),lvals_postLim) #items=[n,lvals]
        mappair=[(binmaps[crosspairs[xind,0]],binmaps[crosspairs[xind,1]]) for (xind,lind) in itertools.product(xrange(Ncross),xrange(Nell_postLim))]
        
        indocross=[xind in docross for (xind,lind) in itertools.product(xrange(Ncross),xrange(Nell_postLim))]
        #put everything into a tuple for the integral wrapper
        argiter = itertools.izip(nl,indocross,mappair,itertools.repeat(cosm),itertools.repeat(rundata.zintlim),itertools.repeat(rundata.epsilon)) #for quad
        #run computations in parallel
        DOPARALLEL=1
        if DOPARALLEL:
            print "  Running Limber approx integrals in parallel."
            pool=Pool()
            results=pool.map_async(LimberCl_intwrapper,argiter)
            limberCl=np.array(results.get())
            pool.close()
            pool.join()
            Clvals[:,Nell_preLim:]=limberCl.reshape(Ncross,Nell_postLim)
        else: #the nonparallel version is for testing that things run
            argiter=list(argiter)
            print "  Running Limber approx integrals (not in parallel)."
            for i in xrange(len(argiter)):
                argtuple=argiter[i]
                nl,indocross,mappair,cosm,zintlim,epsilon=argtuple
                n,lval=nl
                lind=np.where(rundata.lvals==lval)[0][0]
                thiscl=LimberCl_intwrapper(argtuple)
                print 'n,lval',n,lval,thiscl*lval*(1+lval)/(2*np.pi)
                Clvals[n,lind]=thiscl

    cldat.cl=Clvals
                    
    return cldat#Clvals

#------------------------------------------------------------------------
def Clintwrapper(argtuple):
    #nl,bool dothiscross,lnkmin,lnkmax,Pk_array,Igrid,kintlim =argtuple
    nl,dothiscross,lnkmin,lnkmax,Plnkfunc,Ipair_fornl,lnkforIpair,kintlim,epsilon=argtuple
    n,lind=nl
    if not dothiscross: clval= 0
    else:
        ik1=Ipair_fornl[0]
        lnkfori1=lnkforIpair[0]
        ik2=Ipair_fornl[1]
        lnkfori2=lnkforIpair[1]

        #COMMENTED OUT ON 6/1/15; 
        # #find nonzero overlap of the Ilk functions #ADDED 5/19
        #  if less than tolerance, should treat Ilk as zero to avoid noise contrib
        checktol=epsilon
        ISNONZERO=True
        if np.any(ik1>checktol):
            i1minind=np.where(ik1>checktol)[0][0]
            i1maxind=np.where(ik1>checktol)[0][-1]
            ISNONZERO= i1minind!=i1maxind
        else:
            ISNONZERO=False
        if np.any(ik2>checktol):
            i2minind=np.where(ik2>checktol)[0][0]
            i2maxind=np.where(ik2>checktol)[0][-1]
            ISNONZERO= (i2minind!=i2maxind) and ISNONZERO
        else:
            ISNONZERO=False
        if not ISNONZERO:
            return 0.
        i1_minlnk=lnkfori1[i1minind]
        i1_maxlnk=lnkfori1[i1maxind]
        i2_minlnk=lnkfori2[i2minind]
        i2_maxlnk=lnkfori2[i2maxind]
        highermin=max(i1_minlnk,i2_minlnk)
        lowermax=min(i1_maxlnk,i2_maxlnk)
        if highermin>=lowermax: #no overlap
            return 0.
        else:
            lnkmin=max(highermin,lnkmin)
            lnkmax=min(lowermax,lnkmax)
        
        #P_interp = interp1d(lnk_array,Pk_array,kind='cubic')
        if i1maxind-i1minind>3: #need at least 4 pts for cubic interp
            I1_interp= interp1d(lnkfori1[i1minind:i1maxind+1],ik1[i1minind:i1maxind+1],kind='cubic',bounds_error=False,fill_value=0.)
        else: #just do linear interp (the highermin/lowermax stuff above sets things to 1 if they're equal)
            I1_interp= interp1d(lnkfori1[i1minind:i1maxind+1],ik1[i1minind:i1maxind+1],kind='linear',bounds_error=False,fill_value=0.)
        if i2maxind-i2minind>3: #need at least 4 pts for cubic interp
            I2_interp= interp1d(lnkfori2[i2minind:i2maxind+1],ik2[i2minind:i2maxind+1],kind='cubic',bounds_error=False,fill_value=0.)
        else:
            I2_interp= interp1d(lnkfori2[i2minind:i2maxind+1],ik2[i2minind:i2maxind+1],kind='linear',bounds_error=False,fill_value=0.)
        
        #I1_interp= interp1d(lnkfori1,ik1,kind='cubic',bounds_error=False,fill_value=0.)
        #I2_interp= interp1d(lnkfori2,ik2,kind='cubic',bounds_error=False,fill_value=0.)
        clval= quad(lambda lnk: Cl_integrand(lnk,Plnkfunc,I1_interp,I2_interp),lnkmin,lnkmax,limit=kintlim,epsabs=epsilon,epsrel=epsilon,full_output=1)[0]

    return clval*2./np.pi

def Cl_integrand(lnk,Pk_interpfn,Ik1_interpfn,Ik2_interpfn):
    k3=np.exp(3*lnk)
    P = Pk_interpfn(lnk)
    I1 = Ik1_interpfn(lnk)
    I2 = Ik2_interpfn(lnk)
    return k3*P*I1*I2
#-------------------------------------------------------------------------
# Given number of maps, get pairs of indices for unique pairs
# crosspairs[n] holds indices of nth pair of maps, [n,0]<=[n,1]
def get_index_pairs(Nmap):
    #Arranged like 'new=True' ordering in hp.synalm
    Ncross=Nmap*(Nmap+1)/2
    crosspairs=np.zeros((Ncross,2),int) #at location crossind, pair of map ind
    crossinds=np.zeros([Nmap,Nmap],int)#at location [mapind,mapind], crossind
    for w in xrange(Nmap):
        for v in xrange(w,Nmap):
            diff=v-w
            n=w+diff*Nmap - np.sum(np.arange(diff))
            crosspairs[n,:]=w,v
            crossinds[w,v] = n
            crossinds[v,w]=crossinds[w,v]
    return crosspairs,crossinds

def get_index_pairs_old(Nmap): 
    Ncross=Nmap*(Nmap+1)/2
    crosspairs=np.zeros((Ncross,2),int) #at location crossind, pair of map ind
    crossinds = np.zeros((Nmap,Nmap),int)#at location [mapind,mapind], crossind
    u=0
    v=0
    for n in xrange(Ncross):
        crosspairs[n,:]=u,v
        crossinds[u,v]=n
        crossinds[v,u]=n
        v+=1
        if v==Nmap:
            u+=1
            v=u
    return crosspairs,crossinds
#-------------------------------------------------------------------------
# Given list of binmap tags and crossinds, return list of pairs associated with those xinds
def get_pairs_fromcrossind(taglist,docrossind,crosspairs=np.array([]),crossinds=np.array([])):
    if not crosspairs.size or not crossinds.size:
        crosspairs,crossinds=get_index_pairs(len(taglist))
    pairlist=[]
    for n in docrossind:
        i0=crosspairs[n,0]
        i1=crosspairs[n,1]
        pair =(taglist[i0],taglist[i1])
        pairlist.append(pair)
    return consolidate_dotags(pairlist,taglist)
#-------------------------------------------------------------------------
# Given list of BinMaps and dopairs [(tag1,tag2),(,)...] list
# return list of crossinds for which we want to compute C_l
# if addauto=True, autocorrelations will be included even if not in other lists
def get_docross_ind(tagdict,dopairs,crossinds=np.array([]),addauto=False):
    #print 'in get_docross_ind: dopairs',dopairs
    if not crossinds.size:
        crosspairs,crossinds = get_index_pairs(len(tagdict))
    docross=[] #index of cross corrs to do
    #add all autocorrelations to 'do' list
    if addauto:
        #print 'in get_docross_ind: adding autopower cls'
        for i in xrange(len(tagdict)):
            docross.append(crossinds[i,i])
    for pair in dopairs:
        #print 'in get_docross_ind: on pair',pair
        p0=pair[0]
        p1=pair[1]
        i0=i1=-1 #-1 means not in tagdict
        p0isbin= '_bin' in p0
        p1isbin= '_bin' in p1
        #if a tag is for a specific bin, and not in tagdict, won't be computed
        if p0isbin:
            if (p0 in tagdict): i0 =tagdict[p0]
            else: continue
        if p1isbin:
            if (p1 in tagdict): i1 =tagdict[p1]
            else:continue
        # add necessary docross entries to list
        if p0isbin*p1isbin: #both individual bins
            docross.append(crossinds[i0,i1])
        elif p0isbin!=p1isbin: #one individual bin, one type
            if p0isbin:
                pbin=p0
                ptype=p1
                ibin=i0
            elif p1isbin:
                pbin=p1
                ptype=p0
                ibin=i1
            for tag in tagdict:
                new=False
                if tag[:tag.find('_bin')]==ptype:
                    itype = tagdict[tag]
                    new=True
                if new: #if a new maptype match has been found
                    #print 'adding to computations',tag,': ',p0,p1
                    docross.append(crossinds[i0,i1])
        else: #both types of bin
            i0list=[]
            i1list=[]
            for tag in tagdict:
                if tag[:tag.find('_bin')]==p0: i0list.append(tagdict[tag])
                if tag[:tag.find('_bin')]==p1: i1list.append(tagdict[tag])
            i0i1combos= itertools.product(i0list,i1list)
            for combo in i0i1combos:
                docross.append(crossinds[combo[0]][combo[1]])
                
    docross=list(set(docross)) #remove duplicates
    return docross

#------------------------------------------------------------------------
# given two ClData instances returns oldind: array of size newcl.Nmap, where
#         oldind[i] = indix where newcl.bintaglist[i] appears in oldcl.bintaglist
#            that is to say newcl.bintaglist[i]=oldcl.bintaglist[oldind[i]]
#           except: oldind[i]=-1 if tag doesn't appear in oldtaglist
def translate_tag_inds(newcl,oldcl):
    #old = follow indices for maplist in prev existing file
    oldind=-1*np.ones(newcl.Nmap) #for each tag in newcl,bintaglist, its index in oldcl.bintaglist
    #get indices of tags existing in oldbintags
    for t in xrange(newcl.Nmap):
        tag=newcl.bintaglist[t]
        if tag in oldcl.bintaglist:
            oldind[t]=oldcl.tagdict[tag]
    return oldind

#------------------------------------------------------------------------
def combine_old_and_new_Cl(newcl,oldcl,Overwrite=False):
    #combine new and old Cl info to write everything to file
    # if OVERWRITE; new Cl values kept even if old exist for that pair
    Nmap = newcl.Nmap
    Noldmap = oldcl.Nmap
    tagdict=newcl.tagdict
    crossinds=newcl.crossinds
    oldxinds=oldcl.crossinds
    
    oldind=translate_tag_inds(newcl,oldcl)
    
    combotags=oldcl.bintaglist[:] #slicing makes deep copy
    for t in xrange(Nmap): #add any new maptags
        if oldind[t]<0: combotags.append(newcl.bintaglist[t])
    comboNmap = len(combotags)
    combopairs,comboxinds = get_index_pairs(comboNmap)
    comboNcross = combopairs.shape[0]

    #set up arrays to translate between old, new, combo cross indices
    # mapindtranslate[n,0]=old tag ind of map n, [n,1]=new tag ind
    mapindtranslate=-1*np.ones((comboNcross,2))
    mapindtranslate[:oldcl.Nmap,0] = np.arange(oldcl.Nmap)
    for m in xrange(len(combotags)):
        if combotags[m] in newcl.tagdict:
            mapindtranslate[m,1]=newcl.tagdict[combotags[m]]
    # xindtranslate[n,0]=oldxind of combo n, [n,1]=new crossind
    xindtranslate=-1*np.ones((comboNcross,2))
    for n in xrange(comboNcross):
        c0,c1=combopairs[n]
        old0 = mapindtranslate[c0,0]
        new0 = mapindtranslate[c0,1]
        old1 = mapindtranslate[c1,0]
        new1 = mapindtranslate[c1,1]
        if old0>=0 and old1>=0:
            xindtranslate[n,0] = oldcl.crossinds[old0,old1]
        if new0>=0 and new1>=0:
            xindtranslate[n,1] = newcl.crossinds[new0,new1]

    #combine "do" pairs 
    combopairs = consolidate_dotags(newcl.pairs+oldcl.pairs,combotags)

    Nell = newcl.Nell
    comboCl = np.zeros((comboNcross,Nell))
    for n in xrange(comboNcross):
        oldn = xindtranslate[n,0]
        newn = xindtranslate[n,1]
        if Overwrite and oldn>=0 and newn>=0: 
            comboCl[n,:] = newcl.cl[newn,:]
        elif oldn>=0: #if No overwrite, but val was in old file, copy it over
            comboCl[n,:] = oldcl.cl[oldn,:]
        elif newn>=0: #not in old file, but in new
            comboCl[n,:] = newcl.cl[newn,:]
    combocl=ClData(newcl.rundat,combotags,combopairs,clgrid=comboCl)
    return combocl

#------------------------------------------------------------------------
# given list of unique tag pairs [(tag0,tag1),...] all bin tags
#   consoliate so that if tag paired w all bins of
#   replace with (tag0,type) rather than (tag0,type_binX)
#   ->assumes no duplicates in binmaplist
def  consolidate_dotags(pairs,bintaglist):
    #print 'CONSOLIDATING',pairs
    Nmap = len(bintaglist)
    tagdict = {bintaglist[m]:m for m in xrange(Nmap)}
    crosspairs,crossinds = get_index_pairs(Nmap)
    #get list of unique map types
    types=[]
    typedict={}
    binind_fortype=[]# [type][list of indices for bintagss]
    for n in xrange(Nmap):
        tt= bintaglist[n][:bintaglist[n].find('_bin')]
        if tt not in types:
            types.append(tt)
            typedict[tt]=len(types)-1#index of type
            binind_fortype.append([n])
        else:
            binind_fortype[typedict[tt]].append(n)

    #get crosscorr indices for all 'do' pairs. assumes all autocorrs included
    docross=get_docross_ind(tagdict,pairs,crossinds)
    pairedwith=np.zeros((Nmap,Nmap)) #1 if bins assoc w/indices are paired
    accountedfor=np.zeros((Nmap,Nmap)) #1 if this pair is in 'results'
    for n in docross:
        i0 = crosspairs[n,0]
        i1 = crosspairs[n,1]
        pairedwith[i0,i1]=pairedwith[i1,i0]=1

    results=[]
    for t0 in xrange(len(types)):
        binind0 = binind_fortype[t0] #list of bintag indices
        for t1 in xrange(t0,len(types)):
            #print 'looking at type pair:',types[t0],types[t1]
            binind1 = binind_fortype[t1]
            #each b1 index has bool, true if that b1 is paired with all t0
            pairedwithall0=[all([pairedwith[b1,b0] for b0 in binind0]) for b1 in binind1]
            if all(pairedwithall0): #type-type match
                #print '  type-type match!'
                results.append((types[t0],types[t1]))
                #mark those pairs as accounted for
                for b0 in binind0:
                    for b1 in binind1:
                        accountedfor[b0,b1]=accountedfor[b1,b0]=1
            else:
                #add type-bin pairs
                #print '  checking bin-type matches'
                for bi1 in xrange(len(binind1)):
                    if pairedwithall0[bi1]:
                        #print '    adding', (types[t0],bintaglist[binind1[bi1]])
                        results.append((types[t0],bintaglist[binind1[bi1]]))
                        for b0 in binind0:
                            accountedfor[b0,binind1[bi1]]=accountedfor[binind1[bi1],b0]=1
                #check for bin0 bins paired with all t1
                pairedwithall1=[all([pairedwith[b1,b0] for b1 in binind1]) for b0 in binind0]
                for bi0 in xrange(len(binind0)):
                    if pairedwithall1[bi0]:
                        #print '    adding', (types[t1],bintaglist[binind0[bi0]])
                        results.append((types[t1],bintaglist[binind0[bi0]]))
                        for b1 in binind1:
                            accountedfor[b1,binind0[bi0]]=accountedfor[binind0[bi0],b1]=1
    #now, check if there are any bin-bin pairs left
    #print ' checking for leftover bin-bin pairs'
    for n in docross:
        i0 = crosspairs[n,0]
        i1 = crosspairs[n,1]
        if not accountedfor[i0,i1]:
            if i0!=i1:
                #print '    adding', (bintaglist[i0],bintaglist[i1])
                results.append((bintaglist[i0],bintaglist[i1]))
            accountedfor[i0,i1]=accountedfor[i1,i0]=1
    #this is just for testing
    orphans = pairedwith*np.logical_not(accountedfor)
    if np.any(orphans):
        print "MISSING SOME PAIRS IN CONSOLIDATION"

    return results
#------------------------------------------------------------------------
def readCl_file(rundata):
    #return Clarray, lvals, and string ids of all maps cross corr'd
    #will return empty arrays if file doesn't exist or wrong lvals
    outcl= np.array([])
    bintags=[]
    dopairs=[]
    nbar=[]
    if rundata.tag: runtag = '_'+rundata.tag
    else: runtag=''
    infile = ''.join([rundata.cldir,'Cl',runtag,'.dat'])


    if os.path.isfile(infile):
        print "Reading C_l file:", infile
        #open infile and read the first couple lines to get maplist and dopairs
        f=open(infile,'r')
        h0=f.readline() #header line containing list of bin tags
        h0b=f.readline()#header line containting nbar for each bintag (added 6/15)
        h1=f.readline() #header line containing list of pairs of tags to do
        f.close()
        bintags = h0[h0.find(':')+2:].split()
        #Since adding the nbarline is new, check whether h0b is nbar or pairs
        if h0b[:5]=='nbar:':
            hasnbar=True
            nbarstr=h0b#[h0b.find(':')+2:].split()
            nbar=np.array([float(x) for x in nbarstr[nbarstr.find(':')+2:].split()])
        else: #in old format, just has pairs
            hasnbar=False
            #leave nbar as empty array, ClData init will fill in all nbar=-1
            h1=h0b
        dopairs = [(p[:p.find('-')],p[p.find('-')+1:]) for p in h1[h1.find(':')+2:].split()]
        dopairs=consolidate_dotags(dopairs,bintags)
        Nmaps = len(bintags)
        if hasnbar:
            data = np.loadtxt(infile,skiprows=9)
        else:
            data = np.loadtxt(infile,skiprows=8)
        if len(data.shape)>1: #if more than one ell value, more than one row in file
            l = data[:,0].astype(int)
            clgrid = np.transpose(data[:,1:]) #first index is crosspair, second is ell
        else: #just one row
            l= data[0].astype(int)
            clgrid = data[1:].reshape(data[1:].size,1)

        #return clgrid if l values match up, otherwise return empty array
        if l.size==rundata.lvals.size:
            if (l-rundata.lvals<rundata.epsilon).all():
                outcl=clgrid
            else:
                print "  *** unexpected lvals, recompute"
        else:
            print "  *** unexpected size for lvals array, recompute"

    cldat=ClData(rundata,bintags,dopairs,outcl,nbarlist=nbar)
    return cldat#outcl,bintags,dopairs
    
#------------------------------------------------------------------------
def writeCl_file(cldat):
    #cldat= a ClData class instance
    if not cldat.hasClvals:
        print "WARNING: writing file for ClData with empty cl array."
    rundata=cldat.rundat
    crosspairs=cldat.crosspairs
    crossinds=cldat.crossinds
    taglist=cldat.bintaglist
    nbarlist=cldat.nbar
    dopairs=cldat.pairs
    Clgrid=cldat.cl
    
    if rundata.tag: runtag = '_'+rundata.tag
    else: runtag=''
    outfile = ''.join([rundata.cldir,'Cl',runtag,'.dat'])
    lvals = rundata.lvals

    print "Writing C_l data to file:",outfile
    f=open(outfile,'w')
    #write info about cross corr in data; these lists will be checked
    header0 = 'Maps: '+' '.join(taglist)+'\n'
    header0b= 'nbar:'+''.join([' {0:5.3e}'.format(x) for x in nbarlist])+'\n'
    header1 = 'Computed for pairs: '+' '.join([pair[0]+'-'+pair[1] for pair in dopairs])+'\n'
    f.write(header0)
    f.write(header0b)
    f.write(header1)
    #write info about run ; won't be checked but good to have
    f.write(rundata.infostr+'\n')
    f.write('##############################\n') #skiprows = 8
    
    #write column labels
    #crosspairs,crossinds=get_index_pairs(len(taglist))
    Npairs = crosspairs.shape[0]
    colhead0 = ''.join([' {0:23s}'.format(''),''.join([' {0:23s}'.format(taglist[crosspairs[n,0]]) for n in xrange(Npairs)]),'\n'])
    colhead1 = ''.join([' {0:23s}'.format('lvals'),''.join([' {0:23s}'.format(taglist[crosspairs[n,1]]) for n in xrange(Npairs)]),'\n'])
    f.write(colhead0)
    f.write(colhead1)
    
    #write out ell and C_l values, l = rows, pairs= columns
    bodystr=''.join([''.join([' {0:+23d}'.format(lvals[l]),''.join([' {0:+23.16e}'.format(Clgrid[n,l]) for n in xrange(Npairs)]),'\n'])\
                         for l in xrange(lvals.size)])
    f.write(bodystr)
    f.close()

#=========================================================================
# combineCl_twobin:
#   given input cldat containting maps with tags tag1, tag1, combine the Cl from
#     those bins into one larger bin. Only works if nbar are in cldat.
#   newmaptag- binmap tag to be associated with new map made from combo
#        note that it should have _bin# in order to be id's as a binmap tag
#  ouptut: clData object with one less map bin.
def combineCl_twobin(cldat,tag1,tag2,combotag,newruntag='',keept1=False,keept2=False):
    newNmap=cldat.Nmap-1+keept1+keept2
    mapind1=cldat.tagdict[tag1]
    mapind2=cldat.tagdict[tag2]
    xind11=cldat.crossinds[mapind1,mapind1]
    xind22=cldat.crossinds[mapind2,mapind2]
    xind12=cldat.crossinds[mapind1,mapind2]
    nbar1=cldat.nbar[mapind1]
    nbar2=cldat.nbar[mapind2]
    if nbar1<0 or nbar2<0:
        print "***WARNING, no nbar info for one of these maps!"
        return
    nbartot=nbar1+nbar2
    
    # gather info needed to make a new clData object
    newbintaglist=[]
    newnbarlist=[]
    newdocross=[]
    for m in xrange(cldat.Nmap):
        if (keept1 or m!=mapind1) and (keept2 or m!=mapind2):
            newbintaglist.append(cldat.bintaglist[m])
            newnbarlist.append(cldat.nbar[m])
    newbintaglist.append(combotag)
    newnbarlist.append(nbartot)
    combomapind=newNmap-1 #map index of combined map (last entry)
    
    #set up structures for new output dat
    newNcross=newNmap*(newNmap+1)/2
    newcl=np.zeros((newNcross,cldat.Nell))
    newxpairs,newxinds=get_index_pairs(newNmap)
    #fill in values appropriately. Ref: Hu's lensing tomography paper
    for n in xrange(newNcross):
        i,j=newxpairs[n] #in new map index bases
        if i==combomapind and j==combomapind: #both are the new combined map
            newcl[n,:]+=nbar1*nbar1*cldat.cl[xind11,:]
            newcl[n,:]+=nbar2*nbar2*cldat.cl[xind22,:]
            newcl[n,:]+=2.*nbar1*nbar2*cldat.cl[xind12,:]
            newcl[n,:]/=nbartot*nbartot
        elif i==combomapind or j==combomapind: #just 1 is combo
            if i==combomapind:
                k=j #map not in combo, in new basis
            else:
                k=i
            oldmapind=cldat.tagdict[newbintaglist[k]] #in old map basis
            xind1k=cldat.crossinds[mapind1,oldmapind]
            xind2k=cldat.crossinds[mapind2,oldmapind]
            newcl[n,:]+=nbar1*cldat.cl[xind1k,:]
            newcl[n,:]+=nbar2*cldat.cl[xind2k,:]
            newcl[n,:]/=nbartot
        else: #nether are combined map, just translate indices
            oldi=cldat.tagdict[newbintaglist[i]]
            oldj=cldat.tagdict[newbintaglist[j]]
            oldxind=cldat.crossinds[oldi,oldj]
            newcl[n,:]=cldat.cl[oldxind,:]
        if np.any(newcl[n,:]): #not strictly accurate for combo bin; will mark
            #  xind as computed even if only one of the constituent bins were
            newdocross.append(n)

    #construct clData object and return it
    outcldat=ClData(copy.deepcopy(cldat.rundat),newbintaglist,clgrid=newcl,addauto=False,docrossind=newdocross,nbarlist=newnbarlist)
    if newruntag:
        outcldat.rundat.tag=newruntag
    return outcldat
    
#----------------------------------------------------------
# combineCl_binlist:
#   given input cldat, merge all bins in taglist
#           ->taglist bins must be in cldat, and must have nbar!=-1
#   newmaptag- binmap tag to be associated with new map made from combo
#   keeporig - if True, original bins kept, if false, any combined bins dropped
#  ouptut: clData object with one less map bin.
def combineCl_binlist(cldat,taglist,combotag,newruntag='',keeporig=True):
    outcldat=cldat
    origtaglist=taglist[:]
    if newruntag:
        outruntag=newruntag
    else:
        outruntag=cldat.rundat.tag
    while len(taglist)>1:
        tag1=taglist[0]
        tag2=taglist[1]
        keep1=keep2=False
        if tag1 in origtaglist:
            keep1=True
        if tag2 in origtaglist:
            keep2=True
        #print 'tag1,2=',tag1,tag2
        outcldat=combineCl_twobin(outcldat,tag1,tag2,combotag,outruntag,keep1,keep2)
        taglist=taglist[1:]
        taglist[0]=combotag
    return outcldat



