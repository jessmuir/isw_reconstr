#This file contains functions to be used for analyzing how various systematics
# affect our ability to reconstruct ISW maps from galaxy maps

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import itertools
from scipy.optimize import fmin_slsqp,leastsq
import copy

from MapParams import * #Classes w info about map properties
from CosmParams import Cosmology #class w/info about cosm params P(k), etc
from ClRunUtils import * #classes which keep track of k, lvals, etc
from genCrossCor import * #classes which compute C_l for sets of maps
from genMapsfromCor import * #functions which gen g_lm and maps from C_l

###########################################################################
# RecData - contains info used to make ISW reconstruction
#   includeglm is list of tuples of (map,mod,mask) to use glm from
#     if len=1, mod+mask are defaults, if len=2, mask='fullsky'
#   includecl is list map tags (strings) for which we'll use Cl from 
#     for rec, if it is empty, use same set as in includeglm;
#     if not empty, must be the same length as includeglm
#     -> ISW tag will be added to front automatically
#   maptag - string which becomes part of maptag for output glmData
#            should describe simulated maps used for reconstruction
#   rectag - string which becomes modtag for output glmData
#   lmin_forrec - integer identifying the lowest ell value of to be used in 
#                 reconstruction. default is to neglect monopole and dipole
#   NSIDE - healpix NSIDE for reconstructed map
#   recmaptag - string or tuple identifying map to be reconstructed
#             needs to be present in cldat, but not necessarily glmdat
#   useObsCl - use observed Cl's from data for all LSS, use includelcl theory for LSS_ISW and ISW Cls
#   userectags_foriswtrue - if True, uses the recdat modtag and masktag to find the true ISW map for computing rho.
#                               Before 2/1/17, computed used base isw_true map and just added varainces to the large scale structure. Now when creating isw_true maps from Cl (instead of adding Cl to LSS maps), has modtag in filename.
#   
###########################################################################
class RecData(object):
    def __init__(self,includeglm=[],includecl=[],inmaptag='fromLSS',rectag='nonfid',minl_forrec=2,maxl_forrec=-1,
                 NSIDE=32,zerotag='isw_bin0',maptagprefix='iswREC',nolmintag=False,useObsCl=False,userectags_fortrueisw=False,modtaglist=[]):
        self.Nmap=len(includeglm)
        self.inmaptag=inmaptag
        self.maptag=maptagprefix+'.'+inmaptag
        self.rectag=rectag
        if useObsCl:
            self.rectag+='-fromObs' #using Cl from observed maps instead of theory
        self.userectags_fortrueisw=userectags_fortrueisw
        
        self.useObsCl=useObsCl
        self.lmin=minl_forrec
        self.lmax=maxl_forrec #by default, uses max l available
        if not nolmintag: #don't include to be compatible with old data
            lminstr="-lmin{0:02d}".format(self.lmin)
            if self.lmax>0:
                lmaxstr="-lmax{0:02d}".format(self.lmax)
            else:
                lmaxstr=""
            self.masktag='fullsky'+lminstr+lmaxstr

        
        self.NSIDE=NSIDE

        self.includeglm=includeglm
        self.includecl=includecl

        self.zerotag=zerotag
        if type(zerotag)==str: #string, or just one entry; just maptag
            self.zerotagstr=zerotag
        else:
            self.zerotagstr=zerotag[0]#if it is a tuple

        if not includecl:
            if includeglm:
                self.set_fidreccl()
        else:
            if len(includecl)!=len(includeglm):
                print "WARNING: includeglm and includecl are not the same lenght!"
            self.includecl=[self.zerotagstr]+includecl #now len is +1 vs incglm
            
        if not modtaglist:
            self.modtaglist = ['unmod']*len(includeglm)
        else:
            if len(modtaglist)!=len(includeglm):
                if len(modtaglist)!=1:
                    print '\nWARNING: rectaglist and includeglm are not the same length! Using unmod for true maps.'
                    self.modtaglist = ['unmod']*len(includeglm)
                else:
                    self.modtaglist = modtaglist*len(includeglm)
            else:
                self.modtaglist = modtaglist
                
    def set_includeglm(self,newglmlist):
        if newglmlist[0] == self.zerotagstr:  #[zerotag wasn't present when I was running multi tests, resulting in error. NJW 160609]      
            newglmlist.remove(self.zerotagstr)
        self.includeglm=newglmlist
        if not self.includecl:
            self.set_fidreccl()
        self.Nmap=len(self.includeglm)

    def set_fidreccl(self): #set includecl=includeglm
        self.rectag='fid'
        includecl=[]
        includecl.append(self.zerotagstr)
        for x in self.includeglm:
            if type(x)==str and x!=self.zerotagstr:# and x not in bintags:
                includecl.append(x)
            elif type(x)==tuple and x[0]!=self.zerotagstr:# and x[0] not in bintags:
                includecl.append(x[0])
        self.includecl=includecl
        
#================================================
#fitcl_for_constbias - given observed cl and theory cl (1d arrays of same size)
#  find for best fit constant bias via cl_obs = (b0**2)*cl_theory
#  intended to be used on galaxy autopower
def fitcl_forb0_onereal(cltheory,clobs):
    if len(clobs.shape)==1:
        Nreal=1
        b0inv,_=leastsq(b0fit_residuals_onereal,1., args=(cltheory,clobs))
    else:
        Nreal=clobs.shape[1]
        b0inv=np.ones(Nreal)
        for r in xrange(Nreal):
            b0inv[r],_=leastsq(b0fit_residuals_onereal,1., args=(cltheory,clobs[:,r]))
    b0=1./np.fabs(b0inv)#np.sqrt(b0inv)
    return  b0

def b0fit_residuals_onereal(b0inv,cltheory,clobs):
    b02inv=b0inv*b0inv
    ymeas=clobs*b02inv
    err=cltheory - ymeas
    return err

# for inDl = Nellx(NLSS+1)x(NLSS+1) matrix and b0 = NLSS size array, scale
# all cl according to the appropriate b0 values, assuming index order matches
def scale_Dl_byb0(inDl,b0):
    NLSS=b0.size
    Dl=inDl.copy()
    for i in xrange(NLSS):
        Dl[:,i+1,:]=Dl[:,i+1,:]*b0[i]
        Dl[:,:,i+1]=Dl[:,:,i+1]*b0[i]
    return Dl

#================================================
#-------------------------------------------------------------------------
# get_Dl_matrix
#  input: cldat -ClData instance
#         includelist- list of either tags, binmaps, or maptypes to include
#                in estimator construction. if none given, use all
#         zerotag- tag for map we want to construct an estimator for
#                if not in include list, will be added, will be kept index 0
#  output: D: NellxNxN matrix, where Nell=number of l values, N=number of maps
#       where indices go like [ell,maptype,maptype]
#-------------------------------------------------------------------------
def get_Dl_matrix(cldat,includelist=[],zerotag='isw_bin0'):
    #print 'in get dl, includlist:',includelist
    if not includelist:
        includelist=cldat.bintaglist
    #print 'getting D, includelist=',includelist
    dtags=[] #tags for maps included in D_l
    dtags.append(zerotag)
    # get the list of tags set up, corresponding to row/col indices in D
    for x in includelist:
        if isinstance(x,str):
            if x in cldat.bintaglist:
                if x not in dtags: 
                    dtags.append(x)
                elif x!='isw_bin0': print 'Repeat tag {0} ignored for Dl'.format(x,)
            else:
                print x," not in C_l data; disregarding."
        elif isinstance(x,MapType): #this part has not been tested
            for b in x.bintags:
                if b in cldat.bintaglist:
#                    if bintag not in dtags: #allow adding second "map" copy with same properties and name [NJW 160627]
                    dtags.append(b)
                else: print b," not in C_l data; disregarding."
        elif isinstance(x,BinMap):#this part has not been tested
            if x.tag in cldat.bintaglist:
                if x.tag not in dtags: dtags.append(x.tag)
            else: print x.tag," not in C_l data; disregarding."

    Nmap=len(dtags)
    #translate between map indices in D and in C
    clind=-1*np.ones(Nmap,int)
    for t in xrange(Nmap):
        clind[t]=cldat.tagdict[dtags[t]]
#    print clind
#    print 'dtags:',dtags

    #set up D matrix
    Nell=cldat.Nell
    D=np.zeros((Nell,Nmap,Nmap))
    cl=np.zeros((Nell,Nmap,Nmap))
    noise=np.zeros((Nell,Nmap,Nmap))
    for i in xrange(Nmap):
        for j in xrange(i,Nmap): #i<=j
            ci=clind[i] #mapind of i in cl basis
            cj=clind[j] #mapind of j in cl basis
            cxij = cldat.crossinds[ci,cj] #crossind of ij pair in clbasis
            D[:,i,j]=cldat.cl[cxij,:]+cldat.noisecl[cxij,:]
            cl[:,i,j]=cldat.cl[cxij,:]
            noise[:,i,j]=cldat.noisecl[cxij,:]
            if i!=j:
                D[:,j,i]=cldat.cl[cxij,:]+cldat.noisecl[cxij,:] #matrix is symmetric
                cl[:,j,i]=cldat.cl[cxij,:] #matrix is symmetric
                noise[:,j,i]=cldat.noisecl[cxij,:] #matrix is symmetric
#    print '    \n cl[0,:] = \n',cl[4,:,:]
#    print '    \n noise[0,:] = \n',noise[4,:,:]
#    print '    D[00][01][11]=',D[4,0,0],D[4,0,1],D[4,1,1]
    
    #print 'det(D)=',np.linalg.det(D)
    return D,dtags

#-------------------------------------------------------------------------
# invert_Dl: givem Dl matrix, with dim NellxNmapsxNmaps
#            invert each 2d D[lind,:,:] matrix
def invert_Dl(D):
    #Dinv=np.zeros_like(D)
    Nell=D.shape[0]
    Nmap=D.shape[1]
    Dinv=np.zeros((Nell,Nmap,Nmap))

    #check whether any map has all zeros in their column/row: 
    #   these are prolematic: will result in singular matrix
    zerorows=[[]]#for each ell, holds list of indices for rows that are all zero [lind][i]
    Dtoinvert=[np.array([])]
    for lind in xrange(1,Nell):
        zerorows.append([])
        #Dtoinvert.append([])
        tempD=D[lind,:,:]
        tempDind = 0 #if rows/cols deleted from D, keep track of index
        for i in xrange(Nmap):
            if np.all(D[lind,i,:]==0):
                zerorows[lind].append(i)
                tempD=np.delete(tempD,tempDind,0) #delete this row
                tempD=np.delete(tempD,tempDind,1) #delete this col
            else:
                tempDind+=1
        Dtoinvert.append(tempD)
    #print 'zerorows',zerorows

    #for each l, invert with all-zero-rows/cols removed
    #could also just use np.linalg.inv, but found a forum saying this is quicker
    identity=np.identity(Nmap,dtype=D.dtype)
    for l in xrange(1,Nell): #leave ell=0 as all zeros

        #print 'ELL=',l
        #print Dtoinvert[l]
        if identity.size!=Dtoinvert[l].size:
            identity=np.identity(Dtoinvert[l].shape[0])
        thisDinv=np.linalg.solve(Dtoinvert[l],identity) #just for this ell value
        #print thisDinv
        #put values into Dinv, skipping rows with indices in zerorows
        #  so: zerorows in D will also be zero in Dinv so they don't contribute
        #      to the ISW estimator
        tempi=0
        for i in xrange(Nmap):
            if i not in zerorows[l]:
                tempj=i
                for j in xrange(i,Nmap):
                    #print 'j=',j
                    if j not in zerorows[l]: 
                        Dinv[l,i,j]=thisDinv[tempi,tempj]
                        Dinv[l,j,i]=thisDinv[tempj,tempi]
                        tempj+=1
                tempi+=1
    return Dinv
            
#-------------------------------------------------------------------------
# get glm_array_forrec
# given glmData object, get glm for appropriate maps, get indices to match D
#   includelist is list of tuples of (map,mod,mask) to get
#     if len=1, mod+mask are defaults, if len=2, mask='fullsky'
# returns: outglm = array with dim [Nreal,len(includelist),Nlm]
# NOTE includelist should NOT contain map to be reconstructed 
#-------------------------------------------------------------------------
def get_glm_array_forrec(glmdat,includelist=[],zerotag='isw_bin0'):
    #print glmdat.maptaglist
    #loop through tags in includelist, check that they're in glmdat
    # and map between indices appropriately
    if not includelist: #if no includelist, use all unmod maps
        includelist=glmdat.maptaglist
        if zerotag in includelist:
            includelist.remove(zerotag)

    dinds=[]#d[n]=i: n=index for rec data glm, i = index in input
    indexforD=0 #index of data glm to be used in rec
    for x in includelist:
        if type(x)==str:
            x=(x,'unmod','fullsky')
        elif type(x)==tuple:
            if len(x)==1:
                x=(x[0],'unmod','fullsky')
            elif len(x)==2:
                x=(x[0],x[1],'fullsky')
        xind=glmdat.get_mapind_fromtags(x[0],x[1],x[2])
        if xind not in dinds:
            dinds.append(xind)
            indexforD+=1
    #collect appropriate glm info
    print 'Doing rec with following glm indices (based on modtags): ',dinds
    outglm=np.zeros((glmdat.Nreal,len(dinds),glmdat.Nlm),dtype=np.complex)
    for d in xrange(len(dinds)):
        outglm[:,d,:]=glmdat.glm[:,dinds[d],:]
    return outglm,dinds

def calc_isw_est(cldat,glmdat,recdat,writetofile=True,getmaps=True,redofits=True,makeplots=False,dorho=False,fitbias=True):
    """
    calc_isw_est -  given clData and glmData objects, plus list of maps to use
            get the appropriate D matrix and construct isw estimator ala M&D
            but with no CMB temperature info
     input:
       cldat, glmdat - ClData and glmData objects containing maps
                       to be used in reconstructing it (Cl must have isw too)
       recdat- RecData object containing info about what maps, tags, etc to use
       writetofile- if True, writes glm to file with glmfiletag rectag_recnote
       getmaps - if True, generates fits files of maps corresponding to glms
       makeplots - if getmaps==makeplots==True, also generate png plot images
     output: 
       iswrecglm - glmData object containing just ISW reconstructed
                    also saves to file
    -------------------------------------------------------------------------
    """
    useObsCl=recdat.useObsCl
    maptag=recdat.maptag
    rectag=recdat.rectag
    lmin_forrec=recdat.lmin
    lmax_forrec=recdat.lmax
    print "\nComputing ISW estimator maptag,rectag:",maptag,rectag
    if not recdat.includeglm:
        useglm=cldat.bintaglist
        print 'recdat.includeglm not present --> Using cldat.bintaglist for ISW rec: ',useglm
        print recdat.zerotag
        recdat.set_includeglm(useglm) # rectag.set_includeglm(useglm) [rectag is string, no set_.. func, changed to recdat NJW 160609]
    
    #get D matrix dim Nellx(NLSS+1)x(NLSS+1) 
    #   where NLSS is number of LSS maps being used for rec, 
    #       where +1 is from ISW
    Dl,dtags=get_Dl_matrix(cldat,recdat.includecl,recdat.zerotagstr)
    Dinv=invert_Dl(Dl)
    #print 'Dinv',Dinv
    #get glmdata with right indices, dim realzxNLSSxNlm
    glmgrid,dinds=get_glm_array_forrec(glmdat,recdat.includeglm,recdat.zerotag)

    #fit for constant bias for each LSS map (assume all non-zero indices are LSS)
    Nreal=glmgrid.shape[0]
    NLSS=glmgrid.shape[1] 
    Nell=Dl.shape[0] # 170113 - note that if useObsCl, don't actually uise Dl. this assumes Nell from Cl matches observed maps, which should usually be true, but could we tighten this?
    lmin=lmin_forrec
    if lmax_forrec<0 or lmax_forrec>Nell-1:
        lmax=Nell-1
    else:
        lmax=lmax_forrec

    b0=np.ones((Nreal,NLSS))#find best fit bias for each real, for each LSS tracer
    
    #print '------reconstructing isw000-------'
    if fitbias and not useObsCl: #170112 added useObs conidtion
        for i in xrange(NLSS):
            #get cltheory form diagonal entry in Dl
            cltheory=Dl[:,i+1,i+1]#0th entry is ISW
            clobs=np.zeros((Nell,Nreal))
            #for each realization, get clobs from glm and do a fit
            for r in xrange(Nreal):
                clobs[:,r]=hp.alm2cl(glmgrid[r,i,:])
            b0[:,i]=fitcl_forb0_onereal(cltheory[lmin:lmax+1],clobs[lmin:lmax+1,:])#only fit to desired ell
            # print ' theory cl for ell=4:',cltheory[4]
            # print ' observ cl for ell=4:',clobs[4,0]#0th real
            # print '       best fit bias:',b0[0,i],b0[0,i]**2
            # print '    R for ell=4 b0=1:',Dl[4,0,1]/cltheory[4]
            # print '  R for ell=4 wb0fit:',Dl[4,0,1]/cltheory[4]/b0[0,i]

            # plt.figure(0)
            # plt.semilogy(np.arange(Nell),np.fabs(cltheory),label='theory')
            # plt.semilogy(np.arange(Nell),np.fabs(cltheory*b0[0,i]**2),label='theory*b0^2')
            # plt.semilogy(np.arange(Nell),np.fabs(clobs),label='obs')
            # plt.legend()
            # plt.show()

            showtestplot=False
            if showtestplot:
                plt.figure(0)
                col=['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
                for r in xrange(Nreal):
                    plt.plot(cltheory,clobs[:,r],linestyle='none',marker='x',color=col[r%len(col)])
                    plt.plot(cltheory,cltheory*(b0[r,i]**2),color=col[r%len(col)])
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.show()


        print "bbb -- Scaling by best-fit constant bias. Looping through realizations..."
    else:
        print "***Skipping bias fitting step!***"            
    #scaling according to bias will make Dl and Dinv depend on realization
    Dlr=np.zeros((Nreal,Nell,NLSS+1,NLSS+1))
    Dinvr=np.zeros((Nreal,Nell,NLSS+1,NLSS+1))

#    --------- useobs 170112 NJW
    if useObsCl: #calculate cross corr from map realisations and build Dl from that.
        print '----- Using Cl_obs from maps for Estimator...'
        xpairs,xinds = get_index_pairs(NLSS) #corresponds to Cl ordering from healpy
        c_4_obs=[]
        for r in xrange(Nreal):
    #        Dlr[r,:,:,:]=scale_Dl_byb0(Dl,b0[r,:]) #b0=1 if no fitting
    #        Dinvr[r,:,:,:] = invert_Dl(Dlr[r,:,:,:])
            # according to docs, alm2cl can take array of alms, and will return cross spectra in diag order (clA, clB, clC)--> (ClAA, clBB clCC, clAB, clBC, clAC)
            clobs_xspec=np.array(hp.alm2cl(glmgrid[r,:,:])).T #try it as array first, (will it use correct alms by NLSS? not sure) Should be array of dims Nell x NLSS*(NLSS+1)/2. Need to convert to correct order in Cl matrix
#            print 'clobs shape:', clobs_xspec.shape
#            print '\nclobs [(ClAA, clBB clCC, clAB, clBC, clAC)]:'
#            print clobs_xspec[1,:]
#            print 
            c_4_obs.append(clobs_xspec[4,:])
            #dims Nell x N_xinds. Assumes readin maps have same Nell as that derived from passed Cldat
            for i in xrange(NLSS):
                for j in xrange(NLSS):
                    if j<=i: #symmetric matrix
#                        print 'in au.calc_isw_rec: xinds',xinds[i,j]
#                        print (i,j)
#                        print clobs_xspec.shape
                        Dlr[r,:,i+1,j+1]=clobs_xspec[:,xinds[i,j]] # scale_Dl_byb0(Dl,b0[r,:]) #b0=1 if no fitting
                        Dlr[r,:,j+1,i+1]=clobs_xspec[:,xinds[i,j]]
            # get isw-gal xpower from D matrix (indep of realization) for all ell
            Dlr[r,:,:,0] = Dl[:,:,0] ## D matrix dims: Nellx(NLSS+1)x(NLSS+1) 
            Dlr[r,:,0,:] = Dl[:,0,:] 
            assert Dlr[r,4,-1,-2]==Dlr[r,4,-2,-1] #should be symmetric
            Dinvr[r,:,:,:] = invert_Dl(Dlr[r,:,:,:])
#                cltheory=Dl[:,i+1,i+1]#0th entry is ISW
#                clobs=np.zeros((Nell,Nreal))
                #for each realization, get clobs from glm and do a fit
    #            for r in xrange(Nreal):
#                    clobs[:,r]=hp.alm2cl(glmgrid[r,i,:])

#        print Dlr[0,:,0:3,0:3]
#                b0[:,i]=fitcl_forb0_onereal(cltheory[lmin:lmax+1],clobs[lmin:lmax+1,:])#only fit to desired ell
    #------          
    else: #don't use obs Dl from maps, use theory
        for r in xrange(Nreal):
            #print '         bias:',b0[r,:]
            Dlr[r,:,:,:]=scale_Dl_byb0(Dl,b0[r,:]) #b0=1 if no fitting
            Dinvr[r,:,:,:] = invert_Dl(Dlr[r,:,:,:])
        #print '   prescaling:',Dl[4,0,0],Dl[4,0,1],Dl[4,1,1]
        #print '  postscaling:',Dlr[r,4,0,0],Dlr[r,4,0,1],Dlr[r,4,1,1]
        #print '  R4 after scaling func',Dlr[r,4,0,1]/Dlr[r,4,1,1]

    ltest=4
    cosvar=2*np.average(Dlr[:,ltest,0,0])/(2*ltest+1)
    print' \n   ---- N_realz = {0}    '.format(Nreal)
    print '\nC_[l={0}]_isw-isw. (avg, std, sderr)'.format(ltest)
    print '<Clr> = ',(np.average(Dlr[:,ltest,0,0]), np.std(Dlr[:,ltest,0,0]), np.std(Dlr[:,ltest,0,0])/np.sqrt(Nreal))
    print '  Cosmic var: ',cosvar
    print '\n  +/- limits: ',(np.average(Dlr[:,ltest,0,0]) - 2*np.sqrt(cosvar), np.average(Dlr[:,ltest,0,0]) + 2*np.sqrt(cosvar))
    
    print '\nC_[l={0}]_isw-lss. (avg, std, sderr)'.format(ltest)
    print (np.average(Dlr[:,ltest,0,1]), np.std(Dlr[:,ltest,0,1]), np.std(Dlr[:,ltest,0,1])/np.sqrt(Nreal))
    
    print '\nC_[l={0}]_lss-lss. (avg, std, sderr)'.format(ltest)
    print (np.average(Dlr[:,ltest,1,1]), np.std(Dlr[:,ltest,1,1]), np.std(Dlr[:,ltest,1,1])/np.sqrt(Nreal))

    

    #compute estimator; will have same number of realizations as glmdat
    almest=np.zeros((glmgrid.shape[0],1,glmgrid.shape[2]),dtype=np.complex)
    ellvals,emmvals=glmdat.get_hp_landm()
    for lmind in xrange(glmdat.Nlm):
        ell=ellvals[lmind] #note this only has m=[0,l], not -l values, so l*(l+1)/2 total vals
        emm=emmvals[lmind]
        if ell<lmin_forrec or (lmax_forrec>0 and ell>lmax_forrec): 
            continue
        Nl=1./Dinvr[:,ell,0,0] #Nreal sized array
        for i in xrange(recdat.Nmap): #loop through non-isw maps
            almest[:,0,lmind]-=Dinvr[:,ell,0,i+1]*glmgrid[:,i,lmind]
            #print 'just added',-1*Dinv[ell,0,i]*glmgrid[:,i,lmind]
        almest[:,0,lmind]*=Nl
        
    ltest_mask = ellvals == ltest # selects lm_indices where l=ltest
    m0_mask = emmvals==0
    m_non0_mask = emmvals != 0
    
    # OK NEED TO CHECK ALL THESE. THIS IS ALL DEBUG STUFF, BUT NOT FULLY WRITTEN. MAKING SURE ESTIMATED ALMS HAVE SAME MEAN AND VARIANCE AS EXPECTED, EVEN WITH CALIB ERRORS
#    print ellvals[90:130]
    print len(ellvals)
    print sum(ltest_mask)
    print '\nvar(a_[l={0}]_m_est). (avg, (var(real), ar(imag), var(full), sderr)'.format(ltest)
    print 'avg',np.average(almest[:,0,ltest_mask])
    
    clrest = np.average(almest[:,0,ltest_mask*m0_mask], axis=1) #average alm per 
    clrest_var = np.var(clrest) #sample variance. Should = 2*Cl^2/(2l+1)
    print 'len clrest = reals?',len(clrest)
    print np.average(clrest)
    print '<Cl> = ',clrest
    print 'var(Cl) = ',clrest_var
    
    print '(Cl =) var, m=0'
    clrest_m0 = np.var(almest[:,0,ltest_mask*m0_mask], axis=1) #length realz, with variance of Alms, which should = Cl on average
    clrest_non0_real = np.var(almest[:,0,ltest_mask*m_non0_mask].real, axis=1)
    clrest_non0_imag = np.var(almest[:,0,ltest_mask*m_non0_mask].imag, axis=1)
    print '<Cl> (m=0) = ',np.average(clrest_m0)
    print '<Cl> (m!=0, real) = ',np.average(clrest_non0_real)
    print '<Cl> (m!=0, imag) = ',np.average(clrest_non0_imag)
    
    print 
    print (np.var(almest[:,0,ltest_mask*m0_mask].real), np.var(almest[:,0,ltest_mask*m0_mask].imag))
    print np.var(almest[:,0,ltest_mask*m0_mask])
    print '(Cl =) 2*var, m != 0'
    print (2*np.var(almest[:,0,ltest_mask*m_non0_mask].real), 2*np.var(almest[:,0,ltest_mask*m_non0_mask].imag))
    print 'var tot',np.var(almest[:,0,ltest_mask*m_non0_mask])
    print 'std'
    print (np.std(almest[:,0,ltest_mask*m_non0_mask].real),np.std(almest[:,0,ltest_mask*m_non0_mask].imag))
    print np.std(almest[:,0,ltest_mask*m_non0_mask])
    print 'stderr',np.std(almest[:,0,ltest_mask])/np.sqrt(Nreal)
    print
    
    outmaptags=[recdat.maptag]
# Now have useObsCl directly modify modtag in recdat object NJW 170119
#    if useObsCl: #added 170116, sbould only affect ISWrec maps... NJW
#        outmodtags=[recdat.rectag + '-obs']
#    else:
#        outmodtags=[recdat.rectag + '-thry']
    outmodtags=[recdat.rectag]
    print recdat.rectag
    outmasktags=[recdat.masktag]#for now, only handles changing lmin/max, not masks
    almdat=glmData(almest,glmdat.lmax,outmaptags,glmdat.runtag,glmdat.rundir,rlzns=glmdat.rlzns,filetags=[maptag+'.'+rectag],modtaglist=outmodtags,masktaglist=outmasktags)

    if writetofile: #might set as false if we want to do several recons
        #print "WRITING ALM DATA TO FILE"
        write_glm_to_files(almdat)
    if getmaps:
        get_maps_from_glm(almdat,redofits=redofits,makeplots=makeplots,NSIDE=recdat.NSIDE)

        #compute rho
        if dorho:
            print "Computing rho statistics"
            truemapf=glmdat.get_mapfile_fortags(0,recdat.zerotagstr)
            truemapbase=truemapf[:truemapf.rfind('.r')]
            recmapf=almdat.get_mapfile(0,0,'fits')
            recmapbase=recmapf[:recmapf.rfind('.r')]
            if almdat.rlzns.size:
                rlzns=almdat.rlzns
            else:
                rlzns=np.arange(almdat.Nreal)
            rhovals=rho_manyreal(truemapbase,recmapbase,rlzns=rlzns,lmin=recdat.lmin,lmax=recdat.lmax)

    return almdat

#-------------------------------------------------------------------------
# domany_isw_recs- run several sets of isw rec, bundle results into one output file
#    list of cldata objects - if len=1, use same cl for all
#           otherwise should be same length as recinfo
#           also accepts a single ClData object
#    list of glmdata objects - like list of cl for length
#    reclist - list of RecData objects
#    outfiletag, outruntag - to be used in output glm filename 
#       (runtag will also show up in maps made from glmdat)
#    writetofile - if True, writes output to file
#    getmaps - if True, get fits files for maps that go with recs
#    makeplots - if getmaps and True, also make png files
#  Assumes all recs have same Nlm and Nreal
def domany_isw_recs(cldatlist,glmdatlist,reclist,outfiletag='iswREC',outruntag='',writetofile=True,getmaps=True,redofits=True,makeplots=False,dorho=True,fitbias=True):
    print "In domany_isw_recs func"    
    SameCl=False
    Sameglm=False
    if type(cldatlist)!=list:#if a clData object is passed
        cldatlist=[cldatlist]
    if type(glmdatlist)!=list: #if a glmData object is passedo
        glmdatlist=[glmdatlist]
    if len(cldatlist)==1:
        SameCl=True
    if len(glmdatlist)==1:
        Sameglm=True
    i=0

    for rec in reclist:
        if SameCl:
            cldat=cldatlist[0]
        else:
            cldat=cldatlist[i]
        if Sameglm:
            glmdat=glmdatlist[0]
        else:
            glmdat=glmdatlist[i]
        #rec contains expected info here
        #print '-->rec.includeglm',rec.includeglm
        #print '   rec.includecl',rec.includeglm
        #print '   rec.lmin,.lmax',rec.lmin,rec.lmax
        almdat=calc_isw_est(cldat,glmdat,rec,writetofile=False,getmaps=getmaps,redofits=redofits,makeplots=makeplots,dorho=dorho,fitbias=fitbias)

        if i==0:
            outalmdat=almdat
        else:
            outalmdat=outalmdat+almdat
        i+=1

    #print 'IN DOMANY',outalmdat.mapdict
    #assign consistent runtag and filetag
    outalmdat.filetag=[outfiletag]
    outalmdat.runtag=outruntag
    if writetofile:
        write_glm_to_files(outalmdat,setnewfiletag=True,newfiletag=outfiletag)
    return outalmdat
#-------------------------------------------------------------------------   
# get_dummy_recalmdat - returns dummy glmdat object with nreal=0
#     with map/mod/mask lists set up to give the filenames of rec mas which 
#     would be created by feeding the same args into domany_isw_recs
#   input:
#      glmdat - some glmdata object with desired rundir, lmax
def get_dummy_recalmdat(glmdat,reclist,outfiletag='iswREC',outruntag=''):
    outmaptags=[]
    outmodtags=[]
    outmasktags=[]
    for rec in reclist:
        outmaptags.append(rec.maptag)
        outmodtags.append(rec.rectag)
        outmasktags.append(rec.masktag)
    #in domany_isw_rec, the runtag is only set by first glmdat?
    almdat=glmData(np.array([]),glmdat.lmax,outmaptags,outruntag,glmdat.rundir,filetags=[outfiletag],modtaglist=outmodtags,masktaglist=outmasktags)

    return almdat

#-------------------------------------------------------------------------
# rho_mapcorr: compute rho (cross corr) between pixels of two maps
#   #input: two heapy map arrays with equal NSIDE
#   #output: rho = <map1*map2>/(variance(map1)*variance(map2))
#                 where <> means average over all pixels
def rho_onereal(map1,map2):
    if map1.size!=map2.size:
        print "Can't compute correlation between maps with different NSIDE.***"
        return 0
    if (not np.any(map1)) or (not np.any(map2)):
        print "At least one of these maps is all zeros."
        return 0
    product=map1*map2
    avgprod=np.mean(product)
    sig1=np.sqrt(np.var(map1))
    sig2=np.sqrt(np.var(map2))
    #print 'avgprod=',avgprod
    #print 'sig1=',sig1
    #print 'sig2=',sig2
    rho= avgprod/(sig1*sig2)
    #print 'rho=',rho
    return rho
#-------------------------------------------------------------------------
# s: compute s (variance of difference) between pixels of two maps
#   #input: two heapy map arrays with equal NSIDE
#   #output: s = <(map1-map2)^2>/(variance(map1)*variance(map2))
#                 where <> means average over all pixels
def s_onereal(truemap,recmap):
    if truemap.size!=recmap.size:
        print "Can't compute correlation between maps with different NSIDE.***"
        return 0
    if (not np.any(truemap)) or (not np.any(recmap)):
        print "At least one of these maps is all zeros."
        return 0
    #print ' s, recmap std:',np.std(recmap)
    #print ' s, trumap std:',np.std(truemap)
    diff=recmap-truemap
    vardiff=np.mean(diff*diff)
    sigtrue=np.sqrt(np.var(truemap))
    s= np.sqrt(vardiff)/sigtrue
    return s

#-------------------------------------------------------------------------
# chisq: compute chisq (variance of difference) between pixels of two maps
#   #input: two heapy map arrays with equal NSIDE
#   #output: chisq; take out ell using healpy, sum  over l
def chisq_onereal(truemap,recmap):
    if truemap.size!=recmap.size:
        print "Can't compute correlation between maps with different NSIDE.***"
        return 0
    if (not np.any(truemap)) or (not np.any(recmap)):
        print "At least one of these maps is all zeros."
        return 0
    almtrue=hp.map2alm(truemap)
    almrec =hp.map2alm(recmap)
    cltrue=hp.alm2cl(almtrue)
    almdiff2=(np.absolute(almrec-almtrue))**2 #real number
    lmax=cltrue.size - 1.
    lm=hp.sphtfunc.Alm.getlm(lmax)
    l,m=lm
    chisq=0
    for i in xrange(almtrue.size):
        ell=l[i]
        chisq+=almdiff2[i]/cltrue[ell]
    return chisq

#-------------------------------------------------------------------------
# rell_onereal: compute correlation coef between true and rec alm given two maps
# input: two healpy map arrays with equal NSIDE
# output: rell - array of size Nell correlation for each ell value
def rell_onereal(truemap,recmap,varname='rell'):
    #max ell default is 3*Nside-1
    if varname=='rell':
        truecl=hp.sphtfunc.anafast(truemap) #compute Cl's from maps
        #reccl=hp.sphtfunc.anafast(recmap) #compute Cl's from maps
        xcl = hp.sphtfunc.anafast(recmap,truemap)
        rell=xcl/truecl
    elif varname=='chisqell':
        almtrue=hp.map2alm(truemap)
        almrec =hp.map2alm(recmap)
        cltrue=hp.alm2cl(almtrue)
        almdiff2=(np.absolute(almrec-almtrue))**2 #real number
        lmax=cltrue.size - 1.
        lm=hp.sphtfunc.Alm.getlm(lmax)
        l,m=lm
        rell=np.zeros(cltrue.size)
        #rell_lm=np.zeros(almtrue.size)
        for i in xrange(almtrue.size):
            ell=l[i]
            rell[ell]+=almdiff2[i]/(cltrue[ell]*(2*ell+1))
    return rell 


#-------------------------------------------------------------------------
def getmaps_fromCl(cldat,Nreal=1,rlzns=np.array([]),reclist=[],Nglm=1,block=100,
                   glmfiletag='',almfiletag='iswREC',rhofiletag='',justgetrho=False,
                   dorho=True,dos=True,dochisq=False,dorell=False,dochisqell=False,
                   rec_cldat=None, fitbias=True): #170217 added by NJW, so can do reconstruction from different cl_data object (all params except cl_grid must be same)
    """
     The glm files take up a lot of space in memory;
     this function is meant to bundle together:
      -given Cl, generate glm for simulated maps
      -perform some isw reconstructions (maybe for a few lmin?)
      -only save glm data for Nglm realizations (make plots for these too)
    """
    #block=1
    print '======in getmaps_fromCl==========='
    arangereal=not rlzns.size
    if rlzns.size:
        Nreal=rlzns.size

    if rec_cldat==None:
        rec_cldat = cldat #do recreation from same cls as passing in
    #to avoid having giant glm arrays, run in batches, 100ish should be fine
    Nblock=Nreal/block
    remainder=Nreal%block
    
    #rhogrid will hold rho values
    #first index dientifies which recdata, second is block
    rhogrid=[] #will have indices [block][rec][real]

    NEWRHOFILE=(not rlzns.size) or np.all(rlzns==np.arange(rlzns.size))

    #generate maps!
    for n in xrange(Nblock+1):
        rmin=n*block
        if n==Nblock:
            if not remainder:
                continue #nothing left!
            rmax=rmin+remainder
        else:
            rmax=(n+1)*block

        if arangereal:
            nrlzns=np.arange(rmin,rmax)
        else:
            nrlzns=rlzns[rmin:rmax]#np.arange(rmin,rmax)
        #print nrlzns
        
        if Nglm>rmax:
            thisNglm=rmax-rmin-1
            Nglm-=thisNglm
        else:
            thisNglm=Nglm
            Nglm=0

        if not justgetrho:
            print "\nMaking maps for rlzns {0:d}-{1:d}".format(nrlzns[0],nrlzns[-1])
            #print "   thisNglm=",thisNglm
            glmdat=generate_many_glm_fromcl(cldat,rlzns=nrlzns,savedat=False)
            if reclist:
                print "  Doing ISW reconstructions."
                almdat=domany_isw_recs(rec_cldat,glmdat,reclist,writetofile=False,getmaps=True,makeplots=False,outruntag=glmdat.runtag,dorho=False,fitbias=fitbias)
#                almdat=domany_isw_recs(cldat,glmdat,reclist,writetofile=False,getmaps=True,makeplots=False,outruntag=glmdat.runtag,dorho=False)
            #    print '  ************alm after rec:',almdat.maptaglist
            #print '  *****glm generated from healpy:',glmdat.maptaglist
            print 'getting galaxy maps'
            get_maps_from_glm(glmdat,redofits=True,makeplots=False)
            # note that 'getmaps' is done for isw Recs in 'domany_isw_recs'
            if thisNglm:
                print '  Saving glm data for Nreal=',thisNglm 
                saveglm=glmdat.copy(Nreal=thisNglm) #save glm for these
                saveglm= write_glm_to_files(saveglm,setnewfiletag=True,newfiletag=glmfiletag)
                get_maps_from_glm(saveglm,redofits=False,makeplots=True)
                if reclist:
                    savealm=almdat.copy(Nreal=thisNglm)
                    savealm=write_glm_to_files(savealm,setnewfiletag=True,newfiletag=almfiletag)
                    get_maps_from_glm(savealm,redofits=False,makeplots=True)
        elif reclist:
            print "Reading maps for rlzns {0:d}-{1:d}".format(nrlzns[0],nrlzns[-1])
            #need to get almdat and glmdat for filenames
            glmdat=get_glm(cldat,filetag=glmfiletag,Nreal=0,runtag=cldat.rundat.tag)
            almdat=get_dummy_recalmdat(glmdat,reclist,outruntag=glmdat.runtag)
        #for each list, get rho
        if dorho and reclist:
            print "   Computing and saving rho statistics"
            calc_rho_forreclist(glmdat,almdat,reclist,nrlzns,filetag=rhofiletag,overwrite=NEWRHOFILE,varname='rho') #start new file for first block, then add to it
            #print 'done computing rho'
        if dos and reclist:
            print "   Computing and saving s statistics"
            calc_rho_forreclist(glmdat,almdat,reclist,nrlzns,filetag=rhofiletag,overwrite=NEWRHOFILE,varname='s') #start new file for first block, then add to it
        if dochisq and reclist:
            print "   Computing and saving chisq statistics"
            calc_rho_forreclist(glmdat,almdat,reclist,nrlzns,filetag=rhofiletag,overwrite=NEWRHOFILE,varname='chisq') #start new file for first block, then add to it
        if dorell and reclist: #this is slow
            print "  Computing and saving r_ell statistics."
            calc_rell_forreclist(glmdat,almdat,reclist,nrlzns,overwrite=NEWRHOFILE,filetag=rhofiletag)

        if dochisqell and reclist: #this is slow
            print "  Computing and saving chisq_ell statistics."
            calc_rell_forreclist(glmdat,almdat,reclist,nrlzns,overwrite=NEWRHOFILE,filetag=rhofiletag,varname='chisqell')
        
        
        NEWRHOFILE=False
    for recdat in reclist:
        print get_rho_filename(recdat,almdat,filetag=rhofiletag)


def doiswrec_formaps(dummyglm,cldat,Nreal=1,rlzns=np.array([]),reclist=[],Nglm=0,block=100,glmfiletag='',almfiletag='iswREC',rhofiletag='',domaps=True,dorell=False,dos=True,fitbias=True):
    """
    #------------------------------------------------------------------------
    # doiswrec_formaps - given a dummy glmdata with info for existing .fits maps
    #                   make ISW reconstructiosn, compute stats, etc
    # input:
    #  dummyglm - glmData object with Nreal=0, but containing input map names, lmax, etc
    #  cldat - clData object containing info for maps to be used in reconstruction
    #  Nreal, rlzns; tell what or how many realizations to do reconstruction for
    #  reclist - list of recdat objects, containing reconstruction parameters, such as which glm of cldat to us
    #  Nglm - if nonzero, number of realizations for which to save glm data
    #  block - number of realizations to include in each reconstruction; set in order
    #          to avoid manipulating extremely large arrays
    #  glmfiletag,almfiletag - for if you want to save glm data
    #  rhofiletag- appended to filename where rho data is saved
    #  domaps - do we want to generate maps or just look at them? set to false
    #           if we want to just calc rho and other stats
    #  dorell - set to True if we want to compute variance at different ell
    #  
    """
    #block=3
    arangereal=not rlzns.size
    if rlzns.size:
        Nreal=rlzns.size

    #to avoid having giant glm arrays, run in batches, 100ish should be fine
    Nblock=Nreal/block
    remainder=Nreal%block
    
    #rhogrid will hold rho values
    #first index dientifies which recdata, second is block
    rhogrid=[] #will have indices [block][rec][real]

    NEWRHOFILE=(not rlzns.size) or np.all(rlzns==np.arange(rlzns.size))

    #generate maps!
    for n in xrange(Nblock+1):
        rmin=n*block
        if n==Nblock:
            if not remainder:
                continue #nothing left!
            rmax=rmin+remainder
        else:
            rmax=(n+1)*block

        if arangereal:
            nrlzns=np.arange(rmin,rmax)
        else:
            nrlzns=rlzns[rmin:rmax]#np.arange(rmin,rmax)
        #print nrlzns
        
        if Nglm>rmax:
            thisNglm=rmax-rmin-1
            Nglm-=thisNglm
        else:
            thisNglm=Nglm
            Nglm=0

        if domaps:
            print "Getting glm from maps for rlzns {0:d}-{1:d}".format(nrlzns[0],nrlzns[-1])
            #glmdat=generate_many_glm_fromcl(cldat,rlzns=nrlzns,savedat=False)
            glmdat=getglm_frommaps(dummyglm,rlzns=nrlzns)
            print 'have glm, now compute alm'
            almdat=domany_isw_recs(cldat,glmdat,reclist,writetofile=False,getmaps=True,makeplots=False,outruntag=glmdat.runtag,dorho=False,fitbias=fitbias)
            if thisNglm:
                print 'thisNglm in doiswrec_formaps'
                saveglm=glmdat.copy(Nreal=thisNglm) #save glm for these
                saveglm= write_glm_to_files(saveglm,setnewfiletag=True,newfiletag=glmfiletag)
                get_maps_from_glm(saveglm,redofits=False,makeplots=True)
                print 'glms saved, now save alm'
                savealm=almdat.copy(Nreal=thisNglm)
                savealm=write_glm_to_files(savealm,setnewfiletag=True,newfiletag=almfiletag)
                get_maps_from_glm(savealm,redofits=False,makeplots=True)
        else:
            print "Reading maps for rlzns {0:d}-{1:d}".format(nrlzns[0],nrlzns[-1])
            #need to get almdat and glmdat for filenames
            glmdat=dummyglm
            almdat=get_dummy_recalmdat(glmdat,reclist,outruntag=glmdat.runtag)
        #for each list, get rho
        print "   Computing and saving rho and s statistics"
        calc_rho_forreclist(glmdat,almdat,reclist,nrlzns,filetag=rhofiletag,overwrite=NEWRHOFILE,varname='rho') #start new file for first block, then add to it
        if dos:
            print 'dos true'
            calc_rho_forreclist(glmdat,almdat,reclist,nrlzns,filetag=rhofiletag,overwrite=NEWRHOFILE,varname='s') #start new file for first block, then add to it
        if dorell: #this is slow
            print "  Computing and saving r_ell statistics."
            calc_rell_forreclist(glmdat,almdat,reclist,nrlzns,overwrite=NEWRHOFILE,filetag=rhofiletag)
        
        NEWRHOFILE=False
    for recdat in reclist:
        print get_rho_filename(recdat,almdat,filetag=rhofiletag)


#------------------------------------------------------------------------
# calc_rho_forreclist - given glmdat, almdat (can be dummy) plus reclist
#                  return 2d array of [rec][real] of rho values
#      if savedat, writes rho values to file
#          if overwrite, will makenew rho output file
#          otherwise, will add rho data to that in existing file
def calc_rho_forreclist(glmdat,almdat,reclist,rlzns,savedat=True,overwrite=False,filetag='',varname='rho'):
    print "-------Computing ",varname,"statistics-------"
    rhogrid=[]
    for i, rec in enumerate(reclist):
        if rec.userectags_fortrueisw:
            #modtag and masktag from recdat. As of 170101, using modtag for isw_bin0 as well since not using same base isw maps for different variances now
            if rec.useObsCl:
                modtag = rec.rectag[:rec.rectag.rfind('-fromObs')] #truemaps don't have this string, only recreations
            else:
                modtag = rec.rectag
            truemapf=glmdat.get_mapfile_fortags(0,reclist[i].zerotagstr, modtag=modtag)#, masktag=reclist[i].masktag) #recdat masktag has lmin also
        else:
            truemapf=glmdat.get_mapfile_fortags(0,reclist[i].zerotagstr) #mod/mask tags default
        truemapbase=truemapf[:truemapf.rfind('.r')]
        recmapf=almdat.get_mapfile(0,i,'fits') #realization 0, map index i, filetype fits
        #print recmapf
        recmapbase=recmapf[:recmapf.rfind('.r')]
        #print '----'
        #print 'truemapbase',truemapbase
        #print 'recmapbase',recmapbase
        lmin=reclist[i].lmin
        lmax=reclist[i].lmax
        rhovals=rho_manyreal(truemapbase,recmapbase,rlzns=rlzns,savedat=False,varname=varname,lmin=lmin,lmax=lmax)           
        rhogrid.append(rhovals)
        if savedat:
            save_rhodat(rhovals,rlzns,truemapbase,recmapbase,overwrite=overwrite,filetag=filetag,varname=varname)
        
    return np.array(rhogrid)

#------------------------------------------------------------------------
#given list of reconstruction ojbects, and realizations, computes r_ell for them
def calc_rell_forreclist(glmdat,almdat,reclist,rlzns,savedat=True,overwrite=False,filetag='',varname='rell',userectags_fortrueisw=False):
    #print "Computing s statistics"
    rellgrid=[]
    for i, rec in enumerate(reclist):
        if rec.userectags_fortrueisw:
            #modtag and masktag from recdat. As of 170101, using modtag for isw_bin0 as well since not using same base isw maps for different variances now
            if rec.useObsCl:
                modtag = rec.rectag[:rec.rectag.rfind('-fromObs')] #truemaps don't have this string, only recreations
            else:
                modtag = rec.rectag
            truemapf=glmdat.get_mapfile_fortags(0,reclist[i].zerotagstr, modtag=modtag)#, masktag=reclist[i].masktag) #recdat masktag has lmin also
        else:
            truemapf=glmdat.get_mapfile_fortags(0,reclist[i].zerotagstr) #mod/mask tags default
#        truemapf=glmdat.get_mapfile_fortags(0,reclist[i].zerotagstr)
        truemapbase=truemapf[:truemapf.rfind('.r')]
        recmapf=almdat.get_mapfile(0,i,'fits')
        recmapbase=recmapf[:recmapf.rfind('.r')]
        rellvals=rell_manyreal(truemapbase,recmapbase,rlzns=rlzns,savedat=False,varname=varname)
        rellgrid.append(rellvals)
        if savedat:
            save_relldat(rellvals,rlzns,truemapbase,recmapbase,overwrite=overwrite,filetag=filetag,varname=varname)
        
    return np.array(rellgrid)#[reconstruction,realization,ell]
#------------------------------------------------------------------------
# Use this to remove low ell components from a healpy map
#   does so by turning it to glm, setting all glm with l<lmin to zero
#   if lmax passed that is>0, will set l>lmax to zero too
#   then turns it back into a map, which is returned
def remove_lowell_frommap(hpmap,lmin,reclmax=-1):
    nside=hp.npix2nside(hpmap.size)
    alm=hp.map2alm(hpmap)
    lmax=hp.sphtfunc.Alm.getlmax(alm.size)
    l,m=hp.sphtfunc.Alm.getlm(lmax)
    keepell=l>=lmin
    if reclmax>0 and reclmax<lmax:
        keepell*=l<=reclmax
    alm*=keepell
    outmap=hp.alm2map(alm,nside,verbose=False)
    return outmap

#------------------------------------------------------------------------
# rho_manyreal -  find correlations between pairs of maps for many realizations
#  input: mapdir -  directory where the maps are 
#        filebases - filename of maps, up to but not including '.rXXXXX.fits'
#        rlzns, Nreal - if rlzns is empty, rlzns=np.arange(Nreal), otherwise Nreal=rlzns.size

def rho_manyreal(truefilebase,recfilebase,Nreal=1,rlzns=np.array([]),savedat=False,overwrite=False,filetag='',varname='rho',lmin=1,lmax=-1):
    if rlzns.size:
        Nreal=rlzns.size
    else:
        rlzns=np.arange(Nreal)
    rhovals=np.zeros(Nreal)
    #read in the maps
    for r in xrange(Nreal):
        f1=''.join([truefilebase,'.r{0:05d}.fits'.format(rlzns[r])])
        f2=''.join([recfilebase,'.r{0:05d}.fits'.format(rlzns[r])])
        #print f1
        #print f2
        map1orig=hp.read_map(f1,verbose=False)
        map2orig=hp.read_map(f2,verbose=False)
        #print '  pre ell removal rms:',np.std(map1orig),np.std(map2orig)
        #filter out ell<lmin
        map1=remove_lowell_frommap(map1orig,lmin,lmax)
        map2=remove_lowell_frommap(map2orig,lmin,lmax)
        #print '  true/rec map rms:',np.std(map1),np.std(map2)
        
        #compute cross correlations and store the value
        if varname=='rho':
            rhovals[r]=rho_onereal(map1,map2)
        elif varname=='s':
            rhovals[r]=s_onereal(map1,map2)
        elif varname=='chisq':
            rhovals[r]=chisq_onereal(map1,map2)
    if savedat:
        save_rhodat(rhovals,rlzns,truefilebase,recfilebase,overwrite=overwrite,filetag=filetag,varname=varname)
    return rhovals

#------------------------------------------------------------------------------
def rell_manyreal(truefilebase,recfilebase,Nreal=1,rlzns=np.array([]),savedat=False,overwrite=False,filetag='',varname='rell'):
    if rlzns.size:
        Nreal=rlzns.size
    else:
        rlzns=np.arange(Nreal)

    #read in the first map to get info about ell
    f10=''.join([truefilebase,'.r{0:05d}.fits'.format(rlzns[0])])
    f20=''.join([recfilebase,'.r{0:05d}.fits'.format(rlzns[0])])
    map10=hp.read_map(f10,verbose=False)
    map20=hp.read_map(f20,verbose=False)
    rell0=rell_onereal(map10,map20,varname)
    Nell=rell0.size
    #set up array to hold all data
    rellvals=np.zeros((Nreal,Nell))
    rellvals[0,:]=rell0
    #read in the maps
    for r in xrange(1,Nreal):
        #print ' on realization',r
        f1=''.join([truefilebase,'.r{0:05d}.fits'.format(rlzns[r])])
        f2=''.join([recfilebase,'.r{0:05d}.fits'.format(rlzns[r])])
        map1=hp.read_map(f1,verbose=False)
        map2=hp.read_map(f2,verbose=False)
        #compute cross correlations and store the value
        rellvals[r]=rell_onereal(map1,map2,varname)
    if savedat:
        save_relldat(rellvals,rlzns,truefilebase,recfilebase,overwrite=overwrite,filetag=filetag,varname=varname)
    return rellvals

#------------------------------------------------------------------------------
# save_rhodat - save rho data to file
#     NOTE: can also be used to save s data; just pass varname='s'
def save_rhodat(rhovals,rlzns,truefilebase,recfilebase,overwrite=False,filetag='',varname='rho'):
    if filetag:
        tagstr='_'+filetag
    else:
        tagstr=''
    truestr=truefilebase[truefilebase.rfind('/')+1:]
    recstr=recfilebase[recfilebase.rfind('/')+1:]

    outf=''.join([recfilebase.replace('/'+recstr+'/','/'),tagstr,'.'+varname+'.dat'])
    #if overwrite, or if no file exists, write, otherwise append
    NEWFILE=overwrite or not os.path.isfile(outf)
    
    if not NEWFILE: #read in existing data, make sure realizations don't overlap
        dat=np.loadtxt(outf,skiprows=6)
        oldrlzns=dat[:,0]
        oldrho=dat[:,1]
        #check for overlaps bewteen realizations
        duplicates=np.intersect1d(oldrlzns,rlzns)
        Ndup=duplicates.size
        if Ndup:
            NEWFILE=True #we'll combine old and new w/out duplicateds
            # for each duplicated real,store where they are in oldrlzns, rlzns
            overlap=np.array([\
                          [np.where(oldrlzns==r)[0][0],np.where(rlzns==r)[0][0]]\
                          for r in duplicates])

            newdat=np.zeros((oldrlzns.size+rlzns.size - Ndup,2))
            newdat[:oldrlzns.size,0]=oldrlzns
            newdat[:oldrlzns.size,1]=oldrho
            #overwrite duplicates
            for d in xrange(Ndup):
                whereold=overlap[d,0]
                wherenew=overlap[d,1]
                newdat[whereold,:]=np.array([rlzns[wherenew],rhovals[wherenew]])
            #delete overlaps from input data, add to newdat
            rhovals=np.delete(rhovals,overlap[:,1],0)
            rlzns=np.delete(rlzns,overlap[:,1],0)
            newdat[oldrlzns.size:,0]=rlzns
            newdat[oldrlzns.size:,1]=rhovals
            #sort in to order arrays
            newdat.sort(axis=0)
            #change labels to get ready to write to file
            rlzns=newdat[:,0]
            rhovals=newdat[:,1]
    Nreal=len(rhovals)
    if NEWFILE: #write header and data
        truestr=truefilebase[truefilebase.rfind('/')+1:]
        recstr=recfilebase[recfilebase.rfind('/')+1:]
        f=open(outf,'w')
        if varname=='rho':
            f.write('Correlation coefficent rho between true and rec maps\n')
        elif varname=='s':
            f.write('RMS of difference between true and rec maps, in units of true map RMS\n')
        elif varname=='chisq':
            f.write('Chisqared agreement between true and rec maps\n')
        f.write('true isw: '+truestr+'\n')
        f.write('rec isw:  '+recstr+'\n')
        f.write('NReal: '+str(Nreal)+'\n')
        f.write('mean: {0:0.3f}\n'.format(np.mean(rhovals)))
        f.write('{0:5s} {1:s}\n'.format('rlzn',varname))
    else:
        f=open(outf,'a') #just add data to end

    bodystr=''.join(['{0:05d} {1:0.4f}\n'.format(int(rlzns[i]),rhovals[i])\
                         for i in xrange(Nreal)]) ###160516 NJW changed from .3f to .4f for more precision
    f.write(bodystr)
    f.close()
#------------------------------------------------------------------------------
# save_reldat - save rell data to file
def save_relldat(rellvals,rlzns,truefilebase,recfilebase,overwrite=False,filetag='',varname='rell'):
    if filetag:
        tagstr='_'+filetag
    else:
        tagstr=''
    truestr=truefilebase[truefilebase.rfind('/')+1:]
    recstr=recfilebase[recfilebase.rfind('/')+1:]

    outf=''.join([recfilebase.replace('/'+recstr+'/','/'),tagstr,'.'+varname+'.dat'])
    #if overwrite, or if no file exists, write, otherwise append
    NEWFILE=overwrite or not os.path.isfile(outf)
    
    if not NEWFILE: #read in existing data, make sure realizations don't overlap
        dat=np.loadtxt(outf,skiprows=5)
        oldell=dat[0,1:]
        oldrlzns=dat[1:,0]
        oldrell=dat[1:,1:]
        Nell=oldell.size
        #check for overlaps bewteen realizations
        duplicates=np.intersect1d(oldrlzns,rlzns)
        Ndup=duplicates.size
        if Ndup:
            NEWFILE=True #we'll combine old and new w/out duplicateds
            # for each duplicated real,store where they are in oldrlzns, rlzns
            overlap=np.array([\
                          [np.where(oldrlzns==r)[0][0],np.where(rlzns==r)[0][0]]\
                          for r in duplicates]) #index starts at rlzns=0

            newdat=np.zeros((oldrlzns.size+rlzns.size - Ndup,Nell+1))
            newdat[0,1:]=oldell #assumes new and old data have same ell
            newdat[1:oldrlzns.size+1,0]=oldrlzns 
            newdat[1:oldrlzns.size+1,1:]=oldrell
            #overwrite duplicates
            for d in xrange(Ndup):
                whereold=overlap[d,0]+1 #plus one since first row is ell vals
                wherenew=overlap[d,1]+1
                newdat[whereold,:]=np.concatenate(rlzns[wherenew],rhovals[wherenew,:])
            #delete overlaps from input data, add to newdat
            rhovals=np.delete(rellvals,overlap[:,1],0)
            rlzns=np.delete(rlzns,overlap[:,1],0)
            newdat[1+oldrlzns.size:,0]=rlzns
            newdat[1+oldrlzns.size:,:]=rellvals
            #sort in to order arrays
            newdat.sort(axis=0)
            #change labels to get ready to write to file
            rlzns=newdat[1:,0]
            rellvals=newdat[1:,1:]
    Nreal=len(rlzns)
    Nell=rellvals.shape[1]
    if NEWFILE: #write header and data
        truestr=truefilebase[truefilebase.rfind('/')+1:]
        recstr=recfilebase[recfilebase.rfind('/')+1:]
        f=open(outf,'w')
        if varname=='rell':
            f.write('Correlation coefficent r_ell between true and rec alm\n')
        if varname=='chisqell':
            f.write('l specific contrib to chisq between true and rec alm\n')
        f.write('true isw: '+truestr+'\n')
        f.write('rec isw:  '+recstr+'\n')
        f.write('NReal: '+str(Nreal)+'\n')
        f.write('{0:5s} {1:s}\n'.format('rlzn',varname))
        f.write('{0:05d}'.format(0)+''.join([' {0:05d}'.format(l) for l in xrange(Nell)])+'\n')
    else:
        f=open(outf,'a') #just add data to end

    bodystr=''.join(['{0:05d}'.format(int(rlzns[i]))+''.join([' {0:0.3f}'.format(rellvals[i,l]) for l in xrange(Nell)])+'\n' for i in xrange(Nreal)])
    f.write(bodystr)
    f.close()
    
#------------------------------------------------------------------------------
def get_rho_filename(recdat,recalmdat,filetag='',varname='rho'):
    #can also be used for s data and rell data
    if filetag:
        tagstr='_'+filetag
    else:
        tagstr=''
    #recalmdat.get_mapfile(0,0,'fits')
    real0file=recalmdat.get_mapfile_fortags(0,recdat.maptag,recdat.rectag,recdat.masktag)
    recind=recalmdat.get_mapind_fromtags(recdat.maptag,recdat.rectag,recdat.masktag)
    filebase=recalmdat.get_mapfile_base(recind)
    outf=real0file.replace('/'+filebase+'/','/')
    outf=outf[:outf.rfind('.r')]
    outf=outf+tagstr+'.'+varname+'.dat'
    return outf

def read_rhodat(recdat,recalmdat,filetag='',varname='rho'):
    #can also be used for s data ad rell data
    f=get_rho_filename(recdat,recalmdata,filetag,varname)
    dat=np.loadtxt(f,skiprows=6)
    rlzns=dat[:,0]
    rho=dat[:,1]
    return rho

def read_rhodat_wfile(filename):
    #can also be used for s data 
    print 'reading',filename
    dat=np.loadtxt(filename,skiprows=6)
    if len(dat.shape)>1: #more than one realization
        rlzns=dat[:,0]
        rho=dat[:,1]
    else:
        rlzns=np.array([dat[0]])
        rho=np.array([dat[1]])
    #print 'rho.shape in readfunc:',rho.shape
    return rho

def read_relldat_wfile(filename):
    print 'reading',filename
    dat=np.loadtxt(filename,skiprows=5) 
    rlzns=dat[1:,0] #rows are realizations
    ell = dat[0,1:] #col are ell values
    #print 'IN READING FUNC: NELL',ell.size
    rho=dat[1:,1:]
    #print 'rho.shape',rho.shape
    return rho

#compute the expected value of rho, given theoretical Cl
# cldat is the Cl representing the Cl's used for simulation
#  if reccldat is passed as a ClData object, these Cl's are used in estimator
#       Also, currently assumes recdat can be used for both cldat and recldat
#                 etc.
def compute_rho_fromcl(cldat,recdat,reccldat=0,varname='rho',fitbias=True):
    #print 'recdat.includeglm',recdat.includeglm
    #print 'recdat.includecl',recdat.includecl
    #Dl is a matrix of Cls, with isw at zero index
    #  and other maps in order specified by recdat.includecl
    if not reccldat:
        DIFFREC=False
    else:
        #print 'DIFFREC=True'
        DIFFREC=True #are the Cl's for rec and sim different?

    lmin=recdat.lmin
    lmax=recdat.lmax
#    print cldat.noisecl[:,4]
#    print '\ncldat bintaglist:',cldat.bintaglist
#    print 'recdat includeglm:',recdat.includeglm
#    print 'recdat includecl:',recdat.includecl
    (cldat, recdat_true) = handle_dupes(cldat, recdat)#, var='includeglm') #check rec maps to see if duplicate tagnames and create another set of cls for it if so
#    print 'handling dupes'    
#    print '\ncldat bintaglist:',cldat.bintaglist
#    print 'recdat includeglm:',recdat.includeglm
#    print 'recdat includecl:',recdat.includecl
#    # These are the Cl used for simulating maps (hence the recdat.includeglm)
#    Dl,dtags=get_Dl_matrix(cldat,recdat_true.includeglm,recdat.zerotagstr)
    Dl,dtags=get_Dl_matrix(cldat,recdat_true.includecl,recdat.zerotagstr) #i think this is supposed to be includecl, not includeglm -NJW[160819]
#    print 'True Dl:'
#    print '\nMaps ',dtags
#    print '\ncldat bintaglist:',cldat.bintaglist
#    print 'recdat includglm:',recdat_true.includeglm
#    print Dl[4,:,:]
    Dinv=invert_Dl(Dl)
#    print Dinv[4,:,:]

#    print Dinv[4,:,:]
    Nell=Dinv.shape[0]
    lvals=np.arange(Nell)
    Nl=np.zeros(Nell)
    for l in xrange(Nell):
        if Dinv[l,0,0]!=0:
            Nl[l]=1/Dinv[l,0,0]
#    print Nl[:50]
#    print 'N[l=4]=',Nl[4]
    if lmax<0 or  (lmax>(Nell-1)):
        lmax=Nell-1
#    print 'N[l=4]=',Nl[4]
    NLSS=recdat_true.Nmap

    #if DIFFREC, get Dl data for those Cl, these are Cl for making estimator
    if DIFFREC: #assumes cldat and reccldat have same ell info  
#        print '\nrecldat bintaglist:',reccldat.bintaglist
#        print 'recdat_rec includeglm:',recdat.includeglm
#        print 'recdat_rec includecl:',recdat.includecl        
        #should we add these dupetags to both Sim and Rec? Just Rec?? NJW 160627
        (reccldat, recdat_rec) = handle_dupes(reccldat, recdat)#, var='includecl') #check rec maps to see if duplicate tagnames and create another set of cls for it if so
#        print 'handling dupes'    
#        print '\nrecldat bintaglist:',reccldat.bintaglist
#        print 'recdat_rec includeglm:',recdat_rec.includeglm
#        print 'recdat_rec includecl:',recdat_rec.includecl                
        recDl,recdtags=get_Dl_matrix(reccldat,recdat_rec.includecl,recdat_rec.zerotagstr)
#        print 'verify that rec and sim Dl different:',np.any(recDl-Dl)
#        print '\n RecMaps ',recdtags
#        print 'rec Dl BEFORE BIAS FIT:'
#        print recDl[4,:,:]
#        print
#        print invert_Dl(recDl)[4,:,:]
        #fit for b0 for each LSS map by compareing Dl Cl to recDl
        b0=np.ones(NLSS)
        if fitbias:
            for i in xrange(NLSS):
                #only fit to lvalues we want to use
                b0[i]=fitcl_forb0_onereal(recDl[lmin:lmax+1,i+1,i+1],Dl[lmin:lmax+1,i+1,i+1])
        recDl=scale_Dl_byb0(recDl,b0)
        recDinv=invert_Dl(recDl)
        recNl=np.zeros(Nell)

        for l in xrange(Nell):
            if recDinv[l,0,0]!=0:
                recNl[l]=1/recDinv[l,0,0]
            elif l!=0: print 'recDinv[0,0]=0 for l={0}! Cannot invert for recNl!!'.format(l,)
    else:
        recDl=Dl
        recDinv=Dinv
        recNl=Nl
        recdtags=dtags
#    print '\n Rec Dl AFTER BIAS FIT:'
#    print 'Maps ',recdtags
#    print 'recdat_true includeglm:',recdat_true.includeglm
#    if DIFFREC: print '      recdat_rec includecl:',recdat_rec.includecl
#    print recDl[4,:,:]
#    print recDinv[4,:,:]
#    print 'recN[l=4]=',recNl[4]
#    print 'recDinv shape:',recDinv.shape
#    print 'Are rec and sim Dl different?',np.any(recDl-Dl)
    # construct estimator operators
    estop=np.zeros((NLSS,Nell))#"estimator operator" (R^i_l in eq. 11)
#    print 'Recon Map Weights:'
    for i in xrange(NLSS): #estop set to zero outside ell range
        estop[i,lmin:lmax+1]=-1*recNl[lmin:lmax+1]*recDinv[lmin:lmax+1,0,i+1]
#        print (dtags[i+1], estop[i,4])
#    print (dtags[1:], list(estop[:,4]))
    #for sigisw, just sum over l
    sig2iswl=(2.*lvals[lmin:lmax+1]+1)*Dl[lmin:lmax+1,0,0]
    sig2isw=np.sum(sig2iswl)
    
    #for sigrec, sum over LSS maps 2x (ij), then l
    sig2recl=np.zeros(lvals.size)
    for i in xrange(NLSS):
        sig2recli=np.zeros(lvals.size)
        for j in xrange(NLSS):
            sig2recli+=estop[j,:]*Dl[:,j+1,i+1] #estop set to zero outside ell range
        sig2recl+=sig2recli*estop[i,:]
    sig2recl*=(2.*lvals+1)
    sig2rec=np.sum(sig2recl)
    
    if varname=='rho':
        #for each l sum over LSS maps for numerator, the sum over l
        numell = np.zeros(lvals.size)
        for i in xrange(NLSS):
            numell+=estop[i,:]*Dl[:,0,i+1]#estop already zero for unwanted ell
        numell*=(2.*lvals+1)
        numerator=np.sum(numell)

        denom=np.sqrt(sig2isw*sig2rec)
        if denom==0:
            result=0
        else:
#            print '   FINAL   num,demon:',numerator,denom
            result=numerator/denom
    elif varname=='s':
        #for each l sum over LSS maps for numerator, the sum over l
        crosspowerell = np.zeros(lvals.size)
        for i in xrange(NLSS):
            crosspowerell+=estop[i,:]*Dl[:,0,i+1]
        crosspowerell*=(2.*lvals+1)
        numerator=np.sqrt(sig2rec+sig2isw -2*np.sum(crosspowerell))
        denom=np.sqrt(sig2isw)
        result=numerator/denom
    elif varname=='ssym':
        #for each l sum over LSS maps for numerator, the sum over l
        crosspowerell = np.zeros(lvals.size)
        for i in xrange(NLSS):
            crosspowerell+=estop[i,:]*Dl[:,0,i+1]
        crosspowerell*=(2.*lvals+1)
        numerator=np.sqrt(sig2rec+sig2isw -2*np.sum(crosspowerell))
        denom=np.power(sig2isw*sig2rec,.25)
        result=numerator/denom
        # print 's = ',result
        # print '      Dl[4,0,1]=',Dl[4,0,1]
        # print '      Dl[4,1,1]=',Dl[4,1,1]
        # print '      est[gal,4]=',estop[0,4]
        # print '    sig2isw=',sig2isw
        # print '    sig2rec=',sig2rec
        # print '    xpowsum=',2*np.sum(crosspowerell)
        # print '    numerat=',numerator
        # print '    denom  =',denom
    elif varname=='chisq': 
        # chisq is sum over l of |alm_true-alm_rec|^2
        denom=Dl[:,0,0]#true isw autopower
        nonzerodenom=np.fabs(denom)>0 #booleans
        crossnum=np.zeros(lvals.size) #crosspower piece of numerator
        recnum=np.zeros(lvals.size) #gal-gal part of numerator
        for i in xrange(NLSS):
            crossnum+=estop[i,:]*Dl[:,0,i+1]
            for j in xrange(NLSS):
                recnum+=estop[i,:]*estop[j,:]*Dl[:,j+1,i+1]
        chisqell=np.zeros(len(lvals))
        for l in xrange(len(lvals)):
            if nonzerodenom[l]:
                chisqell[l]=(2*lvals[l]+1.)*(1+(-2.*crossnum[l]+recnum[l])/denom[l])
        result =np.sum(chisqell)#sum over ell
    elif varname=='estop':
        result = estop
    elif varname=='Nl_tup':
        result = (Nl, recNl) #Nl is variance of opitmal estimator
    elif varname=='sigNl_tup': #exploring how could get an analytical variance. Not complete
        sig2Nl=Nl*(2.*lvals+1)
        sig2N=np.sum(sig2Nl[lmin:lmax+1])
#        print sig2N.shape
        sig2Nlrec = recNl*(2.*lvals+1)
        sig2Nrec=np.sum(sig2Nlrec[lmin:lmax+1])
        result = (sig2N/sig2isw, sig2Nrec/sig2rec)
#        result = (sig2N/sig2rec, sig2Nrec/sig2isw)
#        result = (sig2N/(sig2isw*sig2rec)**.5, sig2Nrec/(sig2isw*sig2rec)**.5)
#        result = (sig2Nrec/np.sum((sig2Nl*Nl)[lmin:lmax+1])**.5, sig2Nrec/np.sum((sig2Nlrec*recNl)[lmin:lmax+1])**.5)
    return result
    

#compute the expected value of r_galgal between two maps, given theoretical Cl
# cldat is the Cl representing the Cl's used for simulation

def get_r_for_maps(cldat,maptagA,maptagB,lmin,lmax, varname='r', include_nbar=True):
    """return (r_tot, r_ell) - correlation coefficient between two galaxy maps, total and by l-mode.
    varname={'r', 's', 'ssym'}, where ssym is an s symmetric between maps (denom = sqrt(sigmaA*sigmaB) instead of sigmaA)"""
#    lmin=recdat.lmin
#    lmax=recdat.lmax
    N_l = lmax - lmin + 1 #note NOT THE SMAE AS Nell, WHICH JM I THINK USES AS LMAX+1
#    Dl_gal = np.zeros(Nell,2,2) #like Dl matrix but no ISW
    #auto and cross power arrays
    cl_AA = np.zeros(N_l)
    cl_AB = np.zeros(N_l)
    cl_BB = np.zeros(N_l)    
    rell_gal = np.zeros(N_l)#correlation coeff of each mode    
    sell_gal = np.zeros(N_l)#sse of each mode ([1,inf))
    ssymell_gal = np.zeros(N_l)#map-symmetric form of s ([0, inf))
    
    two_l_plus1 = 2*np.arange(lmin, lmax+1) + 1
    for i in xrange(N_l):
        ell = i+lmin
        cl_AA[i] = cldat.get_cl_from_pair(maptagA,maptagA,ell=ell,include_nbar=include_nbar)
        cl_AB[i] = cldat.get_cl_from_pair(maptagA,maptagB,ell=ell,include_nbar=include_nbar)
        cl_BB[i] = cldat.get_cl_from_pair(maptagB,maptagB,ell=ell,include_nbar=include_nbar)

        rell_gal[i] = cl_AB[i]/np.sqrt(cl_AA[i]*cl_BB[i])
        sell_gal[i] = (cl_AA[i] + cl_BB[i] - 2*cl_AB[i]) / cl_AA[i]
        ssymell_gal[i] = (cl_AA[i] + cl_BB[i] - 2*cl_AB[i]) / np.sqrt(cl_AA[i]*cl_BB[i])

    r_gal = np.sum(two_l_plus1*cl_AB) / np.sqrt(np.sum(two_l_plus1*cl_AA)*np.sum(two_l_plus1*cl_BB))
    s_gal = np.sqrt((np.sum(two_l_plus1*cl_AA) + np.sum(two_l_plus1*cl_BB) - 2.*np.sum(two_l_plus1*cl_AB)) / np.sum(two_l_plus1*cl_AA))
    ssym_gal = np.sqrt((np.sum(two_l_plus1*cl_AA) + np.sum(two_l_plus1*cl_BB) - 2.*np.sum(two_l_plus1*cl_AB)) / np.sqrt(np.sum(two_l_plus1*cl_AA)*np.sum(two_l_plus1*cl_AA))) #note the sqrt encompasses the full expression

    if varname=='r':
        return (r_gal, rell_gal)
    elif varname=='s':
        return (s_gal, sell_gal)
    elif varname=='ssym':
        return (ssym_gal, ssymell_gal)
    else:
        print "enter valid varname in r_map function"
        return

def get_r3_for_maps(cldat, ABmaptag_tuple,maptagC, lmin,lmax, include_nbar=True, tot_or_ell='tot'):
    """return (r[AB+c], [r_AB, r_AC, r_BC]) - multiple correlation coefficient of map C with maps A and B; [pairwise map correlations]
    by default return r_tot; if tot_or_ell=='ell': r[AB+C] etc. are themselves arrays of length (lmax-lmin+1)"""
#    lmin=recdat.lmin
#    lmax=recdat.lmax
    (maptagA,maptagB) = ABmaptag_tuple
    (r_ab, rell_ab) = get_r_for_maps(cldat, maptagA, maptagB, lmin, lmax,include_nbar=include_nbar)
    (r_ac, rell_ac) = get_r_for_maps(cldat, maptagA, maptagC, lmin, lmax,include_nbar=include_nbar)
    (r_bc, rell_bc) = get_r_for_maps(cldat, maptagB, maptagC, lmin, lmax,include_nbar=include_nbar)
    if tot_or_ell == 'tot':
        r_ab_c = np.sqrt((r_bc**2 + r_ac**2 - 2*r_ab*r_ac*r_bc)/(1-r_ab**2))
        return (r_ab_c, [r_ab, r_bc, r_ac])
    elif tot_or_ell == 'ell':
        rell_ab_c = np.sqrt((rell_bc**2 + rell_ac**2 - 2*rell_ab*rell_ac*rell_bc)/(1-rell_ab**2))
        return (rell_ab_c, [rell_ab, rell_bc, rell_ac])
    
    
def rho_sampledist(r,rho,NSIDE=32,Nsample=0): #here rho is the expected mean
    # doesn't integrate to 1 and Nsample=NSIDE seems way too big
    if type(r)!=np.ndarray:
        rarray=np.array([r])
        ISARRAY=False
    else:
        rarray=r
        ISARRAY=True
    result=np.zeros_like(rarray)
    i=0
    #print 'rarray size',rarray.size
    for r in rarray:
        #print i
        if np.fabs(r)>=1:
            result[i]=0.
        else:
            if Nsample: 
                N=Nsample
            else:
                N=hp.nside2npix(NSIDE)
            if N<-3:
                result[i]=-10000000000
                i+=1
                continue
            logarg=(1+r)*(1-rho)/(1-r)/(1+rho)
            logsq=np.log(logarg)**2
            exparg=-1*logsq*(N-3.)/8
            expval=np.exp(exparg)
            coef = np.sqrt((N-3.)/(2.*np.pi))
            result[i]= coef*expval/(1.-r**2)
        i+=1
    if not ISARRAY:
        result=result[0]
    return result



# compute expectation value of r_ell, the correlation coefficient between true
#  and rec alm from theoretical cl
def compute_rell_fromcl(cldat,recdat,reccldat=0,varname='rell',fitbias=True):
    #Dl is a matrix of Cls, with isw at zero index
    #  and other maps in order specified by recdat.includecl
    if not reccldat:
        DIFFREC=False
    else:
        #print 'DIFREC=True'
        DIFFREC=True #are the Cl's for rec and sim different?
    
    lmin=recdat.lmin
    lmax=recdat.lmax
    (cldat, recdat_true) = handle_dupes(cldat, recdat)#, var='includeglm') #check rec maps to see if duplicate tagnames and create another set of cls for it if so
    Dl,dtags=get_Dl_matrix(cldat,recdat.includecl,recdat.zerotagstr) 
#    Dl,dtags=get_Dl_matrix(cldat,recdat_true.includeglm,recdat.zerotagstr) #set to includeglm to match compute_rho_fromcl
#    print Dl[5,:,:]
    Dinv=invert_Dl(Dl)
    Nell=Dinv.shape[0]
    lvals=np.arange(Nell)
    Nl=np.zeros(Nell)
    for l in xrange(Nell):
        if Dinv[l,0,0]!=0:
            Nl[l]=1/Dinv[l,0,0]
            
    if lmax<0 or  (lmax>(Nell-1)):
        lmax=Nell-1
    NLSS=recdat_true.Nmap

    rell=np.zeros(Nell)

    clisw=Dl[:,0,0]#true ISW auto power

    #if DIFFREC, get Dl data for those Cl, these are Cl for making estimator
    if DIFFREC: #assumes cldat and reccldat have same ell info
        (reccldat, recdat_rec) = handle_dupes(reccldat, recdat)#, var='includecl') #check rec maps to see if duplicate tagnames and create another set of cls for it if so
        recDl,recdtags=get_Dl_matrix(reccldat,recdat_rec.includecl,recdat.zerotagstr)
        #fit for b0 for each LSS map by compareing Dl Cl to recDl
        b0=np.ones(NLSS)
        if fitbias:
            for i in xrange(NLSS):
                b0[i]=np.squeeze(fitcl_forb0_onereal(recDl[lmin:lmax+1,i+1,i+1],Dl[lmin:lmax+1,i+1,i+1]))
        recDl=scale_Dl_byb0(recDl,b0)
        recDinv=invert_Dl(recDl)
        recNl=np.zeros(Nell)
        for l in xrange(Nell):
            if recDinv[l,0,0]!=0:
                recNl[l]=1/recDinv[l,0,0]
            elif l!=0: print 'recDinv[0,0]=0 for l={0}! Cannot invert for recNl!!'.format(l,)
    else:
        recDl=Dl
        recDinv=Dinv
        recNl=Nl
        recdtags=dtags

    # construct estimator operators
    estop=np.zeros((NLSS,Nell))#"estimator operator"
    for i in xrange(NLSS):
        estop[i,lmin:lmax+1]=-1*recNl[lmin:lmax+1]*recDinv[lmin:lmax+1,0,i+1]

        
    #for rec auto power, sum over LSS maps 2x (ij)
    clrec=np.zeros(lvals.size)
    for i in xrange(NLSS):
        clreci=np.zeros(Nell)
        for j in xrange(NLSS):
            clreci+=estop[j,:]*Dl[:,j+1,i+1]
        clrec+=clreci*estop[i,:]

    #for cross power sum over LSS maps for numerator
    clx = np.zeros(lvals.size)
    for i in xrange(NLSS):
        clx+=estop[i,:]*Dl[:,0,i+1]
    if varname=='rell':
        result= clx
        for l in xrange(lvals.size):
            denom = clisw[l]*clrec[l]
            if denom!=0:
                result[l]=result[l]/np.sqrt(denom)
            else:
                result[l]=0
    #below lines replaced with above since didn't include cl_rec -NJW[160819]
#            if clisw[l]:
#                result[l]=result[l]/clisw[l]
#            else:
#                result[l]*=0
            
    elif varname=='chisqell':
        #result=1./(2.*lvals+1.) #once many real version is fixed
        result=np.ones(len(lvals))
        for l in range(lvals.size):
            if clisw[l]:
                result[l]*=(1.+(-2.*clx[l]+clrec[l])/clisw[l])
            else:
                result[l]*=0
    #print result
    return result

###########################################################################
# plotting functions
###########################################################################
#------------------------------------------------------------------------------
#plot_Tin_Trec  - make scatter plot comparing true to reconstructed isw
#------------------------------------------------------------------------------
def plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir='output/',plotname='',colorlist=[],dotitle=False,filesuffix='pdf'):
    if not colorlist:
        colors=['#404040','#c51b8a','#0571b0','#66a61e','#ffa319','#d7191c']
    else:
        colors=colorlist
    Nmap=len(recmapfiles)
    recmaps=[]
    iswmaps=[]
    for f in recmapfiles:
        mrec=hp.read_map(f)*1.e5
        recmaps.append(mrec)
    for f in iswmapfiles:
        misw=hp.read_map(f)*1.e5
        iswmaps.append(misw)
    rhovals=[rho_onereal(iswmaps[n],recmaps[n]) for n in xrange(Nmap)]
    plt.figure(1)#,figsize=(10,8))
    plt.subplots_adjust(left=0.15, bottom=.17, right=.95, top=.95, wspace=0, hspace=0)
    if dotitle:
        plt.title('Pixel-by-pixel scatterplot',fontsize=18)
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    plt.xlabel(r'$\rm{T}^{\rm ISW}_{\rm true}$  $(10^{-5}\rm{K})$',fontsize=24)
    plt.ylabel(r'$\rm{T}^{\rm ISW}_{\rm rec}$  $(10^{-5}\rm{K})$',fontsize=24)
    #plt.ticklabel_format(style='sci', axis='both', scilimits=(1,0))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.tick_params(axis='both', labelsize=16)
    ax.set_aspect('equal')#make it square
    
    xmax=np.max(np.fabs(iswmaps))
    plt.xlim(-1.1*xmax,1.6*xmax) 
    plt.ylim(-1.1*xmax,1.1*xmax)
    for i in xrange(len(recmaps)):
        rhoi=rhovals[i]
        coli=colors[i]
        labeli=reclabels[i]+'\n$\\rho={0:0.2f}$'.format(rhoi)
        mrec=recmaps[i]
        misw=iswmaps[i]
        plt.plot(misw,mrec,linestyle='None',marker='o',alpha=1,label=labeli,markersize=4.,color=coli,markeredgecolor='None',rasterized=True)#markerfacecolor='None',markeredgecolor=coli
    plt.plot(10*np.array([-xmax,xmax]),10*np.array([-xmax,xmax]),linestyle='--',linewidth=4.,color='black')

    # try doing text boxes instead of legends?
    startboxes=.8
    totlines=3*len(recmaps)
    fperline=1/15. #estimate by eye
    startheight=startboxes
    for i in range(Nmap)[::-1]:
        li=reclabels[i]+'\n$\\rho={0:0.3f}$'.format(rhovals[i])
        Nline=li.count('\n')+1#.5
        leftside=.975#.69
        textbox=ax.text(leftside, startheight, li, transform=ax.transAxes, fontsize=18,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'boxstyle':'round,pad=.3','alpha':1.,'facecolor':'none','edgecolor':colors[i],'linewidth':4})
        startheight-=Nline*fperline+.03
    
    #plt.show()
    #plt.gcf().subplots_adjust(bottom=0.15) #keep x label from being cut off
        
    #plotdir='output/plots_forposter/'
    if not plotname:
        plotname='TrecTisw_scatter_variousRECs'
    print 'saving',plotdir+plotname+'.'+filesuffix
    plt.savefig(plotdir+plotname+'.'+filesuffix)
    plt.close()

#-----------------------------------------------------------------------------
# plot rho hist - plots histograms of rho
#  rhogrid - NrecxNreal array of rho values
#  reclabels - Nrec string labels for legend, order matches rhogrid 0 axis
#  rhopred - if empty, nothing happens. if not, holds Nrec <rho> values 
#-----------------------------------------------------------------------------
def plot_rhohist(rhogrid,reclabels,testname,plotdir,plotname,rhopred=[]):
    varstr=r'\rho'
    Nreal=rhogrid.shape[1]
    title=r'{0:s}: correlation coef. $\rho$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$\rho=\langle T_{{\rm true}}T_{{\rm rec}}\rangle_{{\rm pix}}/\sigma_{{T}}^{{\rm true}}\sigma_{{T}}^{{\rm rec}}$'
    plothist(varstr,rhogrid,reclabels,title,xtitle,plotdir,plotname,rhopred)

def plot_shist(sgrid,reclabels,testname,plotdir,plotname,spred=[]):
    varstr='s'
    Nreal=sgrid.shape[1]
    title=r'{0:s}: RMS of difference $s$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$s=\langle (T_{{\rm true}}-T_{{\rm rec}})^2\rangle_{{\rm pix}}^{{1/2}}/\sigma_{{T}}^{{\rm true}}$'
    plothist(varstr,sgrid,reclabels,title,xtitle,plotdir,plotname,spred)


def plot_chisqhist(grid,reclabels,testname,plotdir,plotname,cspred=[]):
    varstr=r'\chi^2'
    Nreal=grid.shape[1]
    title=r'{0:s}: $\chi^2$ of $a_{{\ell m}}$ for {1:g} rlzns'.format(testname,Nreal)
    xtitle=r'$\chi^2=\sum_{{\ell}}|a_{{\ell m}}^{{\rm ISW}} - a_{{\ell m}}^{{\rm rec}}|^2/C_{{\ell}}^{{\rm ISW}}$'
    #allmean=np.mean(grid)
    #allsig=np.std(grid)
    #spread=1
    #vallim=(allmean-spread*allsig,allmean+spread*allsig)
    vallim=(0,4000)
    plothist(varstr,grid,reclabels,title,xtitle,plotdir,plotname,cspred,vallim=vallim)
#-----------------------------------------------------------------------------
#plot_hists - plot histograms of some quantity
# varstr - string to label variable being plotted in hist, eg '\rho'
# datagrid - Nrec x Nreal data to be used as hist input
# reclabels - Nrec strings to be used as labels for legend
# plottitle - string title of plot
# xlabel - string, to be used to label x axis
# plotdir - directory in which we'll save the plot
# plotname - filename under which we'll save the plot
# predvalues - if we've computed expectation values analytically, pass them here
#              if they're given, they'll print in legend, plot vertical line
#-----------------------------------------------------------------------------
def plothist(varstr,datagrid,reclabels,plottitle,xlabel,plotdir,plotname,predvals=[],vallim=0):
    Nbins=100
    Nrecs=datagrid.shape[0]
    Nreal=datagrid.shape[1]
    maxval=np.max(datagrid)
    minval=np.min(datagrid)
    #print 'min,max for hist',minval,maxval
    if not vallim: #default setup works well for s, rho, but not chisq
        vallim=(minval,maxval)
    #rholim=(0.,maxrho)
    colors=['#1b9e77','#d95f02','#e7298a','#7570b3','#66a61e','#e6ab02']
    plt.figure(0)
    plt.subplots_adjust(left=0.15, bottom=.2, right=.95, top=.95, wspace=0, hspace=0)
    plt.title(plotname)
    plt.xlabel(xlabel,fontsize=26)
    plt.ylabel('Realizations',fontsize=26)
    plt.tick_params(axis='y', which='both', labelsize=16)
    plt.tick_params(axis='x', which='both', labelsize=16)
    for i in xrange(Nrecs):
        mean=np.mean(datagrid[i,:])
        sigma=np.std(datagrid[i,:])
        colstr=colors[i%len(colors)]
        if len(predvals):
            predval=predvals[i]
            plt.axvline(predval,linestyle='-',color=colstr)
            label=r'{0:s}: $\langle {4:s}\rangle={3:0.3f}$; $\bar{{{4:s}}}={1:0.3f}$'.format(reclabels[i],mean,sigma,predval,varstr)
            #label=r'{0:s}'.format(reclabels[i])
            # if varstr==r'\rho':

            # elif varstr=='s':
            #     label=r'{0:s}: $\langle {3:s}\rangle={2:0.3f}$; $\bar{{{3:s}}}={1:0.3f}$'.format(reclabels[i],mean,predval,varstr)
            # elif varstr=='chisq':
            #     label=r'{0:s}: $\langle {3:s}\rangle={2:0.3f}$; $\bar{{{3:s}}}={1:0.3f}$'.format(reclabels[i],mean,predval,varstr)
        else:
            label='{0:s}: $\bar{{{3:s}}}={1:0.3f} $, $\sigma={2:0.3f}$'.format(reclabels[i],mean,sigma,varstr)
        plt.axvline(mean,linestyle='--',color=colstr)
        nvals,evals,patches=plt.hist(datagrid[i,:],bins=Nbins,range=vallim,histtype='stepfilled',label=label)
        plt.setp(patches,'facecolor',colstr,'alpha',0.6)

    if len(predvals):
        plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='mean from sample')
        plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='expectation value')

    if varstr==r'\rho':
        plt.legend(loc='upper left')
    elif varstr=='s':
        plt.legend(loc='upper right')
    elif varstr==r'\chi^2':
        plt.legend(loc='upper right')
        plt.yscale('log')
        #plt.ylim(.1,1.e5)
        plt.ylim(.1,1.e6)
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()

#-----------------------------------------------------------------------------
# plot_relldat- plot r_ell data
#  rellgrid  - Nrec x Nreal x Nell data array
#  rellpred - Nrec x Nell data array of expectation values
#  reclabels - Nrec strings to be used as labels in legend
#  test name - name of test, for title
#  plotdir - output directory
#  plotname -output file name
def plot_relldat(reclabels,testname,plotdir,plotname,rellgrid=[],rellpred=[],varname='rell'):
    #print 'VARNAME=',VARNAME
    Nellexp=0
    Nelldat=0
    Nrec=len(reclabels)
    doexp=rellpred.size
    dodat=rellgrid.size
    #print 'IN PLOTTING FUNC: rellgrid.shape',rellgrid.shape
    if doexp:
        Nellexp=rellpred.shape[1]
    if dodat:
        Nreal=rellgrid.shape[1]
        Nelldat=rellgrid.shape[2]
        #get mean and quartiles for each ell
        rellmean=np.zeros((Nrec,Nelldat))
        rellquartlow=np.zeros((Nrec,Nelldat))
        rellquarthigh=np.zeros((Nrec,Nelldat))
        for r in xrange(Nrec):
            rellmean[r,:]=[np.mean(rellgrid[r,:,l]) for l in xrange(Nelldat)]
            #rellquartlow[r,:]=[np.percentile(rellgrid[r,:,l],10) for l in xrange(Nelldat)]
            #rellquarthigh[r,:]=[np.percentile(rellgrid[r,:,l],90) for l in xrange(Nelldat)]
    #make plot
    colors=['#1b9e77','#d95f02','#e7298a','#7570b3','#66a61e','#e6ab02']
    plt.figure(0)
#    plt.title(r'$\ell$-specific contrib to $\chi^2$ true and rec ISW $a_{{\ell m}}$') 
    plt.xlabel(r'$\ell$',fontsize=14)
    if varname=='rell':
        plt.ylabel(r'$r_{{\ell}}=\langle a_{{\ell m}} \hat{{a}}^*_{{\ell m}} \rangle/ C_{{\ell}}^{{\rm ISW}}$',fontsize=16)
    elif varname=='chisqell':
        plt.ylabel(r'$\chi^2_{{\ell}}=\langle |a_{{\ell m}} -  \hat{{a}}^*_{{\ell m}} |^2\rangle/ C_{{\ell}}^{{\rm ISW}}$',fontsize=16)
        plt.ylim((0,1))

    if dodat:
        for r in xrange(Nrec):
            colstr=colors[r%len(colors)]
            #plot quartiles as shaded region, legend labels here
            #plt.fill_between(np.arange(Nelldat),rellquartlow[r,:],rellquartlow[r,:],facecolor=colstr,edgecolor=None,alpha=0.5)
            #plot mean as dotted line
            #print rellmean[r,:]
            plt.plot(np.arange(Nelldat),rellmean[r,:],linestyle='-',color=colstr,label=reclabels[r])
    if doexp:
        for r in xrange(Nrec):
            colstr=colors[r%len(colors)]
            if dodat: #no legend label
                plt.plot(np.arange(Nellexp),rellpred[r,:],linestyle='--',color=colstr)
            else:
                plt.plot(np.arange(Nellexp),rellpred[r,:],linestyle='--',color=colstr,label=reclabels[r])

    if dodat and doexp:
        plt.plot(np.array([]),np.array([]),linestyle='-',color='black',label='mean from sample')
        plt.plot(np.array([]),np.array([]),linestyle='--',color='black',label='expectation value')
        plt.legend(loc='center right')
    elif dodat:
        plt.legend(title='From {0:d} rlzns'.format(Nreal),loc='center right')
    elif doexp:
        plt.legend(title='Expectation vals',loc='center right')
    
    outname=plotdir+plotname+'.pdf'
    print 'saving',outname
    plt.savefig(outname)
    plt.close()


def getDupes(ls, getloc = True):
    """get list of duplicates in the list. If loc = False, return [dupe0, dupe1,...,dupeN] instead of [(index, dupe0),...,]
    e.g.: getDupes([1,1,3,5,5,5]) --> [(1,1),(4,5),(5,5)])"""
    seen = set()
    dupes = []
    for i,elem in enumerate(ls):
        if elem not in seen:
            seen.add(elem)
        elif getloc: #incldue location of duplicate in array
            dupes.append((i,elem))
        else:
            dupes.append(elem)
    return dupes
                
#from https://stackoverflow.com/questions/6527641/speed-up-python-code-for-computing-matrix-cofactors
def matrix_adjoint(M):
    """returns adj(M), where M^-1 = adj(M) / det(M).
    This avoids the singularity from det(M)==0 when two sets of same LSS Cls, since det(M) cancels out in estimator, AS LONG AS NO CMB CONTRIBUTION"""
    C = np.zeros(M.shape)
    nrows, ncols = C.shape
    minor = np.zeros([nrows-1, ncols-1])
    for row in xrange(nrows):
        for col in xrange(ncols):
            minor[:row,:col] = M[:row,:col]
            minor[row:,:col] = M[row+1:,:col]
            minor[:row,col:] = M[:row,col+1:]
            minor[row:,col:] = M[row+1:,col+1:]
    return C.T #transpose to get adjoint (though should be symmetric anyways, yes?)

def handle_dupes(cldat, recdat, dupesuf='_1'):
    """take in a recdat object and if include_glm or include_cl lists contain duplicates (e.g. using two datasets from same survey with same props and cls)
    add a copy of those cls to the dataset under new mapname with auto-append dupe_suffix."""
    #should we add these dupetags to both Sim and Rec? Just Rec?? NJW 160627
    cldat = copy.deepcopy(cldat) #make copies of these objects so as to not overwrite the original cldat and recdat objects
    recdat = copy.deepcopy(recdat)
    assert(type(recdat.includeglm) != str)
    assert(type(recdat.includecl) != str)
    
    incl_dupes_running = getDupes(recdat.includecl, getloc=True)
    incl_dupes_done = []
    for loc,dupemap in getDupes(recdat.includeglm, getloc=True):
        cldat,newtag = cldat.add_dupemap(dupemap, dupesuf=dupesuf) #update Cldat with a copy of duplicate maptype
        recdat.includeglm[loc]=newtag #change the duplicate tag to the newtag
        for i,(cl_loc, dm) in enumerate(incl_dupes_running):
            if dm == dupemap: #if the same dupemap is in includecl, update it as well
                recdat.includecl[cl_loc]=newtag#change the duplicate tag to the newtag
                incl_dupes_running.pop(i)
                break
    if len(incl_dupes_running)>0:
        for loc,dupemap in getDupes(recdat.includeclm, getloc=True):
            cldat,newtag = cldat.add_dupemap(dupemap, dupesuf=dupesuf) #update Cldat with a copy of duplicate maptype
            recdat.includecl[cl_loc]=newtag
    return (cldat, recdat)

    
#    if var == 'includeglm':
#        include_list = recdat.includeglm
#    if var == 'includecl':
#        include_list = recdat.includecl
#    else: raise TypeError('Please set var="includeglm" or "includecl". Recieved var= {0}'.format(var,))

#    for v in var:
#        if v=='includeglm':
#            include_list = recdat.includeglm
#        if type(include_list)==str:  #added 6/28/16 NJW
#            #recdat.includeglm = [recdat.includeglm]
#            print "WARNING: given include_list is string, not list!"
#            include_list=[include_list] #did it this way so as to not enforce that it be a list in recdat, lest mess up something of JM'S.
#        if len(include_list) != len(set(include_list)): #check for duplicates in includecl
#            print 'includ_list = ',include_list
#            for loc,dupemap in getDupes(include_list, getloc=True):
#                cldat,newtag = cldat.add_dupemap(dupemap, dupesuf=dupesuf) #update Cldat with a copy of duplicate maptype
#    #            print '\nin HANDLE DUPES. CLDAT BINTAGLIST:',cldat.bintaglist
#                if dupemap in 
#                if var=='includeglm':
#                    recdat.includeglm[loc]=newtag #change the duplicate tag to the newtag
#                elif var=='includecl':
#                    recdat.includecl[loc]=newtag #change the duplicate tag to the newtag
#    return (cldat, recdat)
