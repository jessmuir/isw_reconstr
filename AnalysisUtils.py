#This file contains functions to be used for analyzing how various systematics
# affect our ability to reconstruct ISW maps from galaxy maps

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import itertools

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
###########################################################################
class RecData(object):
    def __init__(self,includeglm=[],includecl=[],inmaptag='fromLSS',rectag='nonfid',minl_forrec=1,NSIDE=32,zerotag='isw_bin0',maptagprefix='iswREC'):
        self.Nmap=len(includeglm)
        self.inmaptag=inmaptag
        self.maptag=maptagprefix+'.'+inmaptag
        self.rectag=rectag
        self.lmin=minl_forrec
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

    def set_includeglm(self,newglmlist):
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
    if not includelist:
        includelist=cldat.bintaglist
    #print 'getting D, includelist=',includelist
    dtags=[] #tags for maps included in D_l
    dtags.append(zerotag)
    # get the list of tags set up, corresponding to row/col indices in D
    for x in includelist:
        if isinstance(x,str):
            if x in cldat.bintaglist:
                if x not in dtags: dtags.append(x)
            else:
                print x," not in C_l data; disregarding."
        elif isinstance(x,MapType): #this part has not been tested
            for b in x.bintags:
                if b in cldat.bintaglist:
                    if bintag not in dtags: dtags.append(b)
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

    #set up D matrix
    Nell=cldat.Nell
    D=np.zeros((Nell,Nmap,Nmap))
    for i in xrange(Nmap):
        for j in xrange(i,Nmap): #i<=j
            ci=clind[i] #mapind of i in cl basis
            cj=clind[j] #mapind of j in cl basis
            cxij = cldat.crossinds[ci,cj] #crossind of ij pair in clbasis
            D[:,i,j]=cldat.cl[cxij,:]+cldat.noisecl[cxij,:]
            if i!=j: D[:,j,i]=cldat.cl[cxij,:]+cldat.noisecl[cxij,:] #matrix is symmetric

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
    outglm=np.zeros((glmdat.Nreal,len(dinds),glmdat.Nlm),dtype=np.complex)
    for d in xrange(len(dinds)):
        outglm[:,d,:]=glmdat.glm[:,dinds[d],:]
    return outglm,dinds

#-------------------------------------------------------------------------
# calc_isw_est -  given clData and glmData objects, plus list of maps to use
#        get the appropriate D matrix and construct isw estimator ala M&D
#        but with no CMB temperature info
# input:
#   cldat, glmdat - ClData and glmData objects containing maps
#                   to be used in reconstructing it (Cl must have isw too)
#   recdat- RecData object containing info about what maps, tags, etc to use
#   writetofile- if True, writes glm to file with glmfiletag rectag_recnote
#   getmaps - if True, generates fits files of maps corresponding to glms
#   makeplots - if getmaps==makeplots==True, also generate png plot images
# output: 
#   iswrecglm - glmData object containing just ISW reconstructed
#                also saves to file
#-------------------------------------------------------------------------
def calc_isw_est(cldat,glmdat,recdat,writetofile=True,getmaps=True,redofits=True,makeplots=False,dorho=False):
    maptag=recdat.maptag
    rectag=recdat.rectag
    lmin_forrec=recdat.lmin
    print "Computing ISW estimator maptag,rectag:",maptag,rectag

    if not recdat.includeglm:
        useglm=cldat.bintaglist
        rectag.set_includeglm(useglm)
    
    #get D matrix dim Nellx(NLSS+1)x(NLSS+1) 
    #   where NLSS is number of LSS maps being used for rec, 
    #       where +1 is from ISW
    Dl,dtags=get_Dl_matrix(cldat,recdat.includecl,recdat.zerotagstr)
    Dinv=invert_Dl(Dl)
    #print 'Dinv',Dinv
    #get glmdata with right indices, dim realzxNLSSxNlm
    glmgrid,dinds=get_glm_array_forrec(glmdat,recdat.includeglm,recdat.zerotag)

    #compute estimator; will have same number of realizations as glmdat
    almest=np.zeros((glmgrid.shape[0],1,glmgrid.shape[2]),dtype=np.complex)
    ellvals,emmvals=glmdat.get_hp_landm()
    for lmind in xrange(glmdat.Nlm):
        ell=ellvals[lmind]
        if ell<lmin_forrec: 
            #print 'continuing since ell is too small'
            continue
        Nl=1./Dinv[ell,0,0]
        for i in xrange(recdat.Nmap): #loop through non-isw maps
            almest[:,0,lmind]-=Dinv[ell,0,i+1]*glmgrid[:,i,lmind]
            #print 'just added',-1*Dinv[ell,0,i]*glmgrid[:,i,lmind]
        almest[:,0,lmind]*=Nl

    outmaptags=[recdat.maptag]
    outmodtags=[recdat.rectag]
    outmasktags=['fullsky']
    almdat=glmData(almest,glmdat.lmax,outmaptags,glmdat.runtag,glmdat.rundir,rlzns=glmdat.rlzns,filetags=[maptag+'.'+rectag],modtaglist=outmodtags,masktaglist=outmasktags)

    if writetofile: #might set as false if we want to do several recons
        #print "WRITING ALM DATA TO FILE"
        write_glm_to_files(almdat)
    if getmaps:
        #print "GETTING ISW REC MAPS"
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
            rhovals=rho_manyreal(truemapbase,recmapbase,rlzns=rlzns)

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
#    makeplots - if getmapes and True, also make png files
#  Assumes all recs have same Nlm and Nreal
def domany_isw_recs(cldatlist,glmdatlist,reclist,outfiletag='iswREC',outruntag='',writetofile=True,getmaps=True,redofits=True,makeplots=False,dorho=True):
    SameCl=False
    Sameglm=False
    if type(cldatlist)!=list:#if a clData object is passed
        cldatlist=[cldatlist]
    if type(glmdatlist)!=list: #if a glmData object is passed
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
        almdat=calc_isw_est(cldat,glmdat,rec,writetofile=False,getmaps=getmaps,redofits=redofits,makeplots=makeplots,dorho=dorho)

        if i==0:
            outalmdat=almdat
        else:
            outalmdat=outalmdat+almdat
        i+=1

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
        outmasktags.append('fullsky')
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
# The glm files take up a lot of space in memory;
# this function is meant to bundle together:
#  -given Cl, generate glm for simulated maps
#  -perform some isw reconstructions (maybe for a few lmin?)
#  -only save glm data for Nglm realizations (make plots for these too)
def getmaps_fromCl(cldat,Nreal=1,rlzns=np.array([]),reclist=[],Nglm=1,block=100,glmfiletag='',almfiletag='iswREC',rhofiletag='',justgetrho=False):
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
    truemapbases=['' for i in xrange(len(reclist))]
    recmapbases=['' for i in xrange(len(reclist))]

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
        
        print "Making maps for rlzns {0:d}-{1:d}".format(nrlzns[0],nrlzns[-1])
        #print "   thisNglm=",thisNglm

        if not justgetrho:
            glmdat=generate_many_glm_fromcl(cldat,rlzns=nrlzns,savedat=False)
            almdat=domany_isw_recs(cldat,glmdat,reclist,writetofile=False,getmaps=True,makeplots=False,outruntag=glmdat.runtag,dorho=False)
            get_maps_from_glm(glmdat,redofits=True,makeplots=False)
            if thisNglm:
                saveglm=glmdat.copy(Nreal=thisNglm) #save glm for these
                saveglm= write_glm_to_files(saveglm,setnewfiletag=True,newfiletag=glmfiletag)
                get_maps_from_glm(saveglm,redofits=False,makeplots=True)

                savealm=almdat.copy(Nreal=thisNglm)
                savealm=write_glm_to_files(savealm,setnewfiletag=True,newfiletag=almfiletag)
                get_maps_from_glm(savealm,redofits=False,makeplots=True)
        else:
            #need to get almdat and glmdat for filenames
            glmdat=get_glm(cldat,filetag=glmfiletag,Nreal=0,runtag=cldat.rundat.tag)
            almdat=get_dummy_recalmdat(glmdat,reclist,outruntag=glmdat.runtag)
        #for each list, get rho
        print "   Computing and saving rho statistics"
        calc_rho_forreclist(glmdat,almdat,reclist,nrlzns,rhofiletag=rhofiletag,overwrite=NEWRHOFILE) #start new file for first block, then add to it
        
        NEWRHOFILE=False
    for recdat in reclist:
        print get_rho_filename(recdat,almdat,filetag=rhofiletag)

#------------------------------------------------------------------------
# calc_rho_forreclist - given glmdat, almdat (can be dummy) plus reclist
#                  return 2d array of [rec][real] of rho values
#      if savedat, writes rho values to file
#          if overwrite, will makenew rho output file
#          otherwise, will add rho data to that in existing file
def calc_rho_forreclist(glmdat,almdat,reclist,rlzns,savedat=True,overwrite=False,rhofiletag=''):
    #print "Computing rho statistics"
    rhogrid=[]
    for i in xrange(len(reclist)):
        truemapf=glmdat.get_mapfile_fortags(0,reclist[i].zerotagstr)
        truemapbase=truemapf[:truemapf.rfind('.r')]
        recmapf=almdat.get_mapfile(0,i,'fits')
        recmapbase=recmapf[:recmapf.rfind('.r')]

        rhovals=rho_manyreal(truemapbase,recmapbase,rlzns=rlzns,savedat=False)
        rhogrid.append(rhovals)
        if savedat:
            save_rhodat(rhovals,rlzns,truemapbase,recmapbase,overwrite=overwrite,filetag=rhofiletag)
        
    return np.array(rhogrid)
    
#------------------------------------------------------------------------
# rho_manyreal -  find correlations between pairs of maps for many realizations
#  input: mapdir -  directory where the maps are 
#        filebases - filename of maps, up to but not including '.rXXXXX.fits'
#        rlzns, Nreal - if rlzns is empty, rlzns=np.arange(Nreal), otherwise Nreal=rlzns.size

def rho_manyreal(truefilebase,recfilebase,Nreal=1,rlzns=np.array([]),savedat=False,overwrite=False,rhofiletag=''):
    if rlzns.size:
        Nreal=rlzns.size
    else:
        rlzns=np.arange(Nreal)
    rhovals=np.zeros(Nreal)
    #read in the maps
    for r in xrange(Nreal):
        f1=''.join([truefilebase,'.r{0:05d}.fits'.format(rlzns[r])])
        f2=''.join([recfilebase,'.r{0:05d}.fits'.format(rlzns[r])])
        map1=hp.read_map(f1,verbose=False)
        map2=hp.read_map(f2,verbose=False)
        #compute cross correlations and store the value
        rhovals[r]=rho_onereal(map1,map2)
    if savedat:
        save_rhodat(rhovals,rlzns,truefilebase,recfilebase,overwrite=overwrite,filetag='')
    return rhovals

#------------------------------------------------------------------------------
# save_rhodat - save rho data to file
def save_rhodat(rhovals,rlzns,truefilebase,recfilebase,overwrite=False,filetag=''):
    if filetag:
        tagstr='_'+filetag
    else:
        tagstr=''
    truestr=truefilebase[truefilebase.rfind('/')+1:]
    recstr=recfilebase[recfilebase.rfind('/')+1:]

    outf=''.join([recfilebase.replace('/'+recstr+'/','/'),tagstr,'.rho.dat'])
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
        f.write('Correlation coefficent rho between true and rec maps\n')
        f.write('true isw: '+truestr+'\n')
        f.write('rec isw:  '+recstr+'\n')
        f.write('NReal: '+str(Nreal)+'\n')
        f.write('mean: {0:0.3f}\n'.format(np.mean(rhovals)))
        f.write('{0:5s} {1:s}\n'.format('rlzn','rho'))
    else:
        f=open(outf,'a') #just add data to end

    bodystr=''.join(['{0:05d} {1:0.3f}\n'.format(int(rlzns[i]),rhovals[i])\
                         for i in xrange(Nreal)])
    f.write(bodystr)
    f.close()

def get_rho_filename(recdat,recalmdat,filetag=''):
    if filetag:
        tagstr='_'+filetag
    else:
        tagstr=''
    #recalmdat.get_mapfile(0,0,'fits')
    real0file=recalmdat.get_mapfile_fortags(0,recdat.maptag,recdat.rectag)
    recind=recalmdat.get_mapind_fromtags(recdat.maptag,recdat.rectag)
    filebase=recalmdat.get_mapfile_base(recind)
    outf=real0file.replace('/'+filebase+'/','/')
    outf=outf[:outf.rfind('.r')]
    outf=outf+tagstr+'.rho.dat'
    return outf

def read_rhodat(recdat,recalmdat,filetag=''):
    f=get_rho_filename(recdat,recalmdata,filetag)
    dat=np.loadtxt(f,skiprows=6)
    rlzns=dat[:,0]
    rho=dat[:,1]
    return rho

def read_rhodat_wfile(filename):
    print 'reading',filename
    dat=np.loadtxt(filename,skiprows=6)
    rlzns=dat[:,0]
    rho=dat[:,1]
    return rho

#compute the expected value of rho, given theoretical Cl
# If Nneighbors=-1, use all available data in Cl
#     if it is 0: set everything that's not bin-isw or bin-auto to zero
#     if it is 1: use only bin-isw, bin-auto, and bin-nearest neighbor
#                 etc.
def compute_rho_fromcl(cldat,recdat,Nneighbors=-1):
    #Dl is a matrix of Cls, with isw at zero index
    #  and other maps in order specified by recdat.includecl
    if Nneighbors>-1:
        oldcl=cldat.cl[:,:]#deep copy, hang onto original info
        oldnoisecl=cldat.noisecl[:,:]
        maptypes=[]
        binnums=[]
        for i in xrange(cldat.Nmap):
            b=cldat.bintaglist[i]
            if 'isw' in b:
                iswind=i
            maptypes.append(b[:b.rfind('_bin')])
            binnums.append(int(b[b.rfind('_bin')+4:]))
        for n in xrange(cldat.Ncross):
            i,j=cldat.crosspairs[n]
            keepthisn=False
            if maptypes[i]==maptypes[j] and np.fabs(binnums[i]-binnums[j])<=Nneighbors:
                #print 'keeping',cldat.bintaglist[i],cldat.bintaglist[j]
                keepthisn=True#keep Nneibhors neighboring bins
            elif i==iswind or j==iswind: #keep cross power with isw
                keepthisn=True
            if not keepthisn:#go through and set unnoted cross power to zero
                cldat.cl[n,:]=np.zeros(cldat.Nell)
                cldat.noisecl[n,:]=np.zeros(cldat.Nell)

        
    lmin=recdat.lmin
    Dl,dtags=get_Dl_matrix(cldat,recdat.includecl,recdat.zerotagstr)
    #print Dl[5,:,:]
    Dinv=invert_Dl(Dl)
    Nell=Dinv.shape[0]
    lvals=np.arange(Nell)
    Nl=np.zeros(Nell)
    for l in xrange(Nell):
        if Dinv[l,0,0]!=0:
            Nl[l]=1/Dinv[l,0,0]

    includel=(lvals>=lmin)
    NLSS=recdat.Nmap
    
    #for each l sum over LSS maps for numerator, the sum over l
    numell = np.zeros(lvals.size)
    for i in xrange(NLSS):
        numell+=Dinv[:,0,i+1]*Dl[:,0,i+1]
    numell*=-1*includel*Nl*(2.*lvals+1)
    numerator=np.sum(numell)

    #for sigisw, just sum over l
    sig2iswl=includel*(2.*lvals+1)*Dl[:,0,0]
    sig2isw=np.sum(sig2iswl)

    #for sigrec, sum over LSS maps 2x (ij), then l
    sig2recl=np.zeros(lvals.size)
    for i in xrange(NLSS):
        sig2recli=np.zeros(lvals.size)
        for j in xrange(NLSS):
            sig2recli+=-1*Nl*Dinv[:,0,j+1]*Dl[:,j+1,i+1]
        sig2recl+=sig2recli*(-1)*Nl*Dinv[:,0,i+1]
    sig2recl*=includel*(2.*lvals+1)
    sig2rec=np.sum(sig2recl)
    
    denom=np.sqrt(sig2isw*sig2rec)
    #print '   FINAL   num,demon:',numerator,denom
    result=numerator/denom
    if Nneighbors>-1: #set Cl back to original values
        cldat.cl=oldcl
        cldat.noisecl=oldnoisecl
    #print result
    return result
    
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

###########################################################################
# plotting functions
###########################################################################
#------------------------------------------------------------------------------
#plot_Tin_Trec  - make scatter plot comparing true to reconstructed isw
def plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,plotdir='output/',plotname='',colorlist=[]):
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
    plt.figure(1,figsize=(10,8))
    plt.title('Pixel-by-pixel scatterplot',fontsize=20)
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    plt.xlabel(r'$\rm{T}^{\rm ISW}_{\rm input}$  $(10^{-5}\rm{K})$',fontsize=20)
    plt.ylabel(r'$\rm{T}^{\rm ISW}_{\rm rec}$  $(10^{-5}\rm{K})$',fontsize=20)
    #plt.ticklabel_format(style='sci', axis='both', scilimits=(1,0))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.tick_params(axis='both', labelsize=16)
    ax.set_aspect('equal')#make it square
    
    xmax=np.max(np.fabs(iswmaps))
    plt.xlim(-1.0*xmax,1.8*xmax) 
    plt.ylim(-1.2*xmax,1.2*xmax)
    for i in xrange(len(recmaps)):
        rhoi=rhovals[i]
        coli=colors[i]
        labeli=reclabels[i]+'\n$\\rho={0:0.2f}$'.format(rhoi)
        mrec=recmaps[i]
        misw=iswmaps[i]
        plt.plot(misw,mrec,linestyle='None',marker='o',alpha=1,label=labeli,markersize=4.,color=coli,markeredgecolor='None')#markerfacecolor='None',markeredgecolor=coli
    plt.plot(10*np.array([-xmax,xmax]),10*np.array([-xmax,xmax]),linestyle='--',linewidth=4.,color='grey')

    # try doing text boxes instead of legends?
    startboxes=.7
    totlines=3*len(recmaps)
    fperline=1/20. #estimate by eye
    startheight=startboxes
    for i in range(Nmap)[::-1]:
        li=reclabels[i]+'\n$\\rho={0:0.3f}$'.format(rhovals[i])
        Nline=li.count('\n')+1#.5
        leftside=.975#.69
        textbox=ax.text(leftside, startheight, li, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'boxstyle':'round,pad=.3','alpha':1.,'facecolor':'none','edgecolor':colors[i],'linewidth':4})
        startheight-=Nline*fperline+.03
    
    #plt.show()
    #plt.gcf().subplots_adjust(bottom=0.15) #keep x label from being cut off
        
    #plotdir='output/plots_forposter/'
    if not plotname:
        plotname='TrecTisw_scatter_variousRECs'
    print 'saving',plotdir+plotname+'.png'
    plt.savefig(plotdir+plotname+'.png')
    plt.close()




