#This file contains functions that will generate maps using existing
# data of angular power spectra generated with genCrossCor.py
# It will also contain functions which can postprocesses maps by
#  applying masks or calibration error "screens." 

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib
import itertools,copy
from multiprocessing import Pool, Manager,cpu_count

from MapParams import *
from CosmParams import Cosmology
from ClRunUtils import *
from genCrossCor import *

###########################################################################
# glmData - class object containing g_lm data and info about what went
#               into it for many maps and realizations
#     contains:
#         glm - grid of actual data: indices are [realization,map,lm]
#                map = combo of original map, mod, and mask
#                      orginal map - ISW, gal, desspec, etc + bin number
#                      mod - e.g. application of calibration error screen
#                      mask - mask applied to map
#         maptaglist - list of strings describing maps used, or for 
#                  iswREC, what maps went into reconstruction
#         nbarlist - list of nbar values for each  map. -1 if no value given
#         modtaglist - 
#             for LSS: string describing any modifications to maps, e.g.
#                  calibration error "screen" applied
#                  -> should either have same len as maptaglist, or have 
#                     if len=0, set all default 'nomod', if len=1, use that
#                     tag for all maps
#             for ISW rec, a note signifying what Cl's were used
#                 for reconstruction ('fid'=same Cl used for simulation)
#         masktaglist - like modtaglist, but for masks, w default 'fullsky'
#         rlzns: - array of ints, eitehr the same size as the first dim of glm
#                  or empty.
#                    If nonempty, translates from the first index
#                     of glm to realization number in file names
#                    If empty, will default to np.arange(glm.shape[0])
#         Nreal = glm.shape[0]; number of realizations
#         rundir - directory where data is stored/ came from
#                     glm files will be in rundir/glm_output/
#                     map files will be in rundir/map_output/
#         runtag - like runtag for RunParams class; added to C_lfilename
#                     ->>should not contain '.'
#         filetags- list of strings, like mask/mod tags, refers
#                   to files where glmdata is stored for given map.
#                    if ONEFILE=True, this will have len(1)
#                     ->>should not contain '.'
#         ONEFILE - bool which allows us to quickly check of all items in
#                   filtags are equal
#         mapdict - mapdict[maptag,modtag,masktag] = index in these lists
#                   with the appropriate combo of tags; assumes no
#                   duplicates
#
#     Adding operation has been defined so that you can combine
#       these data object along either the map or rlzn axis, as long as
#       all other properties match (though it doesn't check for duplicates!)
###########################################################################
class glmData(object):
    def __init__(self,glm=np.array([]),lmax=0,maptaglist=[],runtag='',rundir='output/',nbarlist=[],modtaglist=[],masktaglist=[],rlzns=np.array([]),filetags=[]):
        self.isdummy=False
        if not glm.size: #dummy empty array; use if we just want filenames
            self.glm=np.zeros((1,len(maptaglist),hp.sphtfunc.Alm.getsize(lmax)))
            self.isdummy=True
        elif len(glm.shape)==2: #only one realization here
            self.glm=glm.reshape((1,glm.shape[0],glm.shape[1]))
        else:
            self.glm=glm

        #data about realizations
        self.Nreal = self.glm.shape[0]*(not self.isdummy) #0 if dummy
        if rlzns.size and rlzns.size!=self.Nreal:
            print "WARNING! mismatch: rlzns.size!=Nreal. defaulting to arange(Nreal)"
            self.rlzns=np.array([])
        elif np.all(rlzns==np.arange(self.Nreal)):
            self.rlzns=np.array([])
        else:
            self.rlzns=rlzns
        #----  
        #data about maps, modifications, and masks
        self.Nmap=self.glm.shape[1]
        if len(maptaglist)!=self.Nmap:
            print "WARNING! mismatch: len(maptaglist)!=Nmaps. Setting all='?'"
            self.maptaglist=['?']*self.Nmap
        else:
            self.maptaglist=maptaglist
        #nbar is avg density/str for map. used to apply noise and needed for
        #  applying calibration error screens.
        if len(nbarlist)!=self.Nmap:
            if len(nbarlist):
                print "WARNING! mismatch: len(nbarlist)!=Nmaps. Setting all=-1"
            #if zero, or mismatch, set to default
            self.nbarlist=-1*np.ones(self.Nmap)
        else:
            self.nbarlist=np.array(nbarlist)    
        #modtags describe calibration error screens or other post processing
        if len(modtaglist)!=self.Nmap:
            if len(modtaglist)==1: #if only one tag given, apply to all maps
                self.modtaglist=modtaglist*self.Nmap
            else:
                if len(modtaglist):
                    print "WARNING! mismatch: len(modtaglist)!=Nmaps. Setting all='unmod'"
                #if zero, or mismatch, set to default
                self.modtaglist=['unmod']*self.Nmap
        else:
            #if there are empty strings, replace them with the default
            for i in xrange(len(modtaglist)):
                if not modtaglist[i]:
                    modtaglist[i]='unmod'
            self.modtaglist=modtaglist
        #masktags refer to what mask is applied to mask to generate alm
        if len(masktaglist)!=self.Nmap:
            if len(masktaglist)==1:#if only one tag given, apply to all maps
                self.masktaglist=masktaglist*self.Nmap
            else:
                if len(masktaglist):
                    print "WARNING! mismatch: len(masktaglist)!=Nmaps. Setting all='fullsky'"
                #if zero, or mismatch, set to default
                self.masktaglist=['fullsky']*self.Nmap
        else:
            #if there are empty strings, replace them with the default
            for i in xrange(len(masktaglist)):
                if not masktaglist[i]:
                    masktaglist[i]='fullsky'
            self.masktaglist=masktaglist

        #set up dictionary, so that given maptag,modtag,masktaq, can get index
        #assumes no duplicates
        self.mapdict={(self.maptaglist[n],self.modtaglist[n],self.masktaglist[n]):n for n in xrange(self.Nmap)}
        #----
        self.ONEFILE=True #true if all glm data from a single file
        if len(filetags)==self.Nmap:
            self.ONEFILE=all(f==filetags[0] for f in filetags)
            if self.ONEFILE:
                self.filetags=filetags[:1]
            else:
                self.filetags=filetags
        elif len(filetags)==1: #if one filetags, applies to all maps
            self.filetags=filetags
        else: #otherewise, assume all maps have no filetag (default)
            if len(filetags):
                print "WARNING! mismatch: len(filetags)!=Nmaps. Setting all=''"
                #if zero, or mismatch, set to default (empty string)   
            self.filetags=['']
            
        #data about lm values
        self.Nlm = self.glm.shape[2]
        self.lmax=lmax

        #strings describing what data went into makeing this glm
        self.runtag=runtag
        self.rundir=rundir
    #------------------------------------------------------
    # get_glmgrid: if lmax==0 returns self.glm
    #              else, returns what self.glm would be if self.lmax=lmax
    #              -if input lmax<self.lmax, cuts out values
    #              -if input lmax>self.lmax, fills in zeros
    def get_glmgrid(self,lmax=0): 
        if lmax==0 or lmax==self.lmax:
            return self.glm
        else:
            lnew,mnew=hp.sphtfunc.Alm.getlm(lmax)
            lold,mold=self.get_hp_landm()
            Nlmnew=len(lnew)
            outglm=np.zeros((self.Nreal,self.Nmap,Nlmnew),dtype=np.complex)
            #print 'getting glmgrid, self.glm.shape',self.glm.shape #as expected
            #print 'self.glm[:,:,0].shape',self.glm[:,:,0].shape
            #print 'outglm.shape',outglm.shape
            if lmax<self.lmax: #returning fewer values than we have
                for n in xrange(Nlmnew):
                    oldind=np.intersect1d(np.where(lold==lnew[n])[0],np.where(mold==mnew[n])[0])[0]
                    #print 'n=',n,'oldind=',oldind
                    #print 'outglm[:,:,n].shape',outglm[:,:,n].shape
                    #print 'self.glm[:,:,oldind].shape',self.glm[:,:,oldind].shape
                    outglm[:,:,n]=self.glm[:,:,oldind]
            else: #lmax>self.lmax, need to leave some zeros
                for n in xrange(self.Nlm):
                    newind=np.intersect1d(np.where(lold[n]==lnew)[0],np.where(mold[n]==mnew)[0])[0]
                    outglm[:,:,newind]=self.glm[:,:,n]
                    
            return outglm
    #------------------------------------------------------
    #where are the data files? Define as func so that if rundir gets
    #  changed, these will also change accordingly
    def glmdir(self):
        return self.rundir+'glm_output/'
    def mapdir(self):
        return self.rundir+'map_output/'
    #------------------------------------------------------
    #given index of 0th dim of glm, get realization number
    def get_realnum(self,index):
        if index>self.Nreal-1:
            print "**Realization number out of range."
            return 
        elif self.rlzns.size:
            return self.rlzns[index]
        else:
            return index
    #given realization number, get the index where it appears in rlzns
    def get_realind(self,r):
        if self.havethisreal(r):
            if self.rlzns.size:
                return np.where(self.rlzns==r)[0][0]
            else: #is np.arange(nreal)
                return r
        else:
            print "***glmData doesn't contain rlz number",r
            
    #check if we have this realization number in our data
    def havethisreal(self,r):
        if self.rlzns.size:
            return r in self.rlzns
        else: #default rlzns=np.arange(Nreal)
            return 0<=r<self.Nreal
    #------------------------------------------------------
    #get two arrays with size=Nlm, containg l and m vals correspond to lm index
    def get_hp_landm(self):
        return hp.sphtfunc.Alm.getlm(self.lmax)
    #------------------------------------------------------
    def get_mapind_fromtags(self,maptag,modtag='unmod',masktag='fullsky'):
        return self.mapdict[(maptag,modtag,masktag)]
    #------------------------------------------------------
    #given map,mod,mask combo of strings, return filetag
    def get_filetag(self,maptag,modtag='unmod',masktag='fullsky'):
        if self.ONEFILE: return self.filetags[0]
        else:
            n= self.mapdict[(maptag,modtag,masktag)]
            return self.filetags[n]
    #------------------------------------------------------
    #given map index, return filetag
    def get_filetag_fromindex(self,i):
        if self.ONEFILE: return self.filetags[0]
        else:
            return self.filetags[i]
    #------------------------------------------------------
    #given realization int, and index for map (default 0 good for ONEFILE=1)
    # return name of file where that g_lm data is stored
    #   includes dir and realiation info
    def get_glmfile(self,real=0,mapind=0):
        if self.ONEFILE: ftag=self.filetags[0]
        else: ftag=self.filetags[mapind]
        fbase=self.get_glmfile_base(mapind)
        return ''.join([self.glmdir(),fbase,'/',fbase,'.r{0:05d}.dat'.format(real)])

    #given index of map, gives base of filename wehre g_lm data is stored
    # (default 0 good for ONEFILE=1)
    def get_glmfile_base(self,mapind=0):
        if self.ONEFILE: ftag=self.filetags[0]
        else: ftag=self.filetags[mapind]
        
        if ftag:  ftagstr='_'+ftag
        else:  ftagstr=''
        
        if self.runtag: runtag='.'+self.runtag
        else:  runtag=''
        
        return ''.join(['glm',ftagstr,runtag])

    #given maptag,modtag,masktag strings and realization int,
    # return name of file where that glm data is stored. 
    def get_glmfile_fortags(self,real,maptag,modtag='unmod',masktag='fullsky'):
        n= self.mapdict[(maptag,modtag,masktag)]
        return self.get_glmfile(real,n)
    #------------------------------------------------------
    #given realization int, and index for map,
    # return name of file where the corresponding map data goes
    def get_mapfile(self,real=0,mapind=0,ftype='fits'):
        fbase=self.get_mapfile_base(mapind)
        return ''.join([self.mapdir(),fbase,'/',fbase,'.r{0:05d}.'.format(real),ftype])
    
    def get_mapfile_fortags(self,real,maptag,modtag='unmod',masktag='fullsky',ftype='fits'):
        n= self.mapdict[(maptag,modtag,masktag)]
        return self.get_mapfile(real,n,ftype)

    def get_mapfile_base(self,mapind=0):
        if self.runtag: 
            runtagstr='.'+self.runtag
        else:  
            runtagstr=''
        return ''.join([self.maptaglist[mapind],'.',self.modtaglist[mapind],'.',self.masktaglist[mapind],runtagstr])

    #same as above, but doesn't check whether tags are in mapdict
    def get_mapfile_fortags_unchecked(self,real,maptag,modtag='unmod',masktag='fullsky',ftype='fits'):
        if self.runtag: 
            runtagstr='.'+self.runtag
        else:  
            runtagstr=''
        fbase=''.join([maptag,'.',modtag,'.',masktag,runtagstr])
        return ''.join([self.mapdir(),fbase,'/',fbase,'.r{0:05d}.'.format(real),ftype])

    #------------------------------------------------------
    # add_newreal: add glm data for new realizations
    #   input: newglm = array of glmdata to add [newreal,map,lm]
    #           -> requires maps and lm dimensions to have same size as self
    #          newrlzn = array with same size as newglm.shape[0]
    def add_newreal(self,newglm,newrlzns):
        if len(newglm.shape)==2:#only one real, [map,lm] axis
            newglm=newglm.reshape((1,newglm.shape[0],newglm.shape[1]))
            
        if newglm.shape[1]!=self.Nmap or newglm.shape[2]!=self.Nlm:
            print "Cannot add realizations: wrong Nmaps or Nlm!"
            return
        if newglm.shape[0]!=newrlzns.size:
            print "Cannot add realizations: new rlzns labels don't match."
            return
        #make sure we don't add duplicate realizations
        Nnewreal=newrlzns.size
        #check if current and newrlzns give a larger np.array
        #if so: leave self.rlzns as empty array
        if (not self.rlzns.size) and np.all(newrlzns==np.arange(self.Nreal,self.Nreal+Nnewreal)):
            fullglm=np.zeros((self.Nreal+Nnewreal,self.Nmap,self.Nlm),dtype=np.complex)
            fullglm[:self.Nreal,:,:]=self.glm
            fullglm[self.Nreal:,:,:]=newglm
            self.glm=fullglm
            self.Nreal=self.Nreal+Nnewreal
        #otherwise, make sure there aren't overlaps
        else:
            Nadd=Nnewreal
            doadd=np.ones(Nnewreal)
            for i in xrange(Nnewreal):
                newr=newrlzns[i]
                if self.havethisreal(newr):
                    doadd[i]=0
                    Nadd-=1
            if Nadd>0:
                fullglm=np.zeros((self.Nreal+Nadd,self.Nmap,self.Nlm),dtype=np.complex)
                fullglm[:self.Nreal,:,:]=self.glm
                fullrlzns=np.arange(self.Nreal+Nadd)
                if self.rlzns.size:
                    fullrlzns[:self.Nreal]=self.rlzns
                addind=0
                for i in xrange(Nnewreal):
                    if doadd[i]:
                        fullglm[self.Nreal+addind,:,:]=newglm[i,:,:]
                        fullrlzns[self.Nreal+addind]=newrlzns[i]
                        addind+=1
                self.rlzns=fullrlzns
                self.Nreal=fullglm.shape[0]
                if np.all(self.rlzns==np.arange(self.Nreal)):
                    self.rlzns=np.array([])
                self.glm=fullglm
    
    #------------------------------------------------------
    #can combine glmDataGrids if lm vals are the same
    #   if realizations are the same between them, add new maps
    #   if rlzns not equal but maps are, add new realizations
    # return glmData has runtag and rundir of first 
    #------------------------------------------------------
    def __add__(self,other):
        if self.lmax==other.lmax and self.Nlm==other.Nlm:
            # if realization numbers are the same, combine along map axis
            #print 'Nreal:',self.Nreal,other.Nreal
            #print 'rlzns.size:',self.rlzns.size,other.rlzns.size
            if self.Nreal==other.Nreal and self.rlzns.size==other.rlzns.size and np.all(self.rlzns==other.rlzns):
                print "combining along map axis"
                Nmaptot = self.Nmap +other.Nmap
                maptags=self.maptaglist+other.maptaglist
                nbars=np.concatenate((self.nbarlist,other.nbarlist))
                modtags=self.modtaglist+other.modtaglist
                masktags=self.masktaglist+other.masktaglist
                newglm=np.zeros((self.Nreal,Nmaptot,self.Nlm),dtype=np.complex)
                newglm[:,:self.Nmap,:]=self.glm
                newglm[:,self.Nmap:,:]=other.glm
                if self.ONEFILE and other.ONEFILE and self.filetags[0]==other.filetags[0]:
                    newfiletags=self.filetags[0]
                else:
                    if len(self.filetags)==self.Nmap:
                        selfft=self.filetags
                    elif len(self.filetags)==1:
                        selfft=self.filetags*self.Nmap
                    if len(other.filetags)==other.Nmap:
                        otherft=other.filetags
                    elif len(other.filetags)==1:
                        otherft=other.filetags*other.Nmap                       
                    newfiletags=selfft+otherft
                
                newglmData= glmData(newglm,self.lmax,maptags,self.runtag,self.rundir,nbars,modtags,masktags,self.rlzns,newfiletags)
            #if rlzns are different but map/mod/mask info the same, combine
            # to multiple blocks of realizations
            elif self.Nmap==other.Nmap and self.maptaglist==other.maptaglist and np.all(self.nbarlist==other.nbarlist) and self.modtaglist==other.modtaglist and self.masktaglist==other.masktaglist and self.filetags==other.filetags:
                print "combining along realization axis"
                Nrealtot=self.Nreal+other.Nreal
                rlznstot =np.zeros(Nrealtot)
                newglm=np.zeros((Nrealtot,self.Nmap,self.Nlm),dtype=np.complex)
                newglm[:self.Nreal,:,:]=self.glm
                newglm[self.Nreal:,:,:]=other.glm

                if self.rlzns.size:
                    rlznstot[:self.Nreal]=self.rlzns
                else:
                    rlznstot[:self.Nreal]=np.arange(self.Nreal)
                if other.rlzns.size:
                    rlznstot[self.Nreal:]=other.rlzns
                else:
                    rlznstot[self.Nreal:]=np.arange(other.Nreal)
                if np.all(rlznstot==np.arange(Nrealtot)):
                    rlznstot=np.array([])
                #print 'rlznstot=',rlznstot
                newglmData=glmData(newglm,self.lmax,self.maptaglist,self.runtag,self.rundir,self.nbarlist,self.modtaglist,self.masktaglist,rlznstot,self.filetags)
            else:
                print "No valid adding operation, returning first glmData"
                newglmData=self
            return newglmData

    #------------------------------------------------------
    # Return an instance of glmData with identical properties
    #  can change lmax; if made larger, empty spaces fill w 0
    #  If Nreal passed, the new glm will have the first Nreal real 
    #------------------------------------------------------
    def copy(self,lmax=0,Nreal=-1):
        if lmax>0 and lmax!=self.lmax:
            outglm=self.get_glmgrid(lmax=lmax)
            outlmax=lmax
            outdat= glmData(outglm,outlmax,self.maptaglist,self.runtag,self.rundir,self.nbarlist,self.modtaglist,self.masktaglist,self.rlzns,self.filetags)
        else:
            outdat=glmData(self.glm,self.lmax,self.maptaglist,self.runtag,self.rundir,self.nbarlist,self.modtaglist,self.masktaglist,self.rlzns,self.filetags)
        if Nreal>=0:
            outdat.glm=outdat.glm[:Nreal,:,:]
            outdat.Nreal=Nreal
            if outdat.rlzns.size:
                outdat.rlzns=outdat.rlzns[:Nreal]
        elif Nreal==0: #return dummy glm
            outdat.glm=np.zeros((1,len(outdat.maptaglist),outdat.Nlm))
            outdat.Nreal=0
            outdat.isdummy=1
            
        return outdat
    
###########################################################################
#================================================
# get_glm
#    retrieve glmData object either from file or by generating it
#    ->if cldata (a clData object) is given, get the rundir and runtag
#        from it. 
#          >If overwrite=False, check that files exist; read them in if so
#            but if not, generate need glm from Cl data
#    ->If cldata left as an empty string, can read in but not generate
#          glm data.
#    ->If Nreal==0, return dummy glmData object w empty glm array, 
#          might just want this for filename info
#    if array of realization numbers are given, get those
#    otherwise get the Nreal, just starting from rlz=0
#    overwrite - if False, read in glm file if it exists, compute only if needed
#                if true, recompute all glm and overwrite
#================================================
def get_glm(cldata='',rundir='output/',filetag='',runtag='',Nreal=1,rlzns=np.array([]),overwrite=False,matchClruntag=False):
    if cldata:
        rundata=cldata.rundat
        rundir = rundata.rundir
        if matchClruntag:
            runtag=rundata.tag

    if not rlzns.size:
        rlzns=np.arange(Nreal)
    else:
        Nreal=rlzns.size
        
    #set up filename template
    if filetag: ftagstr='_'+filetag
    else:  ftagstr=''
    if runtag: runtagstr='.'+runtag
    else:  runtagstr=''
    glmdir=rundir+'glm_output/'
    fbase=''.join(['glm',ftagstr,runtagstr])
    thisglmdir=glmdir+fbase+'/'
    #make dir if necesseary
    if not os.path.isdir(thisglmdir):
        print "    creating dir",thisglmdir
        os.mkdir(thisglmdir)

    # if Nreal, return dummy glmdat; no glm data, just filenames needed
    if Nreal==0:
        glmdat=glmData(glm=np.array([]),lmax=cldata.rundat.lmax,maptaglist=cldata.bintaglist,runtag=runtag,rundir=cldata.rundat.rundir,filetags=[filetag],nbarlist=cldata.nbar)
        return glmdat

    if overwrite: #generate all these realizations of glm
        if cldata:
            glmdat=generate_many_glm_fromcl(cldata,Nreal,rlzns,filetag=filetag,matchClruntag=matchClruntag,runtag=runtag)
        else:
            print "NEED cldata in order to generate glm!"
    else: #generate glm for the realizations num w/out existing file
        need=[]
        for r in rlzns:
            if not os.path.isfile(''.join([thisglmdir,fbase,'.r{0:05d}.dat'.format(r)])):
                need.append(r)
        #compute if need not empty, don't worry about hanging onto result
        if need:
            if cldata:
                print "Generating glm for realizations:",need
                generate_many_glm_fromcl(cldata,rlzns=np.array(need),filetag=filetag,matchClruntag=matchClruntag,runtag=runtag)
            else:
                print "NEED cldata in order to generate glm!"
        else:
            print "All data already exists. Reading from file."
        #now, read in files and collect glm data
        glmdat=read_glm_files(filetag,runtag,rundir,Nreal,rlzns)
    return glmdat


#================================================
# generate_glm - given cldata and info about realizations, generate glmData
# input:
#    cldata - ClData object (from genCrossCor.py)     
#    Nreal - number of realizations; set to rlzns.size if that is !=0
#    rlzns -array of ints; if empty, set to np.arange(Nreal)
#    savedat - if true, write the generated glm to file
#    filetag - filetag for output glmData, labels saved files
# returns glmData object w mod/masktags set to default 'nomod','fullsky'
#================================================
def generate_many_glm_fromcl(cldata,Nreal=1,rlzns=np.array([]),savedat=True,filetag='',matchClruntag=True,runtag=''):
    wasarange=False
    if not rlzns.size:
        wasarange=True
        rlzns = np.arange(Nreal)
    Nreal=rlzns.size
    print "Generating {0:d} realizations of glm".format(Nreal)
    #generate tthe glm data
    glmlist=[]
    for r in rlzns:
        glmlist.append(generate_glmdat_fromcl(cldata,r,savedat,filetag,False,matchClruntag=matchClruntag,runtag=runtag))
    glmgrid=np.array(glmlist)
    if wasarange:
        rlzns=np.array([])
    if matchClruntag:
        runtag=cldata.rundat.tag
    glmdata=glmData(glm=glmgrid,lmax=cldata.rundat.lmax,maptaglist=cldata.bintaglist,runtag=runtag,rundir=cldata.rundat.rundir,rlzns=rlzns,filetags=[filetag],nbarlist=cldata.nbar)
    if savedat:
        print "writing to files:",glmdata.get_glmfile()[:-9]+'XXXXX.dat'

    return glmdata

#================================================
# generate_glm_fromcl - for a single realization, use healpy to generate
#    appropriately corrleated set of glm.
# input:
#    cldata - ClData object (from genCrossCor.py)     
#    rlz - int, realization number
#    savedat - if True, save glm data to file
#    filetag - string, filetag for glmdata; 
#    retglmData - if true returns a glmData object, otherwise just glm array
#================================================
def generate_glmdat_fromcl(cldata,rlz=0,savedat=True,filetag='',retglmData=True,matchClruntag=True,runtag=''):
    #need contiguous lvals starting at 0 to use healpy.synalm
    rundat=cldata.rundat
    # rlz is the realization number of this generation; just for labeling
    if rundat.lvals.size!=rundat.lmax+1:
        print "**STOPPING! Need contiguous lvals for healpy to give glm"
        return
    if matchClruntag:
        runtag=cldata.rundat.tag
    bintaglist=cldata.bintaglist
        
    #need match between Ncross and Nmaps
    Ncross=cldata.cl.shape[0]
    Nmaps = len(bintaglist)
    if Ncross!=Nmaps*(Nmaps+1)/2:
        print "***STOPPING! # of C_l not consistant with # of maps."
        return
    glmgrid=np.array(hp.synalm(cldata.cl+cldata.noisecl,new=True))#[map,lm]
    if retglmData or savedat:
        #modtag and masktag are defaults 'nomod' and 'fullsky'
        glmdat=glmData(glm=glmgrid,lmax=rundat.lmax,maptaglist=bintaglist,runtag=runtag,rundir=rundat.rundir,rlzns=np.array([rlz]),filetags=[filetag],nbarlist=cldata.nbar)
    #save the data
    if savedat:
        glmdat= write_glm_to_files(glmdat,setnewfiletag=True,newfiletag=filetag)
    if retglmData: #return glmData object
        return glmdat 
    else:
        return glmgrid #just return [map][lm] array


#------------------------------------------------------------------------
# Here glmdata is a glmData object
#      #will write a unique file for each realization
# if setnewfiletag: change filetag for all glmdata to newfiletag
# then, write to file only if glmdata has ONEFILE=True
def write_glm_to_files(glmdata,setnewfiletag=False,newfiletag=''):
    #print glmdata.glm
    if setnewfiletag:
        glmdata.filetags=[newfiletag]
        glmdata.ONEFILE=True
    if not glmdata.ONEFILE:
        print "glmdata.ONEFILE=False: CANNOT WRITE."
        return ''
    lmax=glmdata.lmax
    Nmap = glmdata.Nmap
    Nlm=  glmdata.Nlm
    hplvals,hpmvals = glmdata.get_hp_landm()

    #get length of longest tag string for nice header formatting
    maxlen=10 #make it at least 10 to hold nbar values
    maxlen=max(maxlen,len(max(glmdata.maptaglist,key=len)))
    maxlen=max(maxlen,len(max(glmdata.modtaglist,key=len)))
    maxlen=max(maxlen,len(max(glmdata.masktaglist,key=len)))

    #set up header strings with info about maps/mods/masks
    indheader = ''.join(['mapinds: ',''.join([' map{0:<{1}d}'.format(n,maxlen-3) for n in xrange(Nmap)]),'\n'])
    mapheader = ''.join(['maps:    ',''.join([' {0:{1}s}'.format(glmdata.maptaglist[n],maxlen) for n in xrange(Nmap)]),'\n'])
    modheader = ''.join(['mods:    ',''.join([' {0:{1}s}'.format(glmdata.modtaglist[n],maxlen) for n in xrange(Nmap)]),'\n'])
    maskheader = ''.join(['masks:   ',''.join([' {0:{1}s}'.format(glmdata.masktaglist[n],maxlen) for n in xrange(Nmap)]),'\n'])
    nbarheader = ''.join(['nbar:    ',''.join([' {0:{1}.3e}'.format(glmdata.nbarlist[n],maxlen) for n in xrange(Nmap)]),'\n'])
    colheader = ' {0:>8s} {1:>3s} {2:>3s}'.format('hp.lm','l','m')+''.join([' {0:>18.18s}.real {0:>18.18s}.imag'.format('map{0:<d}'.format(m)) for m in xrange(Nmap)])+'\n'

    #check that output dir exists
    fbase=glmdata.get_glmfile_base()
    outdir=glmdata.glmdir()+fbase+'/'
    if not os.path.isdir(outdir):
        print "    creating dir",outdir
        os.mkdir(outdir)

    #loop through realizations and write appropriate data to file
    outflist=[]
    #print "writing to files:",glmdata.get_glmfile()[:-9]+'XXXXX.dat'
    for r in xrange(glmdata.Nreal):
        realnum=glmdata.get_realnum(r)
        header0=''.join(['lmax=',str(lmax),'; rlzn=',str(realnum),'; glmfiletag=',glmdata.filetags[0],'; runtag=',glmdata.runtag,'\n'])
        outf=glmdata.get_glmfile(glmdata.get_realnum(r))
        outflist.append(outf)
        rowlist = [''.join([' {0:8d} {1:3d} {2:3d}'.format(n,hplvals[n],hpmvals[n]),''.join([' {0:+23.16e} {1:+23.16e}'.format(glmdata.glm[r,m,n].real,glmdata.glm[r,m,n].imag) for m in xrange(Nmap)]),'\n']) for n in xrange(Nlm)]
        bodystr=''.join(rowlist)
        #print 'writing to', outf
        f=open(outf,'w')
        f.write(header0)
        f.write(indheader)
        f.write(mapheader)
        f.write(modheader)
        f.write(maskheader)
        f.write(nbarheader)
        f.write(colheader)
        f.write(bodystr)
        f.close()
    return glmdata #return this, since filetags have changed
#------------------------------------------------------------------------
# given rundir, runtag, filetag, rlzns array, read all appropriate files
#  and return a glmData object
def read_glm_files(filetag='',runtag='',rundir='output/',Nreal=1,rlzns=np.array([])):
    if not rlzns.size:
        rlzns=np.arange(Nreal)
    else:
        Nreal=rlzns.size
    #set up filename template
    if filetag: ftagstr='_'+filetag
    else:  ftagstr=''
    if runtag: runtagstr='.'+runtag
    else:  runtagstr=''
    base=''.join(['glm',ftagstr,runtagstr])
    fbase=''.join([rundir+'glm_output/',base,'/',base])

    #read the first file and get its header data
    f0= ''.join([fbase,'.r{0:05d}.dat'.format(rlzns[0])])
    glmdat=read_glm_onefile(f0)
    newglm=np.zeros((Nreal-1,glmdat.Nmap,glmdat.Nlm),dtype=np.complex)
    print "Reading glm from :",f0[:-9]+'XXXXX.dat'
    for r in xrange(1,Nreal):
        fname=''.join([fbase,'.r{0:05d}.dat'.format(rlzns[r])])
        newglm[r-1,:,:]=read_glm_onefile(fname,False)
    glmdat.add_newreal(newglm,rlzns[1:])
    return glmdat
    
#------------------------------------------------------------------------
# Read a glm data file
#   if getheaderinfo=True: uses filename and first few lines to get all
#                          data needed to make glmData object; returns glmData
#   if getheaderinfo=Fasle: just reads out array of numbers
#                          returns np.array of imaginary glm[map,lm]
def read_glm_onefile(filename,getheaderinfo=True):
    #print "Reading glm from ",filename
    if getheaderinfo:
        rundir=filename[:filename.find('glm_output/')]
        #open the file to get some of the header data
        f=open(filename,'r')
        header0=f.readline()
        indheader=f.readline()
        mapheader=f.readline()
        modheader=f.readline()
        maskheader=f.readline()
        nbarheader=f.readline()
        f.close()
        #parse header strings
        split0=header0.split(';')
        lmax=int(split0[0][5:])
        realnum=int(split0[1][6:])
        filetag=split0[2][12:]
        runtag=split0[3][8:-1]
        
        maptaglist=mapheader.split()[1:]
        modtaglist=modheader.split()[1:]
        masktaglist=maskheader.split()[1:]
        nbarlist=[float(n) for n in nbarheader.split()[1:]]

    #read in data
    gdat = np.loadtxt(filename,skiprows=7)[:,3:]#[lm,map(real and imag alternate)]
    realdat=gdat[:,::2]
    imagdat=gdat[:,1::2]
    glmdat= np.transpose(realdat+np.complex(0,1)*imagdat) #[map,lm]

    if getheaderinfo:
        return glmData(glm=glmdat,lmax=lmax,maptaglist=maptaglist,runtag=runtag,rundir=rundir,nbarlist=nbarlist,modtaglist=modtaglist,masktaglist=masktaglist,rlzns=np.array([realnum]),filetags=[filetag])
    else:
        return glmdat
#------------------------------------------------------------------------
# get_maps_from_glm
#   given a glmData object, use healpy to make maps
#     if rlzns empty, do so for all realizations stord in glmdata
#      otherwise, do only for realization numbers both in rlzns and glmdat.rlzns
#     if redofits=True, will write over existing .fits map files
#       otherwise, will check if the file exists and will just read
#       the map in if it already exists
#     if makeplots=True, will make png files of the maps
#     Can set NSIDE for the healpix maps, default is 32
#
#  returns array w shape Nreal,Nmap,Npix containing helpy maps for each real+map
#------------------------------------------------------------------------
def get_maps_from_glm(glmdata,rlzns=np.array([]),redofits=False,makeplots=False,NSIDE=32):
    if not rlzns.size:
        rlzns=glmdata.rlzns
    if not rlzns.size:
        rlzns=np.arange(glmdata.Nreal)
    
    #check for output directories
    fbases=[glmdata.get_mapfile_base(i) for i in xrange(glmdata.Nmap)]
    for i in xrange(glmdata.Nmap):
        thismapdir=glmdata.mapdir()+fbases[i]+'/'
        #print 'checking for ',thismapdir
        if not os.path.isdir(thismapdir):
            print "    creating dir",thismapdir
            os.mkdir(thismapdir)

    mapgrid=[]
    for r in rlzns:
        mapgrid.append(get_maps_from_glm_1real(glmdata,r,redofits,makeplots,NSIDE,checkdir=False))
    mapgrid=np.array(mapgrid)
    return mapgrid
#------------------------------------------------------------------------
# get_maps_from_glm_1real - uses healpy to get healpix maps for all maps
#      associated with a given realization number
#    input: glmdat - glmData object
#           rlz - realization number, int; will check that glmdat has this real.
#           redofits - bool. if false, reads in data from exsiting files
#                            if true, overwrites any files with matching name
#           makeplot - if true, also makes a plot of map and puts it in a png file
#                      isw maps have red/blue colors, other maps are greyscale
#           NSIDE- the healpix Nside parameter. default is 32 here
#    returns: array of heaplix maps, shape is rlzns,Nmaps,Npix
#------------------------------------------------------------------------
def get_maps_from_glm_1real(glmdat,rlz=0,redofits=False,makeplots=False,NSIDE=32,checkdir=True):
    #print 'in get_maps_from_glm_1real, rlz=',rlz
    #check that realization number is actually in glmdat
    if not glmdat.havethisreal(rlz):
        print "***No map! glmData does not contain realization number",rlz
        return
    realind=glmdat.get_realind(rlz)

    if checkdir:
        fbases=[glmdat.get_mapfile_base(i) for i in xrange(glmdat.Nmap)]
        for i in xrange(glmdat.Nmap):
            thismapdir=glmdat.mapdir()+fbases[i]+'/'
            if not os.path.isdir(thismapdir):
                print "    creating dir",thismapdir
                os.mkdir(thismapdir)
    #get file names for the various maps and loop through them
    outflist = [glmdat.get_mapfile(rlz,i,'fits') for i in xrange(glmdat.Nmap)]
    hpmaplist=[]
    for i in xrange(glmdat.Nmap):
        maptag=glmdat.maptaglist[i]
        modtag=glmdat.modtaglist[i]
        masktag=glmdat.masktaglist[i]
        mapfname = outflist[i]

        if redofits or not os.path.isfile(mapfname):
            #healpy complains if array isn't C contiguous, do this to avoid errors
            #  (errors don't always appear without, may depend on size)
            contigglm=np.ascontiguousarray(glmdat.glm[realind,i,:])
            m = hp.sphtfunc.alm2map(contigglm,NSIDE,verbose=False)
            #m = hp.sphtfunc.alm2map(glmdat.glm[realind,i,:],NSIDE,verbose=False)
            #print 'Writing ',mapfname
            hp.write_map(mapfname,m)
        else:
            #print "Reading existing file ",mapfname
            m=hp.read_map(mapfname,verbose=False)
        hpmaplist.append(m)
        if makeplots:
            plotfname=glmdat.get_mapfile(rlz,i,'png')
            plotmap(m,plotfname,rlz,maptag,modtag,masktag,glmdat.runtag)
    return np.array(hpmaplist) #shape=Nmap,Npix
#------------------------------------------------------------------------
# plotmap - plot map, hanging onto various map/mod/mask info
def plotmap(m,outfname,rlz,maptag,modtag='unmod',masktag='fullsky',titlenote=''):
    #variance=np.var(m)
    maxval=max(np.fabs(m))
    #nsig=6
    scalemax=0.7*maxval#nsig*variance
    scalemin=-0.7*maxval
    plt.figure(0)
    if titlenote:
        titletag=' ('+titlenote+')'
    else: titletag=''
    maptitle=''.join([maptag,'.',modtag,'.',masktag,titletag,', rlzn ',str(rlz)])
    if 'isw' in maptag:
        hp.mollview(m,title=maptitle,min=scalemin,max=scalemax)
    else:#for dm and galmaps, use light for high density, black for low
        mono_cm=matplotlib.cm.Greys_r
        mono_cm.set_under("w") #set background to white
        hp.mollview(m,title=maptitle,min=scalemin,max=scalemax,cmap=mono_cm)
    #print 'Writing plot to',plotfname
    plt.savefig(outfname)
    plt.close()
 
#------------------------------------------------------------------------
# get_fixedvar_errors_formaps
#   given a glmData object, get calibration error maps which are to be combined
#   with appropriate maps, save calibration error maps to .fits files
# input:
#   glmData - glmData object; glm data not actually used, but mapnames,
#             realizations, and rundir info are used for saving.
#   cdatatuple - list of info about which maps should get calib error maps
#                constructed for them, and what stats that error maps has
#                ->though calib maps don't are indep of gal maps, should gen
#                  a unique calib error for each map+real 
#                ->list entries should be tuples (mapstr,fixedvar,clmax,shape)
#                  where mapstr is a string labeling maps
#                  fixedvar gives the variance of the caliberror to make
#                  and clmax gives max l of calib error map (defaults used
#                  if tuple only contains string).
#                  if shape not give, assume 'g' (gaussian), can also pass 'l2' 
#                --->note; calib error map will be generated for each map+real
#                    containing the mapstr; ie, putting 'gal' will apply to 
#                    'gal_bin0', 'galA_bin0', 'gal_bin1',etc
#   overwrite - if True, will write new fits files for all maps
#               if False, checks to see if fits files already exist, if so 
#               uses those; only makes maps when needed
#  returns: for now, nothing. as long as apply_caliberror_toglm uses same
#           naming convention as here, it will open up files
#------------------------------------------------------------------------
def get_fixedvar_errors_formaps(glmdat,cdatalist=[],overwrite=False,NSIDE=32):
    #print 'in get_fixedvar...; glmdat.glm.shape=',glmdat.glm.shape
    rundir=glmdat.rundir
    outdir=glmdat.mapdir()
    outtuplelist=[]
    #print 'outdir:',outdir
    #print 'cdatalist=',cdatalist
    for cdat in cdatalist:
        if type(cdat)==str: #if only string, do this to avoid confusion
            cdat=(cdat,)
        #print cdat
        mapstr=cdat[0]
        nentries=len(cdat)
        if nentries>1:
            cvar=cdat[1]
            #print 'cvar=',cvar
            if nentries>2:
                clmax=cdat[2]
                #print 'clmax=',clmax
                if nentries>3:
                    shape=cldat[3]
                    if nentries>4:
                        width=cldat[4] #sets gaussian width, does nothing to l2
                    else:
                        width=10.
            else: #defaults
                clmax=30
                shape='g'
                width=10.
                #print 'clmax=default=',clmax
        else: #defaults
            cvar=0.1
            clmax=30
            shape='g'
            width=10.
            #print 'cvar=default=',cvar
            #print 'clmax=default2=',clmax
        #collect maptags for which we're generating calib errs
        dothesemaps=[]
        for t in glmdat.maptaglist:
            if (mapstr in t) and (t not in dothesemaps): 
                dothesemaps.append(t)
        #get realization numbers for which we're generating calib errors
        if glmdat.rlzns.size:
            dothesereal=glmdat.rlzns
        else:
            dothesereal=np.arange(glmdat.Nreal)

        if shape=='l2':
            modtag='l2_var{0:.2e}_maxl{1:d}'.format(cvar,clmax)
        elif shape=='g': #currently default width
            modtag='g{1:d}_var{0:.2e}_maxl{2:d}'.format(cvar,int(width),clmax)
        #print 'USING MODTAG',modtag

        #loop through maps and realizations, generating calib maps
        for m in dothesemaps:
            outtuplelist.append((m,modtag)) #for now, assumes no mask
            outbase='caliberror.{0:s}.for_{1:s}'.format(modtag,m)  
            thisoutdir=outdir+outbase+'/'
            if not os.path.isdir(thisoutdir):
                os.mkdir(thisoutdir)
            for r in dothesereal:
                outname=outbase+'.r{0:05d}.fits'.format(r)        
                #only bother generating map if overwrite or file nonexistant
                if overwrite or not os.path.isfile(thisoutdir+outname):
                    if shape=='l2':
                        cmap=gen_error_map_fixedvar_l2(cvar,clmax,NSIDE)
                    elif shape=='g':
                        cmap=gen_error_map_fixedvar_gauss(cvar,clmax,width=width,NSIDE=NSIDE)
                    #write to file
                    hp.write_map(thisoutdir+outname,cmap)
    #return list of map/mod/mask tuples 
    # which can be given to apply_caliberror_toglm
    #print 'outtuplelist',outtuplelis
    #print ' exiting get_fixedvar...; glmdat.glm.shape=',glmdat.glm.shape
    return outtuplelist
            

#------------------------------------------------------------------------
# gen_error_cl_fixedvar_l2
#   variance of calibration error field is fixed to sig2 for 1<l<=caliblmax
#   and calib error is assumed to have l^-2 spectrum
#------------------------------------------------------------------------
def gen_error_cl_fixedvar_l2(sig2=0.1,caliblmax=30,lmin=1):
    invnorm=0 #clcal=norm/l^2, 
    clcal=np.zeros(caliblmax+1)
    for l in xrange(lmin,caliblmax+1):#find using rel between variance and C_l
        invnorm+=(2*l+1.)/(4*np.pi*l*l)
    norm =sig2/invnorm
    #print 'norm=',norm
    for l in xrange(lmin,caliblmax+1):
        clcal[l]=norm/(l*l)#/(2*l+1.)
    return clcal

def getmodtag_fixedvar_l2(sig2,caliblmax):
    modtag='l2_var{0:.2e}_maxl{1:d}'.format(sig2,caliblmax)
    return modtag

def parsemodtag_fixedvar_l2(ctag): #ctag = modtag
    aftervar=ctag.find('_var')+4
    premax=ctag.find('_max')
    variance=float(ctag[aftervar,premax])
    aftermaxl=premax+5
    maxl=int(ctag[aftermaxl:])
    return variance, maxl
    
def gen_error_map_fixedvar_l2(sig2=0.1,caliblmax=30,NSIDE=32):
    modtag=getmodtag_fixedvar_l2(sig2,caliblmax)
    invnorm=0 #clcal=norm/l^2, 
    clcal=gen_error_cl_fixedvar_l2(sig2,caliblmax)
    #now generate map
    cmap=hp.sphtfunc.synfast(clcal,NSIDE,verbose=False)
    return cmap

#------------------------------------------------------------------------
# gen_calib_error_map_fixedvar_gauss
#   variance of calibration error field is fixed to sig2 for 1<l<=caliblmax
#   and calib error is assumed to have exp(-(l/width)**2) spectrum
# this looks like the Clcal shown in figures 5 and 6 in the 'calib errors unleashed'
#------------------------------------------------------------------------
def gen_error_cl_fixedvar_gauss(sig2=0.1,caliblmax=30,lmin=0,width=10.):
    invnorm=0 #clcal=norm/l^2, 
    clcal=np.zeros(caliblmax+1)
    for l in xrange(lmin,caliblmax+1):#find using rel between variance and C_l
        invnorm+=(2*l+1.)*np.exp(-(l/width)**2)/(4*np.pi)
    norm =sig2/invnorm
    #print 'norm=',norm
    for l in xrange(lmin,caliblmax+1):
        clcal[l]=norm*np.exp(-(l/width)**2)
    return clcal

def getmodtag_fixedvar_gauss(sig2,width,caliblmax):
    modtag='g{1:d}_var{0:.2e}_maxl{2:d}'.format(sig2,int(width),caliblmax)
    return modtag

def parsemodtag_fixedvar_l2(ctag): #ctag = modtag
    aftervar=ctag.find('_var')+4
    premax=ctag.find('_max')
    variance=float(ctag[aftervar,premax])
    aftermaxl=premax+5
    maxl=int(ctag[aftermaxl:])
    preunderscore=aftervar-4
    width=float(ctag[1:preunderscore])
    return variance, maxl,width

def gen_error_map_fixedvar_gauss(sig2=0.1,caliblmax=20,lmin=0,width=10.,NSIDE=32):
    modtag=getmodtag_fixedvar_gauss(sig2,width,caliblmax)
    invnorm=0 #clcal=norm/l^2, 
    clcal=gen_error_cl_fixedvar_gauss(sig2,caliblmax,lmin,width)
    #now generate map
    cmap=hp.sphtfunc.synfast(clcal,NSIDE,verbose=False)
    return cmap

#------------------------------------------------------------------------
# apply_additive_caliberror_tocl - for computing <rho> with calib errors
#   input: cldat - ClData object containing ISW and galaxy maps
#          mapmodcombos - list of (maptag,masktag) tuples
#                        where each maptag  only appears once
#
# neglects multiplicative errors
# assumes epsilon propto c_00
# assumes calibration error maps are uncorrelated with each other and galaxies
#------------------------------------------------------------------------
def apply_additive_caliberror_tocl(cldat,mapmodcombs=[]):
    Nmap=cldat.Nmap
    Nell=cldat.Nell
    
    #initialize
    calcl=np.zeros((Nmap,Nell))#should have same number of entries as maps in cldat
    newnbarlist=cldat.nbarlist[:]
    
    #go through mapmod combos, generate Clcal and put it in the appropriate place in calcl
    for c in mapmodcombos:
        mtag=c[0]#maptag
        ctag=c[1]#modtag
        #find index of maptag
        mapind=-1
        for i in xrange(Nmap):
            if mtag==cldat.bintaglist[i]:
                mapind=i
                break
        if mapind==-1:
            print 'map tag not found:',mtag
        else: #parse modtag and get calib error Cl
            if ctag[:2]=='l2':#power law
                var,maxl=parsemodtag_fixedvar_l2(ctag)
                thiscalcl=gen_error_cl_fixedvar_l2(var,maxl)
            elif ctag[:1]=='g':#gaussian
                var,maxl,width=parsemodtag_fixedvar_gauss(ctag)
                thiscalcl=gen_error_cl_fixedvar_gauss(var,maxl,width=width)
            else:
                print "modtag not recognized:",ctag
            #put this cal cl into clcal grid
            thisNell=thiscalcl.size
            calcl[i,:thisNell]=thiscalcl
            #working here
    #epsilon parameter tells us how nbar changes; includign only c00 contrib
    epsilon=cacl[:,0]/np.sqrt(4*np.pi) #is zero if no Clcal input
    newnbarlist=cldat.nbarlist*(1.+epsilon) #if not gal, eps=0, so nbar=-1 still
    
    #make copy of cl data
    outcl=cldat.cl[:,:]
    crosspairs=cldat.crosspairs #[crossind,mapinds] 
    crossinds=cldat.crossinds #[mapind,mapind]
    Ncross=cldat.Ncross
    
    #go through all cross pairs and add appropriate calib error modifications
    for n in Ncross:
        i,j=crosspairs[n]
        if i==j:
            outcl[n,:]+=calcl[i,:] #additive power from calib error auto power
        outcl[n,0]+=-1*calcl[i,0]*calcl[j,0] #from some of the epsilon terms 
        outcl[n,:]/=(1.+epsilon[i])*(1.+epsilon[j]) #no mod if epsilon small

    #creat outcldata object with new outcl and nbar
    outcldat=ClData(copy.deepcopy(cldat.rundat),cldat.bintaglist,docrossind=cldat.docross,nbarlist=newnbarlist)
    return outcldat

#------------------------------------------------------------------------
# apply_caliberror_tomap
#   input: inmap - healpy map array of some given map of N(nhat)/Nbar-1
#          cmap - healpy map array of c(nhat) calib error map
#          innbar - average counts per steradian of input map
#------------------------------------------------------------------------
def apply_caliberror_tomap(inmap,cmap,innbar):
    Nin=innbar*(inmap+1.) #total number count in each direction
    Nobs=(1.+cmap)*Nin #this is how the calibration error map is defined
    outnbar=np.average(Nobs)
    outmap=Nobs/outnbar - 1.
    return outmap,outnbar

#------------------------------------------------------------------------
# apply_calibrerror_toglm
#  input: inglmdat - glmData object
#         mapmodcombos - list of tuples [(maptag,modtag,masktag),...]
#                 where masktag is optional
#         savemaps - if true, the fits file of the combined maps are saved
#         saveplots - if true, plots of the maps are saved as png files
#         newglmfile - if empty string, add new maps to inglmdat, save it all
#                      together, otherwise save just new glm w this as filetag
#  for each pair, if that map/mod/mask combo not already in inglmdata
#  find the glm for that map + mask combo (default mask='fullsky') and
#  apply the calibration error modification associated with modtag to it 
#  to get new map and thus new glm.
# Add new glmdata to glmData, return that
#------------------------------------------------------------------------
def apply_caliberror_toglm(inglmdat,mapmodcombos=[],savemaps=False,saveplots=False,newglmfiletag=''):
    #print 'in apply calerror_toglm, glm shape:',inglmdat.glm.shape
    #for each mapmodcombo, check whether it is already in glmdat
    newmaptags=[]
    newmodtags=[]
    newmasktags=[]
    newnbarlist=[] #to be filled once we start combining maps
    newglm=[] #to be filled once we start combining maps [real,map,lm]
    #print 'mapmodcombos:',mapmodcombos
    for c in mapmodcombos:
        #print 'on c=',c
        # if type(c)==str: #if only string, assume this is the maptag
        #     mapmod=(c,'unmod','fullsky')
        # elif len(c)==1:
        #     mapmod=(c[0],'unmod','fullsky')
        if len(c)==2: #masktag is default
            mapmod=(c[0],c[1],'fullsky')
            #print 'mapmod=',mapmod
        else:
            mapmod=c
            #print '  here for c=',c
        if mapmod in inglmdat.mapdict:
            #print '   already have this one:',mapmod
            continue #we already have added this mod
        else:
            #print '   adding:',mapmod
            newmaptags.append(mapmod[0])
            newmodtags.append(mapmod[1])
            newmasktags.append(mapmod[2])
    print 'len(newmaptags)',len(newmaptags)
    #for each new modification, get the map and mod maps, combine them
    # for each realization contained in inglmdat
    if inglmdat.rlzns.size:
        reals=inglmdat.rlzns
    else:
        reals=np.arange(inglmdat.Nreal)
    realcount=0 #use this so we only add to newnbar once
    
    #For each realization and map/mod combo, get unmod map and appropriate
    # calib error map, then combine them
    #****For now, assumes approp calib error map exists
    print "applying calib erros to maps for",len(reals),' realizations'
    for r in reals:
        glmforr=[]
        for c in xrange(len(newmaptags)):
            #print 'r=',r,': applying',newmodtags[c],'to',newmaptags[c]
            n=inglmdat.mapdict[(newmaptags[c],'unmod',newmasktags[c])] #index of unmodified map

            startmapf=inglmdat.get_mapfile(r,n)
            startmap=hp.read_map(startmapf,verbose=False)
            startnbar=inglmdat.nbarlist[n]

            #read in calib error map 
            calibmapf=inglmdat.mapdir()+'caliberror.{0:s}.for_{1:s}.r{2:05d}.fits'.format(newmodtags[c],newmaptags[c],r)        
            calibmap=hp.read_map(calibmapf,verbose=False)
            newmap,newnbar=apply_caliberror_tomap(startmap,calibmap,startnbar)
            if savemaps:
                newmapf= inglmdat.get_mapfile_fortags_unchecked(r,newmaptags[c],newmodtags[c],newmasktags[c])
                hp.write_map(newmapf,newmap)
            if saveplots:
                plotfname=inglmdat.get_mapfile_fortags_unchecked(r,newmaptags[c],newmodtags[c],newmasktags[c],'png')
                print 'saving to', plotfname
                plotmap(newmap,plotfname,r,newmaptags[c],newmodtags[c],newmasktags[c],inglmdat.runtag)
            if realcount==0:
                newnbarlist.append(newnbar)
            modglm=hp.map2alm(newmap) #lmax won't necessarily be the same
                     #for NSIDE=32, lmax=3*32-1=95; should fill in the 
                     # last five as zero
            glmforr.append(modglm) 
        realcount+=1
        newglm.append(glmforr)
    newglm=np.array(newglm)
    #print 'newglm.shape',newglm.shape 
    newNlm=newglm.shape[2]
    newlmax=hp.Alm.getlmax(newNlm)
    newglmdat_uncheckedlmax=glmData(glm=newglm,lmax=newlmax,maptaglist=newmaptags,runtag=inglmdat.runtag,rundir=inglmdat.rundir,nbarlist=newnbarlist,modtaglist=newmodtags,masktaglist=newmasktags,rlzns=reals,filetags=[newglmfiletag])

    #get lmax to match
    newglmdat=newglmdat_uncheckedlmax.copy(inglmdat.lmax)
    #print 'inglmdat.glm.shape',inglmdat.glm.shape
    #print 'newglmdat.glm.shape',newglmdat.glm.shape
    outglm=inglmdat+newglmdat #should have same nreals; 
    #print 'outglm.glm.shape',outglm.glm.shape
    #print 'newglmdfiletag:',newglmfiletag
    if newglmfiletag: #save new glmdata in its own file
        write_glm_to_files(newglmdat) #should all be one filetag already
    else:
        #add to input glmdata, save with filetag in 0th index
        oldfiletag=inglmdat.filetags[0]
        outglm=write_glm_to_files(outglm,setnewfiletag=True,newfiletag=oldfiletag)
    return outglm
