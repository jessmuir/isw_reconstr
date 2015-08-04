from scipy.integrate import quad
#from scipy.interpolate import interp1d
from scipy.special import erfc
import numpy as np
import itertools
#import copy

#=========================================================================
# get_binmaplist - given list of MapTypes, return a 1d list of all binmaps
#=========================================================================
def get_binmaplist(maptypelist):
    listof_binlists=[maptype.binmaps for maptype in maptypelist]
    listof_taglists=[maptype.bintags for maptype in maptypelist]
    binmaplist = list(itertools.chain(*listof_binlists))
    bintaglist = list(itertools.chain(*listof_taglists))
    return binmaplist,bintaglist

def get_bintaglist(binmaplist):
    #given list of binmaps, return list of their tags
    return [map.tag for map in binmaplist]

def smoothedtophat(x,zminnom,zmaxnom,edge=0.001):
    width=(zmaxnom-zminnom)
    sig=edge*width
    return 0.5*(np.tanh((x-zminnom)/sig)+np.tanh((zmaxnom-x)/sig))

#The classes in this file contain info describing the (mainly radial)
# properties of differnt map types, to be used in computing C_l correlation fns
###########################################################################
# Map - general parent class containing info about a given map
###########################################################################
class MapWrapper(object):
    def __init__(self,idtag,isISW=False,isGal=False,longtag=''):
        self.tag = idtag #string identifier for map, used for filenames
        if longtag:
            self.longtag = longtag #more verbose string id describing map
        else:
            self.longtag = idtag
        #what type of map is this? if not isISW and not isGal, is DM
        self.isISW = isISW 
        self.isGal = isGal 

###########################################################################
# MapType - class containing info about a given map type we want to make
#    Holds info about redshift binning, tag for how file should be saved
#    Galaxy survey maps should use child class SurveyType
#    While ISW and default maps can just use this one
#    ->Note that having isGal=isISW=False does not mean this is a DM map; it just
#      gives a map with a flat dn/dz number count distribution, with no noise;
#      DM window function should be modulated by Hr^2 to tranlate volume to projected density
###########################################################################
class MapType(MapWrapper):
    def __init__(self,idtag,zedges,isISW=False,isGal=False,longtag='',epsilon=1.e-10,sharpness=.01):
        self.epsilon = epsilon #no function except for SurveyType, but included here
                               # so SurveyType and MapType can be created w same syntax
        MapWrapper.__init__(self,idtag,isISW,isGal,longtag)
        self.sharpness=sharpness #how sharp is edge of tophat edges?
        self.zedges=np.array(zedges) #array of redshift values for bin edges
        self.Nbins = self.zedges.size-1 #number of redshift bins
        self.bintags=[''.join([idtag,'_bin',str(n)]) for n in xrange(self.Nbins)]
        if self.isISW:
            self.typestr='ISW'
        elif self.isGal:
            self.typestr='gal'
        else:
            self.typestr='mat'
        if self.longtag != self.tag:
            self.longstr=', {0:s}'.format(self.longtag)
        else: 
            self.longstr=''
        
        if not isGal: #if isGal, this will be run from SurveyType init
            #put together infostr for data file headers
            
            self.infostr = 'Map type {0:s}: tag={1:s}, z-bins {2:s}, sharpness={3:f}, {4:s}'.format(self.typestr,self.tag,self.zedges,self.sharpness,self.longstr)
            #make binmaps
            edgesharpness=sharpness
            #sig=edgesharpness*(zedges[-1]-zedges[0])
            #EDITED zmin/max on 5/8
            zmin = max(0.001,self.zedges[0]-5*edgesharpness*(zedges[1]-zedges[0]))
            zmax = self.zedges[-1]+5*edgesharpness*(zedges[-1]-zedges[-2])
            self.binmaps=[]
            for n in xrange(self.Nbins):
                zminnom = self.zedges[n]
                zmaxnom = self.zedges[n+1]
                #sig=edgesharpness*(zedges[1]-zedges[0])
                #make surveybinmap
                b = BinMap(self.bintags[n],n,zmin,zmax,zminnom,zmaxnom,sharpness,isISW,isGal,self.infostr,epsilon)
                self.binmaps.append(b)
        
###########################################################################
# SurveyParams - Map params for a galaxy survey map.
#      bins, redshift uncertainty, bias, total average source density
#      When initiaized, go through bins and generate new normal
###########################################################################
def nobias(z):
    return 1.

def flatdndz(z):
    return 1.

class SurveyType(MapType):
    # note that dndzargs, biasargs are any argus those fns have other than z
    #   z must be the first argumetn, other should be in order
    def __init__(self,idtag,zedges,sigz0=.01,nbar=1.e9,dndz=flatdndz,bias=nobias,dndzargs=[],biasargs=[],longtag='',epsilon=1.e-10,sharpness=.01,addnoise=False,sigcut=5.,sigatzmin=False,sigatzmax=False):
        isISW=False
        isGal=True
        MapType.__init__(self,idtag,zedges,isISW,isGal,longtag,epsilon=epsilon,sharpness=sharpness)
        self.sigz0 = sigz0 #redshift uncertainty, softens bin edges of Nbins>1
        self.nbar = nbar #avg sources/steradian 
        self.dndz = dndz #function of z, not nec normalized
        self.dndzargs=dndzargs
        self.bias = bias #function of z
        self.biasargs=biasargs
        #put together infostr for data file headers
        self.sigcut=sigcut #how many sig(z) past edge of bin to keep?
        self.sigatzmin=sigatzmin #do we smooth dndz at zmax?
        self.sigatzmax=sigatzmax #do we smooth dndz at zmax?
        self.infostr = 'Map type {0:s}: tag={1:s}, nbar={2:0.4g}, sigz0={3:0.3g}, z-bins= {4:s}, sharpness={5:f}, sigcut={6:f}, {7:s}'.format(self.typestr,self.tag,self.nbar,self.sigz0,self.zedges,self.sharpness,self.sigcut,self.longstr)

        #set up bin maps
        self.binmaps=[]
        #since bins have smoothed edges, go beyond nominal zmin, max
        if self.sigatzmin:
            zmin = max(0.,self.zedges[0]-sigcut*sigz0)
        else:
            zmin=zedges[0]
        if self.sigatzmax:
            zmax = self.zedges[-1]+sigcut*sigz0*(1+zedges[-1])
        else:
            zmax=zedges[-1]
            
        fullint = quad(lambda z: dndz(z,*self.dndzargs),zmin,zmax,epsabs=self.epsilon,epsrel=self.epsilon)[0]      
        #fullint = romberg(lambda z: dndz(z),zmin,zmax,tol=self.epsilon)
        for n in xrange(self.Nbins):
            zminnom = self.zedges[n]
            zmaxnom = self.zedges[n+1]
            binzmin= max(zmin,zminnom-sigcut*sigz0*(1.+zminnom))
            binzmax= min(zmax,zmaxnom+sigcut*sigz0*(1.+zmaxnom))
            #what fraction of the avg source counts go here?
            binint = quad(lambda z: self.F(n,z)*self.dndz(z,*self.dndzargs),binzmin,binzmax,epsabs=self.epsilon,epsrel=self.epsilon)[0]
            sourcefrac=binint/fullint
            binnbar = sourcefrac*nbar
            b=SurveyBinMap(idtag=self.bintags[n],binnum=n,zmin=binzmin,zmax=binzmax,zminnom=zminnom,zmaxnom=zmaxnom,nbar=binnbar,sigz0=sigz0,dndz=self.dndz,bias=self.bias,dndzargs=self.dndzargs,biasargs=self.biasargs,maptypeinfostr=self.infostr,sharpness=self.sharpness,epsilon=self.epsilon,addnoise=addnoise)
            self.binmaps.append(b)

    def sigz(self,z): #assuming some form of z dependence...
        return self.sigz0*(1.+z)

    def F(self,n,z): #function describing how photo-z erros soften bin edges
        minedge = self.zedges[n]
        maxedge = self.zedges[n+1]
            
        if self.sigz0==0:
            result= smoothedtophat(z,minedge,maxedge)#float((z>=minedge)*(z<maxedge)) #tophat fn
        else: #inside full z range, finite z uncertainty
            sigzroot2 = self.sigz(z)*np.sqrt(2.)
            lowerpart=.5*erfc((minedge-z)/sigzroot2)
            upperpart=-.5*erfc((maxedge-z)/sigzroot2)
            result= lowerpart+upperpart
        return result
            
###########################################################################
# BinMap - map params for one bin of a map type
#        has an IDstr used for filenames, bool ISGAL,ISISW
#        IDStr for surveytype, bin number,
#        if ISGAL, initialize by normalizing dndz and combining with bias
#        window function which combines bias with dndz, or is just 1 for ISW
###########################################################################
class BinMap(MapWrapper):
    def __init__(self,idtag,binnum,zmin,zmax,zminnom,zmaxnom,sharpness=0.001,isISW=False,isGal=False,maptypeinfostr='',epsilon=1.e-10):
        MapWrapper.__init__(self,idtag,isISW,isGal,maptypeinfostr)
        self.typetag = self.tag[:self.tag.find('_bin')]
        self.binnum = binnum
        self.zmin = zmin #lower edge of bin (below, window goes to 0)
        self.zmax = zmax #upper edge of bin (above, window goes to 0)
        self.zminnom=zminnom #no difference for sharp bin edges
        self.zmaxnom=zmaxnom
        self.sharpness=sharpness
        self.epsilon=epsilon
        #infostr for data file headers. maptypinfostr should be of full map
        self.binint=1.
        if not isGal:
            self.infostr='{0:s}, z[min,minnom,maxnom,max]=[{1:0.3g} {2:0.3g} {3:0.3g} {4:0.3g} ]; {5:s}'.format(idtag,self.zmin,self.zminnom,self.zmaxnom,self.zmax,maptypeinfostr)

            self.binint =quad(lambda z: self.window(z),zmin,zmax,epsabs=self.epsilon,epsrel=self.epsilon)[0]
            self.nbar=-1.

    def window(self,x): #smoothed tophat
        width=(self.zmaxnom-self.zminnom)/2.
        sig=width*self.sharpness
        result = 0.5*(np.tanh((x-self.zminnom)/sig)+np.tanh((self.zmaxnom-x)/sig))
        if not self.isISW: #matter and number counts get this normalization, but ISW doesn't
            result =result/self.binint
        return result


###########################################################################
# SurveyBinMap - inherits from binmap, survey specific initialization
#        has an IDstr used for filenames, bool ISGAL,ISISW
#        IDStr for surveytype, bin number,
#        if ISGAL, initialize by normalizing dndz and combining with bias
#        window function which combines bias with dndz, or is just 1 for ISW
###########################################################################
class SurveyBinMap(BinMap):
    def __init__(self,idtag,binnum,zmin,zmax,zminnom,zmaxnom,nbar,sigz0,dndz,bias,dndzargs=[],biasargs=[],maptypeinfostr='',sharpness=0.001,epsilon=1.e-10,addnoise=True):
        isGal=True
        isISW=False
        self.dndz=dndz#copy.deepcopy(dndz)
        self.dndzargs=dndzargs
        self.bias=bias#copy.deepcopy(bias)
        self.biasargs=biasargs
        self.sigz0=sigz0
        self.nbar = nbar #source/steradian
        BinMap.__init__(self,idtag,binnum,zmin,zmax,zminnom, zmaxnom,sharpness,isISW,isGal,maptypeinfostr)
        self.infostr='{0:s}, nbar={1:0.4g}, z[min,minnom,maxnom,max]=[{2:0.3g} {3:0.3g} {4:0.3g} {5:0.3g}]; {6:s}'.format(self.tag,self.nbar,self.zmin,self.zminnom,self.zmaxnom,self.zmax,maptypeinfostr)

        self.binint=1.
        self.binint =quad(lambda z: self.F(z)*self.dndz(z,*self.dndzargs),zmin,zmax,epsabs=self.epsilon,epsrel=self.epsilon)[0]
        self.addnoise=addnoise


    def window(self,z):
        #bin n's window function is normalized F*dndz times bias
        result= self.bias(z,*self.biasargs)*self.F(z)*self.dndz(z,*self.dndzargs)
        return result/self.binint 
        
    def sigz(self,z): #assuming some form of z dependence...
        return self.sigz0*(1.+z)

    def F(self,z): #function describing how photo-z erros soften bin edges
        if self.sigz0==0: #tophat
            width=(self.zmaxnom-self.zminnom)/2.
            sig=width*self.sharpness
            result = 0.5*(np.tanh((z-self.zminnom)/sig)+np.tanh((self.zmaxnom-z)/sig))
        else: #inside full z range, finite z uncertainty
            sigzroot2 = self.sigz(z)*np.sqrt(2.)
            lowerpart=.5*erfc((self.zminnom-z)/sigzroot2)
            upperpart=-.5*erfc((self.zmaxnom-z)/sigzroot2)
            result= lowerpart+upperpart
        return result

