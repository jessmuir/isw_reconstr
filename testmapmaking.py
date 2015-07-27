import numpy as np
import matplotlib.pyplot as plt
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *
from AnalysisUtils import *
from scipy.integrate import quad,romberg
from scipy.interpolate import interp1d
from scipy.special import sph_jn
import time 

#---------------------------------------------------
def biastest(z):
    return 1.

def dndztest(z):
    z0=.32
    alpha=.36
    return (z/z0)**(alpha)*np.exp(-alpha*z/z0)

#---------------------------------------------------
def cltest():
    # generated the ilk data for this somewhat accidentally
    REDODATA=0
    zbins = np.array([.01,5.,1.])
    sig = .05
    nbar=1.e8

    iswtest = MapType('isw',np.array([.01,3.]),True,False,'A test ISW map')
    #mattest = MapType('mat',zbins,False,False,'A test DM map')
    galtesta = SurveyType('galA', np.array([.01,5.,1.]),sig,nbar,dndztest,biastest,'A test galaxy map, two bins')
    galtestb = SurveyType('galB', np.array([.01,1.]),sig,nbar,dndztest,biastest,'A test galaxy map, one bin')
    galtestc=  SurveyType('galC', np.array([.01,1.]),sig,nbar/1.e2,dndztest,biastest,'A test galaxy map, one noisy bin')

    ktest = KData(kmin=1.e-2,kmax=1,nperlogk=10,krcut=1)
    outdir='test_output/test/'
    clrun = ClRunData('test',rundir = outdir,kdata=ktest,lmax=30,zmax=5.)

    maplist=iswtest.binmaps+galtesta.binmaps+galtestb.binmaps+galtestc.binmaps
    maptags=get_bintaglist(maplist)
    pairs=['all']
    cldat=getCl(maplist,clrun,pairs,DoNotOverwrite=1,redoIlk=0)

    glmdat=get_glm(cldat,Nreal=3,overwrite=0)
    #print 'glmdat.nbarlist',glmdat.nbarlist
    #print glmdat.glm.shape

    if 0: #testing copying and changing lmax
        glmdatlmax2=glmdat.copy(2)
        glmdatlmax2.filetags=['lmax2']
        #print glmdatlmax2.shape
        write_glm_to_files(glmdatlmax2)
        glmdatlmax50=glmdat.copy(50)
        glmdatlmax50.filetags=['lmax50']
        #print glmdatlmax50.shape
        write_glm_to_files(glmdatlmax50)        

    get_maps_from_glm(glmdat,redofits=0,makeplots=0)#real,map,pix

    if 0: #testing calib error map generation/application
        calinfolist=[('gal')]
        availmapmods=get_fixedvar_errors_formaps(glmdat,calinfolist)
        print availmapmods
        apply_caliberror_toglm(glmdat,availmapmods,savemaps=1,saveplots=1,newglmfiletag='calibtest')

    if 0: #testing estimator covariance matrix selection/inversion
        #make estimator using only galB info #(later do gal A w two bins)
        #includetags=['galA_bin0','galA_bin1']
        includetags=['galB_bin0']
        rectag='iswREC_B'
        #rectag='iswREC_A'
        D,dtags=get_Dl_matrix(cldat,includetags)
        #print D
        print 'D.shape',D.shape
        Dinv=invert_Dl(D)
        #print Dinv
        print 'Dinv.shape',Dinv.shape
        # for l in xrange(D.shape[0]): #test the inversion
            # print '-----'
            # print 'ELL=',l
            # print D[l]
            # print
            # print Dinv[l]
            # print
            # print np.dot(D[l],Dinv[l]) #good enough; small or 0 off-diags
        #passing just includetags should get all unmod maps
        dglmgrid,dinds=get_glm_array_forD(glmdat,includetags)
        print 'dglmgrid.shape',dglmgrid.shape

        #
        iswalmdat=calc_isw_est(cldat,glmdat,includetags,rectag,makeplots=True)
        print 'isw glm shape:',iswalmdat.glm.shape

    if 0: #test cl combination
        print 'original',cldat.bintaglist
        print cldat.nbar
        #newcldat=combineCl_twobin(cldat,'galA_bin0','galA_bin1','galA_bin01')
        #print 'new',newcldat.bintaglist
        #print newcldat.nbar
        newcl2=combineCl_binlist(cldat,['galA_bin0','galA_bin1','galB_bin0','galC_bin0'],'gal_binall')
        print 'new2',newcl2.bintaglist
        print newcl2.nbar
#################################################################
if __name__=="__main__":
    cltest()
