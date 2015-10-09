import numpy as np
import matplotlib.pyplot as plt
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from genMapsfromCor import *
from AnalysisUtils import *
from scipy.integrate import quad,romberg
from binningtest_utils import *
import copy

def make_deslikeplots(destype='spec',Nreal=1,Nrealmaps=1,REDOGLM=True,makeplots=False,DOISWEST=True,dofid=True,doOneBin_zmax1=False,doOneBin_zmax5=False,do6bins=False,do5binz1_calpt1=False,do5binz1_calpt01=False,do5binz1_calpt001=False,do1binz1_calpt1=False,do1binz1_calpt01=False,do1binz1_calpt001=False,doindividualREC=False,NSIDE=32):
    ADDINNBAR=0 #old version of code had no nbar in cl files, this will add it
    outdir='output/'

    if destype=='spec':
        sigma=.001
        runtag='desspec6bin'
        maptypetag='desspec'
    elif destype=='photo':
        sigma=.05
        runtag='desphoto6bin'
        maptypetag='desphoto'
    elif destype=='both':
        iswmaps=get_fullISW_MapType(zmax=10).binmaps
        photomaps=get_DESlike_SurveyType(.05,'desphoto').binmaps
        specmaps=get_DESlike_SurveyType(.001,'desspec').binmaps
        maps=iswmaps+specmaps+photomaps
        runtag='desall6bin'
    if destype=='photo' or destype=='spec':
        iswmaps=get_fullISW_MapType(zmax=10).binmaps
        desmaps=get_DESlike_SurveyType(sigma,maptypetag).binmaps
        maps=desmaps+iswmaps
    #===================
    #set up Cl data and read it in
    rundat=get_generic_rundat(outdir=outdir,tag=runtag,noilktag=True)
    if ADDINNBAR:
        nbarlist=[m.nbar for m in maps]
    maptags=get_bintaglist(maps)
    pairs=['all']    
    #read in C_l
    print 'maps',maps,rundat.tag
    cldat=getCl(maps,rundat,dopairs=pairs,DoNotOverwrite=True)
    if ADDINNBAR:
        cldat.nbar=nbarlist
        writeCl_file(cldat)
    if destype=='both':
        maptypes=['desspec','desphoto']
    else:
        maptypes=[maptypetag]
    #------------------------
    #get cl data for when we lump the des bins together
    # one bin, zmax=1
    if doOneBin_zmax1:
        #combine all but last des-like bin
        for maptypetag in maptypes:
            combinethese=[]
            for i in xrange(5):
                combinethese.append(maptypetag+'_bin'+str(i))
            combotagz1=maptypetag+'1binz1_bin0'
            comboruntagz1=''#maptypetag+'1binz1'
            keeporig=dofid
            if keeporig:
                comboruntagz1=''
            else:
                comboruntagz1=maptypetag+'1binz1'
            cldat_1binz1 = combineCl_binlist(cldat,combinethese,combotagz1,comboruntagz1,keeporig)
            #remove bin5 for easier glm handling
            if not keeporig:
                cldat_1binz1.deletemap(maptypetag+'_bin5')
            if keeporig:
                #origruntag=cldat.rundat.tag
                cldat=copy.deepcopy(cldat_1binz1)
                #cldat.rundat.tag=origruntag
    #------------------------
    #one bin, zmax=5
    if doOneBin_zmax5:
        for maptypetag in maptypes:
            #combine all but last des-like bin
            combinethesez5=[]
            for i in xrange(6):
                combinethesez5.append(maptypetag+'_bin'+str(i))
            combotagz5=maptypetag+'1binz5_bin0'
            keeporig=dofid 
            if keeporig:
                comboruntagz5=''
            else:
                comboruntagz5=maptypetag+'1binz5'
            cldat_1binz5 = combineCl_binlist(cldat,combinethesez5,combotagz5,comboruntagz5,keeporig)
            if keeporig:
                cldat=copy.deepcopy(cldat_1binz5)
    #===================
    #generate glm realizations, make maps if requested
    maprlzns=np.arange(Nrealmaps)
    glmdat=get_glm(cldat,Nreal=Nreal,overwrite=REDOGLM)
    if dofid:
        get_maps_from_glm(glmdat,maprlzns,redofits=REDOGLM,makeplots=makeplots,NSIDE=NSIDE)
        
    if doOneBin_zmax1 and not dofid:
        glmdat_1binz1=get_glm(cldat_1binz1,Nreal=Nreal,overwrite=REDOGLM)
        get_maps_from_glm(glmdat_1binz1,maprlzns,redofits=REDOGLM,makeplots=makeplots,NSIDE=NSIDE)
    elif dofid:
        glmdat_1binz1=glmdat
    if doOneBin_zmax5 and not dofid:
        glmdat_1binz5=get_glm(cldat_1binz5,Nreal=Nreal,overwrite=REDOGLM)
        get_maps_from_glm(glmdat_1binz5,maprlzns,redofits=REDOGLM,makeplots=makeplots,NSIDE=NSIDE)
    elif dofid:
        glmdat_1binz5=glmdat

    #if we want to add calib errors, do so 
    availmapmods=[(glmdat.maptaglist[n],glmdat.modtaglist[n],glmdat.masktaglist[n]) for n in xrange(glmdat.Nmap)]

    if do5binz1_calpt1: #5 bins, with 0.1 var calib field added to each bin
         #if dofid and others shared in glmdat, this handles all         
        calinfolist=[('des',0.1)] #apply to all des-related maps
        dothesemods=get_fixedvar_errors_formaps(glmdat,cdatalist=calinfolist,overwrite=REDOGLM,NSIDE=NSIDE)
        availmapmods+=dothesemods
        #print availmapmods
        glmdat=apply_caliberror_toglm(glmdat,dothesemods,savemaps=1,saveplots=makeplots,newglmfiletag='calpt1')
        #mapmods_5binz1_calpt1=availmapmods
    
    if do5binz1_calpt01:#5 bins, with 0.01 var calib field added to each bin
        calinfolist=[('des',0.01)] #apply to all des-related maps
        dothesemods=get_fixedvar_errors_formaps(glmdat,cdatalist=calinfolist,overwrite=REDOGLM,NSIDE=NSIDE)
        availmapmods+=dothesemods
        glmdat=apply_caliberror_toglm(glmdat,dothesemods,savemaps=1,saveplots=makeplots,newglmfiletag='calpt01')

    if do5binz1_calpt001:#5 bins, with 0.01 var calib field added to each bin
        calinfolist=[('des',0.001)] #apply to all des-related maps
        dothesemods=get_fixedvar_errors_formaps(glmdat,cdatalist=calinfolist,overwrite=REDOGLM,NSIDE=NSIDE)
        availmapmods+=dothesemods
        glmdat=apply_caliberror_toglm(glmdat,dothesemods,savemaps=1,saveplots=makeplots,newglmfiletag='calpt001')

    if do1binz1_calpt1 and not dofid: #1 bin, zmax=1, with 0.1 var calib field added 
        calinfolist=[('des',0.1)] #apply to all des-related maps
        dothesmods=get_fixedvar_errors_formaps(glmdat_1binz1,cdatalist=calinfolist,overwrite=REDOGLM,NSIDE=NSIDE)
        availmapmods+=dothesemods
        glmdat_1binz1=apply_caliberror_toglm(glmdat_1binz1,dothesemods,savemaps=1,saveplots=makeplots,newglmfiletag='1binz1_calpt1')
    elif dofid:
        glmdat_1binz1=glmdat
        
    if do1binz1_calpt01 and not dofid:#1 bin, zmax=1, with 0.1 var calib field added
        calinfolist=[('des',0.01)] #apply to all des-related maps
        dothesemods=get_fixedvar_errors_formaps(glmdat_1binz1,cdatalist=calinfolist,overwrite=REDOGLM,NSIDE=NSIDE)
        availmapmods+=dothesemods

        glmdat_1binz1=apply_caliberror_toglm(glmdat_1binz1,dothesemods,savemaps=1,saveplots=makeplots,newglmfiletag='1binz1_calpt01')
    elif dofid:
        glmdat_1binz=glmdat

    if do1binz1_calpt001 and not dofid:#1 bin, zmax=1, with 0.1 var calib field added
        calinfolist=[('des',0.001)] #apply to all des-related maps
        dothesemods=get_fixedvar_errors_formaps(glmdat_1binz1,cdatalist=calinfolist,overwrite=REDOGLM,NSIDE=NSIDE)
        availmapmods+=dothesemods

        glmdat_1binz1=apply_caliberror_toglm(glmdat_1binz1,dothesemods,savemaps=1,saveplots=makeplots,newglmfiletag='1binz1_calpt001')
    elif dofid:
        glmdat_1binz=glmdat


        #mapmods_1binz1_calpt01=availmapmods
    #print availmapmods
    #===================
    #get desired isw estimators
    if DOISWEST:
        for maptypetag in maptypes:
            if dofid: #5bins, zmax=1 no calib error
                fidtuples=[t for t in availmapmods if ((maptypetag+'_bin' in t[0]) and ('bin5' not in t[0]) and t[1]=='unmod')]
                #print 'ON FID MAP: USING',fidtuples
                recnote=maptypetag+'_5binz1'
                calc_isw_est(cldat,glmdat,includelist=fidtuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)
                    
            if do6bins: #6 bins, zmax=5, no calib error
                recnote=maptypetag+'_6binz5'
                maptuples=[t for t in availmapmods if ((maptypetag+'_bin' in t[0]) and t[1]=='unmod')]
                #print 'ON do6bins: USING',maptuples
                calc_isw_est(cldat,glmdat,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)

            if doOneBin_zmax1: #1 bin, zmax=1, no calib error
                recnote=maptypetag+'_1binz1'
                maptuples=[t for t in availmapmods if ((maptypetag+'1binz1_bin0'==t[0]) and t[1]=='unmod')]
                #print 'ON 1binz1: USING',maptuples
                calc_isw_est(cldat_1binz1,glmdat_1binz1,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)

            if doOneBin_zmax5: #1 bin, zmax=5, no calib error
                recnote=maptypetag+'_1binz5'
                maptuples=[t for t in availmapmods if ((maptypetag+'1binz5_bin0'==t[0]) and t[1]=='unmod')]
                #print 'ON 1binz5: USING',maptuples
                calc_isw_est(cldat_1binz5,glmdat_1binz5,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)

            if do5binz1_calpt1: #5 bins, with 0.1 var calib field added to each bin
                recnote=maptypetag+'_5binz1_calpt1'
                maptuples=[t for t in availmapmods if ((maptypetag+'_bin' in t[0]) and ('bin5' not in t[0]) and t[1]=='fixvar0.1_maxl20')]
                #print 'ON 5binz1_calpt1: USING',maptuples
                calc_isw_est(cldat,glmdat,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)
            
            if do5binz1_calpt01:#5 bins, with 0.1 var calib field added to each bin
                maptuples=[t for t in availmapmods if ((maptypetag+'_bin' in t[0]) and ('bin5' not in t[0]) and t[1]=='fixvar0.01_maxl20')]
                #print 'ON 5binz1_calpt01: USING',maptuples
                recnote=maptypetag+'_5binz1_calpt01'
                calc_isw_est(cldat,glmdat,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)

            if do5binz1_calpt001:#5 bins, with 0.1 var calib field added to each bin
                maptuples=[t for t in availmapmods if ((maptypetag+'_bin' in t[0]) and ('bin5' not in t[0]) and t[1]=='fixvar0.001_maxl20')]
                recnote=maptypetag+'_5binz1_calpt001'
                calc_isw_est(cldat,glmdat,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)
            
            if do1binz1_calpt1: #1 bin, zmax=1, with 0.1 var calib field added
                recnote=maptypetag+'_1binz1_calpt1'
                maptuples=[t for t in availmapmods if ((maptypetag+'1binz1_bin0'==t[0]) and t[1]=='fixvar0.1_maxl20')]
                #print 'ON 1binz1_calpt1: USING',maptuples
                calc_isw_est(cldat_1binz1,glmdat_1binz1,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)

            if do1binz1_calpt01:#1 bin, zmax=1, with 0.01 var calib field added
                recnote=maptypetag+'_1binz1_calpt01'
                maptuples=[t for t in availmapmods if ((maptypetag+'1binz1_bin0'==t[0]) and t[1]=='fixvar0.01_maxl20')]
                #print 'ON 1binz1_calpt1: USING',maptuples
                calc_isw_est(cldat_1binz1,glmdat_1binz1,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)

            if do1binz1_calpt001:#1 bin, zmax=1, with 0.001 var calib field 
                recnote=maptypetag+'_1binz1_calpt001'
                maptuples=[t for t in availmapmods if ((maptypetag+'1binz1_bin0'==t[0]) and t[1]=='fixvar0.001_maxl20')]
                calc_isw_est(cldat_1binz1,glmdat_1binz1,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)

            if doindividualREC: #reconstruct isw for each bin individually
                for i in xrange(6):
                    recnote=maptypetag+'_bin'+str(i)
                    maptuples=[t for t in availmapmods if ((t[0]==recnote) and t[1]=='unmod')]

                    calc_isw_est(cldat,glmdat,includelist=maptuples,rectag='iswREC',recnote=recnote,makeplots=makeplots,NSIDE=NSIDE)    
#===========================================
def makemaps_separate_spec_and_photo(Nreal=1,Nrealmaps=1):
    NSIDE=64
    makeplots=01
    REDOGLM=01
    DOISWEST=1
    dofid=01 
    doOneBin_zmax1=01 
    doOneBin_zmax5=01 
    do6bins=01 
    do5binz1_calpt1=01 
    do5binz1_calpt01=01
    do5binz1_calpt001=01 
    do1binz1_calpt1=01
    do1binz1_calpt01=01
    do1binz1_calpt001=01
    doindividualREC=01 
    
    make_deslikeplots(destype='spec',Nreal=Nreal,Nrealmaps=Nrealmaps,REDOGLM=REDOGLM,makeplots=makeplots,DOISWEST=DOISWEST,dofid=dofid,doOneBin_zmax1=doOneBin_zmax1,doOneBin_zmax5=doOneBin_zmax5,do6bins=do6bins,do5binz1_calpt1=do5binz1_calpt1,do5binz1_calpt01=do5binz1_calpt01,do5binz1_calpt001=do5binz1_calpt001,do1binz1_calpt1=do1binz1_calpt1,do1binz1_calpt01=do1binz1_calpt01,do1binz1_calpt001=do1binz1_calpt001,doindividualREC=doindividualREC,NSIDE=NSIDE)

    make_deslikeplots(destype='photo',Nreal=Nreal,Nrealmaps=Nrealmaps,REDOGLM=REDOGLM,makeplots=makeplots,DOISWEST=DOISWEST,dofid=dofid,doOneBin_zmax1=doOneBin_zmax1,doOneBin_zmax5=doOneBin_zmax5,do6bins=do6bins,do5binz1_calpt1=do5binz1_calpt1,do5binz1_calpt01=do5binz1_calpt01,do5binz1_calpt001=do5binz1_calpt001,do1binz1_calpt1=do1binz1_calpt1,do1binz1_calpt01=do1binz1_calpt01,do1binz1_calpt001=do1binz1_calpt001,doindividualREC=doindividualREC,NSIDE=NSIDE)

#------------------------------------------------------------------------------
def makemaps_correlated_spec_and_photo(Nreal=1,Nrealmaps=1):
    NSIDE=64
    makeplots=01
    REDOGLM=01
    DOISWEST=01
    dofid=01 
    doOneBin_zmax1=01 
    doOneBin_zmax5=01 
    do6bins=01 
    do5binz1_calpt1=01 
    do5binz1_calpt01=01
    do5binz1_calpt001=01 
    do1binz1_calpt1=01
    do1binz1_calpt01=01
    do1binz1_calpt001=01
    doindividualREC=01 
    
    make_deslikeplots(destype='both',Nreal=Nreal,Nrealmaps=Nrealmaps,REDOGLM=REDOGLM,makeplots=makeplots,DOISWEST=DOISWEST,dofid=dofid,doOneBin_zmax1=doOneBin_zmax1,doOneBin_zmax5=doOneBin_zmax5,do6bins=do6bins,do5binz1_calpt1=do5binz1_calpt1,do5binz1_calpt01=do5binz1_calpt01,do5binz1_calpt001=do5binz1_calpt001,do1binz1_calpt1=do1binz1_calpt1,do1binz1_calpt01=do1binz1_calpt01,do1binz1_calpt001=do1binz1_calpt001,doindividualREC=doindividualREC,NSIDE=NSIDE)

#------------------------------------------------------------------------------
#plot_Tin_Trec  - make scatter plot comparing true to reconstructed isw
def plot_Tin_Trec(iswmapfiles,recmapfiles,reclabels,r=0):
    colors=['#404040','#c51b8a','#0571b0','#66a61e','#ffa319','#d7191c']
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
    print rhovals
    plt.figure(1,figsize=(10,8))
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    plt.xlabel(r'$\rm{T}^{\rm ISW}_{\rm input}$  $(10^{-5}\rm{K})$',fontsize=20)
    plt.ylabel(r'$\rm{T}^{\rm ISW}_{\rm rec}$  $(10^{-5}\rm{K})$',fontsize=20)
    #plt.ticklabel_format(style='sci', axis='both', scilimits=(1,0))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.tick_params(axis='both', labelsize=18)
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

    #plt.subplots_adjust(right=.8)
    #plt.legend(loc='lower right',ncol=1,bbox_to_anchor=(1.,.5),numpoints=1,prop={'size':10})
    #leg=plt.legend(loc='lower right',ncol=1,numpoints=1,prop={'size':16},fancybox=True, framealpha=0.)
    #leg.draw_frame(False)
    # try doing text boxes instead of legends?
    startboxes=.7
    totlines=19
    fperline=1/25. #estimate by eye
    startheight=startboxes
    for i in range(Nmap)[::-1]:
        li=reclabels[i]+'\n$\\rho={0:0.3f}$'.format(rhovals[i])
        Nline=li.count('\n')+1
        leftside=.975#.69
        textbox=ax.text(leftside, startheight, li, transform=ax.transAxes, fontsize=15,verticalalignment='top', ha='right',multialignment      = 'left',bbox={'boxstyle':'round,pad=.3','alpha':1.,'facecolor':'none','edgecolor':colors[i],'linewidth':4})
        #bb=textbox.get_bbox_patch()
        #bb.set_edgewidth(4.)
        startheight-=Nline*fperline+.03
    
    #plt.show()
    plotdir='output/plots_forposter/'
    plotname='TrecTisw_scatter_variousRECs.r{0:05d}'.format(r)
    print 'saving',plotdir+plotname
    plt.savefig(plotdir+plotname+'_small.png',dpi=300)
    plt.savefig(plotdir+plotname+'.png',dpi=900)
    #plt.savefig(plotdir+plotname+'.svg', format='svg',dpi=1200)
    plt.close()

#################################################################
def make_TTscaatterplot_forposter(r=0):
    #r=realization number
    rstr='r{0:05d}'.format(r)
    #make scatter plot
    mapdir='output/map_output/'
    trueisw='isw_bin0.unmod.fullsky.desall6bin.'+rstr+'.fits'

    recfiles=[\
              'iswREC.desspec_5binz1_calpt001.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_1binz1.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_5binz1.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desphoto_5binz1.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_1binz5.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_6binz5.fullsky.desall6bin.'+rstr+'.fits'\
    ]
    iswfiles=[trueisw]*len(recfiles)
    reclabels=[\
               '5 $\\rm z$-bins (spec): $\\rm{{z}}_{{\\rm max}}=1$\nCalib. error var. $10^{{-3}}$',\
               '1 $\\rm z$-bin (spec):  $\\rm{{z}}_{{max}}=1$',\
               '5 $\\rm z$-bins (spec): $\\rm{{z}}_{{max}}=1$',\
               '5 $\\rm z$-bins (photo): $\\rm{{z}}_{{max}}=1$',\
               '1 $\\rm z$-bin (spec):  $\\rm{{z}}_{{max}}=5$',\
               '6 $\\rm z$-bins (spec): $\\rm{{z}}_{{max}}=5$'\
    ]
    plot_Tin_Trec([mapdir+iswfile for iswfile in iswfiles],[mapdir+recfile for recfile in recfiles],reclabels,r)

#===========================================
def make_binplots():
    plotdir='output/plots_forposter/'
    desphoto=get_DESlike_SurveyType(.05,'desphoto')
    desspec=get_DESlike_SurveyType(.001,'desspec')
    specmaps=desspec.binmaps
    photomaps=desphoto.binmaps
    Nbins=len(specmaps)

    zmax=1.7#max(zmaxlist)
    nperz=1000 #need a high number here for vertical edges of spec bins
    #nperz=500
    zgrid=np.arange(nperz*zmax)/float(nperz)
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']

    plt.figure(0)
    plt.rcParams['axes.linewidth'] =2
    ax=plt.subplot()
    ax.set_yticklabels([])
    plt.xlabel('Redshift z',fontsize=20)
    plt.ylabel('Source distribution (arbitrary units)',fontsize=20)
    plt.ylim(0.001,1.)
    plt.xlim(0,zmax)
    ax.tick_params(axis='x', labelsize=18)
    for n in xrange(Nbins):
        sm=specmaps[n]
        wgridspec=sm.window(zgrid)*sm.nbar/1.e9
        plt.fill_between(zgrid,0,wgridspec, facecolor=colors[n], alpha=.5)
        plt.plot(zgrid,wgridspec,color=colors[n],linestyle='-',linewidth=2)
    for n in xrange(Nbins):
        pm=photomaps[n]
        wgridphoto=pm.window(zgrid)*pm.nbar/1.e9
        plt.fill_between(zgrid,0,wgridphoto, facecolor='white',edgecolor='none',linewidth=2,alpha=1.)
    for n in xrange(Nbins):
        pm=photomaps[n]
        wgridphoto=pm.window(zgrid)*pm.nbar/1.e9
        plt.fill_between(zgrid,0,wgridphoto, facecolor=colors[n],edgecolor='none',linewidth=2, alpha=0.2)
        plt.plot(zgrid,wgridphoto,color=colors[n],linestyle='--',linewidth=2)
    plt.plot(np.array([]),np.array([]),linestyle='-',linewidth=2,color='k',label='spec-z: \n  $\\sigma_z$=$0.001(1+z)$')
    plt.plot(np.array([]),np.array([]),linestyle='--',linewidth=2,color='k', label='photo-z: \n  $\\sigma_z\\,$=$\\,0.05(1+z)$')
    plt.legend(loc='upper right',fancybox=True, framealpha=0.,prop={'size':16},handlelength=3.5)

    plotname='zbin_comparison'
    plt.savefig(plotdir+plotname+'_small.png')
    plt.savefig(plotdir+plotname+'.png',dpi=900)
    plt.close()
#################################################################
# make plots to illustrate what we're doing when we add calibration errors
def make_calib_demo_plots(r=0): 
    rstr='r{0:05d}'.format(r)
    mapdir='output/map_output/'
    plotdir='output/plots_forposter/'
    unmodmaps=[\
               'desspec_bin0.unmod.fullsky.desall6bin.'+rstr+'.fits']#,'desspec1binz1_bin0.unmod.fullsky.desall6bin.'+rstr+'.fits']
    calvars=[0.1]#,.01]#,.001,.01] #look at all 3, only put one on poster
    FIRSTC=True
    for c in calvars:
        modmaps=[f.replace('.unmod.','.fixvar'+str(c)+'_maxl20.') for f in unmodmaps]
        calmaps=['caliberror.fixvar'+str(c)+'_maxl20.for_desspec_bin0.'+rstr+'.fits']#,'caliberror.fixvar'+str(c)+'_maxl20.for_desspec1binz1_bin0.'+rstr+'.fits' ]
        for n in xrange(len(unmodmaps)):
            #for each calib error and unmod map, make set of plots
            umap=hp.read_map(mapdir+unmodmaps[n])
            mmap=hp.read_map(mapdir+modmaps[n])
            cmap=hp.read_map(mapdir+calmaps[n])

            outprefix='calibDemo_'
            uoutf=outprefix+unmodmaps[n].replace('.fits','.forposter')
            moutf=outprefix+modmaps[n].replace('.fits','.forposter')
            coutf=outprefix+calmaps[n].replace('.fits','.forposter')

            savesmall=1
            savebig=00

            #mapmax=max(np.max(np.fabs(mmap)),np.max(np.fabs(umap)))
            #mapscalemax=0.7*mapmax
            #mapscalemin=-1*mapscalemax
            calmax=np.max(np.fabs(cmap))
            calscalemax=0.7#*calmax
            calscalemin=-1*calscalemax
                        
            mono_cm=matplotlib.cm.Greys_r #greyscale plot
            mono_cm.set_under("w") #set background to white

            #plot umod map
            if FIRSTC:
                mapmax=np.max(np.fabs(umap))
                mapscalemax=.7#.8*mapmax
                mapscalemin=-1*mapscalemax
                plt.figure(0)
                utitle='Original LSS map'
                uscalelabel='$\\delta=n/\\bar{{n}}-1$'
                hp.mollview(umap,title=utitle,min=mapscalemin,max=mapscalemax,cmap=mono_cm,unit=uscalelabel,notext=True)
                if savesmall: plt.savefig(plotdir+uoutf+'_small.png',dpi=300)
                #if savebig: plt.savefig(plotdir+uoutf+'.png',dpi=900)
                plt.close()

            #plot cal error map
            plt.figure(1)
            ctitle='Calibration error field with variance='+str(c)
            calscalelabel='$c(\\bf{{\\hat{{n}}}})$'
            hp.mollview(cmap,title=ctitle,min=calscalemin,max=calscalemax,cmap=mono_cm,unit=calscalelabel,notext=True)
            #hp.mollview(cmap,min=calscalemin,max=calscalemax,cmap=mono_cm,unit=calscalelabel)
            if savesmall: plt.savefig(plotdir+coutf+'_small.png',dpi=300)
            #if savebig: plt.savefig(plotdir+coutf+'.png',dpi=300)
            plt.close()

            #plot mod map
            plt.figure(2)
            mtitle='Observed LSS map'
            mscalelabel='$\\delta^{{\\rm obs}}=n/\\bar{{n}}-1$'
            hp.mollview(mmap,title=mtitle,min=mapscalemin,max=mapscalemax,cmap=mono_cm,unit=mscalelabel,notext=True)
            if savesmall: plt.savefig(plotdir+moutf+'_small.png',dpi=300)
            #if savebig: plt.savefig(plotdir+moutf+'.png',dpi=900)
            plt.close()
        FIRSTC=False
#################################################################
# make plots to illustrate some of the isw reconstriction shown
def make_recon_demo_plots(r=0):
    savesmall=1
    savebig=0
    rstr='r{0:05d}'.format(r)
    mapdir='output/map_output/'
    plotdir='output/plots_forposter/recDemo_mapplots/'
    #--------------
    #plot original isw plot, get scaling for following isw plots
    origiswf='isw_bin0.unmod.fullsky.desall6bin.'+rstr+'.fits'
    iswmap=hp.read_map(mapdir+origiswf)
    iswmax=np.max(np.fabs(iswmap))
    iswscalemax=5.e-5#0.7*iswmax
    iswscalelabel='$T^{{\\rm ISW}}$ (K)'
    iswtitle='True ISW signal'
    iswoutf=origiswf.replace('.fits','_forposter')
    plotandsave_map(iswmap,iswtitle,iswscalelabel,iswscalemax,plotdir,iswoutf,savesmall,savebig,False)
    #--------------
    #make plots for the rec maps that went into our TT plot
    recfiles=[\
              #'iswREC.desspec_5binz1_calpt001.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_1binz1.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_5binz1.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desphoto_5binz1.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_1binz5.fullsky.desall6bin.'+rstr+'.fits',\
              'iswREC.desspec_6binz5.fullsky.desall6bin.'+rstr+'.fits'\
    ]
    reclabels=[\
               #'5 $\\rm z$-bins (spec), $\\rm{{z}}_{{\\rm max}}=1$,Calib. error var. $10^{{-3}}$',\
               '1 $\\rm z$-bin (spec),  $\\rm{{z}}_{{max}}=1$',\
               '5 $\\rm z$-bins (spec), $\\rm{{z}}_{{max}}=1$',\
               '5 $\\rm z$-bins (photo), $\\rm{{z}}_{{max}}=1$',\
               '1 $\\rm z$-bin (spec),  $\\rm{{z}}_{{max}}=5$',\
               '6 $\\rm z$-bins (spec), $\\rm{{z}}_{{max}}=5$'\
    ]
    for f in xrange(len(recfiles)):
        m=hp.read_map(mapdir+recfiles[f])
        t='ISW reconstructed from: '+reclabels[f]
        recoutf=recfiles[f].replace('.fits','_forposter')
        plotandsave_map(m,t,iswscalelabel,iswscalemax,plotdir,recoutf,savesmall,savebig,False)

    #do the calib error reconstriction on its own, since its amp is way higher
    f='iswREC.desspec_5binz1_calpt001.fullsky.desall6bin.'+rstr+'.fits'
    m=hp.read_map(mapdir+f)
    isw_calib_max=7.e-4#.7*np.max(np.fabs(m))
    label='5 $\\rm z$-bins (spec), $\\rm{{z}}_{{\\rm max}}=1$,Calib. error var. $10^{{-3}}$'
    t='ISW reconstructed from: '+label
    recoutf=f.replace('.fits','_forposter')
    plotandsave_map(m,t,iswscalelabel,isw_calib_max,plotdir,recoutf,savesmall,savebig,False)
    #--------------
    #make plots of the galaxy fluctuations that went into the rec maps
    mscalemax=0.5
    mscalelabel='$\\delta=n/\\bar{{n}}-1$'

    #do the individual 6 spec bins
    zedges=[0,.2,.4,.6,.8,1.,5.]
    for i in xrange(6):
        f='desspec_bin'+str(i)+'.unmod.fullsky.desall6bin.'+rstr+'.fits'
        m=hp.read_map(mapdir+f)
        t='Spec. bin: z$\\in$['+str(zedges[i])+','+str(zedges[i+1])+')'
        moutf=f.replace('.fits','_forposter')
        plotandsave_map(m,t,mscalelabel,mscalemax,plotdir,moutf,savesmall,savebig)

    #individual 5 spec bins with calib error of 0.001 
    for i in xrange(5):
        f='desspec_bin'+str(i)+'.fixvar0.001_maxl20.fullsky.desall6bin.'+rstr+'.fits'
        m=hp.read_map(mapdir+f)
        t='Spec. bin: z$\\in$['+str(zedges[i])+','+str(zedges[i+1])+') + calib. error with var=0.001'
        moutf=f.replace('.fits','_forposter')
        plotandsave_map(m,t,mscalelabel,mscalemax,plotdir,moutf,savesmall,savebig)

    #do the individual photo bins (5)
    for i in xrange(6):
        f='desphoto_bin'+str(i)+'.unmod.fullsky.desall6bin.'+rstr+'.fits'
        m=hp.read_map(mapdir+f)
        t='Photo. bin: z$\\in$['+str(zedges[i])+','+str(zedges[i+1])+')'
        moutf=f.replace('.fits','_forposter')
        plotandsave_map(m,t,mscalelabel,mscalemax,plotdir,moutf,savesmall,savebig)

    #one bin zmax=1
    f='desspec1binz1_bin0.unmod.fullsky.desall6bin.'+rstr+'.fits'
    m=hp.read_map(mapdir+f)
    t='Spec. bin: z$\\in$['+str(zedges[0])+','+str(zedges[5])+')'
    moutf=f.replace('.fits','_forposter')
    plotandsave_map(m,t,mscalelabel,mscalemax,plotdir,moutf,savesmall,savebig)

    #1 bin zmax=5
    f='desspec1binz5_bin0.unmod.fullsky.desall6bin.'+rstr+'.fits'
    m=hp.read_map(mapdir+f)
    t='Spec. bin: z$\\in$['+str(zedges[0])+','+str(zedges[6])+')'
    moutf=f.replace('.fits','_forposter')
    plotandsave_map(m,t,mscalelabel,mscalemax,plotdir,moutf,savesmall,savebig)

########################
def plotandsave_map(m,title,scalelabel,scalemax,plotdir,plotf,savesmall=1,savebig=0,isgal=1):
    plt.figure(0)
    #make galaxy plots black and white
    if isgal:
        mono_cm=matplotlib.cm.Greys_r #greyscale plot
        mono_cm.set_under("w") #set background to white
        hp.mollview(m,title=title,min=-1*scalemax,max=scalemax,unit=scalelabel,cmap=mono_cm)
    else:
        hp.mollview(m,title=title,min=-1*scalemax,max=scalemax,unit=scalelabel)
    if savesmall: plt.savefig(plotdir+plotf+'_small.png')
    if savebig: plt.savefig(plotdir+plotf+'.png',dpi=300)
    plt.close()
#################################################################
if __name__=="__main__":
    #makemaps_separate_spec_and_photo() #don't use both this and _correlated_
    Nreal=5
    #makemaps_correlated_spec_and_photo(Nreal=Nreal,Nrealmaps=Nreal)
    for i in xrange(Nreal):
        #make_TTscaatterplot_forposter(i)
        pass
    #make_binplots()
    make_calib_demo_plots(0)
    make_recon_demo_plots()
