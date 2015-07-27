#################################################################
#  This script will contain functions used to test the integrals
#   and calculations in genCrossCorr.py
#################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator
from MapParams import *
from ClRunUtils import *
from genCrossCor import *
from scipy.integrate import quad,romberg
from scipy.interpolate import interp1d
from scipy.special import sph_jn
import time



#################################################################
# Test Cosmology functions and their interpolation
#  Check in values per z needed for accuracy
#  compare different interpolation schemes
#################################################################
   
def test_cosm_tabs():
    GENDATA=0
    outdir = 'test_output/ztabulation/'
    cosmfile = 'testparam.cosm'
    zmax=5.
    maxnperz=1000
    nperzlist = [100,200,300,400,500,1000]
    Nnperz = len(nperzlist)
    maxnperz=nperzlist[-1]
    
    linoutr = outdir+'r_linearinterp.dat'
    quadoutr = outdir+'r_quadinterp.dat'
    cubeoutr = outdir+'r_cubicinterp.dat'
    linoutg = outdir+'g_linearinterp.dat'
    quadoutg = outdir+'g_quadinterp.dat'
    cubeoutg = outdir+'g_cubicinterp.dat'

    if GENDATA:
        #get co_r, D(z) for nperz=1000
        #for the other vals, interpolate; plot, compare differences
        cosmlist=[]
        for n in xrange(Nnperz):
            print "GENERATING VALUES FOR NPERZ=",nperzlist[n]
            cosmlist.append(Cosmology(cosmfile,tabulateZ=True,zmax=zmax,nperz=nperzlist[n]))
        zdata = cosmlist[-1].z_array
        Nz = zdata.size
        #print "Nz=",Nz        
        linr=np.zeros((Nz+1,Nnperz+1)) #comoving radius with linear interp
        linr[1:,0]=zdata
        linr[0,1:]=np.array(nperzlist)
        quadr=np.zeros((Nz+1,Nnperz+1)) # with quadradic interp
        quadr[1:,0]=zdata
        quadr[0,1:]=np.array(nperzlist)
        cuber=np.zeros((Nz+1,Nnperz+1)) # with cubic interp
        cuber[1:,0]=zdata
        cuber[0,1:]=np.array(nperzlist)
        ling=np.zeros((Nz+1,Nnperz+1)) #growth factor with linear interp
        ling[1:,0]=zdata
        ling[0,1:]=np.array(nperzlist)
        quadg=np.zeros((Nz+1,Nnperz+1)) # etc
        quadg[1:,0]=zdata
        quadg[0,1:]=np.array(nperzlist)
        cubeg=np.zeros((Nz+1,Nnperz+1))
        cubeg[1:,0]=zdata
        cubeg[0,1:]=np.array(nperzlist)
        
        for n in xrange(Nnperz):
            print "INTERPOLATING VALUES FOR NPERZ=",nperzlist[n]
            cosm=cosmlist[n]
            origz = cosm.z_array
            rl = interp1d(origz,cosm.r_array,kind='linear')
            rq = interp1d(origz,cosm.r_array,kind='quadratic')
            rc = interp1d(origz,cosm.r_array,kind='cubic')
            gl = interp1d(origz,cosm.g_array,kind='linear')
            gq = interp1d(origz,cosm.g_array,kind='quadratic')
            gc = interp1d(origz,cosm.g_array,kind='cubic')

            linr[1:,n+1] =  rl(zdata)
            quadr[1:,n+1] = rq(zdata)
            cuber[1:,n+1] = rc(zdata)
            ling[1:,n+1] =  gl(zdata)
            quadg[1:,n+1] = gq(zdata)
            cubeg[1:,n+1] = gc(zdata)
        np.savetxt(linoutr,linr)
        np.savetxt(quadoutr,quadr)
        np.savetxt(cubeoutr,cuber)
        np.savetxt(linoutg,ling)
        np.savetxt(quadoutg,quadg)
        np.savetxt(cubeoutg,cubeg)
    else:
        linr=np.loadtxt(linoutr)
        quadr=np.loadtxt(quadoutr)
        cuber=np.loadtxt(cubeoutr)
        ling=np.loadtxt(linoutg)
        quadg=np.loadtxt(quadoutg)
        cubeg=np.loadtxt(cubeoutg)
        
    #plot stuff
    zdata=linr[1:,0]
    Nz = zdata.size
    filelist=[f.replace('.dat','_rel.png') for f in [linoutr,quadoutr,cubeoutr,linoutg,quadoutg,cubeoutg]]
    #filelist=[f.replace('.dat','_abs.png') for f in [linoutr,quadoutr,cubeoutr,linoutg,quadoutg,cubeoutg]]
    varlist=['r','r','r','g','g','g']
    datalist=[linr[1:,1:],quadr[1:,1:],cuber[1:,1:],ling[1:,1:],quadg[1:,1:],cubeg[1:,1:]]
    taglist = ["{0:s}-{1:s}".format(pair[0],pair[1]) for pair in [('r','linear'),('r','quadratic'),('r','cubic'),('g','linear'),('g','quadratic'),('g','cubic')]]
    titlelist=["Accurracy of {0:s} with {1:s} interp".format(pair[0],pair[1]) for pair in [('r','linear'),('r','quadratic'),('r','cubic'),('g','linear'),('g','quadratic'),('g','cubic')]]
    #ignore warnings about dividing by zero, since some of these functions
    #    get close to zero
    np.seterr(divide='ignore', invalid='ignore') 
    for n in xrange(len(datalist)):
        plt.figure(n)
        plt.title(titlelist[n])
        plt.xlabel('redshift z')
        #plt.ylabel('Abs[Frac diff between Nperz and Nperz='+str(maxnperz)+']')
        plt.ylabel('Abs[diff between Nperz and Nperz='+str(maxnperz)+']')
        refdat= datalist[n][:,-1]
        for m in xrange(Nnperz-1):
            diffdat = (datalist[n][:,m] - refdat)/refdat #for rel
            #diffdat = (datalist[n][:,m] - refdat) #for abs
            absdiffdat = np.fabs(diffdat)
            plt.semilogy(zdata,absdiffdat,label=str(nperzlist[m]))
        plt.legend(title='Nperz')
        plt.savefig(filelist[n])
        plt.close()
        

#################################################################
# Test z integrals in Ilk calculation
#################################################################
# Ilklike_justbessel - integrate over spherical Bessel
#                     (shold be like the Il(k) integral with D, H, c=1)
#----------------------------------------------------------------
def Ilklike_justbesselromberg(k,l,rmin,rmax,krcut=-1,epsilon=1.e-10,divmax=10):
    dr = rmax-rmin
    #kmax_fast_osc = krcut/dr #if k*dr>>1, bessel fn is oscillating rapidly
    #kmax_is_ocs = (l+krcut)/rmin #if kr>>l, bessel fn is in osc regime
    if krcut>=0:
        if rmin>0:
            kcut = [max(krcut/dr,(l+krcut)/rmin) for l in lvals]
        else: #for rmin=0, just require that most of bin is in osc regime
            kcut = [max(krcut/dr,(l+krcut)/(.1*dr)) for l in lvals]
    else:
        kcut=k*2 #doesn't matter, as long as its bigger than k
    if k>kcut:
        result=0
    else:
        result=romberg(lambda r: Ilklike_justbessel_integrand(r,l,k),rmin,rmax,tol=epsilon,divmax=divmax)
    return result

def Ilklike_justbesselquad(k,l,rmin,rmax,krcut=-1,epsilon=1.e-10,intlim=10000):
    dr = rmax-rmin
    #kmax_fast_osc = krcut/dr #if k*dr>>1, bessel fn is oscillating rapidly
    #kmax_is_ocs = (l+krcut)/rmin #if kr>>l, bessel fn is in osc regime
    if krcut>=0:
        if rmin>0:
            kcut = [max(krcut/dr,(l+krcut)/rmin) for l in lvals]
        else: #for rmin=0, just require that most of bin is in osc regime
            kcut = [max(krcut/dr,(l+krcut)/(.1*dr)) for l in lvals]
    else:
        kcut=k*2 #doesn't matter, as long as its bigger than k
    if k>kcut:
        result=0
    else:
        result=quad(lambda r: Ilklike_justbessel_integrand(r,l,k),rmin,rmax,epsabs=epsilon,epsrel=epsilon,limit=intlim)[0]
    return result

def Ilklike_justbessel_integrand(r,l,k):
    dI =  sphericalBesselj(l,k*r)
    return dI

#================================================================
# look at ell = 2, 30,100
# look at k = 1.e-2, 1.e-1,1.,10.
# look at z edges = 'thin' [.01,.1],[.4,.5]
#                   'thick' [.01,1],[1,2]
#                   'thinner', [.01,.05],[1.,1.05]
#                   'thicker', [0.1,10]
#================================================================
# Notes on Romberg integration: lots of complaints about hitting divmax
#   Increasing divmax seems to help, but slows things down a lot.
#     seems more probelmatic with the wide bin and ell=100
#     So: more problematic where bessel is integrating fast vs to bin size
def test_bessel_integration_romberg():
    # integrals in Ilklike_justbessel

    outdir = 'test_output/bessel_integration/'
    #tolerance = 1.e-6
    lvals = np.array([3,30,100])
    kvals = np.array([1.e-2, 1.e-1,1.,10.])
    redges=np.array([[50,500],[1600,2000],[3000,6000]])
   
    #divmaxlist=[10,15,20]
    divmax=100 #setting high
    tollist=[1.e-6,1.e-8,1.e-10,1.e-12,1.e-14,1.e-16]
    outfile = outdir+'romberg_varytol.dat'
    f= open(outfile,'w')
    f.write('Integrating Bessel function j_l(kr) for fiducial cosmologyn\n')
    f.write('divmax of integrals: '+str(divmax)+'\n')
    for nperz in [500]:
        f.write('---------------\n r tabulated at N points/z, N='+str(nperz)+'\n')
        resulthead=''.join([' {0:23s}'.format('tol='+str(t)) for t in tollist])
        f.write('{0:10s} {1:10s} {2:3s} {3:7s}'.format('rmin','rmax','l','k')+resulthead+'\n')
        for i in xrange(len(redges)):
            rmin = redges[i,0]
            rmax = redges[i,1]
            for l in lvals:
                for k in kvals:
                    results=[Ilklike_justbesselromberg(k,l,rmin,rmax,epsilon=t,divmax=divmax) for t in tollist]
                    resultstr = ''.join([' {0:+23.16g}'.format(r) for r in results])
                    f.write('{0:10.3g} {1:10.3g} {2:3d} {3:7.1e}'.format(rmin,rmax,l,k)+resultstr+'\n')
        f.write('\n')
    f.close()

#================================================================
# Notes on quad integration:
#   no complaints about non-convergence once I fixed issue w/bessel fn
#   varying inttegral limit from 1000 up to 20000 doesn't change things to 1e-6
#   no complaints when varying epsabs, even all the way down to 1.e-16
#   making epsrel small gives warnings, but most numbers pretty stable
#               some dependence on epsilon for k=1
#            ->We'll probably be ok just setting epsabs=1.e-10
#               or whatever the code is set up for already
#              since the bessel fn gets multiplied modulating fns
#  l=2, k=10 is significantly different than the mathematica results
#         they're more comparable to the mathematica numerical integration

def test_bessel_integration_quad():
    outdir = 'test_output/bessel_integration/'
    tolerance = 1.e-6
    lvals = np.array([3,30,100])
    kvals = np.array([1.e-2, 1.e-1,1.,10.])
    redges=np.array([[50,500],[1600,2000],[3000,6000]])
   
    intlim = 10000
    #limlist=[1000,5000,10000,20000] Doesn't seem to make a difference
    tollist=[1.e-6,1.e-8,1.e-10,1.e-12,1.e-14,1.e-16]
    outfileq = outdir+'quad_varyepsabs.dat'
    f= open(outfileq,'w')
    f.write('Integrating Bessel function j_l(kr) for fiducial cosmology\n')
    f.write('setting epsabs but not epsrel\n')
    for nperz in [500]:
        f.write('---------------\n r tabulated at N points/z, N='+str(nperz)+'\n')
        resulthead=''.join([' {0:23s}'.format('epsabs='+str(eps)) for eps in tollist])
        f.write('{0:10s} {1:10s} {2:3s} {3:7s}'.format('rmin','rmax','l','k')+resulthead+'\n')
        for i in xrange(len(redges)):
            #zmin = zedges[i,0]
            #zmax = zedges[i,1]
            rmin = redges[i,0]
            rmax = redges[i,1]
            for l in lvals:
                for k in kvals:
                    results=[Ilklike_justbesselquad(k,l,rmin,rmax,epsilon=eps,intlim=intlim) for eps in tollist]
                    resultstr = ''.join([' {0:+23.16g}'.format(r) for r in results])
                    f.write('{0:10.3g} {1:10.3g} {2:3d} {3:7.1e}'.format(rmin,rmax,l,k)+resultstr+'\n')
        f.write('\n')
    f.close()

#################################################################
# Look at plots for Il(k) functiosn
#################################################################
#for the same ell values as above, make and save DM Ilk functions
# also check on effect of krcut
def test_Ilk_initial():
    REDODATA=1
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([2,30,100])
    zedges=np.array([[.01,3.]])
    #zedges=np.array([[.01,.1],[1.,1.1],[.01,3.]])
    binlabels=['widez']#['lowz','highz','widez']
    kmin=1.e-5
    kmax=100.0
    krcut=10
    nperlog = 10
    precision=1.e-4

    runtag='test'
    
    kdat = KData(kmin=kmin,kmax=kmax,nperlogk=nperlog,krcut=krcut)
    rundat = ClRunData(rundir=outdir,tag=runtag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=precision)

    maptypes=[[],[]]#[dm,isw][bins]
    for i in xrange(zedges.shape[0]):
        maptypes[0].append(MapType(idtag='mat_'+binlabels[i],zedges=zedges[i,:]))
        maptypes[1].append(MapType(idtag='isw_'+binlabels[i],zedges=zedges[i,:],isISW=True))
    #collect list of binmaps
    dmmaps = get_binmaplist(maptypes[0])[0]
    iswmaps = get_binmaplist(maptypes[1])[0]
    maps = dmmaps#+iswmaps
    Nmaps = len(maps)

    Idata = []
    for m in maps:
        I=getIlk_for_binmap(m,rundat,redo=REDODATA)[0]
        'I.shape',I.shape
        Idata.append(getIlk_for_binmap(m,rundat,redo=REDODATA)[0])
    Idata=np.array(Idata)
    print 'I shape',Idata.shape
    plotdir = outdir+'plots/'
    k=kdat.karray
    print 'k shape',k.shape
    for n in xrange(Nmaps):
        for l in xrange(lvals.size):
            m = maps[n]
            tag=m.typetag
            outf = plotdir+tag+'_l'+str(l)+'_quickplot.png'
            plt.figure(n)
            titlestr = r'{0:s}, ell={1:d} $z\in[{2:g},{3:g})$'.format(tag,int(lvals[l]),m.zmin,m.zmax)
            plt.title(titlestr)
            plt.xlabel(r'$k$ [h Mpc$^{-1}$]')
            plt.ylabel(r'$I_l(k)$')
            plt.semilogx(k,Idata[n,l,:])
            plt.savefig(outf)
            plt.close()

def test_Ilk_krcut():
    REDODATA=0
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    #lvals=np.array([30])
    lvals = np.array([2,30,100])
    #zedges=np.array([[1.,1.1]])
    zedges=np.array([[.01,.1],[1.,1.1],[.01,3.]])
    binlabels=['lowz','highz','widez']
    kmin=1.e-3
    kmax=1.e1
    nperlog = 100
    precision=1.e-10
    krcutlist=[10,50,100,200,500]
    Ncut = len(krcutlist)
    runtaglist=['krcut{0:03d}'.format(c) for c in krcutlist]
    kdatlist=[KData(kmin=kmin,kmax=kmax,nperlogk=nperlog,krcut=c) for c in krcutlist]

    rundatlist = [ClRunData(rundir=outdir,tag=runtaglist[c],cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdatlist[c],epsilon=precision) for c in xrange(Ncut)]

    maptypes=[[],[]]#[dm,isw][bins]
    for i in xrange(zedges.shape[0]):
        maptypes[0].append(MapType(idtag='mat_'+binlabels[i],zedges=zedges[i,:]))
        maptypes[1].append(MapType(idtag='isw_'+binlabels[i],zedges=zedges[i,:],isISW=True))
        
    #collect list of binmaps
    dmmaps = get_binmaplist(maptypes[0])[0]
    iswmaps = get_binmaplist(maptypes[1])[0]
    maps = dmmaps+iswmaps
    Nmaps = len(maps) #number of maps per krcut

    Idata = [[] for c in krcutlist] #[kcut,map,l,k]
    for c in xrange(Ncut):
        print "ON KRCUT=",krcutlist[c]
        for m in maps:
            Idata[c].append(getIlk_for_binmap(m,rundatlist[c],redo=REDODATA))
    Idata=np.array(Idata)
    #print 'Idata.shape',Idata.shape

    plotdir = outdir+'plots/'
    k=kdatlist[0].karray
    diffdat = np.zeros_like(Idata)
    for c in xrange(Ncut):
        diffdat[c,:,:,:] = Idata[c,:,:,:] - Idata[-1,:,:,:]
    for n in xrange(Nmaps):
        for l in xrange(lvals.size):
            #for each map, plot diff between low and high krcut
            m = maps[n]
            tag=m.typetag
            outf = plotdir+'varykrcut_'+tag+'_l{0:03d}.png'.format(lvals[l])
            plt.figure(n)
            titlestr = r'Varying krcut, {0:s}, ell={1:d} $z\in[{2:g},{3:g})$'.format(tag,int(lvals[l]),m.zmin,m.zmax)
            plt.title(titlestr)
            plt.xlabel(r'$k$ [h Mpc$^{-1}$]')
            plt.ylabel(r'$I_{krcut=N} - I_{krcut=max}$')
            for c in xrange(Ncut):
                plt.semilogx(k,diffdat[c,n,l,:],label='{0:g}'.format(krcutlist[c]))
            plt.legend(title='krcut',loc=2)
            plt.savefig(outf)
            plt.close()
#---------------------------------------------------
def test_Ilk_nperlogk():
    REDODATA=0
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    #lvals=np.array([30])
    lvals = np.array([2,30,100])
    #zedges=np.array([[1.,1.1]])
    zedges=np.array([[.01,.1],[1.,1.1],[.01,3.]])
    binlabels=['lowz','highz','widez']
    kmin=1.e-3
    kmax=1.e2
    nperlog = 100
    precision=1.e-10
    krcut=50
    nperlogmax=1000
    keeponeper=[2,1] #keep one k value per X entries in k with max div
    Nnper=len(keeponeper)
    nperlog=[nperlogmax/x for x in keeponeper]

    #Generate Ilk data just for nperlog=1000
    runtag='nperlogk{0:04d}'.format(nperlogmax) 
    kdat=KData(kmin=kmin,kmax=kmax,nperlogk=nperlogmax,krcut=krcut)
    rundat = RunData(rundir=outdir,tag=runtag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=precision) 
    maptypes=[[],[]]#[dm,isw][bins]
    for i in xrange(zedges.shape[0]):
        maptypes[0].append(MapType(idtag='mat_'+binlabels[i],zedges=zedges[i,:]))
        maptypes[1].append(MapType(idtag='isw_'+binlabels[i],zedges=zedges[i,:],isISW=True))
    #collect list of binmaps
    dmmaps = get_binmaplist(maptypes[0])[0]
    iswmaps = get_binmaplist(maptypes[1])[0]
    maps = dmmaps+iswmaps
    Nmaps = len(maps) #number of maps per krcut

    Imaxdiv=[]
    for m in maps:
        Imaxdiv.append(getIlk_for_binmap(m,rundat,redo=REDODATA))
    Imaxdiv=np.array(Imaxdiv)
    #print 'Idata.shape',Idata.shape

    #make sparse datasets and interpolate to original k values
    Iinterp=[]#[nper,map,l,k]
    k=kdat.karray
    print "Generating and interpolating sparse Ilk data."
    for i in xrange(Nnper-1): #all but max nperlog
        print "   keeping on data point per original",keeponeper[i]
        sparsek = k[::keeponeper[i]]
        sparseI = Imaxdiv[:,:,::keeponeper[i]]
        interpfunc=interp1d(sparsek,sparseI,kind='cubic')
        Iinterp.append(interpfunc(k))
    Iinterp.append(Imaxdiv)  #then add max nperlog
    Iinterp=np.array(Iinterp)

    #plot comparisons
    plotdir = outdir+'plots/'
    diffdat = np.zeros_like(Iinterp)
    print "Evalulating differences from interpolation."
    for i in xrange(Nnper):
        diffdat[i,:,:,:] = Iinterp[i,:,:,:] - Iinterp[-1,:,:,:]
    print "Making plots."
    for n in xrange(Nmaps):
        for l in xrange(lvals.size):
            #for each map, plot diff between low and highest nperlogk
            m = maps[n]
            tag=m.typetag
            outf = plotdir+'nperlogk_'+tag+'_l{0:03d}.png'.format(lvals[l])
            plt.figure(n)
            titlestr = r'Varying nperlogk, {0:s}, ell={1:d} $z\in[{2:g},{3:g})$'.format(tag,int(lvals[l]),m.zmin,m.zmax)
            plt.title(titlestr)
            plt.xlabel(r'$k$ [h Mpc$^{-1}$]')
            plt.ylabel(r'$I_{krcut=N} - I_{krcut=max}$')
            for i in xrange(Nnper):
                plt.semilogx(k[:-150],diffdat[i,n,l,:-150],label='{0:g}'.format(nperlog[i]))
            plt.legend(title='nperlogk',loc=2)
            plt.savefig(outf)
            plt.close()

#---------------------------------------------------
# to evaluate whether kmax, kmin are sufficient, look at I^2 k^3 P(k)
def eyeball_Ilk_convergence():
    REDODATA=0
    runtag='eyeball'
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([1,2,30,100,200])
    zedges=np.array([[.01,.1],[1.,1.1],[.01,3.]])
    binlabels=['lowz','highz','widez']
    kdat=KData(kmin=1.e-4,kmax=1.e1,nperlogk=10)
    rundat = ClRunData(rundir=outdir,tag=runtag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10)
    
    #collect list of binmaps
    maptypes=[[],[]]#[dm,isw][bins]
    for i in xrange(zedges.shape[0]):
        maptypes[0].append(MapType(idtag='mat_'+binlabels[i],zedges=zedges[i,:]))
        maptypes[1].append(MapType(idtag='isw_'+binlabels[i],zedges=zedges[i,:],isISW=True))
    dmmaps = get_binmaplist(maptypes[0])[0]
    iswmaps = get_binmaplist(maptypes[1])[0]
    maps = dmmaps+iswmaps
    Nmaps = len(maps) #number of maps per krcut

    #get I functions
    k=kdat.karray
    Idata=[] #[map,l,k]
    for m in maps:
        Idata.append(getIlk_for_binmap(m,rundat,redo=REDODATA))
    Idata=np.array(Idata)

    #get power spectrum
    cosm = rundat.cosm
    cosm.getPk(kdat.kmin,kdat.kmax)#,rerunCAMB=True)
    Pk = cosm.P(k) #interpolates to relevant values

    #plot integrad of auto power spec
    plotdir = outdir+'plots/'
    plt.figure(Nmaps)
    plt.loglog(cosm.k_forPower,cosm.P_forPower,label='cosm arrays')
    plt.loglog(k,Pk,label='interp')
    plt.legend()
    plt.savefig(plotdir+'Pk_interp_test.png')
    plt.close()
    for n in xrange(Nmaps):
        for l in xrange(lvals.size):
            m = maps[n]
            tag=m.typetag
            integrand=Idata[n,l,:]*Idata[n,l,:]*k*k*k*Pk
            
            outf = plotdir+'autoIntegrand_'+tag+'_l{0:03d}.png'.format(lvals[l])
            plt.figure(n)
            titlestr = r'auto-$C_l$ integrand, for {0:s} ell={1:d} $z\in[{2:g},{3:g})$'.format(tag,int(lvals[l]),m.zmin,m.zmax)
            plt.title(titlestr)
            plt.xlabel(r'$k$ [h Mpc$^{-1}$]')
            plt.ylabel(r'$k^3 P(k) [I_l(k)]^2$')
            plt.loglog(k,integrand,)
            plt.savefig(outf)
            plt.close()

#---------------------------------------------------
def get_crosstags(maptags):
    Nmaps=len(maptags)
    crosspairs,crossinds=get_index_pairs(Nmaps)
    Ncross=crosspairs.shape[0]
    crosstags=[]
    for n in xrange(Ncross):
        i=crosspairs[n,0]
        j = crosspairs[n,1]
        if i==j:
            crosstags.append(maptags[i][:-5])
        else:
            crosstags.append(maptags[i][:-5]+'-'+maptags[j][:-5])
    return crosstags
#---------------------------------------------------
def get_testmaplist():
    zedges=np.array([[.01,.1],[1.,1.1],[.01,3.]])
    binlabels=['lowz','highz','widez']
    maptypes=[[],[]]#[dm,isw][bins]
    for i in xrange(zedges.shape[0]):
        maptypes[0].append(MapType(idtag='mat_'+binlabels[i],zedges=zedges[i,:]))
        maptypes[1].append(MapType(idtag='isw_'+binlabels[i],zedges=zedges[i,:],isISW=True))
    dmmaps = get_binmaplist(maptypes[0])[0]
    iswmaps = get_binmaplist(maptypes[1])[0]
    maps = dmmaps+iswmaps
    return maps

#---------------------------------------------------
def makediffplots(clgrid,lvals,crosstags,varyparamlist,varyparamlabel,plotdir,refvalind=-1,shapelist=[],colorlist=[]):
    print 'varyparamlabel',varyparamlabel
    print 'varyparamlist',varyparamlist,'length=',len(varyparamlist)
    print 'clgrid.shape',clgrid.shape
    inttol=1.e-10*2./np.pi
    Ncross=len(crosstags)
    if refvalind<0: #by default, we'll assume last param value is the reference
        refvalind=len(varyparamlist)+refvalind #set to value rather than -1 to check equality
        
    diffgrid=np.zeros((len(varyparamlist)-1,Ncross,len(lvals)))
    fracdiffgrid=np.zeros((len(varyparamlist)-1,Ncross,len(lvals)))
    for n in xrange(len(varyparamlist)):
        if n==refvalind: continue
        else:
            if n<refvalind:i=n
            else: i=n-1
            diffgrid[i,:,:]= np.fabs(clgrid[n,:,:]-clgrid[refvalind,:,:])
            fracdiffgrid[i,:,:]= np.fabs((clgrid[n,:,:]-clgrid[refvalind,:,:])/clgrid[refvalind,:,:])
            fracdiffgrid[i,:,:]=np.nan_to_num(fracdiffgrid[i,:,:])
            #print fracdiffgrid
    #add a small number to avoid logplots having trouble with zeros
    diffgrid=diffgrid+1.e-20*(diffgrid==0)
    fracdiffgrid=fracdiffgrid+1.e-20*(fracdiffgrid==0)

    if not shapelist:
        #shapelist=['o','v','^','<','>','d','*']
        shapelist=['v','^','<','>','*']
    if not colorlist:
        colorlist=[
                   '#1f78b4', '#33a02c','#e31a1c', '#ff7f00','#cab2d6', '#6a3d9a', '#ffff99', '#b15928','#a6cee3','#b2df8a','#fb9a99','#fdbf6f']
    for lind in xrange(lvals.size):
        print 'on ell=',lvals[lind]
        #NOTE: Y LIM are set up so that points can disappear off the bottom
        #      of plots, but not the top. If a point doesn't show, the change
        #      is smaller than the ymin of the plot.
        maxdiff=np.max(diffgrid[:,:,lind])
        mindiff=np.min(diffgrid[:,:,lind])+1.e-18
        maxfdiff = np.max(fracdiffgrid[:,:,lind])
        minfdiff = np.min(fracdiffgrid[:,:,lind])+1.e-18

        plt.figure(lind,figsize=(7.5,10))
        #Absolute value of C_l
        ax1=plt.subplot(311)
        plt.title(r'Dependence of $C_l$ on {0:s}: $\ell={1:d}$'.format(varyparamlabel,lvals[lind]))
        plt.grid(True)
        ismajor=np.where(np.fabs(clgrid[refvalind,:,lind])>inttol)[0]
        ax=plt.gca()
        ax.xaxis.set_major_locator(FixedLocator(ismajor))
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(Ncross)))
        ax.xaxis.grid(b=True, which='both')
        ax.xaxis.grid(True, which='major',linestyle='-',color='grey')
        ax.xaxis.grid(True, which='minor',linestyle=':',color='grey')
        plt.setp(ax.get_xmajorticklabels(), visible=False)
        plt.setp(ax.get_xminorticklabels(), visible=False)
        
        plt.ylabel(r'$\left|C_l^{{[\rm{{{0:s}}}=X]}}\right|$'.format(varyparamlabel))
        plt.ylim((1.e-20,max(1.e-2,np.max(np.fabs(clgrid[:,:,lind])))))
        plt.xlim((-1,Ncross))
        for n in xrange(len(varyparamlist)):
            plt.semilogy(np.arange(Ncross),np.fabs(clgrid[n,:,lind]),marker=shapelist[n%len(shapelist)],color=colorlist[n%len(colorlist)],linestyle='None',label=str(varyparamlist[n]))
        plt.semilogy(np.arange(Ncross+2)-1,inttol*np.ones(Ncross+2),label='int tol') #mark the integral tolerance
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.legend(title='{0:s}'.format(varyparamlabel),loc='center left',ncol=1,bbox_to_anchor=(1.,.5),numpoints=1, prop={'size':9})

        #differences
        ax2=plt.subplot(312, sharex=ax1)
        plt.ylabel(r'$\left|C_l^{{[\rm{{{1:s}}}=X]}} - C_l^{{[{0:s}]}}\right|$'.format(str(varyparamlist[refvalind]),varyparamlabel))
        plt.ylim((1.e-15,max(5*maxdiff,1.e-5)))
        plt.xlim((-1,Ncross))
        plt.grid(True)
        ax=plt.gca()
        ax.xaxis.grid(b=True, which='both')
        ax.xaxis.grid(True, which='major',linestyle="-",color='grey')
        ax.xaxis.grid(True, which='minor',linestyle=':',color='grey')
        plt.setp(ax.get_xmajorticklabels(), visible=False)
        plt.setp(ax.get_xminorticklabels(), visible=False)

        for n in xrange(len(varyparamlist)):
            if n==refvalind: continue
            else:
                i=n-(n>refvalind)
                plt.semilogy(np.arange(Ncross),diffgrid[i,:,lind],marker=shapelist[n%len(shapelist)],color=colorlist[n%len(colorlist)],linestyle='None',label=str(varyparamlist[n]))
        plt.legend(title='{0:s}'.format(varyparamlabel),loc='center left',ncol=1,bbox_to_anchor=(1.,.5),numpoints=1, prop={'size':9})
        
        #relative difference
        ax3=plt.subplot(313, sharex=ax1)
        plt.grid(True)
        plt.ylabel(r'$\left|(C_l^{{[\rm{{{1:s}}}=X]}} - C_l^{{[{0:s}]}})/C_l^{{[{0:s}]}}\right|$'.format(str(varyparamlist[refvalind]),varyparamlabel))
        plt.ylim((1.e-8,max(5*maxfdiff,1.)))
        #plt.ylim((1.e-4,.1))#max(5*maxfdiff,1.)))
        plt.xlim((-1,Ncross))
        for n in xrange(len(varyparamlist)):
            if n==refvalind: continue
            else:
                i=n-(n>refvalind)
                plt.semilogy(np.arange(Ncross),fracdiffgrid[i,:,lind],marker=shapelist[n%len(shapelist)],color=colorlist[n%len(colorlist)],linestyle='None',label=str(varyparamlist[n]))

        #make gridlines solid if Cl>inttol
        ismajor=np.where(np.fabs(clgrid[refvalind,:,lind])>inttol)[0]
        ax=plt.gca()
        ax.yaxis.set_minor_locator(FixedLocator([1.e-3]))

        ax.yaxis.grid(b=True, which='both')

        ax.yaxis.grid(True, which='minor',linestyle='-',color='grey')


        ax.xaxis.grid(b=True, which='both')
        ax.xaxis.grid(True, which='major',linestyle='-',color='grey')
        ax.xaxis.grid(True, which='minor',linestyle=':',color='grey')
        plt.setp(ax.get_xmajorticklabels(), visible=False)
        ax.set_xticklabels(crosstags,rotation='vertical',size=10,minor=True)
        
        #format plot size and legend
        plt.subplots_adjust(right=.85)
        plt.subplots_adjust(bottom=0.2)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(hspace=.05)
        plt.legend(title='{0:s}'.format(varyparamlabel),loc='center left',ncol=1,bbox_to_anchor=(1.,.5),numpoints=1, prop={'size':10})
        plt.savefig(plotdir+'Cl_{0:s}_l{1:03d}.png'.format(varyparamlabel, lvals[lind]))
        plt.close()
        
#---------------------------------------------------
def test_Cl_nperlogk(REDOCL=0,REDOILK=0):
    #this function will generate the Ilk functions to be used for Cl tests
    READONLY=1
    tophatedge=.01
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([1,30,100])
    nperlist=[100,200,300,400,500,600]#,1000]
    runtaglist=['nperlogk{0:04d}'.format(n,tophatedge) for n in nperlist]
    kdatlist=[KData(kmin=1.e-5,kmax=10.,nperlogk=n,krcut=50) for n in nperlist]
    rundatlist=[ClRunData(rundir=outdir,tag=runtaglist[k],cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdatlist[k],epsilon=1.e-10) for k in xrange(len(kdatlist))]

    maps = get_testmaplist()
    Nmaps = len(maps) #number of maps per krcut
    maptags=get_bintaglist(maps)
    pairs=[p for p in itertools.combinations_with_replacement(maptags,2)]
    clgrid=[] #nperlog,crossind,ell
    for n in xrange(len(nperlist)):
        print "ON NPERLOGK=",nperlist[n]
        clgrid.append(getCl(maps,rundatlist[n],dopairs=pairs,DoNotOverwrite=READONLY).cl)#,redoAllCl=REDOCL,redoIlk=REDOILK))
    clgrid=np.array(clgrid)
    crosstags=get_crosstags(maptags)
    plotdir = outdir+'plots/'
    makediffplots(clgrid,lvals,crosstags,nperlist,'nperlogk',plotdir)


#---------------------------------------------------
def test_Cl_varykmax(REDOCL=0,REDOILK=0):
    READONLY=1
    tophatedge=.01
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([1,30,100])
    kmaxlist=[.1,1.,10.,20.,50.,100.]#[1.,10.,20.,50.,100.]
    runtaglist=['kmax{0:0.0e}'.format(kmax,tophatedge) for kmax in kmaxlist]
    #ilktag='nperlogk{0:04d}'.format(200) #use previously computed Ilk
    kdatlist=[KData(kmin=1.e-5,kmax=n,nperlogk=300,krcut=50) for n in kmaxlist]
    rundatlist=[ClRunData(rundir=outdir,tag=runtaglist[k],cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdatlist[k],epsilon=1.e-10) for k in xrange(len(kdatlist))]

    maps = get_testmaplist()
    Nmaps = len(maps) #number of maps per krcut
    maptags=get_bintaglist(maps)
    crosstags=get_crosstags(maptags)
    pairs=[p for p in itertools.combinations_with_replacement(maptags,2)]
    clgrid=[] #nperlog,crossind,ell
    for n in xrange(len(kmaxlist)):
        print "****ON kmax=",kmaxlist[n]
        clgrid.append(getCl(maps,rundatlist[n],dopairs=pairs,DoNotOverwrite=READONLY).cl)#,redoAllCl=REDOCL,redoIlk=REDOILK))
    clgrid=np.array(clgrid)
    plotdir = outdir+'plots/'
    makediffplots(clgrid,lvals,crosstags,kmaxlist,'kmax',plotdir)

def test_Cl_varykmin(REDOCL=0,REDOILK=0):
    READONLY=1
    tophatedge=.01
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([1,30,100])
    kminlist=[1.e-5,5.e-5,1.e-4,5.e-4,1.e-3]
    runtaglist=['kmin{0:0.0e}'.format(kmin,tophatedge) for kmin in kminlist]
    ilktag='nperlogk{0:04d}'.format(300,tophatedge) #use previously computed Ilk
    kdatlist=[KData(kmin=n,kmax=10.,nperlogk=300,krcut=50) for n in kminlist]
    rundatlist=[ClRunData(rundir=outdir,tag=runtaglist[k],ilktag=ilktag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdatlist[k],epsilon=1.e-10) for k in xrange(len(kdatlist))]

    maps = get_testmaplist()
    Nmaps = len(maps) #number of maps per krcut
    maptags=get_bintaglist(maps)
    crosstags=get_crosstags(maptags)
    pairs=[p for p in itertools.combinations_with_replacement(maptags,2)]
    clgrid=[] #nperlog,crossind,ell
    for n in xrange(len(kminlist)):
        print "****ON kmin=",kminlist[n]
        #for m in maps:R
        #    getIlk_for_binmap(m,rundatlist[n],redo=REDOILK)
        clgrid.append(getCl(maps,rundatlist[n],dopairs=pairs,DoNotOverwrite=READONLY,redoAllCl=REDOCL,redoIlk=REDOILK).cl)
    clgrid=np.array(clgrid)
    plotdir = outdir+'plots/'
    makediffplots(clgrid,lvals,crosstags,kminlist,'kmin',plotdir,refvalind=0)

def test_Cl_varykrcut(REDOCL=0,REDOILK=0,factorcut=False,newtag=''):
    READONLY=1
    tophatedge=.01
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([2,30,100])
    if factorcut:
        krcutlist=[1,2,5,10,15,20]
        kdatlist=[KData(kmin=1.e-5,kmax=10.,nperlogk=300,krcutadd=0,krcutmult=n) for n in krcutlist]
        #newtag='New'
    else:
        krcutlist=[1,10,50,100,200,300]#[1,10,50,100,200,500]
        kdatlist=[KData(kmin=1.e-5,kmax=10.,nperlogk=300,krcutadd=n,krcutmult=1) for n in krcutlist]
        #newtag=''
        
    runtaglist=['krcut'+newtag+'{0:03d}_edge{1:0.3f}'.format(c,tophatedge) for c in krcutlist]
    #ilktaglist=['krcut{0:03d}'.format(c,tophatedge) for c in krcutlist]

    rundatlist=[ClRunData(rundir=outdir,tag=runtaglist[k],cosmpfile=cosmfile,lvals=lvals,zmax=10.,kdata=kdatlist[k],epsilon=1.e-10) for k in xrange(len(kdatlist))]

    maps = get_testmaplist()
    Nmaps = len(maps) #number of maps per krcut
    maptags=get_bintaglist(maps)
    crosstags=get_crosstags(maptags)
    pairs=[p for p in itertools.combinations_with_replacement(maptags,2)]
    clgrid=[] #nperlog,crossind,ell
    for n in xrange(len(kdatlist)):
        print "****ON KRCUT=",krcutlist[n]
        #for m in maps:
        #    getIlk_for_binmap(m,rundatlist[n],redo=REDOILK)
        clgrid.append(getCl(maps,rundatlist[n],dopairs=pairs,DoNotOverwrite=READONLY,redoAllCl=REDOCL,redoIlk=REDOILK).cl)
    clgrid=np.array(clgrid)
    plotdir = outdir+'plots/'
    print 'making plots'
    makediffplots(clgrid,lvals,crosstags,krcutlist,'krcut'+newtag,plotdir,refvalind=-1)
#---------------------------
def test_Cl_varykrcut_addmult(REDOCL=0,REDOILK=0,newtag=''):
    READONLY=1
    tophatedge=.01
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([2,30,100])
    krcutaddlist=[10,20,50,100,200]
    krcutmultlist=[0,10,15,20]

    maps = get_testmaplist()
    Nmaps = len(maps) #number of maps per krcut
    maptags=get_bintaglist(maps)
    crosstags=get_crosstags(maptags)
    pairs=[p for p in itertools.combinations_with_replacement(maptags,2)]

    allkrcutlist=[(ca,cm) for ca in krcutaddlist for cm in krcutmultlist]
    allkrcutlist.append('None')
    nocutkdat=KData(krcutadd=-1,krcutmult=-1)
    nocutrundat=ClRunData(rundir=outdir,tag='krcutNone',cosmpfile=cosmfile,lvals=lvals,zmax=10.,kdata=nocutkdat)
    nocutcl=getCl(maps,nocutrundat,dopairs=pairs,DoNotOverwrite=READONLY,redoAllCl=REDOCL,redoIlk=REDOILK).cl
    allclgrid=[]
    for cm in krcutmultlist:
        krcutlist=[(ca,cm) for ca in krcutaddlist]# for cm in krcutmultlist]
        krcutlist.append('None')
        runtaglist=['krcut'+newtag+'a{0:03d}_m{1:03d}_edge{2:0.3f}'.format(ca,cm,tophatedge) for ca in krcutaddlist]# for cm in krcutmultlist]
        kdatlist=[KData(kmin=1.e-5,kmax=10.,nperlogk=200,krcutadd=na,krcutmult=cm) for na in krcutaddlist]# for cm in krcutmultlist]
        rundatlist=[ClRunData(rundir=outdir,tag=runtaglist[k],cosmpfile=cosmfile,lvals=lvals,zmax=10.,kdata=kdatlist[k],epsilon=1.e-10) for k in xrange(len(kdatlist))]
        clgrid=[] #nperlog,crossind,ell
        for n in xrange(len(kdatlist)):
            print "****ON KRCUT=",krcutlist[n]
            #for m in maps:
            #    getIlk_for_binmap(m,rundatlist[n],redo=REDOILK)
            clgrid.append(getCl(maps,rundatlist[n],dopairs=pairs,DoNotOverwrite=READONLY,redoAllCl=REDOCL,redoIlk=REDOILK).cl)
            allclgrid.append(clgrid[-1])
        clgrid.append(nocutcl)
        clgrid=np.array(clgrid)
        plotdir = outdir+'plots/'
        print 'making plots'
        ref=-1
        makediffplots(clgrid,lvals,crosstags,krcutlist,'krcut_am_mult{0:03d}'.format(cm)+newtag,plotdir,refvalind=ref)
    allclgrid.append(nocutcl)
    allclgrid=np.array(allclgrid)
    makediffplots(allclgrid,lvals,crosstags,allkrcutlist,'krcut_am_all'+newtag,plotdir,refvalind=ref)

#---------------------------
def test_cosm_vsclass():
    outdir = 'test_output/Ilktests/plots/'
    cosmfile = 'testparam.cosm'
    
    classfile='test_output/classdir/classcompare_background.dat'
    classdat=np.loadtxt(classfile,skiprows=4)
    classz=classdat[::-1,0]
    cosm=Cosmology(cosmfile,tabulateZ=True,zmax=5,nperz=100)
    z=cosm.z_array
    g=cosm.g_array
    h=cosm.H_array*cosm.h0
    r=cosm.r_array/cosm.h0 #Mpc/h -> Mpc
    f=cosm.f_array-1.
    zmin=z[0]
    zmax=z[-1]
    
    maxind=0
    for i in xrange(len(classz)):
        maxind=i
        if classz[i]>=zmax:
            break
    classz=classz[:maxind]
    print 'class zmax',classz[-1],'zmin',classz[0]
    print 'my zmax',zmax,'zmin',zmin
    classr=classdat[:,4][::-1][:maxind]
    classH=classdat[:,3][::-1][:maxind]*cosm.c #from 1/Mpc to km/s/Mpc
    classg=classdat[:,14][::-1][:maxind]/classdat[-1,14] #normalize to present time
    classf=classdat[:,15][::-1][:maxind]-1.

    names=['comoving radius','Hubble parameter','growth factor','growth rate f-1']
    varlist=['r','H','D','f-1']
    myfuncs=[r,h,g,f]
    classfuncs=[classr,classH,classg,classf]
    Nplots=len(varlist)
    for n in xrange(Nplots):
        print '  on',names[n]
        plt.figure(n,figsize=(8.5,11))
        ax1=plt.subplot(311)
        plt.title(names[n])
        plt.xlim(zmin,zmax)
        plt.gca().yaxis.grid(True)
        plt.gca().xaxis.grid(True)
        plt.ylabel(varlist[n])
        if varlist[n]=='f-1':
            plt.ylim(-.1,.1)
            plt.plot(z,np.zeros_like(z))
        plt.plot(z,myfuncs[n],label='my output')
        plt.plot(classz,classfuncs[n],label='CLASS')
        plt.legend()

        #plot ratios
        plt.subplot(313, sharex=ax1)
        plt.title('fractional difference')
        myinterp=interp1d(z,myfuncs[n],bounds_error=False)
        myproj=myinterp(classz)
        plt.semilogy(classz,np.fabs((myproj-classfuncs[n])/classfuncs[n]))
        plt.xlabel('z')
        plt.gca().yaxis.grid(True)
        plt.gca().xaxis.grid(True)
        plt.ylabel('Abs[(mine-CLASS))/CLASS]')
        plt.xlim(zmin,zmax)

        #plot differences
        #plt.figure(2*Nplots+n)
        plt.subplot(312, sharex=ax1)
        plt.title('absolute difference')
        myinterp=interp1d(z,myfuncs[n],bounds_error=False)
        myproj=myinterp(classz)
        plt.semilogy(classz,np.fabs(myproj-classfuncs[n]))
        plt.gca().yaxis.grid(True)
        plt.gca().xaxis.grid(True)
        plt.ylabel('Abs[mine-CLASS]')
        plt.xlim(zmin,zmax)
        plt.savefig(outdir+'classcomp_cosm_'+varlist[n]+'.png')
        plt.close()

        

#----------------------------------------------------------------
def biastest(z):
    return 1.2
def dndztest(z):
    z0=.32
    alpha=.36
    return (z/z0)**(alpha)*np.exp(-alpha*z/z0)
#----------------------------------------------------------------
def test_new_tophat():
    REDODATA=1
    zedges=np.array([[0.01,0.1],[1.,1.1],[0.01,3.]])
    sig = .010
    nbar=29829
    maps = get_testmaplist()
    binlabels=['lowz','highz','widez']
    galmaptypes=[]
   
    for i in xrange(zedges.shape[0]):
        galmaptypes.append(SurveyType(idtag='gal_'+binlabels[i],zedges=zedges[i,:],sigz0=sig,nbar=nbar,dndz=dndztest,bias=biastest))
    gmaps = get_binmaplist(galmaptypes)[0]
    maps = maps+gmaps

    zmaxlist=[m.zmax for m in maps]
    zmax=max(zmaxlist)*1.5
    print 'overallzmax=',zmax
    cosmfile = 'testparam.cosm'
    lvals=np.array([2,30])
    kdat=KData(kmin=.01,kmax=1.,nperlogk=5,krcut=1.)
    outdir = 'test_output/Ilktests/'
    rundat=ClRunData(rundir=outdir,tag='quicktest',cosmpfile=cosmfile,lvals=lvals,zmax=zmax,kdata=kdat,epsilon=1.e-10)
    #get I functions
    k=kdat.karray
    Idata=[] #[map,l,k]
    plt.figure(0)
    plt.title('window function check')
    plt.xlabel('z')
    for m in maps:
        nperz=100
        zgrid = m.zmin+np.arange(nperz*(m.zmax-m.zmin))/float(nperz)
        wgrid = m.window(zgrid)
        intval=quad(lambda z: m.window(z),m.zmin,m.zmax)[0]
        binlabel=m.tag[:m.tag.rfind('_bin')]+ ', Int={0:g}'.format(intval)
        plt.plot(zgrid,wgrid,label=binlabel)
    plt.legend()
    print 'writing to file',outdir+'windowtest.png'
    plt.savefig(outdir+'plots/Windowtest.png')
    plt.close()
        #Idata.append(getIlk_for_binmap(m,rundat,redo=REDODATA))
    #Idata=np.array(Idata)
#----------------------------------------------------------------
def get_class_cl_data(clshape,lvals,Nmaps,Ncross,ONEBINTEST=0,infile=''):
    #read in class data and plot comparisons
    if not infile:
        if ONEBINTEST:
            classclfile = 'test_output/classdir/classcomp_onebin_cl.dat'
        else:
            classclfile = 'test_output/classdir/classcompare_cl.dat'
    else:
        classclfile=infile
    classindat = np.loadtxt(classclfile,skiprows=7)
    classcl=np.zeros(clshape)
    #first, need to map class output columns to crossindices
    classcol_xinds=[] #classcol_xinds[i] = crossind for ith c_l column
    if ONEBINTEST:
        classcol_xpairs=[[0,0]] 
    else:
        classcol_xpairs=[[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]] 
    crosspairs,crossinds=get_index_pairs(Nmaps)
    for i in xrange(len(classcol_xpairs)):
        p=classcol_xpairs[i]
        classcol_xinds.append(crossinds[p[0],p[1]])
    for lind in xrange(len(lvals)):
        for xind in xrange(Ncross):
            l=lvals[lind]
            classunits=classindat[l-2,1+classcol_xinds[xind]]
            classcl[xind,lind] = classunits*2*np.pi/(l*(l+1))
    return classcl
#----------------------------------------------------------------
def make_classcompare_plots(clvals,classcl,lvals,crosstags,outdir,plottag='classcomp',titlenote='',ONEBINTEST=0,runlabels=[],markers=['o','<','>','^','v','<','>']):
    #clvals have axis [clrun][crossinds][l]
    if len(clvals.shape)>2:
        Nrun=clvals.shape[0]
    else:
        Nrun=1
        clvals=np.expand_dims(clvals,0)
    if Nrun>len(runlabels):
        for i in xrange(nrun):
            if i+1<len(runlabels): continue
            else: runlabels.append(str(i))

    clvals=clvals+1.e-20*(clvals==0) #do this to prevent log plots from complaining about zeros
        
    #make plots
    Nell=len(lvals)
    Ncross=len(crosstags)

    allmycl=clvals.reshape((Nrun,Nell*Ncross))
    allclasscl = np.array([classcl.reshape(Nell*Ncross),]*Nrun) #duplicate class dat 
    alldiff=np.fabs(allmycl-allclasscl)
    allfracdiff=alldiff/np.fabs(allclasscl)
    crosselltags=[]
    for t in crosstags:
        for lind in xrange(Nell):
            crosselltags.append(t+', l={0: 3d}'.format(lvals[lind]))
    for i in xrange(Nell*Ncross):
        xind=i/Nell
        lind=i%Nell
    plottag2=''
    if ONEBINTEST:
        plottag2='onebin'
    plotdir=outdir+'plots/'

    #compare log plots
    maxcl=np.max(np.fabs(allmycl))
    mincl=np.min(np.fabs(allmycl))
    maxcl=max(maxcl,np.max(np.fabs(allclasscl)))
    mincl=min(mincl,np.min(np.fabs(allclasscl)))
    mincl=min(mincl,1.e-16)
    
    plt.figure(0,figsize=(7.5,10))
    ax1=plt.subplot(311)
    plt.title('Comparing my C_l with CLASS'+titlenote)
    plt.ylabel(r'$\left|C_l\right|$')
    plt.xlim((-1,Nell*Ncross))
    plt.ylim((.1*mincl,100*maxcl))
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)

    for n in xrange(Nrun):
        plt.semilogy(np.arange(Ncross*Nell),np.fabs(allmycl[n,:]),markers[n%len(markers)],label=runlabels[n])

    plt.semilogy(np.arange(Ncross*Nell),np.fabs(allclasscl[0,:]),'*',label='full CLASS calc')
    plt.semilogy(np.arange(Ncross*Nell+2)-1,1.e-10*np.ones(Ncross*Nell+2)*2./np.pi,label='int tolerance') #mark the integral tolarance
    plt.legend(loc='lower left', numpoints=1, prop={'size':9})
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    #then plot differences
    maxdiff=np.max(alldiff)
    mindiff=np.min(alldiff)+1.e-28
    maxfdiff = np.max(allfracdiff)
    minfdiff = np.min(allfracdiff)+1.e-28
    
    #difference
    ax2=plt.subplot(312, sharex=ax1)
    plt.ylabel(r'$\left|C_l^{mine}-C_l^{CLASS}\right|$')
    plt.ylim((.1*mindiff,100*maxdiff))
    plt.xlim((-1,Ncross*Nell))
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)
    for n in xrange(Nrun):
        plt.semilogy(np.arange(Ncross*Nell),alldiff[n,:],markers[n%len(markers)],label=runlabels[n])
    plt.legend(numpoints=1, loc='lower right',prop={'size':9})
    plt.setp(ax2.get_xticklabels(), visible=False)
        
    #relative difference
    plt.subplot(313, sharex=ax1)
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)
    plt.ylabel(r'$\left|\left(C_l^{mine}-C_l^{CLASS}\right)/C_l^{CLASS}\right|$')
    plt.ylim((.1*minfdiff,100*maxfdiff))
    plt.xlim((-1,Ncross*Nell))
    for n in xrange(Nrun):
        plt.semilogy(np.arange(Ncross*Nell),allfracdiff[n,:],markers[n%len(markers)],label=runlabels[n])
    plt.legend(numpoints=1,loc='upper left', prop={'size':9})
    plt.xticks(np.arange(Ncross*Nell),crosselltags,rotation='vertical',size=10)

    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(top=.95)
    plt.tight_layout()
    savename=plotdir+plottag+'_Cl'+plottag2+'.png'
    print "writing to file: ",savename
    plt.savefig(savename)
    plt.close()

#----------------------------------------------------------------
def compare_to_class():
    READONLY=1
    REDOCL=0 #only rerun for onebin test, otherwise just read
    REDOILK=0
    ONEBINTEST=0

    DO_mycode_only=1
    DO_mycode_w_classbkgd=0
    DO_mycode_w_classPk=0
    DO_mycode_w_classbkgdPk=0
    DO_mycode_w_sharpkcut=1
    DO_mycode_w_besselxmincut=1
    DO_mycode_w_bothbesselcuts=1
    DO_lowkmax=1
    DO_mycode_w_softtophat=0
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    runtag='classtest'
    newruntag=''
    #runtag='nperlogk0300_edge0.010'

    #set up maps
    if ONEBINTEST:
        lvals = np.array([30])
        runtag+='_1bin'
        kdat = KData(kmin=1.e-5,kmax=10.,nperlogk=10,krcut=10.)
    else:
        lvals = np.array([2,30,100])
        kdat = KData(kmin=1.e-5,kmax=10.,nperlogk=400,krcut=50.)
    if ONEBINTEST:
        zedges=np.array([[.01,.1]])
        binlabels=['lowz']#,'highz','widez']
    else:
        zedges=np.array([[0.01,0.1],[1.,1.1],[.01,3.]])
        binlabels=['lowz','highz','widez']

    maptypes=[]#[bins]
    for i in xrange(zedges.shape[0]):
        maptypes.append(MapType(idtag='mat_'+binlabels[i],zedges=zedges[i,:]))
    dmmaps = get_binmaplist(maptypes)[0]
    maps = dmmaps
    Nmaps = len(maps) #number of maps per krcut
    maptags=get_bintaglist(maps)
    crosstags=get_crosstags(maptags)
    Ncross=len(crosstags)
    pairs=[p for p in itertools.combinations_with_replacement(maptags,2)]
    zmax=max([m.zmax for m in maps])
    
    #---
    mycls=[]
    mycllabels=[]
    #---
    rundat=ClRunData(rundir=outdir,tag=runtag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10)
    if REDOILK:
        print "To get around weird numpy/multiprocessing problem,just running Ilk computation. \n****Rerun with REDOILK=0 to finish."
        for m in maps:
            getIlk_for_binmap(m,rundat,redo=REDOILK)
        return
    else:
        clvals=getCl(maps,rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(clvals)
        mycllabels.append('no CLASS input')

    #===
    if DO_mycode_w_classbkgd or DO_mycode_w_classbkgdPk:
        classbkgdfile='test_output/classdir/classcompare_background.dat'
        classbkgrd = get_class_bkgdcosm(classbkgdfile,rundat.cosm.h0)
    if DO_mycode_w_classPk or DO_mycode_w_classbkgdPk:
        classpkfile='test_output/classdir/classcompare_pk_nl.dat'
        classpk=get_class_pk(classpkfile)
        
    if DO_mycode_w_classbkgd: #use H,r,g,f from class
        print "Getting C_l for my code, class cosmology background."
        classbkgd_rundat=ClRunData(rundir=outdir,tag='classbkgd',cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10,cosm_zrhgf_bkgrd=classbkgrd)
        classbkgd_cl=getCl(maps,classbkgd_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(classbkgd_cl)
        mycllabels.append('CLASS bkgd')
    #---
    if DO_mycode_w_classPk: #use P(k) from class
        print "Getting C_l for my code, class P(k)."
        classpk_rundat=ClRunData(rundir=outdir,tag='classPk',cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10,pk_ext=classpk)
        classpk_cl=getCl(maps,classpk_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(classpk_cl)
        mycllabels.append('CLASS Pk')
    #---
    if DO_mycode_w_classbkgdPk: #use H,r,g,f from class
        print "Getting C_l for my code, class cosmology background."
        classbkgdpk_rundat=ClRunData(rundir=outdir,tag='classbkgdPk',cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10,cosm_zrhgf_bkgrd=classbkgrd,pk_ext=classpk)
        classbkgdpk_cl=getCl(maps,classbkgdpk_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(classbkgdpk_cl)
        mycllabels.append('CLASS bkgd, Pk')
    #---
    if DO_mycode_w_sharpkcut: #past kr=l+krcut, just set bessel function to zero
        print "Getting C_l for my code, sharp k cutoff."
        sharpkcut_rundat=ClRunData(rundir=outdir,tag='sharpkcut'+newruntag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10)
        sharpkcut_cl=getCl(maps,sharpkcut_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        #print sharpkcut_cl
        mycls.append(sharpkcut_cl)
        mycllabels.append('bessel xmax cut')

    #---
    if DO_mycode_w_besselxmincut: #like class, set j_l(x)=0 at low x, when func < epsilon
        print "Getting C_l for my code, besselxmincut."
        besselxmincut_rundat=ClRunData(rundir=outdir,tag='besselxmincut'+newruntag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10)
        besselxmincut_cl=getCl(maps,besselxmincut_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(besselxmincut_cl)
        mycllabels.append('bessel xmin cut')
        
    if DO_mycode_w_bothbesselcuts: #l set j_l(x)=0 at low x, when func < epsilon, and when x>l+krcut
        print "Getting C_l for my code, bothbesselcuts."
        bothbesselcuts_rundat=ClRunData(rundir=outdir,tag='bothbesselcuts'+newruntag,cosmpfile=cosmfile,lvals=lvals,zmax=5.,kdata=kdat,epsilon=1.e-10)
        bothbesselcuts_cl=getCl(maps,bothbesselcuts_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(bothbesselcuts_cl)
        mycllabels.append('both bessel cuts')
    #---
    if DO_lowkmax:
        lowkmax_kdat=KData(kmax=3.,krcutadd=-1,krcutmult=-1)
        lowkmax_rundat=ClRunData(rundir=outdir,tag='lowkmax',cosmpfile=cosmfile,lvals=lvals,zmax=zmax,kdata=lowkmax_kdat,besselxmincut=True)
        lowkmax_cl=getCl(maps,lowkmax_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(lowkmax_cl)
        mycllabels.append('lowkmax')
    #---
    if DO_mycode_w_softtophat: #make tophat edge param .1 rather than .001
        soft_maptypes=[]#[bins]
        for i in xrange(zedges.shape[0]):
            soft_maptypes.append(MapType(idtag='mat_'+binlabels[i],zedges=zedges[i,:],sharpness=.1))
            soft_dmmaps = get_binmaplist(maptypes)[0]
            soft_tophat_maps = soft_dmmaps
        softtophat_rundat=ClRunData(rundir=outdir,tag='softtophat'+newruntag,cosmpfile=cosmfile,lvals=lvals,zmax=zmax,kdata=kdat,epsilon=1.e-10)
        softhat_cl=getCl(soft_tophat_maps,softtophat_rundat,dopairs=pairs,redoAllCl=REDOCL,redoIlk=REDOILK,DoNotOverwrite=READONLY).cl
        mycls.append(softhat_cl)
        mycllabels.append('tophat edge=0.1')
    #---
    softclasshats=0#True
    if softclasshats:
        plottag='classcomp'+newruntag+'_edge0.1'
        titlenote='(CLASS tophat edge=.1)'
        classclfile='test_output/classdir/soft_classcompare_cl.dat'
    else:
        plottag='classcomp'+newruntag+'_edge0.01'
        titlenote='(CLASS tophat edge=.01)'
        classclfile=''# leave blank for default: 'test_output/classdir/classcompare_cl.dat'

    mycls=np.array(mycls)
    classcl=get_class_cl_data(clvals.shape,lvals,Nmaps,Ncross,ONEBINTEST,infile=classclfile)

    make_classcompare_plots(mycls,classcl,lvals,crosstags,outdir,plottag=plottag,titlenote=titlenote,ONEBINTEST=ONEBINTEST,runlabels=mycllabels)

        
#----------------------------------------------------------------
def get_class_bkgdcosm(classfile,h0,zmax=10.):
    #classfile='test_output/classdir/classcompare_background.dat'
    classdat=np.loadtxt(classfile,skiprows=4)
    classz=classdat[::-1,0] #note, -1 because class has z from large to small
    maxind=0
    for i in xrange(len(classz)):
        maxind=i
        if classz[i]>=zmax:
            break
    classz=classz[:maxind]
    c= 299792. #speed of light in km/s
    classr=classdat[::-1,4][:maxind]*h0
    classH=classdat[::-1,3][:maxind]*c/h0 #from 1/Mpc to km/s/Mpc
    classg=classdat[::-1,14][:maxind]/classdat[-1,14] #normalize to present time
    classf=classdat[::-1,15][:maxind]
    zrhgf_fromclass=np.array([classz,classr,classH,classg,classf])
    return zrhgf_fromclass

def get_class_pk(classfile):
    classdat=np.loadtxt(classfile,skiprows=4)
    return classdat #1st col is k, 2nd is P
    
#---------------------------
def test_misc_error():
    REDODATA=0
    outdir = 'test_output/Ilktests/'
    cosmfile = 'testparam.cosm'
    lvals = np.array([2,30])
    kmin=1.e-3
    kmax=10
    nperlogk=100
    krcut=10
    runtag='interptest'
    kdat=KData(kmin=kmin,kmax=kmax,nperlogk=nperlogk,krcut=krcut)
    rundat=ClRunData(rundir=outdir,tag=runtag,cosmpfile=cosmfile,lvals=lvals,zmax=2.,kdata=kdat,epsilon=1.e-10)
    
    mtype=MapType(idtag='mat_widez',zedges=np.array([.01,3.]))
    mtype1=MapType(idtag='mat_lowz',zedges=np.array([.01,.1]))
    binmaps = [mtype.binmaps[0],mtype1.binmaps[0]]

    #getIlk_for_binmap(binmap,rundat,redo=True)
    getCl(binmaps,rundat,redoAllCl=True)

#-----------------
# Looking to answer the question: how high in z should we have
# the ISW Ilk calculations go?
def plot_ISW_Ilk_integrand():
    cosmfile = 'testparam.cosm'
    cosm=Cosmology(cosmfile,tabulateZ=True,zmax=100,nperz=10)
    z=cosm.z_array
    g=cosm.g_array
    h=cosm.H_array
    r=cosm.r_array
    f=cosm.f_array
    H0=cosm.h0*100
    c=cosm.c

    integrand_nobessel=g*h*(1-f)*3*(H0**2)/(c**3)

    plt.figure(0)
    plt.title(r'ISW $I_{\ell}(k)$ integrand sans Bessel and $k^{-2}$')
    plt.ylabel(r'$\left(\frac{3H_0^2}{c^3}\right)\,D(z)\,H(z)\,(1-f(z))$')
    plt.xlabel(r'$z$')
    plt.semilogy(z,integrand_nobessel)
    outdir='test_output/Ilktests/plots/'
    plt.savefig(outdir+'isw_integrandbase.png')
    plt.close()

    kvals=[1.e-5,1.e-3,.1,10.]
    lvals=[1,30,100]#3 values only!
    integrand_klpiece=np.zeros((len(lvals),len(kvals),z.size))
    for lind in xrange(len(lvals)):
        for kind in xrange(len(kvals)):
            integrand_klpiece[lind,kind,:]=np.fabs(sphericalBesselj(lvals[lind],kvals[kind]*r))
    
    plt.figure(1,figsize=(7.5,10))
    ax1=plt.subplot(311)
    #first plot is first ell value
    lind=0
    plt.title('ISW $I_{\ell}(k)$ integrand')
    plt.ylabel(r'$\left(\frac{3H_0^2}{c^3k^2}\right)\,|j_{\ell}(kr)|\,D(z)\,H(z)\,(1-f(z))$')
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)

    for kind in xrange(len(kvals)):
        plt.semilogy(z,integrand_nobessel*integrand_klpiece[lind,kind,:],label='l={0:d}, k={1:f}'.format(lvals[lind],kvals[kind]))

    plt.legend(prop={'size':9})   
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    #second plot is second lval
    ax2=plt.subplot(312, sharex=ax1)
    lind+=1
    plt.ylabel(r'$\left(\frac{3H_0^2}{c^3k^2}\right)\,|j_{\ell}(kr)|\,D(z)\,H(z)\,(1-f(z))$')
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)
    for kind in xrange(len(kvals)):
        plt.semilogy(z,integrand_nobessel*integrand_klpiece[lind,kind,:],label='l={0:d}, k={1:f}'.format(lvals[lind],kvals[kind]))
    plt.legend(prop={'size':9})
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    #last plot is third lval
    ax3=plt.subplot(313, sharex=ax1)
    lind+=1
    plt.ylabel(r'$\left(\frac{3H_0^2}{c^3k^2}\right)\,|j_{\ell}(kr)|\,D(z)\,H(z)\,(1-f(z))$')
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)
    for kind in xrange(len(kvals)):
        plt.semilogy(z,integrand_nobessel*integrand_klpiece[lind,kind,:],label='l={0:d}, k={1:f}'.format(lvals[lind],kvals[kind]))
    plt.legend(prop={'size':9})
  
    plt.tight_layout()
    savename=outdir+'isw_integrand_ex.png'
    print "writing to file: ",savename
    plt.savefig(savename)
    plt.close()
    
#################################################################
if __name__=="__main__":
    #test_cosm_tabs()
    #test_bessel_integration_quad()
    #test_bessel_integration_romberg()
    #test_Ilk_krcut()
    #test_Ilk_nperlogk()
    #eyeball_Ilk_convergence()

    if 0:
        test_Cl_nperlogk() 
        #test_Cl_varykmax()
        test_Cl_varykmin()
        #test_Cl_varykrcut()
        #test_Cl_varykrcut(factorcut=True,newtag='New')
        #test_Cl_varykrcut(factorcut=True,newtag='Newxmincut')
        #test_Cl_varykrcut_addmult()

    #test_new_tophat()
    #test_cosm_vsclass()
    #test_Ilk_initial()

    #compare_to_class()

    #test_misc_error()
    #plot_ISW_Ilk_integrand()
