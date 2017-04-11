import numpy as np
import os,subprocess
import copy_reg, types
from scipy.integrate import quad
from scipy.interpolate import interp1d
###########################################################################
#helper functions for multiprocessing; allows you to call instance methods
###########################################################################
# def _pickle_method(method):
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     return _unpickle_method, (func_name, obj, cls)

# def _unpickle_method(func_name, obj, cls):
#     for cls in cls.mro():
#         try:
#             func = cls.__dict__[func_name]
#         except KeyError:
#             pass
#         else:
#             break
#     return func.__get__(obj, cls)
# ###########################################################################
# copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class Cosmology(object):
    """Class containing cosmological params and functions"""
    c =  299792. #speed of light in km/s. units chosen so c/H is in Mpc
    temp_cmb = 2.7255 #CMB temp in Kelvin #added 1/26/17 -JM
    #-----------------------------------
    def __init__(self,paramfile,tabulateZ=False,needPk=False,zmax=1.,nperz=200.,cambdir='output/camb_output/',kmin=-1.,kmax=-1.,rerunCAMB=False,CAMBkmax=-1.,epsilon=1.e-10,bkgd_zrhgf_ext=np.array([]),pk_ext=np.array([])):
#        print " Initializing instance of Cosmology"
        
        self.paramfile = paramfile
        self.importCosmParams(paramfile)
        self.epsilon=epsilon #used for integral tolerance
        self.zmax=0
        #if tabulated cosmology background functions have been passed, supercede zmax, nperz, tabulateZ
        if bkgd_zrhgf_ext.size:
            tabulateZ=True
        if tabulateZ: #we'll set this to false if we don't need new Iilk
            print " ...tabulating z-dep functions up to zmax=",zmax
            self.tabulateZdep(zmax,nperz,zrhgf_ext=bkgd_zrhgf_ext)
        else:
            self.tabZ = 0
        self.nperz = nperz

        self.cambdir = cambdir #default if no Pk needed
        if pk_ext.size:
            needPk=True
            print "  ...Using external power spectrum!"
        if needPk: #set everything up
            self.getPk(kmin,kmax,cambdir,rerunCAMB,CAMBkmax,pk_ext) 
        else: self.havePk =0
    
        #infostr contains info for data file headers
        self.infostr="CAMBtag '{0:s}': Oc={1:0.3g}, Ob={2:0.3g}, h0={3:0.3g}, w0={4:0.3g}, ns={5:0.3g}, On={6:0.3g} [OL=1-Om, Oc=Om-Ob-On. epsilon={7:0.3g}]".format(self.CAMBtag,self.Oc,self.Ob,self.h0,self.w0,self.ns,self.On,self.epsilon)

    #-----------------------------------
    # read in cosmological paramter files, set up instance params
    def importCosmParams(self,paramfile):
#        print "  Importing cosm params from:",paramfile
        f = open(paramfile,'r')
        lines= f.read().split('\n')
        f.close()
        haveOc=0
        haveOb=0
        haveOn=0
        for l in lines:
            label = l[:l.find('=')].strip()
            data = l[l.find('=')+1:l.find('#')].strip()
            if label=='CAMBtag':
                self.CAMBtag = data
            elif label=='w0':
                self.w0 = float(data)
                if self.w0!=-1: print 'WARNING: w0!=-1; growth function innaccurate.!'
            elif label =='Oc':
                self.Oc = float(data)
                haveOc=1
            elif label =='Ob':
                self.Ob = float(data)
                haveOb=1
            elif label =='On':
                self.On = float(data)
                haveOn=1
            elif label =='Och2':
                self.Och2 = float(data)
            elif label =='Obh2':
                self.Obh2 = float(data)
            elif label =='Onh2':
                self.Onh2 = float(data)
            elif label =='h0':
                self.h0 = float(data)
            elif label =='ns':
                self.ns = float(data)
        if haveOc:
            self.Och2=self.Oc*self.h0*self.h0
        else:
            self.Oc = self.Och2/(self.h0*self.h0)
        if haveOb:
            self.Obh2=self.Ob*self.h0*self.h0
        else:
            self.Ob = self.Obh2/(self.h0*self.h0)
        if haveOn:
            self.Onh2=self.On*self.h0*self.h0
        else:
            self.On = self.Onh2/(self.h0*self.h0)
        self.Om = self.Oc+self.Ob+self.On
        self.OL = 1.-self.Om
    #-----------------------------------
    def tabulateZdep(self,zmax=10,nperz=200,zrhgf_ext=np.array([]),outtag='',overwritefile=False):
        #make z array, tabulate z-dependent quantities, save as member arrays
        # can pass optional array zrhgf [first index specifies z,r,H,g, or f][z index]
        # -if this is non empty, supercedes other argumetns
        self.tabZ=1 #used to see if this fn has been called
        pf = self.paramfile
        cosmtag=pf[pf.rfind('/')+1:pf.rfind('.cosm')]
        if outtag:
            addtotag='_'+outtag
        else:
            addtotag=''
        tabzfile=self.cambdir+'{0:s}_tabz_zmax{1:f}_nperz{2:f}.dat'.format(cosmtag+addtotag,zmax,nperz)

        print "  Tabulating background cosmology fns. zmax={0:g}, nperz={1:g}".format(zmax,nperz)
        if zrhgf_ext.size:
            print '  using external tabulation...'
            self.z_array=zrhgf_ext[0,:]
            self.r_array=zrhgf_ext[1,:]
            #self.H_array=zrhgf_ext[2,:]
            self.g_array=zrhgf_ext[3,:]
            self.f_array=zrhgf_ext[4,:]
            self.zmax=self.z_array[-1]
            self.nperz=self.z_array.size/self.zmax
        else:

            if not overwritefile and os.path.isfile(tabzfile):
                #check for existing file
                print '    using z data from',tabzfile
                #if it exists, read it in
                inzrhgf=np.loadtxt(tabzfile,skiprows=0)#1)#set to 1 if header included
                #for now, assumes that this is the array we want
                
                self.z_array=inzrhgf[:,0]
                self.r_array=inzrhgf[:,1]
                self.H_array=inzrhgf[:,2]
                self.g_array=inzrhgf[:,3]
                self.f_array=inzrhgf[:,4]
                self.zmax=self.z_array[-1]
                self.nperz=self.z_array.size/self.zmax
            else:
                self.zmax=zmax
                self.nperz=nperz
                dz = 1./nperz
                Nz = zmax*nperz+1
                self.z_array = dz*np.arange(Nz)
                self.r_array = np.zeros_like(self.z_array) #Mpc/h units
                self.H_array = np.zeros_like(self.z_array) #h km/s/Mpc
                self.g_array = np.zeros_like(self.z_array)
                self.f_array = np.zeros_like(self.z_array)
                for i in xrange(self.z_array.size):
                    self.r_array[i]=self.comov_r_z(self.z_array[i])
                    self.H_array[i]=self.Hubble(self.z_array[i])
                    self.g_array[i]=self.D1(self.z_array[i])
                    self.f_array[i]=self.fgrowth(self.z_array[i])
                
                self.g_array = self.g_array/self.D1(0) #normalize to 1 today

                #write arrays to file
                zrhgf_grid=np.zeros((self.z_array.size,5))
                zrhgf_grid[:,0]=self.z_array
                zrhgf_grid[:,1]=self.r_array
                zrhgf_grid[:,2]=self.H_array
                zrhgf_grid[:,3]=self.g_array
                zrhgf_grid[:,4]=self.f_array
                #numpy version on flux can't handle header arg, take it out
                #header='columns: z r h g f; cosm file {0:s}; zmax {1:f}; nperz {2:f}'.format(cosmtag,zmax,nperz)
                print '    Saving z tab data to ',tabzfile
                np.savetxt(tabzfile,zrhgf_grid)#,header=header)#header removed for flux's numpy version
                
        #set up interpolating functions
        print '     Setting up interpolating functions.'
        self.co_r = interp1d(self.z_array,self.r_array,kind='cubic',bounds_error=False,fill_value=0.)
        self.z_from_cor= interp1d(self.r_array,self.z_array,kind='cubic',bounds_error=False,fill_value=0.)
        self.hub =self.Hubble#interp1d(self.z_array,self.H_array,kind='cubic') #analytic!
        self.growth =interp1d(self.z_array,self.g_array,kind='cubic',bounds_error=False,fill_value=0.)
        self.growthrate = interp1d(self.z_array,self.f_array,kind='cubic',bounds_error=False,fill_value=0.)
        print '     Tabulation done.'
           
    #==================================================
    # getPk: either read in CAMB file, or run CAMB then read in file
        #by default, CAMB has kmax=100,kmin auto
        # if rerunCAMB=True, CAMBkmax>0, will use camb for up to CAMBkmax
        # if more range needed for run, can set kmin,kmax;
        #  this will stick appropriate power laws onto either end if P(k)
        #if external [k,P] array passed, this overrides everything else
    def getPk(self,kmin=-1.,kmax=-1.,cambdir='',rerunCAMB=False,CAMBkmax=-1.,kperln=0,pk_ext=np.array([])):
        print "  In getPk"
        if not pk_ext.size:
#            print "  ...cambdir=",cambdir,"rerunCAMB=",rerunCAMB
#            print "  ...kmin=",kmin,"kmax=",kmax,"CAMBkmax=",CAMBkmax
            if CAMBkmax<0:
                CAMBkmax=kmax
            self.havePk = 1 #use to check whether this has been run
            if cambdir: #if something given here, make it the instance's cambdir
                self.cambdir=cambdir
            infile = ''.join([self.cambdir,self.CAMBtag,'_matterpower.dat'])
            if not os.path.isdir(self.cambdir): #no camb dir or files exist
                print "  Creating dir for CAMB output:",self.cambdir
                os.mkdir(self.cambdir)
            #camb dir exists, file doesn't
            if (not os.path.isfile(infile)) or rerunCAMB:
                print "  Running CAMB to make file: ",infile
                self.runCAMB(CAMBkmax,int(kperln))
        else:
            print "  ...using extenral P(k) array."    
        #now there will be a CAMB-produced matter power spec file, read it:
        self.importP(infile,kmin,kmax,pk_ext) # in Mpc/h units
        
    #-----------------------------------
    #once getPk has been run, use this as interpolating function
    # can be passed either a value of k or array of k values
    # returns interpolated power spectrum for those values
    #def P(self,k):
    #    return np.interp(k,self.k_forPower,self.P_forPower)

    #==================================================
    #Functions computed from cosm. params
    def comov_r_z(self,z):
        #given redshift, compute comoving radial coordinate in Mpc
        r,dr = quad(lambda x: self.c/self.Hubble(x),0,z,epsabs=self.epsilon,epsrel=self.epsilon)
        #r = romberg(lambda x: self.c/self.Hubble(x),0,z,tol=self.epsilon)
        return r
    #-----------------------------------
    def Hubble(self,z):
        #given redshift and cosm. params, compute H(z) in h*km/s/Mpc units
        H0 = 100
        ainv = 1.+z
        ev2 = self.Om*ainv**3.+self.OL*ainv**(3.*(1.+self.w0))
        return H0*np.sqrt(ev2)
    #-----------------------------------
    def D1(self,z):
        #linear growth function
        H0 = 100
        a = 1./(1.+z)
        #doing integral over a to have finite edges
        I,dI = quad(lambda x: (H0/(x*self.Hubble(-1.+1./x)))**3,0,a,epsabs=self.epsilon,epsrel=self.epsilon)
        #smalla = 1.e-10
        #I = romberg(lambda x: (H0/(x*self.Hubble(-1.+1./x)))**3,smalla,a,tol=self.epsilon)
        D1 = 5.*self.Om/2.
        D1*= self.Hubble(z)/H0
        D1*=I
        return D1
    #-----------------------------------
    def fgrowth(self,z): #dlnD/dlnA
        ainv = 1.+z
        ev2 = self.Om*ainv**3.+self.OL*ainv**(3.*(1.+self.w0))#(H/H0)^2
        D = self.D1(z)
        A = -1.5
        B = -1.5*self.w0*self.OL*ainv**(3.*(1.+self.w0))/ev2
        C = 2.5*self.Om*(ainv**2)/(D*ev2)
        return A+B+C
    #-----------------------------------
    # reads in P(k) from an external file, adds power laws on high and lowk ends
    #  if necessary. if external array passed, use that instead
    def importP(self,infile='',kmin=-1.,kmax=-1.,pk_ext=np.array([])):
        #kmax=kmax #add enough power law to make sure we cover high k end
        if pk_ext.size:
            print "  Getting P(k) from an external array."
            k=pk_ext[:,0]
            P=pk_ext[:,1]
        else:
            print "  Reading in P(k) from CAMB file: \n       ",infile
            x = np.loadtxt(infile)
            k = x[:,0]   #in units of h/Mpc
            P = x[:,1]   #in units of h^{-3} Mpc^3
        kperlog=300
        if kmin>0 and k[0]>kmin: #add some P\propto k^ns to low k region
            print "  ...appending power law to  low-k end of P(k)"
            addtostart = kperlog*(np.int(np.log10(k[0])-np.log10(kmin))+1)
            A = P[0]*(k[0]**(-self.ns)) #normalizaiton for primordial
            knew = np.zeros(addtostart+k.size)
            Pnew = np.zeros(addtostart+k.size)
            for n in xrange(knew.size):
                if n<addtostart:
                    knew[n]= 10**(np.log10(k[0]) - (addtostart-n)/kperlog)
                    #knew[n] = 10**(np.log10(kmin)+n/kperlog)
                    Pnew[n] = A*(knew[n]**self.ns) #primordial power spec
                else:
                    knew[n] = k[n-addtostart]
                    Pnew[n] = P[n-addtostart]
            k = knew
            P = Pnew
        if kmax>0 and k[-1]<kmax:
            print "  ...appending power law to high-k end of P(k)"
            addtoend = kperlog*(np.int(np.log10(kmax)-np.log10(k[-1]))+1)
            #here power spec is nonlinear, somewhere between -2 and -3
            #just get some approximation... P\propto k^x
            x = np.log10(P[-10]/P[-1])/np.log10(k[-10]/k[-1])
            B = P[-1]/(k[-1]**x) #matching normalization
            knew = np.zeros(addtoend+k.size)
            Pnew = np.zeros(addtoend+k.size)
            for n in xrange(knew.size):
                if n<k.size:
                    knew[n]=k[n]
                    Pnew[n]=P[n]
                else:
                    knew[n] =  10**(np.log10(k[-1])+(1+n-k.size)/kperlog)
                    Pnew[n] = B*(knew[n]**x)
            k = knew
            P = Pnew
        self.k_forPower = k
        self.P_forPower = P
        #set up interpolating function
        self.P = interp1d(self.k_forPower,self.P_forPower,bounds_error=False,fill_value=0.)
        return self.k_forPower,self.P_forPower
    #----------------------------------------------------------
    def runCAMB(self,kmax=-1,kperlog=0):
        #given instance of Cosmology class, cosm
        #runs CAMB with approcpriate paramters
        cosm = self
        kmax = kmax
        if kmax<0:
            kmax=100. #100 h/Mpc 
        outdir = cosm.cambdir
        inifile = outdir+cosm.CAMBtag+'_params.ini'
        print '  ...Creating parameter file:',inifile
        f = open(inifile,'w')
        f.write(''.join(["output_root = "+outdir,cosm.CAMBtag,"\n",\
                             "get_scalar_cls = F\n","get_vector_cls = F\n",\
                             "get_tensor_cls = F\n", "get_transfer   = T\n",\
                             "do_lensing     = F\n","do_nonlinear = 1\n"]))
        #These may only be necessary for Cl's...
        f.write(''.join(["l_max_scalar      = 2200\n","l_max_tensor = 1500\n",\
                             "k_eta_max_tensor  = 3000\n"]))
        #set up paramters
        f.write(''.join(["use_physical   = T\n","hubble = ",str(cosm.h0*100.),\
                             " \n", "w = ",str(cosm.w0),"\n","cs2_lam  = 1\n",\
                             "ombh2   = ",str(cosm.Obh2),\
                             "\n","omch2      = ",str(cosm.Och2),"\n",\
                             "omk   = 0\n",\
                             "omnuh2 = ",str(cosm.Onh2),"\n",\
                             "temp_cmb = 2.7255\n","helium_fraction = 0.24\n"]))
        #For neutrinos, using defaults from CAMB's example file
        f.write(''.join(["massless_neutrinos = 3.046 \n",\
                             "nu_mass_eigenstates = 1 \n",\
                             "massive_neutrinos  = 0\n",\
                             "share_delta_neff = T\n",\
                             "nu_mass_fractions = 1\n\n"]))
        #Initial power spectrum, amplitude, spectral index and running
        f.write(''.join(["initial_power_num = 1\n","pivot_scalar = 0.05 \n",\
                             "pivot_tensor = 0.05 \n",\
                             "scalar_amp(1) = 2.1e-9\n",\
                             "scalar_spectral_index(1)  = ",str(cosm.ns),"\n",\
                             "scalar_nrun(1) = 0\n","scalar_nrunrun(1)  = 0\n",\
                             "tensor_spectral_index(1)  = 0\n",\
                             "tensor_nrun(1)  = 0\n",\
                             "tensor_parameterization   = 1 \n",\
                             "initial_ratio(1)  = 1\n",\
                             "reionization         = F \n"]))
        #didn't include several reionization params, as marking it F
        f.write(''.join(["initial_condition   = 1\n"])) #adiabatic
        f.write(''.join(["vector_mode = 0\n","COBE_normalize = F\n",\
                             "CMB_outputscale = 7.42835025e12\n"]))
        #Transfer function settings
        f.write(''.join(["transfer_high_precision = T\n",\
                             "transfer_kmax = ",str(kmax),"\n",\
                             "transfer_k_per_logint   = ",str(kperlog),"\n",\
                             "transfer_num_redshifts= 1\n",\
                             "transfer_interp_matterpower = T\n",\
                             "transfer_redshift(1)= 0 \n",\
                             "transfer_filename(1) = transfer_out.dat\n"]))
        #Matter power spectrum output against k/h in units of h^{-3} Mpc^3
        f.write(''.join(["transfer_matterpower(1) = matterpower.dat \n",\
                             "feedback_level = 1\n"]))#1 to print out some info
        f.write(''.join(["derived_parameters = T\n","massive_nu_approx = 1\n",\
                             "accurate_polarization = F\n",\
                             "accurate_reionization = F\n",\
                             "do_tensor_neutrinos = F\n",\
                             "do_late_rad_truncation   = T\n",\
                             "number_of_threads = 0\n", \
                             "high_accuracy_default=T \n",\
                             "accuracy_boost = 1 \n","l_accuracy_boost  = 1 \n"
                         "l_sample_boost          = 1 \n"]))
        f.close()
        print "  ...Calling CAMB."
        p =subprocess.call(''.join(['camb ' , inifile,' > ',outdir,cosm.CAMBtag,'.log']), shell=True)
        print p
        print cosm.CAMBtag
    #----------------------------------------------------------
