
import scipy as sp
from numpy import fft
from scipy import special
import scipy.interpolate

from . import myGamma

nmuk = 1000
muk=(sp.arange(nmuk)+0.5)/nmuk
dmuk = 1./nmuk
muk=muk[:,None]

def sinc(x):
    return sp.sin(x)/x

def Pk2Mp(ar,k,pk,ell_max=None):
    """
    Implementation of FFTLog from A.J.S. Hamilton (2000)
    assumes log(k) are equally spaced
    """

    k0 = k[0]
    l=sp.log(k.max()/k0)
    r0=1.

    N=len(k)
    emm=N*fft.fftfreq(N)
    r=r0*sp.exp(-emm*l/N)
    dr=abs(sp.log(r[1]/r[0]))
    s=sp.argsort(r)
    r=r[s]

    xi=sp.zeros([ell_max//2+1,len(ar)])

    for ell in range(0,ell_max+1,2):
        pk_ell=sp.sum(dmuk*L(muk,ell)*pk,axis=0)*(2*ell+1)*(-1)**(ell//2)
        mu=ell+0.5
        n=2.
        q=2-n-0.5
        x=q+2*sp.pi*1j*emm/l
        lg1=myGamma.LogGammaLanczos((mu+1+x)/2)
        lg2=myGamma.LogGammaLanczos((mu+1-x)/2)

        um=(k0*r0)**(-2*sp.pi*1j*emm/l)*2**x*sp.exp(lg1-lg2)
        um[0]=sp.real(um[0])
        an=fft.fft(pk_ell*k**n/2/sp.pi**2*sp.sqrt(sp.pi/2))
        an*=um
        xi_loc=fft.ifft(an)
        xi_loc=xi_loc[s]
        xi_loc/=r**(3-n)
        xi_loc[-1]=0
        spline=sp.interpolate.splrep(sp.log(r)-dr/2,sp.real(xi_loc),k=3,s=0)
        xi[ell//2,:]=sp.interpolate.splev(sp.log(ar),spline)

    return xi

def Pk2Xi(ar,mur,k,pk,ell_max=None):
    xi=Pk2Mp(ar,k,pk,ell_max)
    for ell in range(0,ell_max+1,2):
        xi[ell//2,:]*=L(mur,ell)
    return sp.sum(xi,axis=0)

def Pk2MpRel(ar,k,pk):
    k0 = k[0]
    l=sp.log(k.max()/k0)
    r0=1.

    N=len(k)
    emm=N*fft.fftfreq(N)
    r=r0*sp.exp(-emm*l/N)
    dr=abs(sp.log(r[1]/r[0]))
    s=sp.argsort(r)
    r=r[s]

    nu=sp.zeros([2,len(ar)])

    for ell in [1,3]:
        mu=ell+0.5
        n=1.
        q=2-n-0.5
        x=q+2*sp.pi*1j*emm/l
        lg1=myGamma.LogGammaLanczos((mu+1+x)/2)
        lg2=myGamma.LogGammaLanczos((mu+1-x)/2)

        um=(k0*r0)**(-2*sp.pi*1j*emm/l)*2**x*sp.exp(lg1-lg2)
        um[0]=sp.real(um[0])
        an=fft.fft(pk*k**n*sp.sqrt(sp.pi/2))
        an*=um
        nu_loc=fft.ifft(an)
        nu_loc=nu_loc[s]
        nu_loc/=r**(3-n)
        nu_loc[-1]=0
        spline=sp.interpolate.splrep(sp.log(r)-dr/2,sp.real(nu_loc),k=3,s=0)
        nu[ell//2,:]=sp.interpolate.splev(sp.log(ar),spline)

    return nu

def Pk2XiRel(ar,mur,k,pk,kwargs):
    nu=Pk2MpRel(ar,k,pk)
    return nu[0,:]*L(mur,1)*kwargs["Arel1"] + nu[1,:]*L(mur,3)*kwargs["Arel3"]

### Legendre Polynomial
def L(mu,ell):
    return special.legendre(ell)(mu)

def bias_beta(kwargs, tracer1, tracer2):

    growth_rate = kwargs["growth_rate"]

    beta1 = kwargs["beta_{}".format(tracer1['name'])]
    bias1 = kwargs["bias_{}".format(tracer1['name'])]
    bias1 *= growth_rate/beta1

    beta2 = kwargs["beta_{}".format(tracer2['name'])]
    bias2 = kwargs["bias_{}".format(tracer2['name'])]
    bias2 *= growth_rate/beta2

    return bias1, beta1, bias2, beta2

def ap_at(kwargs):
    if kwargs['SB']:
        ap = 1.
        at = 1.
    else:
        ap = kwargs['ap']
        at = kwargs['at']
    return ap, at

def aiso_epsilon(kwargs):
    if kwargs['SB']:
        ap = 1.
        at = 1.
    else:
        aiso = kwargs['aiso']
        eps = kwargs['1+epsilon']
        ap = aiso*eps*eps
        at = aiso/eps
    return ap, at

def convert_instance_to_dictionary(inst):
    dic = dict((name, getattr(inst, name)) for name in dir(inst) if not name.startswith('__'))
    return dic
