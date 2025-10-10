#!/usr/bin/env python3


def GetBeta(T):
    """Returns 1/kT in units of (kcal/mol)^{-1}

    Parameters
    ----------
    T : float
        Temperature (K)

    Returns
    -------
    beta : float
        The value of 1/kT, where k is Boltzmann's constant
    """
    import scipy.constants
    Jperkcal = scipy.constants.calorie * 1000 / scipy.constants.Avogadro
    boltz = scipy.constants.Boltzmann / Jperkcal
    return 1./(boltz*T)


def CptSSCSched(N,alpha):
    """Returns a N-point schedule that scales between SSC(0) and SSC(2)

    If alpha == 0, then a SSC(0) schedule is returned.
    If alpha == 1, then a SSC(1) schedule is returned.
    If alpha == 2, then a SSC(2) schedule is returned.
    If 0 < alpha < 1, the schedule is a mixture of SSC(0) and SSC(1).
    If 1 < alpha < 2, the schedule is a mixture of SSC(1) and SSC(2).
    
    The schedule is determined by finding all unique real roots of the
    the N polynomials, where i = [0,N-1].

    SSC(x;alpha,i) = c_{5}(alpha) x^5 + c_{4}(alpha) x^4 + c_{3}(alpha) x^3 
                     + c_{2}(alpha) x^2 + c_{1}(alpha) x - i/(N-1)
    
    c_{1}(alpha) = { 1,             if alpha = 0
                     1-alpha,       if 0 < alpha < 1
                     0,             if 1 < alpha }

    c_{2}(alpha) = { 0,             if alpha = 0
                     3 alpha,       if 0 < alpha < 1
                     3 (2-alpha),   if 1 < alpha }

    c_{3}(alpha) = { 0,             if alpha = 0
                     -2 alpha,      if 0 < alpha < 1
                     10 (alpha-1) - 2 (2-alpha), if 1 < alpha }

    c_{4}(alpha) = { 0,             if alpha <= 1
                     -15 (alpha-1), if 1 < alpha }

    c_{5}(alpha) = { 0,             if alpha <= 1
                     6 (alpha-1),   if 1 < alpha }

    Parameters
    ----------
    N : int
        The size of the schedule. N >= 2

    alpha : float
        The schedule type. 0 <= alpha <= 2

    Returns
    -------
    lams : numpy.ndarray, shape=(N,), dtype=float
        The sorted list of lambda values
    """

    import numpy as np

    if N < 2:
        raise Exception(f"Expected N>=2, but received {N}")
    if alpha < 0 or alpha > 2:
        raise Exception(f"Expected 0 <= alpha <= 2, but received {alpha}")

    x = np.linspace(0,1,N)
    lams = [0.]
    for i in range(1,N-1):
        if alpha <= 1.:
            a_uniform = np.array([  0., 0., 1., -x[i] ])
            a_ssc1 =    np.array([ -2., 3., 0., -x[i] ])
            a = alpha * a_ssc1 + (1.-alpha) * a_uniform
        else:
            a_ssc1 =    np.array([ 0.,   0., -2., 3., 0., -x[i] ])
            a_ssc2 =    np.array([ 6., -15., 10., 0., 0., -x[i] ])
            a = (alpha-1.) * a_ssc2 + (2.-alpha) * a_ssc1
        r = np.roots(a)
        for j in range(len(r)):
            if (np.iscomplex(r[j]) == 0) & ((r[j] in lams) == 0):
                val = np.real(r[j])
                if val >= 0 and val <= 1:
                    lams.append(val)
        #print(i,lams)
        lams[i] = round(lams[i],6)
    lams.append(1.)
    lams = np.array(lams)
    lams.sort()
    lams[0] = 0.
    lams[-1] = 1.
    #print("lams=",alpha,lams)
    if len(lams) != N:
        raise Exception(f"Failed to obtain a {N}-point schedule."+
                        f" Produced {len(lams)} values")

    return lams


def CptGeneralizedSSCSched(N,alpha0,alpha1=None):
    """Returns a N-point schedule that based on a generalized SSC function.
    The SSC(alpha) function uses a fixed value of alpha to obtain construct
    a symmetric schedule. The generalized form calculated here uses a
    lambda-dependent alpha; e.g., SSC(alpha(lambda)), which scales from
    alpha(0) = alpha0 to alpha(1) = alpha1

    If alpha == 0, then a SSC(0) schedule is returned.
    If alpha == 1, then a SSC(1) schedule is returned.
    If alpha == 2, then a SSC(2) schedule is returned.
    If 0 < alpha < 1, the schedule is a mixture of SSC(0) and SSC(1).
    If 1 < alpha < 2, the schedule is a mixture of SSC(1) and SSC(2).
    
    The schedule is determined by finding all unique real roots of the
    the N polynomials, where i = [0,N-1].

    SSC(x;alpha_i,i) = c_{5}(alpha_i) x^5 + c_{4}(alpha_i) x^4 
                     + c_{3}(alpha_i) x^3 
                     + c_{2}(alpha_i) x^2 + c_{1}(alpha_i) x - i/(N-1)
    
    where alpha_i = alpha0 + (alpha1-alpha0) * i / (N-1), and the
    coefficients are:

    c_{1}(alpha) = { 1,             if alpha = 0
                     1-alpha,       if 0 < alpha < 1
                     0,             if 1 < alpha }

    c_{2}(alpha) = { 0,             if alpha = 0
                     3 alpha,       if 0 < alpha < 1
                     3 (2-alpha),   if 1 < alpha }

    c_{3}(alpha) = { 0,             if alpha = 0
                     -2 alpha,      if 0 < alpha < 1
                     10 (alpha-1) - 2 (2-alpha), if 1 < alpha }

    c_{4}(alpha) = { 0,             if alpha <= 1
                     -15 (alpha-1), if 1 < alpha }

    c_{5}(alpha) = { 0,             if alpha <= 1
                     6 (alpha-1),   if 1 < alpha }

    Parameters
    ----------
    N : int
        The size of the schedule. N >= 2

    alpha0 : float
        The schedule type at lambda=0. 0 <= alpha <= 2

    alpha1 : float, default=None
        The schedule type at lambda=1. 0 <= alpha <= 2
        If unused, it is set to alpha0

    Returns
    -------
    lams : numpy.ndarray, shape=(N,), dtype=float
        The sorted list of lambda values
    """

    import numpy as np

    if alpha1 is None:
        alpha1 = alpha0
    
    if N < 2:
        raise Exception(f"Expected N>=2, but received {N}")
    if alpha0 < 0 or alpha0 > 2:
        raise Exception(f"Expected 0 <= alpha <= 2, but received {alpha}")
    if alpha1 < 0 or alpha1 > 2:
        raise Exception(f"Expected 0 <= alpha <= 2, but received {alpha}")

    x = np.linspace(0,1,N)
    alphas = np.linspace(alpha0,alpha1,N)
    lams = [0.]
    for i in range(1,N-1):
        alpha = alphas[i]
        if alpha <= 1.:
            a_uniform = np.array([  0., 0., 1., -x[i] ])
            a_ssc1 =    np.array([ -2., 3., 0., -x[i] ])
            a = alpha * a_ssc1 + (1.-alpha) * a_uniform
        else:
            a_ssc1 =    np.array([ 0.,   0., -2., 3., 0., -x[i] ])
            a_ssc2 =    np.array([ 6., -15., 10., 0., 0., -x[i] ])
            a = (alpha-1.) * a_ssc2 + (2.-alpha) * a_ssc1
        r = np.roots(a)
        for j in range(len(r)):
            if (np.iscomplex(r[j]) == 0) & ((r[j] in lams) == 0):
                val = np.real(r[j])
                if val >= 0 and val <= 1:
                    lams.append(val)
                    break
        lams[i] = round(lams[i],6)
    lams.append(1.)
    
    lams = np.array(lams)
    lams.sort()
    lams[0] = 0.
    lams[-1] = 1.
    #print("lams=",alpha,lams)
    if len(lams) != N:
        raise Exception(f"Failed to obtain a {N}-point schedule."+
                        f" Produced {len(lams)} values")

    return lams



def TransformLL2UV(lam1,lam2):
    """Performs a coordinate transformation between lam1,lam2 to u,v
    where u and v are the distances perpendicular and parallel to the
    diagonal

    Parameters
    ----------
    lam1 : float
        Lambda 1 (the x-axis). Restriction: 0 <= lam1 <= 1

    lam2 : float
        Lambda 2 (the y-axis). Restriction: 0 <= lam2 <= 1
    
    Returns
    -------
    u : float
        The distance perpendicular to the diagonal

    v : float
        The distance parallel to the diagonal
    """
    import numpy as np
    c = 1./np.sqrt(2)
    u = c * ( lam1 - lam2 )
    v = c * ( lam1 + lam2 )
    return u,v


def TransformUV2LL(u,v):
    """Performs a reverse coordinate transformation between u,v and lam1,lam2
    where u and v are the distances perpendicular and parallel to the
    diagonal

    Parameters
    ----------
    u : float
        The distance perpendicular to the diagonal

    v : float
        The distance parallel to the diagonal

    Returns
    -------
    lam1 : float
        Lambda 1 (the x-axis)

    lam2 : float
        Lambda 2 (the y-axis)
    """
    import numpy as np
    c = 1./np.sqrt(2)
    lam1 = c * ( u + v )
    lam2 = c * ( -u + v )
    return lam1, lam2


def CptGaussianExp(u,s):
    """Returns the exponent, z, satisfying: exp(-z*u**2) = s

    Parameters
    ----------
    u : float
        The distance perpendicular to the lam1,lam2 diagonal

    s : float
        The value of the Gaussian

    Returns
    -------
    z : float
        The Gaussian exponent fit to match the value
    """
    import numpy as np
    # exp(-z*u**2) = s
    # z*u**2 = -log(s)
    # z = -log(s) / u**2
    return -np.log(s+1.e-60) / (u*u)


def CptSlaterExp(u,s):
    """Returns the exponent, z, satisfying: exp(-z*|u|) = s

    Parameters
    ----------
    u : float
        The distance perpendicular to the lam1,lam2 diagonal

    Returns
    -------
    z : float
        The Slater exponent
    """
    import numpy as np
    # exp(-z*|u|) = s
    # z*|u| = -log(s)
    # z = -log(s) / |u|
    return -np.log(s+1.e-60) / np.abs(u)


def CptGaussianFit(xs):
    """Returns the mean and variance of a series of observations

    Parameters
    ----------
    xs : list of float
        The observations

    Returns
    -------
    mean : float
        The mean observation
    var : float
        The variance of the observations
    """
    import numpy as np
    return np.mean(xs),np.var(xs,ddof=1)


def CptGaussianOverlap(xs,ys):
    """Approximates the distribution overlap by fitting each set of
    observations to a Normal distribution and analytically calculating
    their overlap

    Parameters
    ----------
    xs : list of float
        The observations from distribution 1

    ys : list of float
        The observations from distribution 2

    Returns
    -------
    s12 : float
        The overlap between the normal distributions
    """
    import numpy as np
    ma,va = CptGaussianFit(xs)
    mb,vb = CptGaussianFit(ys)
    za = 0.5 / va
    zb = 0.5 / vb
    zab = za*zb/(za+zb)
    return np.sqrt(zab/np.pi) * np.exp( -zab * (ma-mb)**2 )


def CptPhaseSpaceOverlapIndex(xs,ys):
    """Calculates the phase space overlap index, pso = s12 / max(s11,s22)

    Parameters
    ----------
    xs : list of float
        The observations from distribution 1

    ys : list of float
        The observations from distribution 2

    Returns
    -------
    pso : float
        The phase space overlap index
    """
    saa = CptGaussianOverlap(xs,xs)
    sbb = CptGaussianOverlap(ys,ys)
    sab = CptGaussianOverlap(xs,ys)
    return sab / max( saa,sbb )


def AvgAcceptanceRatio(xs,T):
    """Returns the average Boltzmann factor <min(1,exp(-beta*x))>

    Parameters
    ----------
    xs : list of float
        The potential energies

    T : float
        The temperature (K)

    Returns
    -------
    avg : float
        The average Boltzmann factor
    """
    import numpy as np
    beta = GetBeta(T)
    zs = -beta*xs
    zs = np.where( zs > 300., 300., zs )
    px = np.exp(zs)
    px = np.where( px > 1, 1, px )
    mx = np.mean(px)
    return mx


def CptKullbackLeibler(xs,ys):
    """Calculates the Kullback-Leibler Divergence of two
    normal distributions. Note: This is not symmetric.

    Given probability distributions p(x) and q(x),

    KL(p,q) = \int p(x) ln(q(x)/p(x)) dx

    Parameters
    ----------
    xs : list of float
        The observations from distribution 1

    ys : list of float
        The observations from distribution 2

    Returns
    -------
    KL : float
        The KL Divergence
    """
    import numpy as np
    ma,va = CptGaussianFit(xs)
    mb,vb = CptGaussianFit(ys)
    KL = np.log(np.sqrt(vb/va)) + (va+(ma-mb)**2)/(2*vb) - 0.5
    return KL


def CptSymmetricKullbackLeibler(xs,ys):
    """Calculates the symmetrized Kullback-Leibler Divergence 
    of two normal distributions.

    The symmetrized form computed here is:

    SKL(p,q) = ( KL(p,q) + KL(q,p) )/2

    Parameters
    ----------
    xs : list of float
        The observations from distribution 1

    ys : list of float
        The observations from distribution 2

    Returns
    -------
    KL : float
        The KL Divergence
    """
    import numpy as np
    ma,va = CptGaussianFit(xs)
    mb,vb = CptGaussianFit(ys)
    KL1 = np.log(np.sqrt(vb/va)) + (va+(ma-mb)**2)/(2*vb) - 0.5
    KL2 = np.log(np.sqrt(va/vb)) + (vb+(ma-mb)**2)/(2*va) - 0.5
    SKL = (KL1+KL2)/2
    return SKL




def DigitizeSched(x,N):
    """Returns a list of formatted strings that express each float with %.Nf,
    where N is provided as an input parameter

    Parameters
    ----------
    x : list of float
        The list of numbers to format

    N : int
        The number of trailing digits

    Returns
    -------
    ss : list of string
        The string-representation of the numbers
    """
    n = len(x)
    fmt = "%." + "%if"%((N))
    ss = [ fmt%(x) for x in x ]
    return ss


def DigitizeSymSched(x,ndigits):
    """Returns a list of formatted strings that express each float with %.Nf,
    where N is provided as an input parameter. The values x[i] and 1-x[-i] are
    averaged before printing. This is used to print symmetric schedules.

    Parameters
    ----------
    x : list of float
        The list of numbers to format

    N : int
        The number of trailing digits

    Returns
    -------
    ss : list of string
        The string-representation of the numbers
    """
    n = len(x)
    fmt = "%." + "%if"%((ndigits+1))
    slams = [ fmt%(a) for a in x ]
    for i in range(n//2):
        a = x[i]
        b = 1-x[-1-i]
        tfmt = "%." + "%if"%((ndigits))
        fa = float(tfmt%(a))
        fb = float(tfmt%(b))
        f = 0.5*(fa+fb)
        slams[i] = fmt%(f)
        slams[-1-i] = fmt%(1-f)
    return slams



##############################################################################
##############################################################################

class SchedInfo(object):
    def __init__(self,lams,vals):
        import numpy as np
        self.lams = lams
        self.N = len(self.lams)
        self.midpts = np.array( [ 0.5*(lams[i+1]+lams[i])
                                  for i in range(self.N-1) ] )
        self.gaps = np.array( [ (lams[i+1]-lams[i])
                                for i in range(self.N-1) ] )
        self.maxgap = np.amax(self.gaps)
        self.vals   = np.array(vals,copy=True)
        self.mean   = np.mean(self.vals)
        self.std    = np.std(self.vals,ddof=1)
        self.minval = np.amin(self.vals)
        self.maxval = np.amax(self.vals)

        
    def Save(self,fname):
        fh = open(fname,"w")
        for lam in self.lams:
            fh.write("%.8f\n"%(lam))
        fh.close()
        
        
    def __str__(self):
        msg = "Schedule:\n"
        for lam in self.lams:
            msg += "  %.8f\n"%(lam)
        msg += "\n"
        msg += "Midpoints, Gaps, and Gap-Values:\n"
        for i in range(len(self.vals)):
            msg += "  %8.4f %8.4f %8.4f\n"%(self.midpts[i],self.gaps[i],self.vals[i])
        msg += "\n"
        msg += "Max Gap:    %.4f\n"%(self.maxgap)
        msg += "Mean Value: %.4f\n"%(self.mean)
        msg += "Std Value:  %.4f\n"%(self.std)
        msg += "Max Value:  %.4f\n"%(self.maxval)
        msg += "Min Value:  %.4f\n"%(self.minval)
        # string to be copied into input file
        msg += f"lambdas = [{' '.join(f'{lam:.8f}' for lam in self.lams)}]"
        return msg

            
        
##############################################################################
##############################################################################

class ParamMap(object):
    """A class that controls the mapping between the free parameters and the
    lambda schedule. This version uses N-2 parameters for an asymmetric 
    schedule with fixed lambda=0 and lambda=1 end-points.

    Attributes
    ----------
    N : int
        The size of the schedule

    Nfree : int
        The number of free parameters

    Xfree0 : numpy.ndarray, shape=(N-2,), dtype=float
        The initial free parameter values

    Methods
    -------
    """
    def __init__(self,N):
        """Construct an asymmetric schedule consisting of N points"""
        import numpy as np
        self.N = N
        self.Nfree = N-2
        self.Xfree0 = np.linspace(0,1,self.N)[1:-1]

    def GetNumFree(self):
        return self.Nfree

    def GetSize(self):
        return self.N

    def GetInitGuess(self):
        import numpy as np
        return np.array(self.Xfree0,copy=True)

    def SetGuess(self,x):
        """Resets the initial free parameters

        Parameters
        ----------
        x : list of float
            The schedule consisting of N points

        Returns
        -------
        None
            This method modifies Xfree0 as a side-effect
        """
        import numpy as np
        if len(x) != self.N:
            raise Exception(f"Expected schedule of length {self.N},"+
                            f" but received {len(x)}")
        self.Xfree0 = np.array(x,copy=True)[1:1+self.Nfree]

        
    def GetSched(self,xfree):
        """Returns the N-point schedule from a list of Nfree parameters

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters

        Returns
        -------
        x : list of float, len(x) == N
            The lambda schedule
        """
        import numpy as np
        return np.array( [0] + [a for a in xfree] + [1] )

    
    def Digitize(self,x,ndigits):
        """Returns a schedule with rounded values

        Parameters
        ----------
        x : list of float, len(x) == N
            The schedule

        ndigits : int
            The number of trailing digits in the format specification

        Returns
        -------
        ss : list of str, len(ss) == N
            The rounded values
        """
        import numpy as np
        n = len(x)
        if n != self.N:
            raise Exception(f"Expected size {self.N}, received {n}")
        return np.array( [float(a) for a in DigitizeSched(x,ndigits)] )

    
    def DigitizeFromFreeParams(self,xfree,ndigits):
        """Returns a schedule with rounded digits

        Parameters
        ----------
        xfree : list of float, len(x) == Nfree
            The free parameters

        ndigits : int
            The number of trailing digits in the format specification

        Returns
        -------
        ss : list of float, len(ss) == N
            The rounded values
        """
        import numpy as np
        if len(xfree) != self.Nfree:
            raise Exception(f"len(xfree) is {len(xfree)}, "+
                            f"but was expecting {self.Nfree}")
        return np.array( [float(a) for a in
                          self.Digitize(self.GetSched(xfree),ndigits)] )

    
    def GetLowerBounds(self):
        return None

    def GetUpperBounds(self):
        return None
    
    def CptMapPenalty(self,xfree):
        """A penalty factor when the free parameters are outside the
        acceptable range

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters
        
        Returns
        -------
        pen : float
            The penalty value
        """
        return 0
    
##############################################################################
##############################################################################

class SymParamMap(ParamMap):
    """A class that controls the mapping between the free parameters and the
    lambda schedule. This version uses (N-2)//2 parameters for a symetric 
    schedule with fixed lambda=0 and lambda=1 end-points and lambda=0.5
    mid-point.

    Attributes
    ----------
    N : int
        The size of the schedule

    Nfree : int
        The number of free parameters

    Xfree0 : numpy.ndarray, shape=((N-2)//2,), dtype=float
        The initial free parameter values

    Methods
    -------
    """
    def __init__(self,N):
        """Construct a schedule consisting of N points"""
        super(SymParamMap,self).__init__(N)
        self.Nfree = self.Nfree // 2
        self.Xfree0 = self.Xfree0[:self.Nfree]

    def GetSched(self,xfree):
        """Returns the N-point schedule from a list of Nfree parameters

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters

        Returns
        -------
        x : list of float, len(x) == N
            The lambda schedule
        """
        import numpy as np
        if self.N % 2 == 0:
            x = [0] + [a for a in xfree] + \
                [1-a for a in xfree[::-1]] + [1]
        else:
            x = [0] + [a for a in xfree] + [0.5] \
                + [1-a for a in xfree[::-1]] + [1]
        return np.array(x)

    def Digitize(self,x,ndigits):
        """Returns a schedule with rounded values

        Parameters
        ----------
        x : list of float, len(x) == N
            The schedule

        ndigits : int
            The number of trailing digits in the format specification

        Returns
        -------
        ss : list of float, len(ss) == N
            The rounded values
        """
        import numpy as np
        n = len(x)
        if n != self.N:
            raise Exception(f"Expected size {self.N}, received {n}")
        return np.array( [float(a) for a in DigitizeSymSched(x,ndigits)] )


    
##############################################################################
##############################################################################

class GaussianParamMap(SymParamMap):
    """A class that controls the mapping between the free parameters and the
    lambda schedule. This version uses 2 parameters (a and d) to define the 
    following density function:

    \rho(x) = { 1, if x < d or x > 1-d
                \frac{ (1-2d) \sqrt{a/\pi} }{ erf( \sqrt{a} (1-2d)/2 ) }
                \times exp(-a (x-1/2)^2), 
                    otherwise

    The integrated density function is:

    idf(x) = \int_{0}^{x} \rho(y) dy
           = { x, if  x < d or x > 1-d
               (1/2) (1 + (2d-1) erf(\sqrt{a}(1-2x)/2)/erf(\sqrt{a}(1-2d)/2)),
                    otherwise

    The inverse function of the of the integrated density is:

    idf^{-1}(x) 
           = { x, if x < d or x > 1-d
               1/2 - InverseErf[((2x-1) Erf[\sqrt{a}(1-2d)/2])/(2d-1)]/\sqrt{a},
                    otherwise

    So, given a linear spacing, x0, x1, x2 ..., then mapped points are:
    idf^{-1}(x0), idf^{-1}(x1), idf^{-1}(x2), ...

    Attributes
    ----------
    N : int
        The size of the schedule

    Nfree : int
        The number of free parameters. This is always 2.

    Xfree0 : numpy.ndarray, shape=(2,), dtype=float
        The initial free parameter values. This is always [1.0,0.01].

    Methods
    -------
    """
    def __init__(self,N):
        """Construct an Gaussian schedule consisting of N points"""
        import numpy as np
        super(GaussianParamMap,self).__init__(N)
        self.Nfree = 2
        self.Xfree0 = np.array([1.0,0.01])
        
    def SetGuess(self,x):
        """This resets the initial parameters; the input argument is ignored"""
        import numpy as np
        self.Xfree0 = np.array([1.0,0.01])
    
    def GetSched(self,xfree):
        """Returns the N-point schedule from a list of Nfree parameters

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters

        Returns
        -------
        x : list of float, len(x) == N
            The lambda schedule
        """
        import numpy as np
        from scipy.special import erf
        from scipy.special import erfinv

        a = max(0,xfree[0])
        d = min(0.49999,max(0,xfree[1]))
        
        x = np.linspace(0,1,self.N)
        if abs(a) > 1.e-7:
            sa = np.sqrt(a)
            spi = np.sqrt(np.pi)
            tdm1 = 2*d-1
            for i in range(self.N):
                y = x[i]
                tym1 = 2*y-1
                if y <= d:
                    x[i] = y
                elif y >= 1-d:
                    x[i] = y
                else:
                    c = (tym1/tdm1) * erf(-0.5*sa*tdm1)
                    x[i] = abs(0.5-erfinv(c)/sa)
        return x


    def GetLowerBounds(self):
        return [0.,0.]

    def GetUpperBounds(self):
        return [100.,0.499]
    
    def CptMapPenalty(self,xfree):
        """A penalty factor when the free parameters are outside the
        acceptable range

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters
        
        Returns
        -------
        pen : float
            The penalty value
        """
        pen = 0
        if xfree[0] < 0:
            pen += 10*xfree[0]**2
        if xfree[1] < 0:
            pen += 10*xfree[1]**2
        elif xfree[1] > 0.499:
            pen += 10*(0.499-xfree[1])**2
        return pen



##############################################################################
##############################################################################

class SccParamMap(SymParamMap):
    """A class that controls the mapping between the free parameters and the
    lambda schedule. This version uses 1 parameter for a symmetric 
    SCC(alpha) schedule, where alpha is a parameter from 0 to 2.

    Attributes
    ----------
    N : int
        The size of the schedule

    Nfree : int
        The number of free parameters. This is always 1.

    Xfree0 : numpy.ndarray, shape=(1,), dtype=float
        The initial free parameter values. This is always [1.]

    Methods
    -------
    """
    
    def __init__(self,N):
        """Construct an asymmetric schedule consisting of N points"""
        import numpy as np
        super(SccParamMap,self).__init__(N)
        self.Nfree = 1
        self.Xfree0 = np.array([1.0])
        
    def SetGuess(self,x):
        """This resets the initial parameters; the input argument is ignored"""
        import numpy as np
        self.Xfree0 = np.array([1.0])
    
    def GetSched(self,xfree):
        """Returns the N-point schedule from a list of Nfree parameters

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters

        Returns
        -------
        x : list of float, len(x) == N
            The lambda schedule
        """
        import numpy as np
        alpha = min(2,max(0,xfree[0]))
        #x = CptSSCSched(self.N,alpha)
        x = CptGeneralizedSSCSched(self.N,alpha,alpha1=None)
        #print("x=",x)
        return x

    def GetLowerBounds(self):
        return [0.]

    def GetUpperBounds(self):
        return [2.]
    
    def CptMapPenalty(self,xfree):
        """A penalty factor when the free parameters are outside the
        acceptable range

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters
        
        Returns
        -------
        pen : float
            The penalty value
        """
        pen = 0
        if xfree[0] < 0:
            pen += 10*xfree[0]**2
        elif xfree[0] > 2:
            pen += 10*(2-xfree[0])**2
        return pen


##############################################################################
##############################################################################

class AsymSccParamMap(ParamMap):
    """A class that controls the mapping between the free parameters and the
    lambda schedule. This version uses 2 parameters for an asymmetric 
    SCC schedule. The two parameters are the alpha at lambda=0 and the alpha
    at lambda=1.

    Attributes
    ----------
    N : int
        The size of the schedule

    Nfree : int
        The number of free parameters. This is always 2.

    Xfree0 : numpy.ndarray, shape=(2,), dtype=float
        The initial free parameter values. This is always [1.,1.]

    Methods
    -------
    """
    
    
    def __init__(self,N):
        """Construct an asymmetric schedule consisting of N points"""
        import numpy as np
        super(AsymSccParamMap,self).__init__(N)
        self.Nfree = 2
        self.Xfree0 = np.array([1.0,1.0])
        
    def SetGuess(self,x):
        """This resets the initial parameters; the input argument is ignored"""
        import numpy as np
        self.Xfree0 = np.array([1.0,1.0])
    
    def GetSched(self,xfree):
        """Returns the N-point schedule from a list of Nfree parameters

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters

        Returns
        -------
        x : list of float, len(x) == N
            The lambda schedule
        """
        import numpy as np
        alpha1 = min(2,max(0,xfree[0]))
        alpha2 = min(2,max(0,xfree[1]))
        # x1 = CptSSCSched(self.N,alpha1)
        # x2 = CptSSCSched(self.N,alpha2)
        # x = [a for a in x1[:self.N//2]] + [a for a in x2[self.N//2:]]
        x = CptGeneralizedSSCSched(self.N,alpha1,alpha2)
        return x

    def GetLowerBounds(self):
        return [0.,0.]

    def GetUpperBounds(self):
        return [2.,2.]
    
    def CptMapPenalty(self,xfree):
        """A penalty factor when the free parameters are outside the
        acceptable range

        Parameters
        ----------
        xfree : list of float, len(xfree) == Nfree
            The free parameters
        
        Returns
        -------
        pen : float
            The penalty value
        """
        pen = 0
        if xfree[0] < 0:
            pen += 10*xfree[0]**2
        elif xfree[0] > 2:
            pen += 10*(2-xfree[0])**2
            
        if xfree[1] < 0:
            pen += 10*xfree[1]**2
        elif xfree[1] > 2:
            pen += 10*(2-xfree[1])**2
            
        return pen


##############################################################################
##############################################################################


def AutoFindLambdaValues(dname):
    """Determine lambda values by looking for the dvdl_*.dat files in the
    specified directory

    Parameters
    ----------
    dname : str
        The directory to search for dvdl_*.dat files. The wildcard is
        assumed to be a float corresponding to a lambda value

    Returns
    -------
    lams : list of str
        The list of lambda values
    """
    import glob
    import pathlib
    dvdls = glob.glob( str(pathlib.Path(dname).joinpath("dvdl_*.dat")) )
    dvdls.sort()
    lams=[]
    for f in dvdls:
        fname = pathlib.Path(f).parts[-1]
        lam = fname.replace("dvdl_","").replace(".dat","")
        lams.append(lam)
    return lams


def GetEFEPPath(dname,tlam,hlam):
    """Returns the Path object of a efep_*_*.dat energy time-series. The
    file may or may not exist.

    Parameters
    ----------
    dname : str
        The directory where the file resides

    tlam : str
        The lambda producing the trajectory

    hlam : str
        The lambda used to evaluate the potential energy
    
    Returns
    -------
    fname : pathlib.Path
        The Path object refering to the file
    """
    import pathlib
    return pathlib.Path(dname).joinpath("efep_%s_%s.dat"%(tlam,hlam))


##############################################################################
##############################################################################

class PotEneData(object):
    """A class that stores the observed PSO and RER data from a series of
    burn-in simulations

    Attributes
    ----------
    lams : list of str, len(lams) = nlam
        The lambda values

    fnames : list of list of str (forming a nlam*nlam matrix)
        The matrix of filenames containing the potential energy data.
        The fast index is the lambda producing the sampling.
        The slow index is the lambda defining the potential energies.

    T : float
        The simulation temperature (K)

    traj : list of numpy.ndarray
        The raw potential energies. The list index is the lambda
        producing the trajectory. The ndarray is a (nsample,nlam) matrix
        of potential energies, where nsample is the number of samples 
        in the trajectory and nlam indexes the potential energy function.

    pso : numpy.ndarray, shape=(nlam,nlam)
        The phase space overlap estimates made from the observed samples

    rer : numpy.ndarray, shape=(nlam,nlam)
        The replica exchange ratio estimates made from the observed samples

    skl : numpy.ndarray, shape=(nlam,nlam)
        The symmetrized Kullback-Leibler divergence

    Methods
    -------
    """
    def __init__(self,lams,enenames,T,fstart,fstop):
        """Reads energy files and populated the pso and rer matrices"""
        import copy
        from os.path import exists
        import numpy as np
        
        self.lams = copy.deepcopy(lams)
        self.fnames = copy.deepcopy(enenames)
        self.T = T
        self.fstart = fstart
        self.fstop = fstop
        self.traj = []

        nlam = len(self.lams)
        self.pso = np.zeros( (nlam,nlam) )
        self.rer = np.zeros( (nlam,nlam) )
        self.skl = np.zeros( (nlam,nlam) )

        nlam = len(lams)
        for ilam,lam in enumerate(lams):
            if len(self.fnames[ilam]) != nlam:
                raise Exception(f"Expected {nlam} filenames for traj {lam}")

        for ilam,lam in enumerate(lams):
            for fname in self.fnames[ilam]:
                if not exists(fname):
                    raise Exception(f"File {fname} does not exist")

        self.trajs = []
        for ilam,lam in enumerate(lams):
            enemat = []
            for fname in self.fnames[ilam]:
                enemat.append( np.loadtxt(fname)[:,1] )
            nmin = min( row.shape[0] for row in enemat )
            nlo = int(nmin*self.fstart)
            nhi = min(nmin,int((nmin-1)*self.fstop)+1)
            for irow in range(nlam):
                enemat[irow] = enemat[irow][nlo:nhi]
            self.trajs.append( np.array( enemat ).T )

        for ilam in range(nlam):
            for jlam in range(ilam):
                self.pso[ilam,jlam] = self.pso[jlam,ilam] = self._CptPSO(ilam,jlam)
                self.rer[ilam,jlam] = self.rer[jlam,ilam] = self._CptRER(ilam,jlam,self.T)
                self.skl[ilam,jlam] = self.skl[jlam,ilam] = self._CptSKL(ilam,jlam)
            self.pso[ilam,ilam] = 1.
            self.rer[ilam,ilam] = 1.
            self.skl[ilam,ilam] = 1.

            
        #for ilam in range(nlam):
        #    print(" ".join(["%11.2e"%(x) for x in self.skl[ilam,:]]))
            
            
    def _CptPSO(self,ilam,jlam):
        dE1 = self.trajs[ilam][:,jlam] - self.trajs[ilam][:,ilam]
        dE2 = self.trajs[jlam][:,jlam] - self.trajs[jlam][:,ilam]
        return CptPhaseSpaceOverlapIndex(dE1,dE2)

    
    def _CptRER(self,ilam,jlam,T):
        n = min( self.trajs[ilam].shape[0], self.trajs[jlam].shape[0] )
        E0 = self.trajs[ilam][:n,ilam]+self.trajs[jlam][:n,jlam]
        E1 = self.trajs[ilam][:n,jlam]+self.trajs[jlam][:n,ilam]
        return AvgAcceptanceRatio(E1-E0,T)

    def _CptSKL(self,ilam,jlam):
        import numpy as np
        dE1 = self.trajs[ilam][:,jlam] - self.trajs[ilam][:,ilam]
        dE2 = self.trajs[jlam][:,jlam] - self.trajs[jlam][:,ilam]
        return np.exp(-CptSymmetricKullbackLeibler(dE1,dE2))
    

class PotEneData_MBAR(PotEneData):
    def __init__(self, lams, enemat, T):
        """Reads energy files and populated the pso and rer matrices"""
        import copy
        import numpy as np
        
        self.lams = copy.deepcopy(lams)
        #self.fnames = copy.deepcopy(enenames)
        self.T = T
        self.trajs = enemat

        nlam = len(self.lams)
        self.pso = np.zeros( (nlam,nlam) )
        self.rer = np.zeros( (nlam,nlam) )
        self.skl = np.zeros( (nlam,nlam) )

        for ilam in range(nlam):
            for jlam in range(ilam):
                self.pso[ilam,jlam] = self.pso[jlam,ilam] = self._CptPSO(ilam,jlam)
                self.rer[ilam,jlam] = self.rer[jlam,ilam] = self._CptRER(ilam,jlam,self.T)
                self.skl[ilam,jlam] = self.skl[jlam,ilam] = self._CptSKL(ilam,jlam)
            self.pso[ilam,ilam] = 1.
            self.rer[ilam,ilam] = 1.
            self.skl[ilam,ilam] = 1.

            
        #for ilam in range(nlam):
        #    print(" ".join(["%11.2e"%(x) for x in self.skl[ilam,:]]))
            
            



##############################################################################
##############################################################################
    
class SlaterInterp(object):
    def __init__(self,lams,data,clean):
        import copy
        import numpy as np
        from scipy.interpolate import RBFInterpolator
        
        nlam = len(lams)
        self.lams = copy.deepcopy(lams)
        self.lamvals = np.array( [float(x) for x in self.lams] )
        self.data = np.array(data,copy=True)
        if self.data.shape[0] != nlam or self.data.shape[1] != nlam:
            raise Exception(f"Expected square matrix ({nlam},{nlam}), "+
                            f"but received {self.data.shape}")

        SlaterZ = np.zeros( (nlam,nlam) )
        self.Z  = np.zeros( (nlam,nlam) )
        for i in range(nlam):
            for j in range(i+1,nlam):
                u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                self.Z[i,j] = self.Z[j,i] = self.CptExponent(u,self.data[i,j])
                SlaterZ[i,j] = SlaterZ[j,i] = CptSlaterExp(u,self.data[i,j])
            if clean:
                for j in range(i+1,nlam):
                    u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                    z = max( SlaterZ[i,j],  SlaterZ[i,j-1] )
                    SlaterZ[i,j] = SlaterZ[j,i] = z
                    s = np.exp(-z*abs(u))
                    self.data[i,j] = self.data[j,i] = s
                    self.Z[i,j] = self.Z[j,i] = self.CptExponent(u,self.data[i,j])
                for j in range(i-1,-1,-1):
                    u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                    z = max( SlaterZ[i,j],  SlaterZ[i,j+1] )
                    SlaterZ[i,j] = SlaterZ[j,i] = z
                    s = np.exp(-z*abs(u))
                    self.data[i,j] = self.data[j,i] = s
                    self.Z[i,j] = self.Z[j,i] = self.CptExponent(u,self.data[i,j])

        #print("")
        #for i in range(nlam):
        #    print(" ".join(["%11.2e"%(x) for x in self.Z[i,:]]))

        pts = []
        vals = []
        for i in range(nlam):
            for j in range(nlam):
                if i == j:
                    continue
                if self.data[i,j] > 1.e-14:
                    u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                    pts.append( [u,v] )
                    vals.append( self.Z[i,j] )
        pts=np.array(pts)
        vals=np.array(vals)
        self.rbf = RBFInterpolator(pts,vals,kernel='multiquadric',epsilon=100)


    # to be overridden
    def CptExponent(self,u,s):
        return CptSlaterExp(u,s)

    
    # to be overridden
    def CptValueFromExponent(self,u,z):
        import numpy as np
        return np.exp(-z*abs(u))
    
    
    def InterpExponentFromLambdas(self,lam1,lam2):
        import numpy as np
        u,v = TransformLL2UV(lam1,lam2)
        return self.rbf( [u,v] )[0]

    
    def InterpValueFromLambdas(self,lam1,lam2):
        u1,v1 = TransformLL2UV(lam1,lam2)
        u2,v2 = TransformLL2UV(lam2,lam1)
        vals = self.rbf([ [u1,v1], [u2,v2] ])
        s1 = self.CptValueFromExponent(u1,vals[0])
        s2 = self.CptValueFromExponent(u2,vals[1])
        return 0.5*(s1+s2)

    def CptNeighborValues(self,x):
        import numpy as np
        n=len(x)
        return np.array([ self.InterpValueFromLambdas( x[i], x[i+1] )
                          for i in range(n-1) ])

    def GetSummary(self,x):
        return SchedInfo(x,self.CptNeighborValues(x))
    

##############################################################################
##############################################################################
    
class GaussianInterp(SlaterInterp):
    def __init__(self,lams,data,clean):
        super(GaussianInterp,self).__init__(lams,data,clean)

    # to be overridden
    def CptExponent(self,u,s):
        return CptGaussianExp(u,s)
    
    # to be overridden
    def CptValueFromExponent(self,u,z):
        import numpy as np
        return np.exp(-max(0,z)*u*u)


##############################################################################
##############################################################################
    
class QuadraticInterp(object):
    def __init__(self,lams,data,clean):
        import copy
        import numpy as np
        from scipy.interpolate import RBFInterpolator
        
        nlam = len(lams)
        self.lams = copy.deepcopy(lams)
        self.lamvals = np.array( [float(x) for x in self.lams] )
        self.data = np.array(data,copy=True)
        if self.data.shape[0] != nlam or self.data.shape[1] != nlam:
            raise Exception(f"Expected square matrix ({nlam},{nlam}), "+
                            f"but received {self.data.shape}")

        self.Z  = np.zeros( (nlam,nlam) )
        for i in range(nlam):
            for j in range(i+1,nlam):
                u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                self.Z[i,j] = self.Z[j,i] = self.CptExponent(u,self.data[i,j])
            if clean:
                for j in range(i+1,nlam):
                    u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                    z = max( self.Z[i,j],  self.Z[i,j-1] )
                    s = z*u**2
                    self.data[i,j] = self.data[j,i] = s
                    self.Z[i,j] = self.Z[j,i] = self.CptExponent(u,self.data[i,j])
                for j in range(i-1,-1,-1):
                    u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                    z = max( self.Z[i,j],  self.Z[i,j+1] )
                    s = z*u**2
                    self.data[i,j] = self.data[j,i] = s
                    self.Z[i,j] = self.Z[j,i] = self.CptExponent(u,self.data[i,j])

        # print("")
        # for i in range(nlam):
        #    print(" ".join(["%11.2e"%(x) for x in self.Z[i,:]]))


        pts = []
        vals = []
        for i in range(nlam):
            for j in range(nlam):
                if i == j:
                    continue
                if self.data[i,j] > 1.e-14:
                    u,v = TransformLL2UV(self.lamvals[i],self.lamvals[j])
                    pts.append( [u,v] )
                    vals.append( self.Z[i,j] )
        pts=np.array(pts)
        vals=np.array(vals)
        self.rbf = RBFInterpolator(pts,vals,kernel='multiquadric',epsilon=100)


    # to be overridden
    def CptExponent(self,u,s):
        # s = c*u**2
        # c = s/u**2
        c = 0
        u2 = u*u
        if u2 > 0:
            c = s/u2
        return c

    
    # to be overridden
    def CptValueFromExponent(self,u,z):
        import numpy as np
        return z*u**2
    
    
    def InterpExponentFromLambdas(self,lam1,lam2):
        import numpy as np
        u,v = TransformLL2UV(lam1,lam2)
        return self.rbf( [u,v] )[0]

    
    def InterpValueFromLambdas(self,lam1,lam2):
        u1,v1 = TransformLL2UV(lam1,lam2)
        u2,v2 = TransformLL2UV(lam2,lam1)
        vals = self.rbf([ [u1,v1], [u2,v2] ])
        s1 = self.CptValueFromExponent(u1,vals[0])
        s2 = self.CptValueFromExponent(u2,vals[1])
        return 0.5*(s1+s2)

    def CptNeighborValues(self,x):
        import numpy as np
        n=len(x)
        return np.array([ self.InterpValueFromLambdas( x[i], x[i+1] )
                          for i in range(n-1) ])

    def GetSummary(self,x):
        return SchedInfo(x,self.CptNeighborValues(x))
    
##############################################################################
##############################################################################

def OptSchedObjective( xfree, mapper, interp, minval, maxgap ):
    import numpy as np
    x = mapper.GetSched(xfree)
    n = len(x)
    ss = np.array([ interp.InterpValueFromLambdas( x[i], x[i+1] )
                    for i in range(n-1) ])
    
    objval = np.var(ss,ddof=0)
    
    objpen = mapper.CptMapPenalty( xfree )
    for s in ss:
        if s < minval:
            objpen += 100*(s-minval)**2

    for i in range(n-1):
        d = x[i+1]-x[i]
        if d > maxgap:
            objpen += 200 * (d-maxgap)**2

    return objval + objpen


def OptSched( mapper, interp, minval, maxgap,
              xfree0=None, eps=5.e-5 ):

    import numpy as np
    from scipy.optimize import minimize
    from scipy.optimize import Bounds

    nfree = mapper.Nfree
    lbs = mapper.GetLowerBounds()
    ubs = mapper.GetUpperBounds()
    if lbs is not None and ubs is not None:
        bounds = Bounds( lbs, ubs )
    else:
        bounds = None

    if xfree0 is None:
        xfree0 = np.array( mapper.Xfree0, copy=True )

    if len(xfree0) != nfree:
        raise Exception(f"Expected {nfree} parameters, received {len(xfree0)}")
    
    
    res = minimize( OptSchedObjective, xfree0,
                    args=(mapper, interp, minval, maxgap),
                    #method='COBYLA',
                    method='L-BFGS-B', jac='3-point',
                    bounds=bounds,
                    tol=1.e-14,
                    options=
                    {
                        'maxiter': 10000,
                        'eps': eps
                    })
    
    return res



def OptSchedNoGrad( mapper, interp, minval, maxgap,
                    xfree0=None ):

    import numpy as np
    from scipy.optimize import minimize
    from scipy.optimize import Bounds

    nfree = mapper.Nfree
    
    if xfree0 is None:
        xfree0 = np.array( mapper.Xfree0, copy=True )

    if len(xfree0) != nfree:
        raise Exception(f"Expected {nfree} parameters, received {len(xfree0)}")
    
    res = minimize( OptSchedObjective, xfree0,
                    args=(mapper, interp, minval, maxgap),
                    method='COBYLA',
                    #method='L-BFGS-B', jac='3-point',
                    tol=1.e-16,
                    options=
                    {
                        'maxiter': 10000,
                        'rhobeg': 5.e-3,
                        'tol':1.e-30
                    })
    
    return res


def old_ReadDataFiles(dname,T,fstart,fstop):
    lams = AutoFindLambdaValues(dname)
    fmat = []
    for tlam in lams:
        frow = []
        for hlam in lams:
            frow.append(GetEFEPPath(dname,tlam,hlam))
        fmat.append(frow)
    return PotEneData(lams,fmat,T,fstart,fstop)


def ReadDataFiles(dname,T,fstart,fstop):
    import glob
    import pandas as pd
    from alchemlyb.parsing.amber import extract_u_nk
    import numpy as np

    folders = glob.glob(f'{dname}*')
    # dname is provided to be a directory with a trailing letter
    # e.g. include v00, v01, v02, etc. in the folder names
    # sort folders by the trailing number to ensure the order is correct
    folders = [x for x in folders if x.split(dname)[-1].isdigit()]
    folders.sort(key=lambda x: int(x.split(dname)[-1]))  # Sort by the trailing number
    nlams = len(folders)
    print(f"Found {nlams} folders for lambda values in directory: {dname}")
    # read with extract_u_nk

    #Upot = np.load(f"{work_dir}/sdr/data/Upot_{comp}_all.npy")
    df_list = []
    for win_i, folder in enumerate(folders):
        mdin_out_files = glob.glob(f'{folder}/mdin*.out')
        mdin_out_files.sort()
        fstart = np.max([0, fstart])
        fstop = np.min([len(mdin_out_files), fstop])
        print(f'Reading {len(mdin_out_files)} mdin files from {folder} for window {win_i+1}/{nlams}')
        print(f"Reading {fstart} to {fstop} mdin files")
        df = pd.concat([extract_u_nk(mdin_f,
                                    T=T,
                                    reduced=False
                                    ) for mdin_f in mdin_out_files[fstart:fstop]])
        df.to_csv(f'{folder}/potential_energy.csv', index=False)  # Save for debugging
        df_list.append(df)

    full_df = pd.concat(df_list)
    lams = full_df.index.get_level_values('lambdas').unique()
    print(f"Found {nlams} lambda values: {lams.values}")  # Print unique lambda values for verification
    enemat = [] 
    # generate enemat
    # enemat is a list of
    for df in df_list:
        # For each window, extract the potential energy for each lambda
        # and create a matrix of potential energies
        # where each row corresponds to a lambda producing the trajectory
        # and each column corresponds to a lambda defining the potential energy
        temp_enemat = np.zeros((len(df), nlams))
        #print(f"Generating potential energy matrix for window {len(enemat)+1}/{nlams} with shape: {temp_enemat.shape}")
        for i, lam in enumerate(lams):
            # convert kT back to kcal/mol
            kT = T * 1.98720425864083e-3
            temp_enemat[:, i] = df[lam].values * kT
        enemat.append(temp_enemat)
    #np.save(f'{dname}potential_energy_matrix.npy', enemat)  # Save for debugging
    
    return PotEneData_MBAR(lams,enemat,T)
 

def CalcOptRE( N, pedata, minval, maxgap, clean, sym ):

    interp = SlaterInterp(pedata.lams,pedata.rer,clean)
    gmap   = GaussianParamMap(N)
    gres   = OptSched( gmap, interp, minval, maxgap, 
                       xfree0=None, eps=5.e-4 )

    if sym:
        pmap = SymParamMap(N)
    else:
        pmap = ParamMap(N)

    # first optimize without maxgap penalty
    pmap.SetGuess( gmap.GetSched(gres.x) )
    pres = OptSched( pmap, interp, minval, 1., 
                     xfree0=None, eps=5.e-5 )
    
    # restart the optimization with the maxgap penalty
    pmap.SetGuess( pmap.GetSched(pres.x) )
    pres = OptSched( pmap, interp, minval, maxgap, 
                     xfree0=None, eps=5.e-5 )

    return pres, pmap, interp


def CalcOptPSO( N, pedata, minval, maxgap, clean, sym ):

    interp = GaussianInterp(pedata.lams,pedata.pso,clean)
    gmap   = GaussianParamMap(N)
    gres   = OptSched( gmap, interp, minval, maxgap, 
                       xfree0=None, eps=5.e-4 )

    if sym:
        pmap = SymParamMap(N)
    else:
        pmap = ParamMap(N)

    # first optimize without maxgap penalty
    pmap.SetGuess( gmap.GetSched(gres.x) )
    pres = OptSched( pmap, interp, minval, 1., 
                     xfree0=None, eps=5.e-5 )
    
    # restart the optimization with the maxgap penalty
    pmap.SetGuess( pmap.GetSched(pres.x) )
    pres = OptSched( pmap, interp, minval, maxgap, 
                     xfree0=None, eps=2.e-5 )

    return pres, pmap, interp



def CalcOptKL( N, pedata, minval, maxgap, clean, sym ):

    interp = GaussianInterp(pedata.lams,pedata.skl,clean)
    gmap   = GaussianParamMap(N)
    gres   = OptSched( gmap, interp, minval, maxgap, 
                       xfree0=None, eps=5.e-4 )

    if sym:
        pmap = SymParamMap(N)
    else:
        pmap = ParamMap(N)

    # first optimize without maxgap penalty
    pmap.SetGuess( gmap.GetSched(gres.x) )
    pres = OptSched( pmap, interp, minval, 1., 
                     xfree0=None, eps=5.e-5 )
    
    # restart the optimization with the maxgap penalty
    pmap.SetGuess( pmap.GetSched(pres.x) )
    pres = OptSched( pmap, interp, minval, maxgap, 
                     xfree0=None, eps=2.e-5 )

    return pres, pmap, interp




def CalcOptRE_SSC( N, pedata, minval, maxgap, clean, sym ):

    interp = SlaterInterp(pedata.lams,pedata.rer,clean)
    pmap = SccParamMap(N)

    # first optimize without maxgap penalty
    pres = OptSchedNoGrad( pmap, interp, minval, 1, 
                           xfree0=[2.0] )

    # restart the optimization with the maxgap penalty
    pres = OptSchedNoGrad( pmap, interp, minval, maxgap, 
                           xfree0=[pres.x[0]] )

    if not sym:
        dmap = AsymSccParamMap(N)
        dres = OptSchedNoGrad( dmap, interp, minval, maxgap, 
                               xfree0=[pres.x[0],pres.x[0]] )
        pmap = dmap
        pres = dres
    
    return pres, pmap, interp



def CalcOptPSO_SSC( N, pedata, minval, maxgap, clean, sym ):

    interp = GaussianInterp(pedata.lams,pedata.pso,clean)
    pmap = SccParamMap(N)

    # first optimize without maxgap penalty
    pres = OptSchedNoGrad( pmap, interp, minval, 1, 
                           xfree0=[2.0] )

    # restart the optimization with the maxgap penalty
    pres = OptSchedNoGrad( pmap, interp, minval, maxgap, 
                           xfree0=[pres.x[0]] )

    if not sym:
        dmap = AsymSccParamMap(N)
        dres = OptSchedNoGrad( dmap, interp, minval, maxgap, 
                               xfree0=[pres.x[0],pres.x[0]] )
        pmap = dmap
        pres = dres
    
    return pres, pmap, interp




def CalcOptKL_SSC( N, pedata, minval, maxgap, clean, sym ):

    interp = GaussianInterp(pedata.lams,pedata.skl,clean)
    pmap = SccParamMap(N)

    # first optimize without maxgap penalty
    pres = OptSchedNoGrad( pmap, interp, minval, 1, 
                           xfree0=[2.0] )

    # restart the optimization with the maxgap penalty
    pres = OptSchedNoGrad( pmap, interp, minval, maxgap, 
                           xfree0=[pres.x[0]] )

    if not sym:
        dmap = AsymSccParamMap(N)
        dres = OptSchedNoGrad( dmap, interp, minval, maxgap, 
                               xfree0=[pres.x[0],pres.x[0]] )
        pmap = dmap
        pres = dres
    
    return pres, pmap, interp


##############################################################################
##############################################################################

def SimpleScheduleOpt( nlam,
                       directory, temp, fstart, fstop,
                       ssc, sym, pso, re, kl, clean,
                       maxgap, minval, digits, verbose ):

    import numpy as np
    import os

    if verbose:
        print("Entered SimpleScheduleOpt with the following inputs:")
        print("  Working dir: %s"%(os.getcwd()))
        print("  nlam:        %i"%(nlam))
        print("  directory:   %s"%(directory))
        print("  temp:        %.2f"%(temp))
        print("  fstart:      %.3f"%(fstart))
        print("  fstop:       %.3f"%(fstop))
        print("  ssc:         %s"%(str(ssc)))
        print("  sym:         %s"%(str(sym)))
        print("  pso:         %s"%(str(pso)))
        print("  re:          %s"%(str(re)))
        print("  kl:          %s"%(str(kl)))
        print("  clean:       %s"%(str(clean)))
        print("  maxgap:      %.3f"%(maxgap))
        print("  minval:      %.3e"%(minval))
        print("  digits:      %i"%(digits))
        print("  verbose:     %s"%(str(verbose)))
        print("")
    
    fstart = int(max(0,fstart))
    fstop  = int(max(1,max(fstart,fstop)))
    pedata = ReadDataFiles(directory,temp,fstart,fstop)

    method = None
    if pso:
        if ssc:
            method = CalcOptPSO_SSC
        else:
            method = CalcOptPSO
    elif re:
        if ssc:
            method = CalcOptRE_SSC
        else:
            method = CalcOptRE
    elif kl:
        if ssc:
            method = CalcOptKL_SSC
        else:
            method = CalcOptKL
            

    info = None
    if method is not None:
        res, pmap, interp = method( nlam, pedata, minval,
                                    maxgap, clean, sym )
        flams = pmap.Digitize( pmap.GetSched(res.x),digits)
        if verbose:
            print("Optimization result:")
            print(res)
            print("")
            print("Optimized free parameters:")
            for p in res.x:
                print("  %.8f"%(p))
            print("")
    # elif pso_ssc is not None:
    #     interp = GaussianInterp(pedata.lams,pedata.pso,clean)
    #     flams = np.array([float(a) for a in
    #                       DigitizeSymSched(CptSSCSched(nlam,pso_ssc),digits)])
    #     if verbose:
    #         print("Unoptimized parameters:")
    #         print("  %.8f"%(pso_ssc))
    #         print("")
    # elif re_ssc is not None:
    #     interp = SlaterInterp(pedata.lams,pedata.rer,clean)
    #     flams = np.array([float(a) for a in
    #                       DigitizeSymSched(CptSSCSched(nlam,re_ssc),digits)])
    #     if verbose:
    #         print("Unoptimized parameters:")
    #         print("  %.8f"%(re_ssc))
    #         print("")
    # elif kl_ssc is not None:
    #     interp = QuadraticInterp(pedata.lams,pedata.rer,clean)
    #     flams = np.array([float(a) for a in
    #                       DigitizeSymSched(CptSSCSched(nlam,re_ssc),digits)])
    #     if verbose:
    #         print("Unoptimized parameters:")
    #         print("  %.8f"%(re_ssc))
    #         print("")

    if interp is not None:
        info = interp.GetSummary( flams )
        if verbose:
            print(str(info))
            
    return info, pedata, interp
        

def SimpleScheduleRead( read,
                        directory, temp, fstart, fstop,
                        ssc, alpha0, alpha1, pso, re, kl,
                        clean, digits, verbose ):

    import numpy as np
    import os

    if verbose:
        print("Entered SimpleScheduleRead with the following inputs:")
        print("  Working dir: %s"%(os.getcwd()))
        print("  read:        %s"%(read))
        print("  directory:   %s"%(directory))
        print("  temp:        %.2f"%(temp))
        print("  fstart:      %.3f"%(fstart))
        print("  fstop:       %.3f"%(fstop))
        print("  ssc:         %s"%(str(ssc)))
        print("  alpha0:      %.5f"%(alpha0))
        if alpha1 is not None:
            print("  alpha1:      %.5f"%(alpha1))
        else:
            print("  alpha1:      %s"%(str(alpha1)))
        print("  pso:         %s"%(str(pso)))
        print("  re:          %s"%(str(re)))
        print("  kl:          %s"%(str(kl)))
        print("  clean:       %s"%(str(clean)))
        print("  digits:      %i"%(digits))
        print("  verbose:     %s"%(str(verbose)))
        print("")


    fstart = max(0,fstart)
    fstop  = max(1,max(fstart,fstop))
    pedata = ReadDataFiles(directory,temp,
                           fstart,fstop)

    ################################################
    if ssc:
        # evaluate
        nlam = int(read)
        if alpha1 is None:
            sched = CptSSCSched(nlam,alpha0)
                
            print("Unoptimized parameters:")
            print("  %.8f"%(alpha0))
            print("")
        else:
            sched = CptGeneralizedSSCSched(nlam,alpha0,alpha1)
            
            print("Unoptimized parameters:")
            print("  %.8f"%(alpha0))
            print("  %.8f"%(alpha1))
            print("")
    else:
        # read
        sched = np.loadtxt(read)
        if len(sched.shape) == 2:
            sched = sched[:,0]
                
        print("Unoptimized parameters:")
        for lam in sched:
            print("  %.8f"%(lam))
        print("")
    ################################################
        
    flams = np.array([float(a) for a in
                      DigitizeSched(sched,digits)])
    if pso:
        interp = GaussianInterp(pedata.lams,pedata.pso,clean)
    elif re:
        interp = SlaterInterp(pedata.lams,pedata.rer,clean)
    elif kl:
        interp = GaussianInterp(pedata.lams,pedata.skl,clean)
    else:
        raise Exception("Expected pso, re, or kl but none are true")
    info = interp.GetSummary( flams )
    print(str(info))

    return info, pedata, interp
        
    
##############################################################################
##############################################################################
