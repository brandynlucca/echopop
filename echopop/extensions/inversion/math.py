"""
Mathematical and numerical utility functions.
"""

from scipy.special import _spherical_bessel as ssb

def spherical_hn(n, z, derivative=False):
    """
    Spherical Bessel function of the third kind (Hankel function) or its derivative
    
    Defined as [1]_,
    
    .. math:: h_n^{(1)}(z)=j_n(z)+in_n(z),
    
    where :math:`h_n^{(1)}` is the spherical Bessel function of the third kind (or Hankel function 
    of the first kind), :math:`j_n` is the spherical Bessel function of the first kind, :math:`n_n` 
    is the spherical Bessel function of the second kind (or Neumann function), :math:`n` is the 
    order of the function (:math:`n>=0`), :math:`z` is the Bessel function argument value, and 
    :math:`i` is an imaginary number.
    
    Parameters
    ----------
    n: int
        Order of the Bessel function (n >= 0)
    z: Union[float, np.complex]
        Argument of the Bessel function
    derivative: Optional[bool]
        When True, the derivative is computed
        
    Notes
    -----
    The derivative is computed using the relations [2]_,
    
    .. math::
        \frac{n}{z} h^{(1)}_n - h^{(1)}_{n+1}(z)
    
    References
    ----------    
    .. [1] https://dlmf.nist.gov/10.47#E5
    .. [2] https://dlmf.nist.gov/10.51#E2
    
    """
    
    # Define internal function
    def _spherical_hn(n, z):
        return ssb.spherical_jn(n, z) + 1j * ssb.spherical_yn(n, z)
    
    # Computing derivative
    if derivative:
        return (n/z) * _spherical_hn(n, z) - _spherical_hn(n+1, z)
    else:
        return _spherical_hn(n, z)