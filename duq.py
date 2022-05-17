import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv
from scipy.integrate import romberg
from copy import copy

default_cutoff = erfinv(1.-1.e-8)

class DUQ():
    """Distribution of uncertainty quality (DUQ). 
    Compute the DUQ by comparing residuals and uncertainties to an ideal histogram.
    See https://doi.org/10.1016/j.commatsci.2021.110916 for details on the method. 

    Parameters
    ----------
    cutoff_val :    float, otpional
                    The cutoff value for the histogram i.e. the maximum positive bin edge. 
                    Also used for negative bin edge.
    max_bins :      int, optional
                    The maximum number of bins to consider during the optimisation procedure 
                    for the number of bins.

    Methods
    -------
    __call__
    __adj_term
    gen_adjustments
    calc_err_ratio
    dist_error

    Examples
    --------
    Generate some random example data then calculate DUQ and plot it. 
    >>> from duq import DUQ,plot_duq
    >>> import numpy as np
    >>> from numpy.random import normal
    >>> # Well estimated uncertainties.
    >>> y_true = np.zeros(100)
    >>> y_unc = np.abs(normal(scale=1.0,size=100))
    >>> y_pred = normal(scale=y_unc/np.sqrt(2),size=100)
    >>> duq_calc = DUQ()
    >>> duq,bins,ideal_hist,hist = duq_calc(y_true,y_pred,y_unc)
    >>> print(duq)
    >>> plot_duq(bins,ideal_hist,hist)
    >>> # Overestimated uncertainties.
    >>> y_true = np.zeros(100)
    >>> y_unc = np.abs(normal(scale=1.0,size=100))
    >>> y_pred = normal(scale=0.4*y_unc,size=100)
    >>> duq_calc = DUQ()
    >>> duq,bins,ideal_hist,hist = duq_calc(y_true,y_pred,y_unc)
    >>> print(duq)
    >>> plot_duq(bins,ideal_hist,hist)
    >>> # Underestimated uncertainties.
    >>> y_true = np.zeros(100)
    >>> y_unc = np.abs(normal(scale=1.0,size=100))
    >>> y_pred = normal(scale=1.4*y_unc,size=100)
    >>> duq_calc = DUQ()
    >>> duq,bins,ideal_hist,hist = duq_calc(y_true,y_pred,y_unc)
    >>> print(duq)
    >>> plot_duq(bins,ideal_hist,hist)
    """
    def __init__(self,cutoff_val=default_cutoff,max_bins=21):
        self.cutoff_val = cutoff_val
        self.max_bins = max_bins
        self.gen_adjustments()

    def __adj_term(self,N_bins):
        """
        For a given number of bins, calculate the adjustment term used when optimising
        the number of bins. This is the difference in area between the ideal histogram 
        and the true normal distribution. 
        """
        # Area of ideal histogram bins.
        A = 1./N_bins
        cum_areas = (np.arange(1,(N_bins+1)//2)-.5)*A*2
        ideal_bins = erfinv(cum_areas) # Ideally the Nth bin has height 0 and infinite width
        real_bins  = np.append(ideal_bins,self.cutoff_val)
        # Need to double up on these
        ideal_bins = np.append(-np.flip(ideal_bins),ideal_bins)
        real_bins = np.append(-np.flip(real_bins),real_bins)
        bin_widths = np.diff(real_bins)
        # Heights of the ideal bins
        id_bin_hs  = A/bin_widths
        # Calulate the adjustment to the error
        adj_error = romberg(lambda z: np.square(id_bin_hs[np.digitize(z,real_bins)-1]-1/np.sqrt(np.pi)*np.exp(-z**2)),
                                -self.cutoff_val*.999,self.cutoff_val*.999,divmax=100,tol=1.e-6,vec_func=True)
        return adj_error,real_bins,id_bin_hs
    
    def gen_adjustments(self):
        """
        Generate the adjustment terms for all the number of bins being considered. 
        """
        self.adjustments = [self.__adj_term(N) for N in np.arange(5,self.max_bins,2)]
         
    @staticmethod
    def calc_err_ratio(y_true,y_pred,y_unc):
        """
        The quantity that is binned. Given its own function so that it can be 
        overriden by child classes.
        """ 
        return (y_pred-y_true)/y_unc
    
    def dist_error(self,y_true,y_pred,y_unc,bins):
        """
        For a given number of bins, calculate the DUQ.

        Parameters
        ----------
        y_true :    ndarray, shape (n,) 
                    True values of data that has been predicted.
        y_pred :    ndarray, shape (n,) 
                    Predicted data. 
        y_unc :     ndarray, shape (n,) 
                    Uncertainty associated with each prediction. 
        bins :      int
                    Number of bins to use in histogram. 

        Returns
        -------
        duq :   float
                duq for input data and number of bins.
        hist :  ndarray (n,)
                Bin heights for histogram of input data. 
        """
        A = 1./(len(bins)-1)
        bin_widths = np.diff(bins)
        N_bins = len(bin_widths)
        # The thing we want to bin
        err_ratio = self.calc_err_ratio(y_true,y_pred,y_unc)
        err_ratio = err_ratio[~np.isnan(err_ratio)]
        max_err = err_ratio.max()
        # Bin errors and get the histogram.
        hist,_ = np.histogram(err_ratio,bins=bins,density=True)
        # Calculate the distribution error here.
        duq = 0.5*np.abs(hist*bin_widths-A).sum()
        duq *= N_bins/(N_bins-1.)
        return duq,hist
    
    def __call__(self,y_true,y_pred,y_unc):
        """
        Calculate the DUQ for different numbers of bins and choose optimal value 
        by minimising the (DUQ + adjustment term). 

        Parameters
        ----------
        y_true :    ndarray, shape (n,) 
                    True values of data that has been predicted.
        y_pred :    ndarray, shape (n,) 
                    Predicted data. 
        y_unc :     ndarray, shape (n,) 
                    Uncertainty associated with each prediction. 

        Returns
        -------
        opt_duq :       float
                        Optimal DUQ value.
        opt_bins :      ndarray (n+1,)
                        Array of bin edges for optimal binning.
        opt_ideal_hist :ndarray (n,)
                        Array of bin heights for histogram of ideal gaussian. 
        opt_hist :      ndarray (n,)
                        Array of bin heights for input data. 
        """
        best_error = 1.e6
        for i,(adj,bins,ideal_hist) in enumerate(self.adjustments):
            duq,hist = self.dist_error(y_true,y_pred,y_unc,bins)
            error = duq+adj
            if error < best_error:
                best_error = copy(error)
                # Things we actually want to get out
                opt_duq = copy(duq)
                opt_hist = copy(hist)
                opt_bins = copy(bins)
                opt_ideal_hist = copy(ideal_hist)
        return opt_duq,opt_bins,opt_ideal_hist,opt_hist

class DUQ_pos(DUQ):
    """DUQ for positive bins.
    Variation of the DUQ where |residual/uncertainty| is binned so all bins are positive.
    Comparison is made to the histogram of the half-normal distribution. 
    """
    def __init__(self,cutoff_val=default_cutoff,max_bins=20):
        super(DUQ_pos,self).__init__(cutoff_val,max_bins)
        
    def __adj_term(self,N_bins):
        """
        For a given number of bins, calculate the adjustment term used when optimising
        the number of bins. This is the difference in area between the ideal histogram 
        and the true normal distribution. 
        """
        # Area of ideal histogram bins.
        A = 1./N_bins
        cum_areas = np.arange(N_bins)*A
        ideal_bins = erfinv(cum_areas) # Ideally the Nth bin has height 0 and infinite width
        real_bins  = np.append(ideal_bins,self.cutoff_val)
        bin_widths = np.diff(real_bins)
        # Heights of the ideal bins
        id_bin_hs  = A/bin_widths
        # Calulate the adjustment to the error
        adj_error = romberg(lambda z: np.square(id_bin_hs[np.digitize(z,real_bins)-1]-2/np.sqrt(np.pi)*np.exp(-z**2)),
                                0.,self.cutoff_val*.999,divmax=100,tol=1.e-6,vec_func=True)
        return adj_error,real_bins,id_bin_hs
    
    def gen_adjustments(self):
        """
        Generate the adjustment terms for all the number of bins being considered. 
        """
        self.adjustments = [self.__adj_term(N,self) for N in np.arange(3,self.max_bins)]

    @staticmethod
    def calc_err_ratio(y_true,y_pred,y_unc): return np.abs(y_pred-y_true)/y_unc

def plot_duq(bins,ideal_hist,hist,ax=None):
    """
    Plot the histogram produced from a duq calculation. 

    Parameters
    ----------
    bins :          ndarray, (n+1,)
                    ndarray of bin edges.
    hist :          ndarray, (n,)
                    ndarray of histogram heights for true data
    ideal_hist :    ndarray, (n,)
                    ndarray of histogram heights for ideal gaussian.
    ax :            axes.Axes, optional
                    Matplotlib axis to plot onto. 
    """
    if not ax:
        fig,ax = plt.subplots()
    # Plot histograms
    ax.bar((bins[1:]+bins[:-1])/2,hist,(bins[1:]-bins[:-1]),ec="c",fc="c",zorder=10)
    ax.bar((bins[1:]+bins[:-1])/2,ideal_hist,(bins[1:]-bins[:-1]),ec="k",fill=False,hatch="//",zorder=15)
    # Plot gaussian
    lim = min(2*bins[-2],bins[-1])
    z_pts = np.linspace(-lim,lim,100)
    ax.plot(z_pts,1/np.sqrt(np.pi)*np.exp(-z_pts**2),"k",zorder=20)
    ax.set_xlim([-lim,lim])
    ax.set_ylim([0,1.1*max(hist.max(),ideal_hist.max())])
    ax.set_xlabel("(resdiual/uncertainty)")
    ax.set_ylabel("Density")