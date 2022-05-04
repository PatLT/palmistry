#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Nov 09 2021

@author: Pat Taylor (pt409)
"""
#%% Libraries
from dataclasses import replace
from distutils.log import warn
from typing_extensions import runtime
import numpy as np
import pandas as pd
import dill
import warnings
import ast
import functools

from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats import pearsonr

import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.decomposition import PCA
from sklearn.utils.optimize import _check_optimize_result
from joblib import Parallel,delayed
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import minimize
from scipy.linalg import logm,norm,block_diag,inv
from scipy.special import erf,erfc
from ase import data
from ase.build import bulk
# PyTorch, GPyTorch
import torch, gpytorch

import matplotlib.pyplot as plt

from copy import deepcopy,copy

from pprint import pprint

from shells import shell_radii

v = 2 # global output verbosity
if v < 3:
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

#%% Data processing
def check_if_valid(input_,allow_all_null=False):
    """
    Helper function to check if database rows are valid to process.

    Parameters
    ----------
    input_ (pd.Series)      : data series to check.
    allow_all_null (bool)   : Whether a full row of null entries ("-") is to count as valid.

    Return
    ------
    input_ (pd.Series)  : float64 version of input_ parameter.
    rtn_code (int)      : A return code denoting validity of input. 
    """
    if np.prod("-" == input_) and not allow_all_null:
        return input_, 0 # Invalid input, all entries are -
    else :
        input_ = np.where(input_ == "-",0,input_).astype(np.float64)
        if np.isnan(input_.sum()):
            return input_, 1 # Invalid input, (at least one) empty column
        else :
            return input_, 2 # Valid input

def get_microstructure_data(df,drop_dupli=False,shuffle_seed=None):
    """
    Extract all entries from databse with complete microstructure information. 
    Here "complete" means complete composition + precipitate fraction + phase
    composition data. 

    Parameters
    ----------
    df (pd.DataFrame)   : dataframe to extract complete entries from.
    drop_dupli (bool)   : Whether to retain any duplicate compositions that are found.
    shuffle_seed (int)  : Random seed to use to shuffle database entries' orders.
    
    Return
    ms_df (pd.Dataframe): dataframe of complete entries only. 
    """
    ms_df = df.copy()
    drop_indices = []
    for index,row in df.iterrows():
        ht, ht_code = check_if_valid(row[("Precipitation heat treatment")],allow_all_null=True)
        if ht_code == 2:
            comp, comp_code = check_if_valid(row[("Composition","at. %")])
            if comp_code == 2:
                frac, frac_code = check_if_valid(row[("γ’ fraction","at. %")])
                if frac_code == 2:
                    prc, prc_code   = check_if_valid(row[("γ’ composition","at. %")])
                    if prc_code == 2:
                        mtx, mtx_code   = check_if_valid(row[("γ composition","at. %")])
                        if mtx_code == 2:
                            continue
        drop_indices += [index]
    ms_df.drop(drop_indices,inplace=True)
    if drop_dupli:
        ms_df = ms_df.loc[ms_df[("Composition","at. %")].drop_duplicates().index]
    # Shuffle database and select a specified fraction of it:
    ms_df=ms_df.sample(frac=1.,random_state=shuffle_seed).reset_index(drop=True)
    return ms_df

def get_Xy(df,y_header,drop_els=[],
           min_max=None,drop_na=True,flatten=False,ht=False,log_y=False,
           ht_function = None):
    """
    Use in conjunction with get_microstructure to get the X,y data for ML.
    
    Parameters:
    -----------
    df (pd.DataFrame)   : Dataframe to process. 
    y_header (tuple,str): DataFrame column name for y data to extract. Entered as a tuple for multiindex compatibility.
                          If y_header=None, this function just returns all the X data.
    drop_els (list,str) : List of element names to drop from X (composition) data to extract
    min_max (list,float): Min and max cutoff values for databse entries to extract.
    drop_na (bool)      : Drop empty rows (in y) from returned database.
    flatten (bool)      : Whether to return y as shape (n,1) [FALSE] or (n,) [TRUE].
    ht (bool)           : Whether to include heat treatment as well as composition data in X.
    log_y (bool)        : Return logarithm of y data.
    ht_function (lambda): Lambda function to apply to heat treatment data if used.

    Return:
    -------
    X (ndarray) : X data extracted from DataFrame. 
    y (ndarray) : y data extracted from DataFrame. Not returned if y_header=None. 
    """

    if len(drop_els)==0: 
        drop_els=None
    elif drop_els[0]=="": 
        drop_els=None
    elif drop_els[-1]=="": 
        drop_els=drop_els[:-1]
    # Enter header as tuple in case of multiindex
    if y_header:
        # drop rows less/greater than certain min/max values
        if drop_na:
            sub_df = df.dropna(subset=y_header)
        else:
            sub_df = df.copy()
        if min_max:
            min_, max_ = tuple(min_max)
            if isinstance(min_,float): 
                condition_0 = (sub_df != False)
                condition = sub_df[y_header].astype("float64") > min_ # Min
                condition_0.update(condition)
                sub_df = sub_df[condition_0].dropna(subset=y_header)
            if isinstance(max_,float): 
                condition_0 = (sub_df != False)
                condition = sub_df[y_header].astype("float64") < max_ # Max
                condition_0.update(condition)
                sub_df = sub_df[condition_0].dropna(subset=y_header)
        # Now drop empty rows
        # Start getting data here:
        y = sub_df.loc[:,y_header].astype("float64").values
        if flatten and len(y.shape) > 1 and y.shape[-1] == 1:
            y = y.flatten()
        if log_y:
            y = np.log(y)
    else:
        sub_df = df.copy()
    if drop_els:
        X1 = 0.01*(sub_df.loc[:,("Composition","at. %")].drop(drop_els,axis=1).astype("float64").values)
    else:
        X1 = 0.01*(sub_df.loc[:,("Composition","at. %")].astype("float64").values)
    if ht:
        X0 = sub_df.loc[:,("Precipitation heat treatment")]
        col_order = sorted(X0.columns.tolist(),key = lambda h: h[1])
        X0 = X0[col_order].replace("-",0.0).astype(np.float64).values
        X0[:,:3] += 273.
        if ht_function:
            X0 = ht_function(X0)
        X = np.append(X0,X1,axis=1)
    else:
        X = X1
    if y_header:
        return X,y
    else:
        return X

#%% GPR
class customGPR(GaussianProcessRegressor):
    """
    Modification of the sklearn parent class that adds explicit max_iter argument.
    """
    def __init__(self, 
        kernel=None,
        *,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        max_iter=15000):
        super().__init__(kernel,
                alpha=alpha,
                optimizer=optimizer,
                n_restarts_optimizer=n_restarts_optimizer,
                normalize_y=normalize_y,
                copy_X_train=copy_X_train,
                random_state=random_state)
        self.max_iter = max_iter
    
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter":self.max_iter}
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min

#%% Kernels
# All of the following are modified from sklearn parent classes.

class Linear(gp.kernels.DotProduct):
    """
    Modification of DotProduct kernel from sklearn. Allows for simple dimensionality
    reduction to account for only part of X being used. 
    """
    def __init__(self,sigma_0=1.0,sigma_0_bounds=(1.e-5,1.e5),
                 dims=15,dim_range=None,comp=False):
        super(Linear,self).__init__(sigma_0,sigma_0_bounds)
        self.dims = dims
        self.dim_range = dim_range
        self.comp = comp
        self.constr_trans()
        
    def constr_trans(self):
        A = np.eye(self.dims)
        if self.dim_range: 
            A = A[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            A = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],A]
        A = A.T # Use transpose since vectors are represented by rows not columns.
        self.A = A
        
    def trans(self,X):
        return X@self.A
        
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = self.trans(X)
        if Y is None:
            K = np.inner(X, X) + self.sigma_0 ** 2
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y = self.trans(Y)
            K = np.inner(X, Y) + self.sigma_0 ** 2

        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                K_gradient = np.empty((K.shape[0], K.shape[1], 1))
                K_gradient[..., 0] = 2 * self.sigma_0 ** 2
                return K, K_gradient
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K
    
    def diag(self, X):
        X = self.trans(X)
        return np.einsum('ij,ij->i', X, X) + self.sigma_0 ** 2
    
class PCALinear(Linear):
    """
    Further modification of sklearn DotProduct kernel. Allows for further dimensionality
    reduction via an explicit PC projection matrix. 
    """
    def __init__(self,sigma_0=1.0,sigma_0_bounds=(1.e-5,1.e5),
                 pca_components=None,use_inv=False,
                 dims=15,dim_range=None,comp=False):
        self.pca_components = pca_components
        self.use_inv = use_inv
        super(PCALinear,self).__init__(sigma_0,sigma_0_bounds,dims,dim_range,comp=comp)
        
    def constr_trans(self):
        Ilike = np.eye(self.dims)
        if self.dim_range: 
            Ilike = Ilike[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            Ilike = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],Ilike]        # Subspace projection part of matrix
        self.A = (self.pca_components @ Ilike).T
        
    def trans(self,X):
        if self.use_inv:
            return (X@self.A)**-1
        else:
            return X@self.A
        
class L2RBF(gp.kernels.RBF):
    """
    Modification of RBF kernel from sklearn. Allows for simple dimensionality
    reduction to account for only part of X being used. 
    """
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 dims=15,dim_range=None,comp=False):
        super(L2RBF,self).__init__(length_scale,length_scale_bounds)
        # Matrix used to transform vectors in call.
        self.dims = dims
        self.dim_range = dim_range
        self.comp = comp
        self.constr_trans()
        
    def constr_trans(self):
        A = np.eye(self.dims)
        if self.dim_range: 
            A = A[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            A = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],A]
        A = A.T # Use transpose since vectors are represented by rows not columns.
        self.A = A
        
    def trans(self,X):
        return X@self.A
        
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = self.trans(X)
        length_scale = gp.kernels._check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y = self.trans(Y)
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :])**2 \
                    / length_scale
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

class PCAL2RBF(L2RBF):
    """
    Further modification of sklearn RBF kernel. Allows for further dimensionality
    reduction via an explicit PC projection matrix. 
    """
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 pca_components=None,
                 dims=15,dim_range=None,comp=False):
        self.pca_components = pca_components
        super(PCAL2RBF,self).__init__(length_scale,length_scale_bounds,
                 dims,dim_range,comp)
        
    def constr_trans(self):
        Ilike = np.eye(self.dims)
        if self.dim_range: 
            Ilike = Ilike[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            Ilike = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],Ilike]        # Subspace projection part of matrix
        self.A = (self.pca_components @ Ilike).T

#%% SCALER
class PartScaler():
    """
    Based on sklearn.preprocessing.StandardScaler . Allows for scaling of part of an 
    input. 

    Parameters
    ----------
    scale_range (list,int)  : Start end end indices for parts of input that are to be scaled.
    copy_ (bool)            : Whether to make a copy of the input when transforming.
    with_mean (bool)        : Whether to remove mean value during transformation.
    with_std (bool)         : Whether to divide by std during transformation. 
    """
    def __init__(self, scale_range=None, copy_=True,with_mean=True,with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy_ = copy_
        self.range_ = scale_range
        
    def _reset(self):
        if hasattr(self,'scale_'):
            del self.scale_
            del self.offset_
            
    def fit(self,X):
        """
        Fit scaling transformation to some input data.

        Parameters
        ----------
        X (ndarray) : data to fit transformation to. 
        """
        self._reset()
        if self.with_mean:
            if self.range_:
                self.offset_ = np.zeros(X.shape[1])
                self.offset_[self.range_[0]:self.range_[1]] = np.mean(X[:,self.range_[0]:self.range_[1]],axis=0)
            else:
                self.offset_ = np.mean(X,axis=0)                
        else: 
            self.offset_ = 0.0
        
        if self.with_std:
            if self.range_:
                self.scale_ = np.ones(X.shape[1])
                self.scale_[self.range_[0]:self.range_[1]] = np.std(X[:,self.range_[0]:self.range_[1]],axis=0)
            else:
                self.scale_ = np.std(X,axis=0)
            self.scale_ = np.where(self.scale_==0.0,1.0,self.scale_)
        else:
            self.scale_ = 1.0
        return self
    
    def transform(self,X,copy_=None):
        """
        Carry out transformation.
        
        Parameters
        ----------
        X (ndarray) : Input data to transform. 

        Return
        ------
        X (ndarray) : Transformed data. 
        """
        copy_ = copy_ if copy_ is not None else self.copy_
        if copy_:
            X = X.copy()
        X -= self.offset_
        X /= self.scale_
        return X
        
    def inverse_transform(self,X,copy_=None):
        """
        Carry out inverse transformation. 

        Parameters
        ----------
        X (ndarray) : Transformed data to return to original representation. 

        Return
        ------
        X (ndarray) : Data in original representation. 
        """
        copy_ = copy_ if copy_ is not None else self.copy_
        if copy_:
            X = X.copy()
        X *= self.scale_
        X += self.offset_
        return X
    
    def fit_transform(self,X,copy_=None):
        """
        Fit and transform data in a single step. 
        """
        self.fit(X)
        return self.transform(X,copy_)

#%% MODEL CLASSES
"""
REDACTED. See microstructure_gpr.py for these two classes.
They were based on an older idea of what the codebase should look like. 
"""

#%% PLOTTING
# Use to get colours for different models for each datapt, e.g. for plotting 
def gen_colours(values):
    """
    Use to get colours for different models for each datapoint, e.g. for plotting.

    Parameters
    ----------
    values (ndarray)    : A code value for each point in a dataset.

    Returns
    -------
    colours (ndarray)   : Colour index for each point in dataset. 
                        Will be as many colours as there were unique codes.
    key2col (dict)      : Dictionary mapping codes (unique entries in values) to colour indices. 
    """
    values = np.squeeze(values)
    if len(values.shape) > 1:
        colour_dict = {code.tobytes():i for i,code in enumerate(np.unique(values,axis=0))}
        key2col = lambda code: colour_dict.get(code.tobytes())
    else:
        colour_dict = {code:i for i,code in enumerate(np.unique(values))}
        key2col = lambda code: colour_dict.get(code)
    colours = np.array(list(map(key2col,values)))
    return colours,key2col

def plot_byModel(f_true,f_pred,f_stds,
                 name="precipitate fraction",
                 colour_src=None,
                 colourmap="brg",
                 lims=None,
                 label_outliers=None,
                 data_labels=None):
    """
    Quick function to plot predicted vs. true values. 

    Parameters
    ----------
    f_true (ndarray)    : The true values to plot.
    f_pred (ndarray)    : The predicted values to plot.
    f_stds (ndarray)    : Uncertainties for each prediciton.
    name (string)       : Name of variable being plotted.
    colour_src (ndarray): Optional array of colour codes for each datapoint. 
    colourmap (string)  : Pyplot colourmap to use.
    lims (list)         : Lower and upper limits for plot axes.
    label_outliers (float): If not none, outliers greater than this value will be labelled. 
    data_labels (ndarray): Labels for each datapoint, will only be used if label_outliers!=None.
    """
    fig,axs=plt.subplots()
    plt.errorbar(f_true,f_pred,yerr=f_stds,fmt=".",ecolor="k",elinewidth=0.5,zorder=0)
    plt.scatter(f_true,f_pred,marker=".",c=colour_src,cmap=colourmap,zorder=10)
    if label_outliers:
        offset=0.05*np.median(f_pred)
        for f_true_i,f_pred_i,f_std_i,alloy_name in zip(f_true,f_pred,f_stds,data_labels):
            f_tol = f_std_i if isinstance(label_outliers,str) else label_outliers
            if np.abs(f_true_i-f_pred_i) > f_tol: 
                axs.annotate(alloy_name,(f_true_i+offset,f_pred_i+offset),annotation_clip=False)
    if lims is None:
        lims = [min(axs.get_xlim()+axs.get_ylim()),max(axs.get_xlim()+axs.get_ylim())]
    axs.set_xlim(lims)
    axs.set_ylim(lims)
    axs.plot(lims,lims,"--k")
    axs.set_aspect("equal","box")
    axs.set_xlabel("Actual "+name)
    axs.set_ylabel("Predicted "+name)
    return fig,axs

#%% Hume-Rothery transformations.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DECORATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Design of decorators stolen from: 
# https://stackoverflow.com/questions/11731136/class-method-decorator-with-self-arguments
def inputOutputRule(transformRule):
    @functools.wraps(transformRule)
    def wrapper(*args,**kwargs):
        # First get relevant object instance.
        _self = args[0]
        # Check if special flag has been passed. 
        dflag = kwargs.pop("dryrun",False)
        if not _self.Xy_share_labels:
            raise RuntimeError("Cannot use transformation rules that assume a relation between input-output labels for datasets without such relations.")
        else:
            if dflag:
                # Present some fake data and get shape of transformed data
                fake_X = np.c_[[np.random.rand(_self.aux_dims)],np.array([[0.95,.05]])]
                fake_nums = [29,50] 
                fake_y = np.ones((1,1))
                out = transformRule(_self,fake_X,fake_nums,fake_y,**kwargs)
                out_dim = 1 if len(out.shape)==1 else out.shape[-1]
                return out_dim
            else:
                return transformRule(*args,**kwargs)
    return wrapper

def inputRule(transformRule):
    @functools.wraps(transformRule)
    def wrapper(*args,**kwargs):
        # First get relevant object instance.
        _self = args[0]
        # Check if special flag has been passed. 
        dflag = kwargs.pop("dryrun",False)
        if _self.Xy_share_labels:
            if dflag:
                # Present some fake data and get shape of transformed data
                fake_X = np.c_[[np.random.rand(_self.aux_dims)],np.array([[0.95,.05]])]
                fake_nums = [29,50] 
                fake_y = np.ones((1,1))
                out = transformRule(_self,fake_X,fake_nums,fake_y,**kwargs)
                out_dim = 1 if len(out.shape)==1 else out.shape[-1]
                return out_dim
            else:
                X,at_nums,y = args[1:]
                X,y = _self.extend_original(X,y)
            return transformRule(_self,X,at_nums,y,**kwargs)
        elif _self.multi_output:
            if dflag:
                # Present some fake data and get shape of transformed data
                fake_X = np.c_[[np.random.rand(_self.aux_dims-1)],np.array([[0.95,.05]])]
                fake_nums = [29,50] 
                fake_y = np.ones((1,_self.output_cats))
                fake_X,fake_y = _self.multi_2_single_output(fake_X,fake_y)
                out = transformRule(_self,fake_X,fake_nums,fake_y,**kwargs)
                out_dim = 1 if len(out.shape)==1 else out.shape[-1]
                return out_dim
            else:
                X,at_nums,y = args[1:]
                X,y = _self.multi_2_single_output(X,y)
            return transformRule(_self,X,at_nums,y,**kwargs)
        else:
            if dflag:
                fake_X = np.c_[[np.random.rand(_self.aux_dims)],np.array([[0.95,.05]])]
                fake_nums = [29,50] 
                fake_y = np.ones((1,1))
                out = transformRule(_self,fake_X,fake_nums,fake_y,**kwargs)
                out_dim = 1 if len(out.shape)==1 else out.shape[-1]
                return out_dim
            else:
                return transformRule(*args,**kwargs)
    return wrapper

class HRrep_parent():
    """
    Parent class that does NOT implement the transform(X[,y]) method!!!


    This class carries out the transformation into the Hume-Rothery basis.
    It has been written to work with sklearn.pipeline.Pipeline
    """
    def __init__(self,*features,
                        Xy_share_labels=False,
                        aux_dims=None,
                        multi_output=False,
                        rdf_rep_params={}):
        """
        Parameters
        ----------
        Xy_share_labels (bool)  : Whether or not inputs and outputs are linked due to common labelling e.g. input = composition, output = partitioning coefficients
        aux_dims (int)          : Number of dims at the START of the input that do not correspond to composition.
        multi_output (int)      : Number of target columns, optional. Can't be used with Xy_share_labels

        Args
        ----
        features (strings)  : strings correpsonding to the names of desired features in the representation.
        """
        self.Xy_share_labels=Xy_share_labels
        self.aux_dims = aux_dims if aux_dims is not None else 0

        ################################# LEGACY #################################
        self.pt = np.genfromtxt("hr_table.csv",delimiter=",",
                            missing_values=("nan",),filling_values=(np.nan,),
                            skip_header=1,usecols=(2,3,4,5,8,9,10,11,12,13,14,
                                                    15,16,17,18,19,20,21,22,23,24,25,
                                                    26,27,28,29,30,31,32,
                                                    34,35))
        # Atomic properties
        self.cr = self.pt.T[0] # Covalent radius
        self.en = self.pt.T[2] # Electronegativities
        self.sg = self.pt.T[3].astype("int") # Structure groups
        self.pf = self.pt.T[4] # Atomic packing factor
        self.wf = self.pt.T[5] # Work functions
        self.va = self.pt.T[6] # Valence (non-core electrons)
        self.mp = self.pt.T[29]# Modified pettifor scale
        self.am = self.pt.T[30]# Atomic mass
        self.orbitals = dict(zip(["s","p","d","f"],self.pt.T[7:11]))
        # Pseudopotential radii
        self.pp_radii = dict(zip(["1s","2s","2p","3s","3p","3d","4s","4p","4d","5s","5p",
                                    "5d","5f","6s","6p","6d","7s","4f"],
                                self.pt.T[11:29])) # girth is the only relevant feature 
        s = [self.pp_radii["1s"],self.pp_radii["2s"],self.pp_radii["3s"],self.pp_radii["4s"],self.pp_radii["5s"],self.pp_radii["6s"],self.pp_radii["7s"]]
        p = [self.pp_radii["2p"],self.pp_radii["3p"],self.pp_radii["4p"],self.pp_radii["5p"],self.pp_radii["6p"]]
        d = [self.pp_radii["3d"],self.pp_radii["4d"],self.pp_radii["5d"],self.pp_radii["6d"]]
        f = [self.pp_radii["4f"],self.pp_radii["5f"]]
        self.pp_radii["s"] = np.sum(s,axis=1)/np.count_nonzero(s,axis=1)
        self.pp_radii["p"] = np.sum(p,axis=1)/np.count_nonzero(p,axis=1)
        self.pp_radii["d"] = np.sum(d,axis=1)/np.count_nonzero(d,axis=1)
        self.pp_radii["f"] = np.sum(f,axis=1)/np.count_nonzero(f,axis=1)
        #########################################################################
        self.elemental_props = pd.read_csv("hr_table.csv",index_col=0)
        self.mod_pettifor_M  = pd.read_csv("mod_pettifor_matrix.csv",skiprows=[0],index_col=1).drop("Unnamed: 0",1)

        # Elements used in fit.
        self.els_in_fit = None
        # Features used in representation. 
        self.ft_names = features 
        self.m_out = len(features)
        # create a basic rdf class instance.
        self.rdf_0 = self.RDF(self,**rdf_rep_params)
        # Multi-output stuff
        if isinstance(multi_output,bool) or multi_output==1:
            self.multi_output = multi_output
            self.output_cats = 0
        else:
            if self.Xy_share_labels:
                raise RuntimeError("Cannot have Xy_share_labels=True and multi_output > 1.")
            self.multi_output = True
            self.output_cats = multi_output
            self.aux_dims += 1
    
    @staticmethod
    def get_els(dataframe,rtn_numbers=True):
        """
        Work out which elements are in the dataframe.

        Parameters
        ----------
        dataframe (pd.Dataframe): Input dataframe
        rtn_numbers (bool)      : Whether to return atomic numbers or element names.
        """
        nlevels = dataframe.columns.nlevels
        if nlevels==3:
            els = dataframe.columns[
                dataframe.columns.get_locs(("Composition","at. %"))].get_level_values(2).to_numpy()
        elif nlevels==2:
            els = dataframe.columns[
                dataframe.columns.get_locs(("Composition",))].get_level_values(1).to_numpy()
        elif nlevels==1:
            els = dataframe.columns.to_numpy()
        nums = [data.atomic_numbers.get(el) for el in els]
        if rtn_numbers:
            return nums
        else:
            return els
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~ TRANSFORMER METHODS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fit(self,X,y):
        """
        Doesn't do anything except work out which elements were included in 
        the initial fit, and maybe work out number of output categories. 
        """
        self.els_in_fit = self.get_els(X,rtn_numbers=False)
        if self.multi_output:
            self.output_cats = self.multi_2_single_output(X.values,y.values,True)
        return self

    def transform(self,X,y):
        """
        Raises a runtime error: this method is purposefully not implemented in 
        this class. Use child class HRrep instead. 
        """
        raise RuntimeError("Use HRrep class instead")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RESHAPE METHODS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def extend_original(self,X,y=None):
        """
        Maps the original representation from (N,m) array to (~N*m,m) array.
        """
        N,m = X.shape
        m -= self.aux_dims
        mask = self._gen_mask(X)
        if y is None:
            return np.repeat(X,m,axis=0)[mask],None
        else:
            if isinstance(y,tuple):
                return tuple([np.repeat(X,m,axis=0)[mask]]\
                    +[y_.flatten()[mask] for y_ in y])
            else:
                return np.repeat(X,m,axis=0)[mask],y.flatten()[mask]

    def multi_2_single_output(self,X,y,get_cat_num_only=False):
        """
        Similar to extend_original, except this works explicitly for any type of multi-output and generates a new
        label column.
        """
        self.multi_output = True
        if y is not None:
            N,m = y.shape
            mask = ~(np.any([np.isnan(y.flatten()),np.isinf(y.flatten())
            ],0))
        else:
            N = X.shape[0]
            m = self.output_cats
            mask = np.ones(N*m).astype(bool)
        if get_cat_num_only:
            return m
        if y is not None:
            return np.c_[np.arange(N*m)%m,np.repeat(X,m,axis=0)][mask],y.flatten()[mask]
        else:
            return np.c_[np.arange(N*m)%m,np.repeat(X,m,axis=0)][mask],y

    def revert_shape(self,X_orig,y_out):
        """
        Take an array with entries corresponding to non-zeros in original
        data (e.g. from ML predictions) and reshape to match original shape.
        """
        if self.Xy_share_labels:
            N,m = X_orig.shape
            m -= self.aux_dims
            mask = self._gen_mask(X_orig.values)
            y_rev = np.zeros(N*m)
            y_rev[np.where(mask)] = y_out
            y_df = X_orig.copy()
            y_df = y_df.iloc[:,self.aux_dims:]
            if y_df.columns.nlevels > 1:
                y_df.columns = y_df.columns.droplevel(list(range(y_df.columns.nlevels-1)))
            y_df.loc[:] = y_rev.reshape((N,m))
            return y_df
        else:
            return y_out

    def reshape_cov2sub(self,X_orig,cov,y=None):
        """
        Reshape the covariance array for predictions into a list of sub-covariance
        matrices, each corresponding to the covariance of predictions for a single
        given entry. Note this returns a ragged list of matrices, i.e. matrices
        do NOT contain entries for elements not present in input. 

        Parameters
        ----------
        X_orig  (pd.Dataframe)  : The original input data. Used to get zero entries. 
        cov     (ndarray)       : Covariance matrix to reshape. Provide as tuples to get joined arrays as outputs.
        y       (ndarray)       : Optional. Provide predictions and reshape these in the same way too.
        """
        if self.Xy_share_labels:
            tuple_flag = True if isinstance(cov,tuple) else False
            at_nums = self.get_els(X_orig)
            m = len(at_nums)
            mask = self._gen_mask(X_orig.values)
            locs = mask.reshape(-1,m) # Locations of non-zero components
            sub_cov = [] # sub-covariance matrix list
            if y is not None:
                sub_y = []
            start_ind = 0
            for entry in locs:
                end_ind = start_ind + entry.sum()
                if tuple_flag:
                    sub_covs = (cov_[start_ind:end_ind,start_ind:end_ind] for cov_ in cov)
                    sub_cov += [block_diag(*sub_covs)]
                else:
                    sub_cov += [cov[start_ind:end_ind,start_ind:end_ind]]
                if y is not None:
                    if tuple_flag:
                        sub_ys = tuple(y_[start_ind:end_ind] for y_ in y)
                        sub_y += [np.concatenate(sub_ys)]
                    else:
                        sub_y += [y[start_ind:end_ind]]
                start_ind = copy(end_ind)
            if y is not None:
                return sub_cov,sub_y
            else:    
                return sub_cov
        else:
            return

    def _gen_mask(self,X):
        """
        Generates a mask to apply to remove entries corresponding to component with zero composition.
        """
        X_ = self._c_dims(X).flatten()
        return ~(X_==0.)

    def _c_dims(self,X):
        """
        Return the dimensions of the input corresponding to composition.
        """
        return X[:,self.aux_dims:]

    def _a_dims(self,X,dims2use="all"):
        """
        Return the dimensions of the input correponding to auxiliary componenents (i.e. not composition).
        """
        if dims2use=="all":
            return X[:,int(self.multi_output):self.aux_dims]
        else:
            return X[:,np.array(dims2use)+int(self.multi_output)]
        

    #~~~~~~~~~~~~~~~~~~~~~~~~ TRANSFORMATION RULES ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # _mu suffix -> chemical potential-type rule
    # _g  suffix -> free energy-type rule
    # _struc suffix -> structure-type rule
    # _atom -> atom species info rule.

    # _tr_ prefix is required for child class to auto-find all relevant transformation rules
    
    # kwargs should be used to implement adiditonal parameters instead of args. 
    # dryrun should NOT be used as a kwarg - it is popped out by the decorator. 

    # Chemical potential-like transformations
    @inputOutputRule
    def _tr_mix_mu(self,X,at_nums,y=None):
        """
        New rule: entropy of mixing of the alloy.
        """
        X = self._c_dims(X)
        S = np.ma.log(X).flatten()
        return S[~S.mask].data

    def get_radii(self,radius_type,at_nums):
        if radius_type=="covalent":
            m = self.elemental_props.iloc[at_nums].loc[:,"Covalent radius"].values
        elif radius_type.lower()=="vdw":
            m = self.elemental_props.iloc[at_nums].loc[:,"Vdw radius"].values
        elif radius_type=="fcc":
            m = self.elemental_props.iloc[at_nums].loc[:,"DFT fcc lattice param"].values
        elif radius_type=="hcp":
            m = self.elemental_props.iloc[at_nums].loc[:,"DFT hcp lattice param"].values
        return m

    @inputOutputRule
    def _tr_mis_mu(self,X,at_nums,y=None,**kwargs):
        """
        Atomic size misfit of the alloy.
        """
        # Extra parameter
        method = kwargs.get("method","hooke")
        radius = kwargs.get("radius","covalent")

        m = self.get_radii(radius,at_nums)
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        if method=="hooke":
            out = (np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T.flatten()
        elif method=="enthalpy":
            out = (-(np.repeat(m.reshape(-1,1),X.shape[0],axis=1))**2+(X@m)**2).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_eng_mis(self,X,at_nums,y=None,**kwargs):
        """
        Electronegativity misfit of the alloy.
        """
        m = self.elemental_props.iloc[at_nums].loc[:,"Electronegativity"].values
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        out = (np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T.flatten()
        return out[mask]
    
    @inputOutputRule
    def _tr_val_mu(self,X,at_nums,y=None,**kwargs):
        """
        Mean valence of the alloy.
        """
        # Extra parameter
        method = kwargs.get("method","valence")

        m = self.elemental_props.iloc[at_nums].loc[:,"Valence"].values
        N,n = X.shape
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        if method=="valence":
            out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
            return out[mask]
        elif method=="sommerfeld":
            pass
    
    @inputOutputRule
    def _tr_eng_mu(self,X,at_nums,y=None):
        """
        Electronegativities of the alloy.
        """
        m = self.elemental_props.iloc[at_nums].loc[:,"Electronegativity"].values
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        M = (m.reshape(-1,1)-(m.T))**2
        out = np.einsum("kj,ij->ki",X,M).flatten()
        return out[mask]

    @inputOutputRule
    def _tr_prp_rep(self,X,at_nums,y=None):
        """
        Pettifor replacement probability for a given species in an alloy.
        """
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        M = self.mod_pettifor_M.iloc[at_nums,at_nums].values
        out = np.einsum("kj,ij->ki",X,M).flatten()
        return out[mask]

    # Free energy-like transformations
    @inputRule
    def _tr_mix_g(self,X,at_nums,y=None):
        """
        New rule: entropy of mixing of the alloy.
        """
        X = self._c_dims(X)
        S = (X*np.ma.log(X).filled(0.)).sum(axis=1)
        return S

    @inputRule
    def _tr_mis_g(self,X,at_nums,y=None,**kwargs):
        """
        Atomic size misfit of the alloy.
        """
        # Extra parameter
        method = kwargs.get("method","std")
        radius = kwargs.get("radius","covalent")

        m = self.get_radii(radius,at_nums)
        X = self._c_dims(X)
        if method=="hooke":
            out = (X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**2).sum(axis=1)
        elif method=="enthalpy":
            out = (X*(-(np.repeat(m.reshape(-1,1),X.shape[0],axis=1))**3+(X@m)**3).T).sum(axis=1)
        elif method=="std":
            out = (X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**2).sum(axis=1)
            out = np.sqrt(out)
        return out
    
    @inputRule
    def _tr_eng_std(self,X,at_nums,y=None,**kwargs):
        """
        Electronegativity std for the alloy. 
        """
        m = self.elemental_props.iloc[at_nums].loc[:,"Electronegativity"].values
        X = self._c_dims(X)
        out = (X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**2).sum(axis=1)
        return np.sqrt(out)

    @inputRule
    def _tr_val_g(self,X,at_nums,y=None,**kwargs):
        """
        Mean valence of the alloy.
        """
        # Extra parameter
        method = kwargs.get("method","valence")

        m = self.elemental_props.iloc[at_nums].loc[:,"Valence"].values
        X = self._c_dims(X)
        if method=="valence":
            return X@m
        elif method=="sommerfeld":
            pass

    @inputRule
    def _tr_eng_g(self,X,at_nums,y=None):
        """
        Electronegativities of the alloy.
        """
        X = self._c_dims(X)
        m = self.elemental_props.iloc[at_nums].loc[:,"Electronegativity"].values
        M = (m.reshape(-1,1)-(m.T))**2
        return np.einsum("kj,ki,ij->k",X,X,M)

    @inputRule
    def _tr_prp_m(self,X,at_nums,y=None):
        """
        Mean pettifor replacement probability for all of the atomic species.
        """
        X = self._c_dims(X)
        M = self.mod_pettifor_M.iloc[at_nums,at_nums].values
        return np.einsum("kj,ki,ij->k",X,X,M)

    @inputRule
    def _tr_eng_m(self,X,at_nums,y=None):
        """
        Mean electronegativity of the alloy.
        """
        m = self.elemental_props.iloc[at_nums].loc[:,"Electronegativity"].values
        X = self._c_dims(X)
        return X@m

    @inputRule
    def _tr_rad_m(self,X,at_nums,y=None,**kwargs):
        """
        Mean atomic radius of the alloy.
        """
        radius = kwargs.get("radius","covalent")

        m = self.get_radii(radius,at_nums)
        X = self._c_dims(X)
        return X@m

    @inputRule
    def _tr_lel_g(self,X,at_nums,y=None,**kwargs):
        """
        Mean number of l orbital electrons.
        """
        l = kwargs.get("l","s")
        m = self.elemental_props.iloc[at_nums].loc[:,"Number {} electrons".format(l)].values
        X = self._c_dims(X)
        return X@m

    @inputRule
    def _tr_lel_std(self,X,at_nums,y=None,**kwargs):
        """
        Variance of number of l orbital electrons.
        """
        l = kwargs.get("l","s")
        m = self.elemental_props.iloc[at_nums].loc[:,"Number {} electrons".format(l)].values
        X = self._c_dims(X)
        out = (X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**2).sum(axis=1)
        return np.sqrt(out)

    def get_pp_radii(self,quantum_num,at_nums,rtn_n=False):
        n_map = {"s":1,"p":2,"d":3,"f":4}
        highest_n = {"s":7,"p":6,"d":6,"f":5} # highest occupied n-level
        if len(quantum_num)>1:
            # assume function has been supplied 1s, 2s, 2p, etc.
            m = self.elemental_props.iloc[at_nums].loc[:,"{} pp radii".format(quantum_num)].values
        else:
            # only handles case where l has been supplied.
            # Takes value for corresponding l and highest n
            cols = ["{}{} pp radii".format(n,quantum_num) 
                for n in range(n_map.get(quantum_num),highest_n.get(quantum_num)+1)]
            all_vals = self.elemental_props.iloc[at_nums].loc[:,cols].values
            locs = np.array(list(
                zip(np.arange(len(at_nums)),
                    len(cols) - np.argmax((all_vals > 0.)[:,::-1],axis=1) - 1
                )
            ))
            m = all_vals[tuple(locs.T)
            ]
        if rtn_n:
            return (locs[:,1] + n_map.get(quantum_num)) * ~np.all(all_vals==0.,1)
        else:
            return m


    @inputRule
    def _tr_lpp_m(self,X,at_nums,y=None,**kwargs):
        """
        Mean radius for the nl-level pseduopotentials in this alloy.
        """
        nl = kwargs.get("nl","1s")
        m = self.get_pp_radii(nl,at_nums)
        non_zero = 1.*(m > 0.)
        X = self._c_dims(X)
        # mean radii 
        with np.errstate(divide="ignore",invalid="ignore"):
            r = (X@m)/(X@non_zero)
            r[r == np.inf] = 0.
            r = np.nan_to_num(r)
        return r

    @inputRule
    def _tr_lel_bins(self,X,at_nums,y=None,**kwargs):
        """
        Bin the at. % into (user-specified)  l-orbital electron bins. 
        """
        bin_ledges = copy(kwargs.get("bin_ledges",[2.0]))
        l = kwargs.get("l","s")
        n_redges = len(bin_ledges)
        bin_ledges += [np.inf]
        X = self._c_dims(X)
        m = self.elemental_props.iloc[at_nums].loc[:,"Number {} electrons".format(l)].values
        mat = np.zeros((n_redges,len(m)))
        for i,(ledge,redge) in enumerate(zip(bin_ledges[:-1],bin_ledges[1:])):
            mat[i] = (m >= ledge) * (m < redge)
        return X@mat.T

    @inputRule
    def _tr_lpp_bins(self,X,at_nums,y=None,**kwargs):
        """
        Bin the at. % into (user-specified) nl-level psudopot radius bins.
        """
        bin_ledges = copy(kwargs.get("bin_ledges",[0.0,1.0,2.0]))
        nl = kwargs.get("nl","1s")
        n_redges = len(bin_ledges)
        bin_ledges += [np.inf]
        X = self._c_dims(X)
        m = self.get_pp_radii(nl,at_nums)
        mat = np.zeros((n_redges,len(m)))
        for i,(ledge,redge) in enumerate(zip(bin_ledges[:-1],bin_ledges[1:])):
            mat[i] = (m > ledge) * (m <= redge)
        return X@mat.T

    @inputOutputRule
    def _tr_lel_exclb(self,X,at_nums,y=None,**kwargs):
        """
        Bin the at. % into (user-specified) l orbital electron bins. Excludes the 
        element corresponding to this entry. 
        """
        bin_ledges = copy(kwargs.get("bin_ledges",[2.0]))
        l = kwargs.get("l","s")
        n_redges = len(bin_ledges)
        bin_ledges += [np.inf]
        m = self.elemental_props.iloc[at_nums].loc[:,"Number {} electrons".format(l)].values
        mat = np.zeros((n_redges,len(m)))
        for i,(ledge,redge) in enumerate(zip(bin_ledges[:-1],bin_ledges[1:])):
            mat[i] = (m >= ledge) * (m < redge)
        # Extend X to the correct shape, set to zero for corresponding elements
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        N,n = X.shape
        X_ext = np.repeat(X,n,axis=0)
        d1_ind,d2_ind = np.diag_indices(n)
        d2_ind = d2_ind.repeat(N)
        d1_ind = ((n*np.arange(N).reshape(-1,1)).repeat(n,axis=1)+d1_ind).T.flatten()
        X_ext[(d1_ind,d2_ind)] = 0.
        X_ext = X_ext[mask]
        return X_ext@mat.T    

    # Higher order (skewness,kurtosis) based transformations    
    @inputRule
    def _tr_rad_cmn(self,X,at_nums,y=None,**kwargs):
        """
        Atomic size skewness of the alloy.
        """
        # Extra parameter
        radius = kwargs.get("radius","covalent")
        moment = kwargs.get("moment",3)

        m = self.get_radii(radius,at_nums)
        X = self._c_dims(X)
        cmn = (X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**moment).sum(axis=1)
        std = np.sqrt((X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**2).sum(axis=1))
        return cmn/std**moment
    
    @inputRule
    def _tr_eng_cmn(self,X,at_nums,y=None,**kwargs):
        """
        Electronegativity skewness for the alloy. 
        """
        moment = kwargs.get("moment",3)

        m = self.elemental_props.iloc[at_nums].loc[:,"Electronegativity"].values
        X = self._c_dims(X)
        cmn = (X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**moment).sum(axis=1)
        std = np.sqrt((X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**2).sum(axis=1))
        return cmn/std**moment

    @inputRule
    def _tr_lel_cmn(self,X,at_nums,y=None,**kwargs):
        """
        Variance of number of l orbital electrons.
        """
        l = kwargs.get("l","s")
        moment = kwargs.get("moment",3)
        m = self.elemental_props.iloc[at_nums].loc[:,"Number {} electrons".format(l)].values
        X = self._c_dims(X)
        cmn = (X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**moment).sum(axis=1)
        std = np.sqrt((X*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T**2).sum(axis=1))
        return cmn/std**moment

    # atomic species based transformations.
    @inputOutputRule
    def _tr_amu_atom(self,X,at_nums,y=None,**kwargs):
        """
        Atomic mass of each species
        """
        N,n = X.shape
        m = self.elemental_props.iloc[at_nums].loc[:,"Atomic mass"].values
        mask = self._gen_mask(X)
        out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_mps_atom(self,X,at_nums,y=None,**kwargs):
        """
        Modified pettifor scale number for each species. 
        """
        N,n = X.shape
        m = self.elemental_props.iloc[at_nums].loc[:,"Modified Pettifor"].values
        mask = self._gen_mask(X)
        out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_atn_atom(self,X,at_nums,y=None,**kwargs):
        """
        Atomic number for each species. 
        """
        N,n = X.shape
        mask = self._gen_mask(X)
        out = np.repeat(at_nums.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]
    
    @inputOutputRule
    def _tr_rad_atom(self,X,at_nums,y=None,**kwargs):
        """
        Radius of this atomic species
        """
        radius = kwargs.get("radius","covalent")
        N,n = X.shape
        m = self.get_radii(radius,at_nums)
        mask = self._gen_mask(X)
        out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_eng_atom(self,X,at_nums,y=None,**kwargs):
        """
        Electronegativity of this atomic species
        """
        N,n = X.shape
        m = self.elemental_props.iloc[at_nums].loc[:,"Electronegativity"].values
        mask = self._gen_mask(X)
        out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_lsn_atom(self,X,at_nums,y=None,**kwargs):
        """
        Highest quantum number n associated with a given ang. mom. quantum number l
        """
        l = kwargs.get("l","s")
        m = self.get_pp_radii(l,at_nums,True)
        N,n = X.shape
        mask = self._gen_mask(X)
        out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]


    @inputOutputRule
    def _tr_lel_atom(self,X,at_nums,y=None,**kwargs):
        """
        # l orbital electrons for each species. 
        """
        l = kwargs.get("l","s")
        N,n = X.shape
        m = self.elemental_props.iloc[at_nums].loc[:,"Number {} electrons".format(l)].values
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_lpp_atom(self,X,at_nums,y=None,**kwargs):
        """
        # nl-level pseudopot radius for each species. 
        """
        nl = kwargs.get("nl","1s")
        N,n = X.shape
        m = self.get_pp_radii(nl,at_nums)
        mask = self._gen_mask(X)
        out = np.repeat(m.reshape(-1,1),N,axis=1).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_lex_atom(self,X,at_nums,y=None,**kwargs):
        """
        Excess l orbital electrons for each element vs. alloy mean.
        """
        l = kwargs.get("l","s")
        m = self.elemental_props.iloc[at_nums].loc[:,"Number {} electrons".format(l)].values
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        out = (np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-X@m).T.flatten()
        return out[mask]

    @inputOutputRule
    def _tr_lpp_dif(self,X,at_nums,y=None,**kwargs):
        """
        Misfit for this atom's nl-level pseudopot radius vs. mean.
        """
        nl = kwargs.get("nl","1s")
        m = self.get_pp_radii(nl,at_nums)
        non_zero = 1.*(m > 0.)
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        # mean radii 
        with np.errstate(divide="ignore",invalid="ignore"):
            r = (X@m)/(X@non_zero)
            r[r == np.inf] = 0.
            r = np.nan_to_num(r)
        out = (non_zero*\
            (np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-\
                r).T).flatten()
        return out[mask]

    @inputRule
    def _tr_lpp_mis(self,X,at_nums,y=None,**kwargs):
        """
        Mean squared nl-level pseudopotential radius misfit of the alloy.
        """
        nl = kwargs.get("nl","1s")
        m = self.get_pp_radii(nl,at_nums)
        non_zero = 1.*(m > 0.)
        X = self._c_dims(X)
        # mean radii 
        with np.errstate(divide="ignore",invalid="ignore"):
            r = (X@m)/(X@non_zero)
            r[r == np.inf] = 0.
            r = np.nan_to_num(r)
        out = (X*non_zero*(np.repeat(m.reshape(-1,1),X.shape[0],axis=1)-\
                r).T**2).sum(axis=1)
        return out

    @inputOutputRule
    def _tr_cmp_atom(self,X,at_nums,y=None,**kwargs):
        """
        Compare rdf for each element to the averaged alloy rdfs. 
        """
        # Extra parameter
        crystal = kwargs.get("crystal","fcc")
        return np.linalg.norm(
            self._tr_elm_struc(X,at_nums,y)-self._tr_aly_struc(X,at_nums,y,crystal=crystal),
            axis=1)

    # Structure-based transformations
    @inputOutputRule
    def _tr_elm_struc(self,X,at_nums,y=None):
        """
        Find rdf for each element in the alloys.
        """
        n_els = len(at_nums)
        mask = self._gen_mask(X)
        X = self._c_dims(X)
        N,n = X.shape
        out = np.zeros((N*n,self.rdf_0.r_coarse.shape[0]))     
        for a,el in enumerate(np.array(data.chemical_symbols)[at_nums]):
            out[a::n_els,:] = self.rdf_0(el,at_nums)
        return out[mask]

    @inputRule
    def _tr_aly_struc(self,X,at_nums,y=None,**kwargs):
        """
        Find an averaged rdf for each alloy. 
        """
        # Extra parameter
        crystal = kwargs.get("crystal","fcc")
        X = self._c_dims(X)
        return self.rdf_0(X,at_nums,crystal=crystal)

    @inputRule
    def _tr_lambda(self,X,at_nums,y=None,pd_input=True,apply2comp=True):
        """
        Add a lambda function-type transformation. 
        Uses functions from a list attribute of the class. 
        """
        if pd_input:
            X = pd.DataFrame(data=X,
                            columns=[data.chemical_symbols[at_n] for at_n in at_nums],
                            index=np.arange(X.shape[0]))
        if apply2comp:
            X = self._c_dims(X)
        else:
            X = self._a_dims(X)
        out = self.lambda_methods[0](X)
        if len(out.shape) > 1:
            out = out.reshape(-1)
        return out

    # auxiliary transformation rules
    @inputRule
    def _tr_ht_aux(self,X,at_nums,y=None,**kwargs):
        """
        Transforms the heat treatments into a format with correct exchange properties.
        Assumes a temperature,time, temperature,time, ... format
        """
        use_aux_dims = kwargs.get("use_aux_dims","all")
        in_celsius = kwargs.get("in_celsius",True)
        num_ht = (self.aux_dims-int(self.multi_output))//2 if use_aux_dims=="all" else len(use_aux_dims)//2
        X = self._a_dims(X,use_aux_dims)
        T = X[:,::2]
        if in_celsius:
            T += 273.
        t = X[:,1::2]
        rep0 = (T*t).sum(axis=1,keepdims=True)
        out = np.c_[T,rep0]
        for i in range(1,num_ht):
            repi = rep0 + ((T[:,i-1]-T[:,i])*t[:,i]).reshape(-1,1)
            out = np.c_[out,repi]
        return out

    @inputRule
    def _tr_1hot(self,X,at_nums,y=None,**kwargs):
        """
        Only for use with multi-output models. Transforms the multi-output categoricals features to a one-hot
        represented feature.
        """
        N,_ = X.shape
        m = self.output_cats
        X_cat = X[:,0].astype(int) # First column will always be used for categoricals
        out = np.zeros((N,m))
        out[np.arange(N),X_cat] = 1.
        return out

    @inputRule
    def _tr_id(self,X,at_nums,y=None,**kwargs):
        """
        Identity transformation. Only really meant for use with auxiliary features. 
        """
        use_aux_dims = kwargs.get("use_aux_dims","all")
        use_comp = kwargs.get("composition",True)
        if use_aux_dims is not None:
            X_out = self._a_dims(X,use_aux_dims)
            if use_comp:
                X_out = np.c_[X_out,self._c_dims(X)]
        else:
            X_out = self._c_dims(X)
        return X_out

    @inputRule
    def _tr_lmp_aux(self,X,at_nums,y=None,**kwargs):
        """
        Transforms test conditions (temp., stress) into more LMP-friendly features. 
        """
        use_aux_dims = kwargs.get("use_aux_dims","all")
        in_celsius = kwargs.get("in_celsius",True)
        X = self._a_dims(X,use_aux_dims)
        T = X[:,0:1]
        if in_celsius:
            T += 273.
        s = X[:,1:2]
        return np.c_[T**-1,s]


    class RDF():
        # Pre-calc Thomas-Fermi wavevector for shielding
        k_per_n = np.sqrt(4*(3/np.pi)**(1/3)/0.5291772)

        """
        Generate the representation of the crystal structure
        of a given alloy.
        """
        def __init__(self,outer,
                    dr=0.5,cutoff=6.0,
                    smear_factor=0.1,
                    use_shielding=True,
                    use_valence="non-core",
                    coarse=True):
            self.outer = outer
            self.dr = dr
            self.rc = cutoff
            self.coarse = coarse
            self.outsize=1+int((self.rc-1.)//self.dr)
            self.r_coarse = np.linspace(1.,self.rc,self.outsize)
            self.r = np.linspace(1.,self.rc,self.outsize*100) if self.coarse else self.r_coarse.copy()
            self.smear_f = smear_factor
            self.shield = use_shielding
            self.use_v = use_valence
            # Pre-calc a few common shell numbers/radii
            fcc = bulk("X","fcc",1.)
            R_fcc,N_fcc = shell_radii(fcc,self.rc/2.)
            bcc = bulk("X","bcc",1.)
            R_bcc,N_bcc = shell_radii(bcc,self.rc/2.)
            hcp = bulk("X","hcp",1.,1.633)
            R_hcp,N_hcp = shell_radii(hcp,self.rc/1.5)
            self.R_fcc = R_fcc ; self.N_fcc = N_fcc
            self.R_bcc = R_bcc ; self.N_bcc = N_bcc
            self.R_hcp = R_hcp ; self.N_hcp = N_hcp
            # Densities
            self.n_fcc = fcc.get_global_number_of_atoms()/fcc.get_volume()
            self.n_bcc = bcc.get_global_number_of_atoms()/bcc.get_volume()
            self.n_hcp = hcp.get_global_number_of_atoms()/hcp.get_volume()

        def __call__(self,X,at_nums,crystal="fcc"):
            """
            Calculate the representation of an alloy X.

            Parameters
            ----------
            X (ndarray) :   ndarray representing the alloy. 
                            Alternatively use s str representing an element.
            crystal (str):  Type of crystal.
            """
            X_vec = 1.*(np.array(at_nums)==data.atomic_numbers.get(X)) if isinstance(X,str) else X.copy()
            X_vec = np.atleast_2d(X_vec)
            mean_v = X_vec@self.outer.va[at_nums] if self.use_v else 1.
            mean_r = X_vec@self.outer.cr[at_nums]
            # Check what crystal we're using.
            if isinstance(X,str):
                struc = bulk(X)
                R,N = shell_radii(struc,self.rc)
                R = R.reshape(1,-1) ; N = N.reshape(1,-1)
                n = mean_v*struc.get_global_number_of_atoms()/struc.get_volume()
            elif crystal=="fcc":
                mean_a = mean_r*4/np.sqrt(2)
                R = self.R_fcc.reshape(1,-1)*mean_a.reshape(-1,1)
                N = np.repeat(self.N_fcc.reshape(-1,1),mean_a.shape[0],axis=1).T
                n = mean_v*self.n_fcc/mean_a**3
            elif crystal=="hcp":
                mean_a = mean_r*2
                R = self.R_hcp.reshape(1,-1)*mean_a.reshape(-1,1)
                N = np.repeat(self.N_hcp.reshape(-1,1),mean_a.shape[0],axis=1).T
                n = mean_v*self.n_hcp/mean_a**3
            elif crystal=="bcc":
                mean_a = mean_r*4/np.sqrt(3)
                R = self.R_bcc.reshape(1,-1)*mean_a.reshape(-1,1)
                N = np.repeat(self.N_bcc.reshape(-1,1),mean_a.shape[0],axis=1).T
                n = mean_v*self.n_bcc/mean_a**3
            else:
                # Needs to be modified to work properly
                mean_a = 2.*mean_r # <- Probably wrong
                struc = bulk("X",crystal,mean_a)
                R,N = shell_radii(struc,self.rc)
                # density of electrons
                n = mean_v*struc.get_global_number_of_atoms()/struc.get_volume()
            # Find Thomas-Fermi wavenumber for shielding
            k_s = self.k_per_n*n**(1/6) if self.shield else np.zeros_like(mean_v)
            # Calculation of rep starts here.
            r = self.r
            rdf = np.einsum("ij,ijk->ijk",
                    (4*np.pi*R)**-2*N*mean_v.reshape(-1,1)\
                        /(self.smear_f*mean_r.reshape(-1,1)*np.sqrt(2*np.pi)),
                            np.exp(-0.5*((np.atleast_2d(r)-np.atleast_3d(R))\
                                /(self.smear_f*mean_r).reshape(-1,1,1))**2)).sum(axis=1)\
                                    * np.exp(-k_s.reshape(-1,1)*r.reshape(1,-1))
            if self.coarse:
                rdf = rdf.reshape(rdf.shape[0],-1,100).mean(axis=2)
            return rdf

def str2method_kwargs(ft_string):
    """
    Parse an input string representing a HR transformation function.
    """
    parts = ft_string.split("|")
    method = "_tr_"+parts[0]
    kwgs_l = parts[1:]
    kwargs = {}
    for k in kwgs_l:
        kw,v = k.split("=")
        kwargs[kw] = ast.literal_eval(v)
    return method,kwargs


class HRrep(HRrep_parent):
    """
    This class carries out the transformation into the Hume-Rothery basis.
    It has been written to work with sklearn.pipeline.Pipeline.
    Feature names can be suffixed with kwargs by using commas "|" e.g. the 
    string "mis_mu|method=enthalpy" will add the mis_mu method as a transform 
    with the kwarg method=enthalpy always being passed to it. 

    The currently implemented features are:
        mix_mu      Chem. pot.-like term for free energy of mixing. 
        mis_mu      Chem. pot.-like term for atomic size misfit energy.
        val_mu      Chem. pot.-like term for electronic free energy (=valence).
        eng_mu      Chem. pot.-like term for enthalpy of mixing (uses electroneg).
        mix_g       Free energy of mixing.
        mis_g       Atomic size misfit energy.
        val_g       Electronic free energy (= mean valence).
        eng_g       Enthalpy of mixing using electronegativites.
        sel_atom    # s electrons.
        pel_atom    # p electrons.
        del_atom    # d electrons.
        fel_atom    # f electrons.
        cmp_atom    2-norm similairty of atom-species RDF to (averaged) alloy RDF.
        elm_struc   Atom-species RDF (ndarray).
        aly_struc   Averaged RDF for the alloy (ndarray).
    """
    def __init__(self,*features,
                        Xy_share_labels=False,
                        aux_dims=None,
                        multi_output=False,
                        rdf_rep_params={}):
        super().__init__(*features,
                            Xy_share_labels=Xy_share_labels,
                            aux_dims=aux_dims,
                            multi_output=multi_output,
                            rdf_rep_params=rdf_rep_params)
        self._get_feature_methods()

    def add_feature(self,new_features):
        """
        Add new features to the class. Added to END of feature list. 
        
        Parameters
        ----------
        new_features (list, str)    : A list of new features to add. See __init__ for details.
        """
        lambda_methods = []
        ft_methods = []
        ft_kwargs = []
        ft_list = []
        groups = []
        n_found = 0
        m_found = 0
        all_methods = dir(HRrep_parent)
        for ft in new_features:
            if isinstance(ft,str):
                gp_start = copy(m_found)
                method,kwargs = str2method_kwargs(ft)
                if method in all_methods:
                    n_found += 1
                    this_method = getattr(HRrep,method)
                    ft_methods += [this_method]
                    ft_kwargs += [kwargs]
                    # Calling functions with dryrun kwarg returns output shape
                    m = this_method(self,dryrun=True,**kwargs)
                    m_found += m
                    ft_list += ["[{}]__{}".format(ft,i) for i in range(m)]
                gp_end = copy(m_found)
            # Also valid to supply feature names pre-grouped using tuples/lists
            elif hasattr(ft,"__len__"):
                ft_group = copy(ft)
                gp_start = copy(m_found)
                for ft in ft_group:
                    method,kwargs = str2method_kwargs(ft)
                    if method in all_methods:
                        n_found += 1
                        this_method = getattr(HRrep,method)
                        ft_methods += [this_method]
                        ft_kwargs += [kwargs]
                        # Calling functions with dryrun kwarg returns output shape
                        m = this_method(self,dryrun=True,**kwargs)
                        m_found += m
                        ft_list += ["[{}]__{}".format(ft,i) for i in range(m)]
                gp_end = copy(m_found)
            elif callable(ft):
                gp_start = copy(m_found)
                method = "_tr_lambda"
                n_found += 1
                this_method = getattr(HRrep,method)
                ft_methods += [this_method]
                ft_kwargs += [{}] # Can't use kwargs with lambda method (for now)
                lambda_methods += [ft] # Only calls first one for now.
                m = this_method(self,dryrun=True)
                m_found += m
                ft_list += ["[{}]__{}".format("lambda",i) for i in range(m)]
                gp_end = copy(m_found)
            else:
                warnings.warn("Method {} not found.".format(ft),UserWarning)
                continue
            if gp_start!=gp_end: # Have found at least one valid feature
                groups += [[gp_start,gp_end]]
        self.nft_out += n_found
        self.dim_out += m_found
        self.features += ft_methods
        self.ft_kwargs += ft_kwargs
        self.ft_list += ft_list
        self.groups += groups
        self.lambda_methods += lambda_methods

    def _get_feature_methods(self):
        self.nft_out = 0
        self.dim_out = 0
        self.features = []
        self.ft_kwargs = []
        self.ft_list = []
        self.groups = []
        self.lambda_methods = []
        self.add_feature(self.ft_names)

    def transform(self,X,y=None):
        """
        Transform data according to selected rules. 
        """
        at_nums = self.get_els(X)
        X = X.values
        y = y if y is None else y.values
        if self.Xy_share_labels:
            # Start the output array
            mask = self._gen_mask(X)
            X_out = np.zeros((mask.sum(),0))
            for i,(ft,kwargs) in enumerate(zip(self.features,self.ft_kwargs)):
                X_out = np.c_[X_out,ft(self,X,at_nums,y,**kwargs)]
            if y is not None:
                _,y_out = self.extend_original(X,y)
            else:
                y_out = None
        elif self.multi_output:
            # Start the output array
            if y is not None:
                _,y_out = self.multi_2_single_output(X,y)
                X_out = np.zeros((y_out.shape[0],0))
            else:
                X_out = np.zeros((X.shape[0]*self.output_cats,0))
                y_out = None
            for i,(ft,kwargs) in enumerate(zip(self.features,self.ft_kwargs)):
                X_out = np.c_[X_out,ft(self,X,at_nums,y,**kwargs)]
        else:
            # Start the output array
            X_out = np.zeros((X.shape[0],0))
            for i,(ft,kwargs) in enumerate(zip(self.features,self.ft_kwargs)):
                X_out = np.c_[X_out,ft(self,X,at_nums,y,**kwargs)]
            if y is not None:
                y_out = y.reshape(-1,1)
            else:
                y_out = None
        return X_out,y_out
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X,y)

class Logy():
    """log transformation of predicted property."""
    def __init__(self,trans_distribution=False):
        self.trans_dist = trans_distribution

    def fit(self,k):
        return self

    @staticmethod
    def transform(k):
        return np.log(k)

    @staticmethod
    def inverse_transform(p,std=None):
        return np.exp(p)

    @staticmethod
    def inverse_std(p,p_std):
        p_std = p_std.reshape(p.shape)
        return p_std*np.exp(p)

    @staticmethod
    def inverse_cov(p,p_cov):
        p = p.reshape(-1,1)
        return (np.exp(p)@np.exp(p).T)*p_cov

    @staticmethod
    def transform_cov(k,k_cov):
        k = k.reshape(-1,1)
        return k_cov/(k@k.T)

class ArcTanh():
    """arctanh transformation of predicted property."""
    def __init__(self,trans_distribution=False):
        self.trans_dist = trans_distribution

    def fit(self,f):
        return self

    @staticmethod
    def transform(f):
        return np.arctanh(2*f-1.)

    @staticmethod
    def inverse_transform(q,q_std=None):
        return 0.5*(1. + np.tanh(q))

    @staticmethod
    def inverse_std(q,q_std):
        q_std = q_std.reshape(q.shape)
        return 0.5*q_std*np.cosh(q)**-2

    @staticmethod
    def inverse_cov(q,q_cov):
        q = q.reshape(-1,1)
        return 0.25*((np.cosh(q)**-2)@(np.cosh(q)**-2).T)*q_cov
    
    @staticmethod
    def transform_cov(f,f_cov):
        f = f.reshape(-1,1)
        return 0.25*f_cov/((f*(1-f))@((f*(1-f)).T))

class SplitByGeq():
    """
    Split X,y data into two datasets, depending on value of y. 
    """
    def __init__(self,split=1.0,max_n_per_split=True):
        """
        Parameters
        ----------
        split           (float) : Value which is used to split dataset by y. 
        max_n_per_split (bool)  : Whether to add the values==split data pts into each split.
        """
        self.num_splits = 2
        self.split_ = split
        self.max_ = max_n_per_split


    def split(self,X,y):
        y = y.reshape(-1)
        inds_u = (y >= self.split_)
        inds_l = (y <= self.split_) if self.max_ else ~inds_u        
        return (X[inds_l],y[inds_l]),(X[inds_u],y[inds_u])

class Bagging():
    """
    Implements bagging via a random split of the features. 
    """
    def __init__(self,num_splits=24,max_features=0.8,min_features=3,seed=1921):
        self.num_splits = num_splits
        self.max_features = max_features
        self.min_features = min_features
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.split_inds = []

    def reseed(self,seed=None):
        new_seed = self.seed+1 if seed is None else seed
        self.rng = np.random.default_rng(seed)

    def _sample(self,a,n,X,cov):
        inds = np.arange(X.shape[1])
        used = [a]
        # Use covariance to weight choice of next feature
        for i in range(n):
            wts = np.abs(cov[used].sum(0))**-1
            wts[used] = 0.
            wts /= wts.sum()
            nxt = self.rng.choice(inds,p=wts)
            used += [nxt]
        return used

    def fit(self,X,y):
        N,n_fts = X.shape
        max_fts = int(np.around(self.max_features*n_fts,0)) if isinstance(self.max_features,float) else self.max_features
        min_fts = int(np.around(self.min_features*n_fts,0)) if isinstance(self.min_features,float) else self.min_features
        # Estimate covariance in order to choose least correlated features
        cov = np.cov(X,rowvar=False)
        cov /= np.sqrt(np.diag(cov).reshape(-1,1)@np.diag(cov).reshape(1,-1))
        starts = self.rng.integers(0,X.shape[1],self.num_splits)
        lens = self.rng.integers(min_fts,max_fts,self.num_splits)
        for a,n in zip(starts,lens):
            self.split_inds += [self._sample(a,n,X,cov)]

    def transform(self,X,y):
        splits = ()
        for inds in self.split_inds:
            splits += ((X[:,inds],y),)
        return splits

class ModelClass():
    """
    A simple model class, something like sklearn.pipeline.Pipeline,
    but with explicit pipeline steps. 

    Parameters
    ----------
    phys_transformer:   Should be something with all the methods of HRrep or None.
    splitter        :   Should have a fit method. Also a transform method, taking X,y as inputs, returning ((X,y),...). Also needs a .num_splits attribute
    y_transformer   :   Simple transformer, needs inverse_transform method.
    y_scaler        :   Simple scaler, needs inverse_transform method.
    X_scaler        :   Like sklearn.preprocessing.StandardScaler
    X_transformer   :   Simple transformer (needs fit,transform methods).
    regressor       :   Any sklearn regression model class should work.
    committee_method:   Either "weighted" or "best". Only used when there is a y_splitter
    """
    def __init__(self,phys_transformer=None,
                    splitter = None,
                    y_transformer=None,
                    y_scaler=None,
                    X_scaler=None,
                    X_transformer=None,
                    regressor=None,
                    n_jobs=-1,
                    bagging_method = "weighted"):
        self.phys_transformer=deepcopy(phys_transformer)
        self.y_transformer=deepcopy(y_transformer)
        # y_splitter stuff
        self.bagging_method = bagging_method
        self.splitter = splitter
        if self.splitter is not None:
            self.num_splits = self.splitter.num_splits
        else:
            self.num_splits = 1
        # Add dicts for all other methods
        self.y_scaler = deepcopy(y_scaler)
        self.X_scaler = deepcopy(X_scaler)
        self.X_transformer = deepcopy(X_transformer)
        self.regressor = {}
        if regressor is not None:
            for a in range(self.num_splits):
                self.regressor[a] = deepcopy(regressor)
        self.n_jobs = n_jobs
        # Flags
        self.fitted = False

    def _fit_by_split(self,X,y,a):
        if self.regressor[a] is not None:
            self.regressor[a].fit(X,y)

    def fit(self,X,y):
        """
        Fit model.
        """
        X = X.copy() ; y = y.copy()
        if self.phys_transformer is not None:
            X,y = self.phys_transformer.fit(X,y).transform(X,y)
        if self.y_transformer is not None:
            y = self.y_transformer.fit(y).transform(y)
        if self.y_scaler is not None:
                y = y.reshape(-1,1)
                y = self.y_scaler.fit(y).transform(y)
        if self.X_scaler is not None:
            X = self.X_scaler.fit(X).transform(X)
        if self.X_transformer is not None:
            X = self.X_transformer.fit(X).transform(X)
        if self.splitter is not None:
            self.splitter.fit(X,y)
            splits = self.splitter.transform(X,y)
        else:
            splits = ((X,y),)
        if self.n_jobs != 1:
            Parallel(n_jobs=self.n_jobs,backend="threading")(delayed(self._fit_by_split)(X,y,a) 
                    for a,(X,y) in enumerate(splits))
        else:
            for a,(X,y) in enumerate(splits):
                self._fit_by_split(X,y,a)
        self.fitted = True
        return self
    
    def _predict_by_split(self,X,a,complete_transform=True):
        if not self.fitted:
            raise RuntimeError("This ModelClass object has not been fitted.")
        X = X.copy()
        if self.regressor[a] is not None:
            y_prd_0,cov = self.regressor[a].predict(X,return_std=False,
                                                    return_cov=True)
            y_unc_0 = np.sqrt(np.diag(cov))
        if self.y_scaler is not None:
            y_prd_0 = self.y_scaler.inverse_transform(y_prd_0)
            y_prd_0 = y_prd_0.reshape(-1)
            # Uncertainty
            scaler = deepcopy(self.y_scaler)
            scaler.set_params(with_mean=False)
            y_unc_0 = scaler.inverse_transform(y_unc_0)
            y_unc_0 = y_unc_0.reshape(-1)
            # Covariance
            cov *= scaler.scale_**2
        cov_inv = inv(cov)
        # Can skip transformation step.
        if not complete_transform:
            return y_prd_0.reshape(-1),y_unc_0.reshape(-1),cov,cov_inv
        if self.y_transformer is not None:
            y_prd_1 = self.y_transformer.inverse_transform(y_prd_0,y_unc_0)
            y_unc_1 = self.y_transformer.inverse_std(y_prd_0,y_unc_0)
            y_prd_0 = y_prd_1.copy() ; y_unc_0 = y_unc_1.copy()
            return y_prd_0.reshape(-1),y_unc_0.reshape(-1)

    def predict(self,X,return_std=False,return_cov=False):
        """
        Make predictions using fitted model. 
        """
        if not self.fitted:
            raise RuntimeError("This ModelClass object has not been fitted.")
        X_orig = X.copy()
        X = X.copy()
        # Pre-regressor transformations
        if self.phys_transformer is not None:
            X,_ = self.phys_transformer.transform(X)
        if self.X_scaler is not None:
            X = self.X_scaler.transform(X)
        if self.X_transformer is not None:
            X = self.X_transformer.transform(X)
        if self.splitter is not None:
            splits = self.splitter.transform(X,None)
        else:
            splits = ((X,None),)
        if self.n_jobs != 1:
            p_out = Parallel(n_jobs=self.n_jobs,backend="threading")(delayed(
                self._predict_by_split)(X_a,a,complete_transform=False)
                for a,(X_a,_) in enumerate(splits)
            )
        else:
            p_out = []
            for a,(X_a,_) in enumerate(splits):
                p_out += [self._predict_by_split(X_a,a,complete_transform=False)]

        # recombine
        if self.splitter is not None:
            y_prd = np.zeros(X.shape[0])
            cov_inv = np.zeros((X.shape[0],X.shape[0]))
            cov = cov_inv.copy()
            if self.bagging_method == "weighted":
                for model_result in p_out:
                    y_prd_a,y_unc_a,cov_a,cov_inv_a = model_result
                    cov_inv += cov_inv_a
                    y_prd += cov_inv_a@y_prd_a
                cov = inv(cov_inv)
                y_prd = cov@y_prd
                y_unc = np.diag(cov)
            elif self.bagging_method == "average":
                for model_result in p_out:
                    y_prd_a,y_unc_a,cov_a,cov_inv_a = model_result
                    y_prd += y_prd_a
                    cov += cov_a
                y_prd /= self.num_splits
                cov /= self.num_splits
                y_unc = np.diag(cov)
        else:
            y_prd,y_unc,cov,cov_inv = p_out[0]
        # Finally use y_transformer
        if self.y_transformer is not None:
            y_prd_1 = self.y_transformer.inverse_transform(y_prd,y_unc)
            y_unc_1 = self.y_transformer.inverse_std(y_prd,y_unc)
            y_prd = y_prd_1.copy() ; y_unc = y_unc_1.copy() 
            cov_1 = self.y_transformer.inverse_cov(y_prd,cov)
            cov = cov_1.copy()
        # self.phys_transformer appears twice because y needs to be reshaped.
        # cov matrix not reshaped. 
        if self.phys_transformer is not None:
            y_out = self.phys_transformer.revert_shape(X_orig,y_prd)
            u_out = self.phys_transformer.revert_shape(X_orig,y_unc)
        else: 
            y_out = None
            u_out = None
        # Return
        if self.phys_transformer is not None and self.phys_transformer.Xy_share_labels:
            if return_std:
                if return_cov:
                    return y_prd,y_out,y_unc,u_out,cov
                else:
                    return y_prd,y_out,y_unc,u_out
            else:
                if return_cov:
                    return y_prd,y_out,cov
                else:
                    return y_prd,y_out
        else:
            if return_std:
                if return_cov:
                    return y_prd,y_unc,cov
                else:
                    return y_prd,y_unc
            else:
                if return_cov:
                    return y_prd,cov
                else:
                    return y_prd

    def score(self,X,y_true,method=r2_score,bcl_ignore_0s=True):
        """
        Score the model. 

        Parameters
        ----------
        X (pd.Dataframe)        :   Data descriptors
        y_true (pd.Dataframe)   :   Data labels. 
        method                  :   Method of scoring data. 
        bcl_ignore_0s (bool)    :   When data shares common labels, whether or not to ignore 0s for purposes of calculating R^2. 
        """
        
        if self.phys_transformer is not None and self.phys_transformer.Xy_share_labels:
            y_prd_0,y_out = self.predict(X)
            _,y_tru_0 = self.phys_transformer.extend_original(X.values,y_true.values)
            r2_0 = method(y_tru_0,y_prd_0)
            # Calculate R^2 for common labels. 
            r2_bcl = pd.Series(dtype=np.float64)
            X = X.loc[:,("Composition",)]
            for cl in X.columns:
                if bcl_ignore_0s:
                    x = X[cl].values.copy()
                    y_t_i = y_true[cl].values.copy() ; y_p_i = y_out[cl].values.copy() 
                    y_t_i = y_t_i[~(x==0.)]
                    y_p_i = y_p_i[~(x==0.)]
                    if y_t_i.shape[0] < 2:
                        r2 = np.nan
                    else:
                        r2 = method(y_t_i,y_p_i)
                else:
                    r2 = method(y_true[cl].values,y_out[cl].values)
                r2_bcl = r2_bcl.append(pd.Series({cl:r2}))
        else:
            y_prd_0 = self.predict(X)
            r2_0 = method(y_true,y_prd_0)
            r2_bcl = None
        return r2_0, r2_bcl

    def save(self):
        return deepcopy(self)

class GModelClass(ModelClass):
    def __init__(self,phys_transformer=None,
                    splitter = None,
                    y_transformer=None,
                    y_scaler=None,
                    X_scaler=None,
                    X_transformer=None,
                    regressor_cls=None,
                    likelihood = None,
                    mean_model = None,
                    covar_model = None,
                    optimizer_lr = 0.1,
                    max_iter = 1e5,
                    conv_tol = 1.e-4,
                    optimizer_method = "adam",
                    n_restarts = 1,
                    restart_method = "random",
                    scheduler_dr = 0.999,
                    n_jobs=1,
                    bagging_method = "weighted",
                    use_cuda = True,
                    seed=1958):
        super(GModelClass,self).__init__(phys_transformer,splitter,y_transformer,y_scaler,
                                        X_scaler,X_transformer,None,n_jobs,bagging_method)
        
        if use_cuda:
            if torch.cuda.is_available():
                self.use_cuda = True
                self.n_jobs = 1
            else:
                self.use_cuda = False
                warnings.warn("CUDA is not currently available on this device.",UserWarning)
        else:
            self.use_cuda = False
        self.regressor_cls = regressor_cls
        self.likelihood = likelihood
        self.mean_model = mean_model
        self.covar_model = covar_model
        self.max_iter = int(max_iter)
        self.optimizer_lr = optimizer_lr if hasattr(optimizer_lr,"__len__") else [optimizer_lr]
        self.scheduler_dr = scheduler_dr
        self.conv_tol = conv_tol
        self.n_restarts = n_restarts
        self.restart_method = restart_method
        self.optimizer_method = optimizer_method.lower()
        self.states = {}
        self.training_data = {}
        self.fit_history = {}

        self.rng = np.random.default_rng(seed)

    def _fit_by_split(self,X,y,a):
        if self.regressor_cls is not None:
            X = torch.from_numpy(X)
            y = torch.from_numpy(y.reshape(-1))
            likelihood = deepcopy(self.likelihood)
            if self.mean_model is not None and self.covar_model is not None:
                model = self.regressor_cls(X,y,likelihood,
                                            self.mean_model,self.covar_model)
            elif self.mean_model is not None:
                model = self.regressor_cls(X,y,likelihood,
                                            mean_module=self.mean_model)
            elif self.covar_model is not None:
                model = self.regressor_cls(X,y,likelihood,
                                            covar_module=self.covar_model)
            else:
                model = self.regressor_cls(X,y,likelihood)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
                likelihood = likelihood.cuda()
                model = model.cuda()
            # Put into training mode
            model.train()
            likelihood.train()
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            # Def optimization to run in loop.
            def optimise(lr):
                if self.optimizer_method == "adam":
                    # Use the adam optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
                elif self.optimizer_method == "lbfgs":
                    # Use the L-BFGS-B optimiser. 
                    optimizer = torch.optim.LBFGS(model.parameters(),lr=lr)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,self.scheduler_dr)
                iter_ = 0 ; old_loss = 1.e8 ; rel_change = 1.e8
                while iter_<self.max_iter and rel_change>self.conv_tol:
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = model(X)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y)
                    loss.backward()
                    if self.optimizer_method=="lbfgs":
                        # Need this for lbfgs optimizer
                        def closure():
                            optimizer.zero_grad()
                            output = model(X)
                            loss = -mll(output,y)
                            loss.backward()
                            return loss
                        optimizer.step(closure)
                    else:
                        optimizer.step()
                    scheduler.step()
                    iter_ += 1
                    # stopping criteria
                    new_loss = loss.item()
                    rel_change = abs((old_loss - new_loss)/new_loss)
                    old_loss = 1.*new_loss
                if iter_ == self.max_iter:
                    warnings.warn("Max number of iteration ({:.2e}) reached.".format(self.max_iter),UserWarning)
            # Run optimisation, restart as many times as n_restarts
            # Store these as we loop over lr, restarts
            init_r_lengths = model.state_dict().get("covar_module.base_kernel.raw_lengthscale",None).clone()
            loss_list = np.zeros((self.n_restarts,len(self.optimizer_lr)))
            r_lengths_dict = {}
            for lr_trial,lr in enumerate(self.optimizer_lr):
                model.initialize(
                        **{"covar_module.base_kernel.raw_lengthscale":init_r_lengths.clone()})
                for repeat in range(self.n_restarts):
                    optimise(lr)
                    r_lengths = model.state_dict().get("covar_module.base_kernel.raw_lengthscale",None).clone()
                    constraint = model.covar_module.base_kernel.raw_lengthscale_constraint
                    t_lengths = constraint.transform(r_lengths)
                    loss_list[repeat,lr_trial] = -mll(model(X), y) + (t_lengths < 4.e3).sum()
                    r_lengths_dict[(repeat,lr_trial)] = r_lengths
                    # if r_lengths is None:
                    #     break
                    # Transform the true lengthscales
                    if self.restart_method == "random":
                        new_r_lengths = r_lengths[:,self.rng.permutation(r_lengths.shape[1])]
                    else:
                        new_t_lengths = t_lengths**1.5
                        new_r_lengths = constraint.inverse_transform(new_t_lengths)
                    model.initialize(
                        **{"covar_module.base_kernel.raw_lengthscale":new_r_lengths.clone()})
            # Use the best lengths for final model.
            if r_lengths is not None:
                use_rep = tuple(np.argwhere(loss_list==loss_list.min())[0])
                model.initialize(
                    **{"covar_module.base_kernel.raw_lengthscale":r_lengths_dict[use_rep]})
            if self.use_cuda:
                self.regressor[a] = (deepcopy(model.cpu()),deepcopy(likelihood.cpu()))
                self.states[a] = (model.cpu().state_dict(),likelihood.cpu().state_dict())
            else:
                self.regressor[a] = (deepcopy(model),deepcopy(likelihood))
                self.states[a] = (model.state_dict(),likelihood.state_dict())
            self.training_data[a] = [X.cpu(),y.cpu()]
            # Store fit history
            self.fit_history[a] = {"learning rates":self.optimizer_lr,
                                    "losses":loss_list,
                                    "kernel_lengthscales":r_lengths_dict}

    def _predict_by_split(self,X,a,complete_transform=True):
        if not self.fitted:
            raise RuntimeError("This GModelClass object has not been fitted.")
        X = torch.from_numpy(X)
        model,likelihood = self.regressor[a]
        # Switch into predictive posterior mode
        model.eval()
        likelihood.eval()
        with torch.no_grad(),gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X))
        y_prd_0 = observed_pred.mean
        cov     = observed_pred.covariance_matrix
        # Convert to numpy now
        y_prd_0 = y_prd_0.numpy()
        cov = cov.detach().numpy() # Unsure why this is needed. It appeared in an error code once. 
        y_unc_0 = np.sqrt(np.diag(cov))
        # ~~~~~~~~~~ Copied from parent method ~~~~~~~~~~ #
        if self.y_scaler is not None:
            y_prd_0 = self.y_scaler.inverse_transform(y_prd_0)
            y_prd_0 = y_prd_0.reshape(-1)
            # Uncertainty
            scaler = deepcopy(self.y_scaler)
            scaler.set_params(with_mean=False)
            y_unc_0 = scaler.inverse_transform(y_unc_0)
            y_unc_0 = y_unc_0.reshape(-1)
            # Covariance
            cov *= scaler.scale_**2
        cov_inv = inv(cov)
        # Can skip transformation step.
        if not complete_transform:
            return y_prd_0.reshape(-1),y_unc_0.reshape(-1),cov,cov_inv
        if self.y_transformer is not None:
            y_prd_1 = self.y_transformer.inverse_transform(y_prd_0,y_unc_0)
            y_unc_1 = self.y_transformer.inverse_std(y_prd_0,y_unc_0)
            y_prd_0 = y_prd_1.copy() ; y_unc_0 = y_unc_1.copy()
            return y_prd_0.reshape(-1),y_unc_0.reshape(-1)

    def save(self):
        """
        Return a pickle-able version of the model instance.
        """
        inter_copy = deepcopy(self)
        if self.fitted:
            del inter_copy.regressor
            inter_copy.regressor = {}
        return inter_copy
    
    def reload(self):
        """
        After loading a 'saved' model instance, recreate the 
        models from the stored state_dicts. 
        """
        for a in range(self.num_splits):
            X,y = self.training_data[a]
            likelihood = deepcopy(self.likelihood)
            if self.mean_model is not None and self.covar_model is not None:
                model = self.regressor_cls(X,y,likelihood,
                                            self.mean_model,self.covar_model)
            elif self.mean_model is not None:
                model = self.regressor_cls(X,y,likelihood,
                                            mean_module=self.mean_model)
            elif self.covar_model is not None:
                model = self.regressor_cls(X,y,likelihood,
                                            covar_module=self.covar_model)
            else:
                model = self.regressor_cls(X,y,likelihood)
            m_state,l_state = self.states[a]
            model.load_state_dict(m_state)
            likelihood.load_state_dict(l_state)
            self.regressor[a] = (model.cpu(),likelihood.cpu())


def part_coeffs(c1,c2,tol=0.001,composition=None):
    """
    Calculate partitioning coeffiicents for arbitrary phases.
    
    Parameters
    ----------
    c1 (pd.Dataframe)           :   Numerator phase composition.
    c2 (pd.Dataframe)           :   Denominator phase composition.
    tol (float)                 :   Assumed tolerance or uncertainty in experimental measurements.
    composition (pd.Dataframe)  :   Corresponding alloy compositions. 
    """
    c1_ = c1.values ; c2_ = c2.values
    pc_ = np.ma.masked_where(c1_==0.,c1_).filled(tol)\
        /np.ma.masked_where(c2_==0.,c2_).filled(tol)
    if composition is not None:
        pc = np.ma.masked_where(composition.values==0.,pc_)
    pc = c1.copy()
    pc.loc[:] = pc_
    return pc

#%% Precipitate fraction inferral
def bayesian_f(f_prior,f_uncp,X_p1,X_u1,X_p2,X_u2,X):
    """
    Bayesian inferral of new value for the precipitate fraction, as proposed in 
    microstructure paper.

    Parameters
    ----------
    f_prior (ndarray)   : Direct predictions for the fraction of 2nd phase, prior for this method.
    f_uncp (ndarray)    : Uncertainty associated with prediction above.
    X_p1 (ndarray)      : Predictions for the compositions of 1st phase
    X_u1 (ndarray)      : Uncertainties associated with above prediction. 
    X_p2 (ndarray)      : Predictions for the composiiton of 2nd phase. 
    X_u2 (ndarray)      : Uncertianties associated with above prediction. 
    X (ndarray)         : Input composition of alloys. 

    Returns
    -------
    f_infr (ndarray)    : New inferred values for fractions of 2nd phase.
    f_unci (ndarray)    : Associated uncertainties.
    """
    unc_est2 = f_prior[:,np.newaxis]**2*X_u2**2\
        +(1.-f_prior[:,np.newaxis])**2*X_u1**2
    denom = (1+np.nansum((X_p2-X_p1)**2*f_uncp**2/unc_est2,axis=1))**-1
    f_infr = (f_prior+np.nansum((X-X_p1)*(X_p2-X_p1)*f_uncp**2/unc_est2,axis=1))\
        *denom
    f_unci = f_uncp*np.sqrt(denom)
    return f_infr,f_unci

#%% Structure comparison
class CrystalRep():
    # Pre-calc Thomas-Fermi wavevector for shielding
    k_per_n = np.sqrt(4*(3/np.pi)**(1/3)/0.5291772)

    """
    Generate the representation of the crystal structure
    of a given alloy.
    """
    def __init__(self,elements,
                dr=0.01,cutoff=10.0,
                smear_factor=0.1,
                use_shielding=True,
                use_valence="non-core"):
        self.dr = dr
        self.rc = cutoff
        self.smear_f = smear_factor
        self.shield = use_shielding
        self.use_v = use_valence
        # Elements
        self.els = np.array(elements)
        self.at_ns = [data.atomic_numbers.get(el) for el in self.els]
        # Pre-calc a few common shell numbers/radii
        fcc = bulk("X","fcc",1.)
        R_fcc,N_fcc = shell_radii(fcc,self.rc/2.)
        bcc = bulk("X","bcc",1.)
        R_bcc,N_bcc = shell_radii(bcc,self.rc/2.)
        hcp = bulk("X","hcp",1.,1.633)
        R_hcp,N_hcp = shell_radii(hcp,self.rc/1.5)
        self.R_fcc = R_fcc ; self.N_fcc = N_fcc
        self.R_bcc = R_bcc ; self.N_bcc = N_bcc
        self.R_hcp = R_hcp ; self.N_hcp = N_hcp
        # Densities
        self.n_fcc = fcc.get_global_number_of_atoms()/fcc.get_volume()
        self.n_bcc = bcc.get_global_number_of_atoms()/bcc.get_volume()
        self.n_hcp = hcp.get_global_number_of_atoms()/hcp.get_volume()
        # Get "valences" and radii
        self.pt = np.genfromtxt("hr_table.csv",delimiter=",",
                            missing_values=("nan",),filling_values=(np.nan,),
                            skip_header=1,usecols=(2,3,4,5,8,9,10))
        self.r = self.pt.T[0][self.at_ns]
        if self.use_v!="non-core":
            self.v = self.pt.T[1][self.at_ns]
        else:
            self.v = self.pt.T[6][self.at_ns]  

    def __call__(self,X,crystal="fcc"):
        """
        Calculate the representation of an alloy X.

        Parameters
        ----------
        X (ndarray) :   ndarray representing the alloy. 
                        Alternatively use s str representing an element.
        crystal (str):  Type of crystal.
        """
        X_vec = 1.*(self.els==X) if isinstance(X,str) else X.copy()
        mean_v = X_vec@self.v if self.use_v else 1.
        mean_r = X_vec@self.r
        # Check what crystal we're using.
        if isinstance(X,str):
            struc = bulk(X)
            R,N = shell_radii(struc,self.rc)
            n = mean_v*struc.get_global_number_of_atoms()/struc.get_volume()
        elif crystal=="fcc":
            mean_a = mean_r*4/np.sqrt(2)
            R,N = self.R_fcc*mean_a,self.N_fcc
            n = mean_v*self.n_fcc/mean_a**3
        elif crystal=="hcp":
            mean_a = mean_r*2
            R,N = self.R_hcp*mean_a,self.N_hcp
            n = mean_v*self.n_hcp/mean_a**3
        elif crystal=="bcc":
            mean_a = mean_r*4/np.sqrt(3)
            R,N = self.R_bcc,self.N_bcc
            n = mean_v*self.n_bcc/mean_a**3
        else:
            mean_a = 2.*mean_r # <- Probably wrong
            struc = bulk("X",crystal,mean_a)
            R,N = shell_radii(struc,self.rc)
            # density of electrons
            n = mean_v*struc.get_global_number_of_atoms()/struc.get_volume()
        # Keep shells below cutoff
        N = N[R<self.rc] ; R = R[R<self.rc]
        # Find Thomas-Fermi wavenumber for shielding
        k_s = self.k_per_n*n**(1/6) if self.shield else 0.
        # Calculation of rep starts here.
        r_vals = np.arange(0.,self.rc,self.dr)
        rdf = np.vectorize(
            lambda r: ((4*np.pi*R)**-2*N*mean_v\
                /(self.smear_f*mean_r*np.sqrt(2*np.pi))\
                *np.exp(-0.5*((r-R)/(self.smear_f*mean_r))**2)).sum()\
                * np.exp(-k_s*r)
        )
        return r_vals,rdf(r_vals)

# %% Microstructure predictions - exact & bayesian corrections
def exact_corr(x,k,f,cov,lambda_):
    """
    Apply an exact correction to the microstructure. For a single alloy.

    Parameters
    ----------
    x   (ndarray)   : Alloy composition
    k   (ndarray)   : Full vector of partitioning coefficients
    f   (float)     : Precipitate fraction.
    cov (ndarray)   : Covariance of *transformed variables* i.e. p,q.
    lambda_ (float) : Ridge parameter.
    """
    m = 2 ; n = len(k)//m
    K = np.zeros((n*m+1+m,n*m+1+m))
    eps = np.zeros(n*m+1+m)
    for phi in range(m):
        # matrix terms
        k_phi = k[n*phi:n*(phi+1)]
        # Exact correction terms
        K[n*m+1+phi,n*phi:n*(phi+1)] = 0.5*k_phi*x
        K[n*phi:n*(phi+1),n*m+1+phi] = 0.5*k_phi*x
        eps[n*m+1+phi] = k_phi@x-1.
    # Reconstruct Phi matrix, g vector
    f_p_2 = np.concatenate((1.-f,f))
    f_c_2 = np.array([[1,-1],[-1,1]])*cov[-1,-1]
    f_c_2 = ArcTanh.transform_cov(f_p_2,f_c_2)
    Phi = np.zeros((m*n+1+m,m*n+1+m))
    g = np.zeros(m*n+1+m)
    for i in range(n):
        k_i = k[i:m*n:n]
        k_cov_i = cov[i:m*n:n,i:m*n:n]
        k_cov_i = Logy.transform_cov(k_i,k_cov_i)
        sig_i2 = (np.diag(k_cov_i)*f_p_2**-2 + np.diag(f_c_2)*k_i**-2).sum()
        delta_i = (f_p_2*k_i).sum()-1.
        # matrix terms
        Phi11_i = (delta_i*np.diag(f_p_2*k_i)\
            +((f_p_2*k_i).reshape(-1,1))*((f_p_2*k_i).reshape(1,-1)))\
                /sig_i2
        Phi22_i = ((-4*delta_i*np.diag((f_p_2-0.5)*f_p_2*k_i)\
            +4*(((1.-f_p_2)*f_p_2*k_i).reshape(-1,1))\
                *(((1.-f_p_2)*f_p_2*k_i).reshape(1,-1)))\
                    *np.array([[1.,-1.],[-1.,1.]])/sig_i2).sum(keepdims=True)
        Phi12_i = ((delta_i*2*np.diag((1.-f_p_2)*f_p_2*k_i)\
            +2*(((1.-f_p_2)*f_p_2*k_i).reshape(-1,1))\
                *((f_p_2*k_i).reshape(1,-1))\
                    +2*((f_p_2*k_i).reshape(-1,1))\
                        *(((1.-f_p_2)*f_p_2*k_i).reshape(1,-1)))\
                            *np.array([[-1.,-1.],[1,1]])/sig_i2).sum(axis=0,keepdims=True)
        # vector terms
        g1_i = -delta_i*f_p_2*k_i/sig_i2
        g2_i = -(2*delta_i*f_p_2*k_i*(1.-f_p_2)*np.array([-1.,1.])/sig_i2).sum(keepdims=True)
        # Place these terms directly into the Phi matrix, g vector
        # vector
        g[i:n*m:n] = g1_i
        g[n*m:n*m+1] += g2_i
        # matrix
        Phi[i:n*m:n,i:n*m:n] = Phi11_i
        Phi[n*m:n*m+1,i:n*m:n] = Phi12_i
        Phi[i:n*m:n,n*m:n*m+1] = Phi12_i.T
        Phi[n*m:n*m+1,n*m:n*m+1] += Phi22_i
    Phi += np.diag(np.diag(Phi)) ; Phi *= 0.5 
    r_term = lambda_*block_diag(inv(cov),np.zeros((m,m)))
    cov_like_inv = r_term + r_term.T + Phi + Phi.T - K - K.T
    b = 2*g - eps
    sol = inv(cov_like_inv)@b
    p_corr = Logy.transform(k) + np.log(1.+sol[:n*m]) # Technically this is what was constrained. 
    q_corr = ArcTanh.transform(f) + sol[n*m]
    k_out = Logy.inverse_transform(p_corr)
    f_out = ArcTanh.inverse_transform(q_corr)
    return k_out,f_out

with open("correction_Ab_system.dill","rb") as infile:
    correction_Ab_system = dill.load(infile)

with open("correction_hc_Ab_system.dill","rb") as infile:
    correction_hc_Ab_system = dill.load(infile)

def sc_corr_2p(X,k_prd,k_scov,f_prd,f_unc,lambda_=1.,rtn_cov=False):
    """
    Apply a correction to the microstructure predictions for
    2-phase alloys. Essentially treats each part as a constraint as a 
    soft constraint with certain weightings, in order to produce corrected 
    predictions for the partitioning coefficients.

    Note that this no longer incorporates the constraint-fulfillment 
    probabilities. 

    Parameters
    ----------
    k_scov  (list)          : ragged list of covariance matrices for each alloy.
    k_prd   (list)          : ragged list of predicted pcs for each alloy.
    X       (pd.Dataframe)  : Dataframe of input alloy compositions.
    f_prd   (ndarray)       : Predicted phase fractions (phase 2)
    f_unc   (ndarray)       : Associated phase uncertainties.
    lambda_ (float)         : Controls the weighting of ridge parameter.
    rtn_cov (bool)          : Whether or not to return covariance matrices as output. 
    """
    k_prd_new = [[],[]] ; k_unc_new = [[],[]]
    f_prd_new = [] ; f_unc_new = []
    cov_list = []
    for x_full,k_c,k_p,f_p,f_c in zip(X.values,k_scov,k_prd,f_prd,f_unc):
        f_p = np.atleast_1d(f_p) ; f_c = np.atleast_2d(f_c)
        m = 2 # number of phases present
        n = len(k_p)//m # number of elements present.
        x = x_full[x_full>0.] # input composition
        x1 = x*k_p[:n]
        x2 = x*k_p[n:]
        # initial covariance matrix, converted to transformed variable form
        cov = block_diag(Logy.transform_cov(k_p,k_c),
                        ArcTanh.transform_cov(f_p,f_c))
        # Simply import relevant matrix / vector for solutions as lambda functions
        A,b = correction_Ab_system[n]
        
        cov_new_inv  = lambda_*inv(cov) + A(*x,*x1,*x2,*f_p,1.0,1.0)
        cov_new = inv(cov_new_inv)
        var_corr = cov_new @ b(*x,*x1,*x2,*f_p,1.0,1.0)
        # Transform to correct variables. 
        var_corr = var_corr.flatten()
        p_corr = Logy.transform(k_p) + var_corr[:n*m]
        q_corr = ArcTanh.transform(f_p) + var_corr[-1:]
        k_out = Logy.inverse_transform(p_corr).flatten()
        f_out = ArcTanh.inverse_transform(q_corr).flatten()
        # Transform covariance 
        tr_vec = np.concatenate((k_out,2*f_out*(1.-f_out)))
        cov_out = (tr_vec.reshape(-1,1)@tr_vec.reshape(1,-1))*cov_new
        # Store
        k_prd_new[0].append(k_out[:n])
        k_unc_new[0].append(np.diag(cov_out[:n]))
        k_prd_new[1].append(k_out[n:n*2])
        k_unc_new[1].append(np.diag(cov_out[n:n*2]))
        f_prd_new.append(f_out)
        f_unc_new.append(np.diag(cov_out)[-1:])
        cov_list.append(cov)
    k_prd_new[0] = np.concatenate(k_prd_new[0])
    k_unc_new[0] = np.concatenate(k_unc_new[0])
    k_prd_new[1] = np.concatenate(k_prd_new[1])
    k_unc_new[1] = np.concatenate(k_unc_new[1])
    f_prd_new = np.concatenate(f_prd_new)
    f_unc_new = np.concatenate(f_unc_new)
    if rtn_cov:
        return tuple(k_prd_new),tuple(k_unc_new),f_prd_new,f_unc_new,cov_list
    else:
        return tuple(k_prd_new),tuple(k_unc_new),f_prd_new,f_unc_new

def bayesian_corr_2p(X,k_prd,k_scov,f_prd,f_unc,rtn_cov=False,tol=0.05):
    """
    NOW A MODIFICATION OF bayesian_corr(). Apply a Bayesian correction to the microstructure predictions for
    2-phase alloys. This incorporates the likelihood of the model
    fulfilling certain physical constraints with the prior from the 
    ML models in order to produce corrected predictions for the part-
    -itioning coefficients.

    Parameters
    ----------
    k_scov  (list)          : ragged list of covariance matrices for each alloy.
    k_prd   (list)          : ragged list of predicted pcs for each alloy.
    X       (pd.Dataframe)  : Dataframe of input alloy compositions.
    f_prd   (ndarray)       : Predicted phase fractions (phase 2)
    f_unc   (ndarray)       : Associated phase uncertainties.
    rtn_cov (bool)          : Whether or not to return covariance matrices as output. 
    tol     (float)         : Tolerance for soft constraint on element-total as % of overall amount. 
    """
    k_prd_new = [[],[]] ; k_unc_new = [[],[]]
    f_prd_new = [] ; f_unc_new = []
    cov_list = []
    for x_full,k_c,k_p,f_p,f_c in zip(X.values,k_scov,k_prd,f_prd,f_unc):
        f_p = np.atleast_1d(f_p) ; f_c = np.atleast_2d(f_c)**2
        m = 2 # number of phases present
        n = len(k_p)//m # number of elements present.
        x = x_full[x_full>0.] # input composition
        x1 = x*k_p[:n]
        x2 = x*k_p[n:]
        # initial covariance matrix, converted to transformed variable form
        cov = block_diag(Logy.transform_cov(k_p,k_c),
                        ArcTanh.transform_cov(f_p,f_c))
        # Simply import relevant matrix / vector for solutions as lambda functions
        A_l,b_l = correction_hc_Ab_system[n]
        A = A_l(*x,*x1,*x2,*f_p,tol)
        b = b_l(*x,*x1,*x2,*f_p,tol)
        corr  = inv(block_diag(inv(cov),np.zeros((m,m))) + A)@b
        # The covariance is a bit different though
        cov_new_inv  = inv(cov) + A[:-m,:-m]
        cov_new = inv(cov_new_inv)
        # Transform to correct variables. 
        corr = corr.flatten()
        dp = corr[:n*m]
        dq = corr[n*m:n*m+1]
        k_out = k_p * (1+dp)
        f_out = (f_p*(1+np.tanh(dq)))/(1+(2*f_p-1)*np.tanh(dq))
        k_out = k_out.flatten() ; f_out = f_out.flatten()
        # Same transformations for uncertainties
        p_unc = np.sqrt(np.diag(cov_new[:n*m,:n*m]))
        q_unc = np.sqrt(cov_new[n*m:n*m+1,n*m:n*m+1])
        k_unc_out = p_unc*k_out
        f_unc_out = q_unc*2*f_out*(1.-f_out)
        k_unc_out = k_unc_out.flatten() ; f_unc_out = f_unc_out.flatten()
        # Transform covariance 
        tr_vec = np.concatenate((k_out,2*f_out*(1.-f_out)))
        cov_out = (tr_vec.reshape(-1,1)@tr_vec.reshape(1,-1))*cov_new[:n*m+1,:n*m+1]
        # Store
        k_prd_new[0].append(k_out[:n])
        k_unc_new[0].append(k_unc_out[:n])
        k_prd_new[1].append(k_out[n:n*2])
        k_unc_new[1].append(k_unc_out[n:n*2])
        f_prd_new.append(f_out)
        f_unc_new.append(f_unc_out)
        cov_list.append(cov_out)
    k_prd_new[0] = np.concatenate(k_prd_new[0])
    k_unc_new[0] = np.concatenate(k_unc_new[0])
    k_prd_new[1] = np.concatenate(k_prd_new[1])
    k_unc_new[1] = np.concatenate(k_unc_new[1])
    f_prd_new = np.concatenate(f_prd_new)
    f_unc_new = np.concatenate(f_unc_new)
    if rtn_cov:
        return tuple(k_prd_new),tuple(k_unc_new),f_prd_new,f_unc_new,cov_list
    else:
        return tuple(k_prd_new),tuple(k_unc_new),f_prd_new,f_unc_new

def bayesian_corr(X,k_prd,k_scov,f_prd,f_scov):
    """
    Apply a Bayesian correction to the microstructure predictions for
    multi-phase alloys. This incorporates the likelihood of the model
    fulfilling certain physical constraints with the prior from the 
    ML models in order to produce corrected predictions for the part-
    -itioning coefficients and phases. 

    Parameters
    ----------
    X       (pd.Dataframe)  : Dataframe of input alloy compositions.
    k_scov  (list)          : ragged list of covariance matrices for each alloy's partitioning coeffs.
    k_prd   (list)          : ragged list of predicitions for each alloy's partitioning coeffs.
    f_scov  (ndarray)       : ragged list of covariance matrices for each alloy's phases. 
    f_prd   (ndarray)       : ragged list of predicitions for each alloy's phases.
    """
    k_prd_new = [] ; k_unc_new = [] ; f_prd_new = [] ; f_unc_new = []
    for x_full,k_c,k_p,f_p,f_c in zip(X.values,k_scov,k_prd,f_prd,f_scov):
        m = len(f_p) # number of phases present
        n = len(k_p)//m # number of elements present.
        x = x_full[x_full>0.] # input composition
        # initial covariance matrix, converted to transformed variable form
        cov = block_diag(Logy.transform_cov(k_p,k_c),
                        ArcTanh.transform_cov(f_p,f_c))
        # Correction terms relating to sum(components in phase) = 1.
        Xi_subs = [] ; y_subs = []
        for phi in range(m):
            # matrix terms
            k_cov_phi = k_c[n*phi:n*(phi+1),n*phi:n*(phi+1)]
            k_phi = k_p[n*phi:n*(phi+1)]
            sig_phi2 = x@k_cov_phi@x.T
            eps_phi = k_phi@x-1.
            Xi_phi = (((k_phi*x).reshape(-1,1))@((k_phi*x).reshape(1,-1))\
                        + eps_phi * np.diag(k_phi*x))\
                            /sig_phi2
            # vector terms
            y_phi = -eps_phi*k_phi*x/sig_phi2
            Xi_subs += [Xi_phi]
            y_subs += [y_phi]
        # Add on zero terms to these
        Xi_subs += [np.zeros((m,m))]
        y_subs += [np.zeros(m)]
        Xi = block_diag(*Xi_subs)
        Xi += np.diag(np.diag(Xi)) ; Xi *= 0.5
        y = np.concatenate(tuple(y_subs))
        # Correction terms relating to physical sum. 
        Phi = np.zeros((m*(n+1),m*(n+1)))
        g = np.zeros(m*(n+1))
        for i in range(n):
            k_cov_i = k_c[i::n,i::n]
            k_i = k_p[i::n]
            sig_i2 = (np.diag(k_cov_i)*f_p**-2 + np.diag(f_c)*k_i**-2).sum()
            delta_i = (f_p*k_i).sum()-1.
            # matrix terms
            Phi11_i = (delta_i*np.diag(f_p*k_i)\
                +((f_p*k_i).reshape(-1,1))*((f_p*k_i).reshape(1,-1)))\
                    /sig_i2
            Phi22_i = (-4*delta_i*np.diag((f_p-0.5)*f_p*k_i)\
                +(((2.-2.*f_p)*f_p*k_i).reshape(-1,1))\
                    *(((2.-2.*f_p)*f_p*k_i).reshape(1,-1)))/sig_i2
            Phi12_i = (delta_i*np.diag((2.-2.*f_p)*f_p*k_i)\
                +(((1.-f_p)*f_p*k_i).reshape(-1,1))\
                    *((f_p*k_i).reshape(1,-1))\
                        +((f_p*k_i).reshape(-1,1))\
                            *(((1.-f_p)*f_p*k_i).reshape(1,-1)))/sig_i2
            # vector terms
            g1_i = -delta_i*f_p*k_i/sig_i2
            g2_i = -delta_i*f_p*k_i*(2.-2.*f_p)/sig_i2 
            # Place these terms directly into the Phi matrix, g vector
            # vector
            g[i:n*m:n] = g1_i
            g[-m:] += g2_i
            # matrix
            Phi[i:n*m:n,i:n*m:n] = Phi11_i
            Phi[-m:,i:n*m:n] = Phi12_i
            Phi[i:n*m:n,-m:] = Phi12_i
            Phi[-m:,-m:] += Phi22_i
        Phi += np.diag(np.diag(Phi)) ; Phi *= 0.5
        # Correction terms relating to sum of phases
        Delta = f_p.sum()-1.
        sig_2 = f_c.sum()
        h_ = -2.*Delta*f_p*(2.-2.*f_p)/sig_2
        Theta_ = (4.*Delta*np.diag(f_p*(f_p-0.5))\
            +(((2.-2.*f_p)*f_p).reshape(-1,1))\
                    *(((2.-2.*f_p)*f_p).reshape(1,-1)))/sig_2
        # Need to be bigger
        h = np.concatenate((np.zeros(n*m),h_))
        Theta = block_diag(np.zeros((n*m,n*m)),Theta_)
        Theta += np.diag(np.diag(Theta)) ; Theta *= 0.5
        # Can now calculate the corrections (transformed variables units)
        cov_new_inv  = inv(cov)+Xi+Phi+Theta
        cov_new = inv(cov_new_inv)
        var_corr = cov_new@(y+g+h)
        # Transform to correct variables. 
        p_corr = Logy.transform(k_p) + var_corr[:n*m]
        q_corr = ArcTanh.transform(f_p) + var_corr[-m:]
        k_out = Logy.inverse_transform(p_corr)
        f_out = ArcTanh.inverse_transform(q_corr)
        # Transform covariance 
        tr_vec = np.concatenate((k_out,2*f_out*(1.-f_out)))
        cov_out = tr_vec.reshape(-1,1)@tr_vec.reshape(1,-1)*cov_new
        # Store
        k_prd_new.append(k_out)
        k_unc_new.append(np.diag(cov_out[:n*m]))
        f_prd_new.append(f_out)
        f_unc_new.append(np.diag(cov_out)[-m:])
    return k_prd_new,k_unc_new,f_prd_new,f_unc_new

def reshape_cov2sub(X_orig,cov,y=None):
        """
        Reshape the covariance array for predictions into a list of sub-covariance
        matrices, each corresponding to the covariance of predictions for a single
        given entry. Note this returns a ragged list of matrices, i.e. matrices
        do NOT contain entries for elements not present in input. 

        Parameters
        ----------
        X_orig  (pd.Dataframe)  : The original input data. Used to get zero entries. 
        cov     (ndarray)       : Covariance matrix to reshape. Provide as tuples to get joined arrays as outputs.
        y       (ndarray)       : Optional. Provide predictions and reshape these in the same way too.
        """
        tuple_flag = True if isinstance(cov,tuple) else False
        at_nums = HRrep.get_els(X_orig)
        m = len(at_nums)
        mask = ~(X_orig.loc[:,"Composition"].values.flatten()==0.)
        locs = mask.reshape(-1,m) # Locations of non-zero components
        sub_cov = [] # sub-covariance matrix list
        if y is not None:
            sub_y = []
        start_ind = 0
        for entry in locs:
            end_ind = start_ind + entry.sum()
            if tuple_flag:
                sub_covs = (cov_[start_ind:end_ind,start_ind:end_ind] for cov_ in cov)
                sub_cov += [block_diag(*sub_covs)]
            else:
                sub_cov += [cov[start_ind:end_ind,start_ind:end_ind]]
            if y is not None:
                if tuple_flag:
                    sub_ys = tuple(y_[start_ind:end_ind] for y_ in y)
                    sub_y += [np.concatenate(sub_ys)]
                else:
                    sub_y += [y[start_ind:end_ind]]
            start_ind = copy(end_ind)
        if y is not None:
            return sub_cov,sub_y
        else:    
            return sub_cov