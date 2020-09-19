__author__ = 'lisette.espin'

#######################################################################
# Global dependencies (system)
#######################################################################
import gc
import sys
import time
import matplotlib
import collections
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore
from statsmodels.formula.api import ols

#######################################################################
# Local dependencies
#######################################################################
import utils

#######################################################################
# MRQAP
#######################################################################
INTERCEPT = 'Intercept'

class MRQAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, X=None, npermutations=-1, diagonal=False, logfile=None, standarized=False, node_regression=False, seed=None):
        '''
        Initialization of variables
        :param Y: numpy array depended variable
        :param X: dictionary of numpy array independed variables
        :param npermutations: int number of permutations
        :param diagonal: boolean, False to delete diagonal from the OLS model
        :return:
        '''
        np.random.seed(seed)
        self.X = X                                  # independent variables: dictionary of numpy.array
        self.target = list(Y.keys())[0]             # dependent variable: string
        self.Y = Y[self.target]                     # dependent variable: numpy.array
        self.n = self.Y.shape[0]                    # number of nodes
        self.npermutations = npermutations          # number of permutations
        self.diagonal = diagonal                    # False then diagonal is removed
        self.data = None                            # Pandas DataFrame
        self.model = None                           # OLS Model y ~ x1 + x2 + x3 (original)
        self.v = collections.OrderedDict()          # vectorized matrices, flatten variables with no diagonal
        self.betas = collections.OrderedDict()      # betas distribution
        self.tvalues = collections.OrderedDict()    # t-test values
        self.logfile = logfile                      # logfile path name
        self.standarized = standarized              # standarize variables
        self.node_regression = node_regression      # whether the variables are vectors (node regression) or not (dyad)
        self.duration = None                        # how long mrqap took to run
        
    def init(self):
        '''
        Generating the original OLS model. Y and Xs are flattened.
        Also, the betas and tvalues dictionaries are initialized (key:independent variables, value:[])
        :return:
        '''
        self.v[self.target] = self._getFlatten(self.Y)
        self._initCoefficients(INTERCEPT)
        for k,x in self.X.items():
            if k == self.target:
                utils.printf('ERROR: Idependent variable cannot be named \'{}\''.format(self.target), self.logfile)
                sys.exit(0)
            self.v[k] = self._getFlatten(x)
            self._initCoefficients(k)
        self.data = pd.DataFrame(self.v)
        self.model = self._fit(self.v.keys(), self.data)
        del(self.X)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def mrqap(self):
        '''
        MultipleRegression Quadratic Assignment Procedure
        :return:
        '''
        self.duration = time.time()
        self.init()
        self._shuffle()
        self.duration = time.time() - self.duration

    def _shuffle(self):
        '''
        Shuffling rows and columns npermutations times.
        beta coefficients and tvalues are stored.
        :return:
        '''
        for p in range(self.npermutations):
            self.Ymod = self.Y.copy()
            self._rmperm()
            model = self._newfit()
            self._update_betas(model._results.params)
            self._update_tvalues(model.tvalues)
            self.Ymod = None
        gc.collect()


    def _newfit(self):
        '''
        Generates a new OLS fit model
        :return:
        '''
        newv = collections.OrderedDict()
        newv[self.target] = self._getFlatten(self.Ymod)
        for k,x in self.v.items():
            if k != self.target:
                newv[k] = x
        newdata = pd.DataFrame(newv)
        newfit = self._fit(newv.keys(), newdata)
        del(newdata)
        del(newv)
        return newfit


    #####################################################################################
    # Handlers
    #####################################################################################

    def _fit(self, keys, data):
        '''
        Fitting OLS model
        v a dictionary with all variables.
        :return:
        '''
        if self.standarized:
            data = data.apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis=0) #axis: 0 to each column, 1 to each row

        formula = '{} ~ {}'.format(self.target, ' + '.join([k for k in keys if k != self.target]))
        return ols(formula, data).fit()

    def _initCoefficients(self, key):
        self.betas[key] = []
        self.tvalues[key] = []

    def _rmperm(self, duplicates=True):
        shuffle = np.random.permutation(self.Ymod.shape[0])
        np.take(self.Ymod,shuffle,axis=0,out=self.Ymod)
        if not self.node_regression:
            np.take(self.Ymod,shuffle,axis=1,out=self.Ymod)
        del(shuffle)

    def _update_betas(self, betas):
        for idx,k in enumerate(self.betas.keys()):
                self.betas[k].append(round(betas[idx],6))

    def _update_tvalues(self, tvalues):
        for k in self.tvalues.keys():
            self.tvalues[k].append(round(tvalues[k],6))

    def _getFlatten(self, original):
        return self._deleteDiagonalFlatten(original)

    def _deleteDiagonalFlatten(self, original):
        tmp = original.flatten()
        if not self.diagonal:
            tmp = np.delete(tmp, [i*(original.shape[0]+1)for i in range(original.shape[0])])
        return tmp

    def _zeroDiagonalFlatten(self, original):
        tmp = original.copy()
        if not self.diagonal:
            np.fill_diagonal(tmp,0)
        f = tmp.flatten()
        del(tmp)
        return f


    #####################################################################################
    # Prints
    #####################################################################################

    def summary(self, verbose=False):
        '''
        Prints the OLS original summary and beta and tvalue summary.
        :return:
        '''
        self._summary_ols(verbose)
        self._summary_betas()

    def _summary_ols(self, verbose=False):
        '''
        Print the OLS summary
        :return:
        '''
        if verbose:
            print('# of Permutations: {}'.format(self.npermutations))
            print(self.model.summary())
        else:
            print('')
            print(self.model.model.formula.center(78))
            print('==============================================================================')

            df = pd.DataFrame({'R-Square':[round(self.model.rsquared,4)],
                               'Adj. R-Sqr.':[round(self.model.rsquared_adj,4)],
                               'Obs.':[round(self.model.nobs,4)],
                               'Perms.':[self.npermutations],
                               'Duration (sec.)':int(round(self.duration)),
                              })
            print(df)

    def _summary_betas(self):
        '''
        Summary of beta coefficients
        :return:
        '''
        print('')
        print('Summary Permutation Test'.center(78))
        print('==============================================================================')
        print()
        
        cols = ['MIN','MAX','MEDIAN','MEAN','STD','BETA','P(+)', 'P(-)']
        df = pd.DataFrame(index=self.betas.keys(), columns=cols)
        for k,v in self.betas.items():
            beta = self.model.params[k]
            df.loc[k,'MIN'] = round(min(v),4)
            df.loc[k,'MAX'] = round(max(v),4)
            df.loc[k,'MEDIAN'] = round(np.median(v),4)
            df.loc[k,'MEAN'] = round(np.mean(v),4)
            df.loc[k,'STD'] = round(np.std(v),4)
            df.loc[k,'BETA'] = round(beta,4)
            df.loc[k,'P(+)'] = round(np.mean([int(c >= beta) for c in v]),4)
            df.loc[k,'P(-)'] = round(np.mean([int(c <= beta) for c in v]),4)
        
        print(df)

    def _summary_tvalues(self):
        '''
        Summary t-values
        :return:
        '''
        print('')
        utils.printf('=== Summary T-Values ===')
        
        cols = ['MIN','MEDIAN','MEAN','MAX','STD. DEV.','T-TEST','P(+)', 'P(-)']
        df = pd.DataFrame(index=self.tvalues.keys(), columns=cols)
        for k,v in self.tvalues.items():
            tstats = self.model.tvalues[k]
            df.loc[k,'MIN'] = round(min(v),4)
            df.loc[k,'MEDIAN'] = round(np.median(v),4)
            df.loc[k,'MEAN'] = round(np.mean(v),4)
            df.loc[k,'MAX'] = round(max(v),4)
            df.loc[k,'STD. DEV.'] = round(np.std(v),4)
            df.loc[k,'T-TEST'] = round(tstats,4)
            df.loc[k,'P(+)'] = round(np.mean([int(c >= tstats) for c in v]),4)
            df.loc[k,'P(-)'] = round(np.mean([int(c <= tstats) for c in v]),4)
        
        print(df)
        
    def _ttest(self):
        print('')
        utils.printf('========== T-TEST ==========')
        
        cols = ['BETA', 'T-STAT', 'P-VALUE']
        df = pd.DataFrame(index=self.betas.keys(), columns=cols)
        for k,v in self.betas.items():
            beta = self.model.params[k]
            t = stats.ttest_1samp(v,beta)
            df.loc[k,'BETA'] = round(beta,4)
            df.loc[k,'T-STAT'] = t[0] #round(float(t[0]),4)
            df.loc[k,'P-VALUE'] = t[1] #round(float(t[1]),4)
        
        print(df)
        
            
            
            
    #####################################################################################
    # Plots
    #####################################################################################

    def plot(self, coef='betas', nrows=None, fn=None):
        '''
        Plots frequency of pearson's correlation values
        :param coef: string \in {betas, tvalues}
        :return:
        '''

        ### Data
        if coef == 'betas':
            dict_data = self.betas
        elif coef == 'tvalues':
            dict_data = self.tvalues

        ### Plot
        nplots = len(self.betas.keys())
        maxc = 4 if nrows is None else int(np.ceil(nplots/nrows))
        
        nrows = int(np.ceil(nplots/maxc)) if nrows is None else nrows
        ncols = 1 if nplots/maxc <= 1 and nrows is None else maxc if nrows is None else maxc
        
        fig, axes = plt.subplots(nrows,ncols,figsize=(3*ncols,3*nrows))
        
        counter = 0
        for var, data in dict_data.items():
            row = int(counter/maxc)
            col = counter%maxc
            ax = axes[col] if nrows==1 else axes[row,col]
            counter += 1
            
            ax.hist(data)
            ax.axvline(self.model.params[var], c='orange', lw=2, ls='--')
            
            ax.set_xlabel('regression coefficients', fontsize=8)
            ax.set_ylabel('frequency' if col==0 else '', fontsize=8)
            ax.set_title(var)
            ax.grid(True)

        while counter < (nrows*ncols):
            row = int(counter/maxc)
            col = counter%maxc
            ax = axes[col] if nrows==1 else axes[row,col]
            ax.axis("off")
            counter += 1
            
        plt.tight_layout()

        if fn is not None:
            plt.savefig(fn)

        plt.show()
        plt.close()
