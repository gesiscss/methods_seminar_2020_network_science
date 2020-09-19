__author__ = 'lisette.espin'

#######################################################################
# Global dependencies (system)
#######################################################################
import time
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats.stats import pearsonr
from statsmodels.stats.weightstats import ztest

#######################################################################
# Local dependencies
#######################################################################
import utils


#######################################################################
# QAP
#######################################################################
class QAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, X=None, npermutations=-1, diagonal=False, seed=None):
        '''
        Initialization of variables
        :param Y: numpy array depended variable
        :param X: numpy array independed variable
        :return:
        '''
        np.random.seed(seed)
        self.Y = Y
        self.X = X
        self.npermutations = npermutations
        self.diagonal = diagonal
        self.beta = None
        self.Ymod = None
        self.betas = []
        self.duration = None

    def init(self):
        '''
        Shows the correlation of the initial/original variables (no shuffeling)
        :return:
        '''
        self.beta = self.correlation(self.X, self.Y)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def qap(self):
        '''
        Quadratic Assignment Procedure
        :param npermutations:
        :return:
        '''
        self.duration = time.time()
        self.init()
        self._shuffle()
        self.duration = time.time() - self.duration

    def _shuffle(self):
        self.Ymod = self.Y.copy()
        for t in range(self.npermutations):
            self._rmperm()
            self._addBeta(self.correlation(self.X, self.Ymod, False))

    def correlation(self, x, y, show=True):
        '''
        Computes Pearson's correlation value of variables x and y.
        Diagonal values are removed.
        :param x: numpy array independent variable
        :param y: numpu array dependent variable
        :param show: if True then shows pearson's correlation and p-value.
        :return:
        '''
        if not self.diagonal:
            xflatten = np.delete(x, [i*(x.shape[0]+1)for i in range(x.shape[0])])
            yflatten = np.delete(y, [i*(y.shape[0]+1)for i in range(y.shape[0])])
            pc = pearsonr(xflatten, yflatten)
        else:
            pc = pearsonr(x.flatten(), y.flatten())
            
        if show:
            utils.printf("Observed Pearson's correlation: {}".format(round(pc[0],4)))
        return pc

    #####################################################################################
    # Handlers
    #####################################################################################

    def _addBeta(self, p):
        '''
        frequency dictionary of pearson's correlation values
        :param p: person's correlation value
        :return:
        '''
        p = round(p[0],6)
        self.betas.append(p)

    def _rmperm(self):
        shuffle = np.random.permutation(self.Ymod.shape[0])
        np.take(self.Ymod,shuffle,axis=0,out=self.Ymod)
        np.take(self.Ymod,shuffle,axis=1,out=self.Ymod)


    #####################################################################################
    # Plots & Prints
    #####################################################################################

    def summary(self):
        utils.printf('')
        utils.printf('# Permutations: {}'.format(self.npermutations))
        utils.printf('Duration (sec.): {}'.format(int(round(self.duration))))
        utils.printf('')
        utils.printf('- Sum all betas: {}'.format(sum(self.betas)))
        utils.printf('- Min betas: {}'.format(min(self.betas)))
        utils.printf('- Max betas: {}'.format(max(self.betas)))
        utils.printf('- Average betas: {}'.format(np.average(self.betas)))
        utils.printf('- Std. Dev. betas: {}'.format(np.std(self.betas)))
        utils.printf('')
        
        gt = np.mean([int(b >= self.beta[0]) for b in self.betas ]) # right tail: >= beta
        utils.printf('prop >= {}: {} P(Large)'.format(round(self.beta[0],4), gt))
        
        lt = np.mean([int(b <= self.beta[0]) for b in self.betas ]) # left tail: <= -beta
        utils.printf('prop <= {}: {} P(Small)'.format(round(self.beta[0],4), lt))

    def plot(self, fn=None):
        '''
        Plots frequency of pearson's correlation values
        :return:
        '''
        plt.hist(self.betas)
        plt.axvline(self.beta[0], c='orange', lw=2, ls='--')
        plt.xlabel('regression coefficients')
        plt.ylabel('frequency')
        plt.title('QAP')
        plt.grid(True)
        
        if fn is not None:
            plt.savefig(fn, bbox_inches='tight')
            
        plt.show()
        plt.close()
