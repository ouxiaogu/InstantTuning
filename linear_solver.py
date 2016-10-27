# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:34:51 2016

@author: hshi
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def snaptogrid(v, dataSeries):
    vp = dataSeries[np.abs(dataSeries - v).argmin()]
    # print 'snap from {} to {}'.format(v, vp)
    return vp

def RMS(error, wt):
    return np.sqrt(np.sum((np.power(error, 2) * wt))/np.sum(wt))

class Solver(object):
    def __init__(self, signalpath, regressor='linear'):
        self.data = pd.read_csv(signalpath, sep='\t')
        self.__preprocess()
        self.__defineTerms()
        self.__initTarget()
        if regressor == 'linear':
            self.regresssor = linear_model.LinearRegression()
        elif regressor == 'ridge':
            self.regresssor = linear_model.Ridge(alpha=0.5)

    def __defineTerms(self):
        self.ptdterms = ['Mav', 'A', 'AG1', 'AG2', 'Ap', 'Bp','Am', 'Bn', 'MG1', 'MG2', 'Slope']
        self.ptdterms_inner = ['Mav', 'A', 'AG1','MG1', 'Slope']
        self.term_to_inner = {}
        for term in self.ptdterms:
            if term in self.ptdterms_inner:
                self.term_to_inner[term] = term
            elif term == 'AG2':
                self.term_to_inner[term] = 'AG1'
            elif term == 'MG2':
                self.term_to_inner[term] = 'MG2'
            elif term in ['Ap', 'Am']:
                self.term_to_inner[term] = 'ap'
            elif term in ['Bp', 'Bn']:
                self.term_to_inner[term] = 'bp'
        # hardcode this for speed
        self.truncrange = np.arange(0.1, 2, 0.2)

    def __getSigmaName(self, innerterm):
        if re.match(r'^[ab]p\d$', innerterm):
            return 'sigma_{}'.format(innerterm)
        else:
            return 'sigma{}'.format(innerterm)
    def __getSignalName(self, innterterm, sigkind):
        return '{}_{}'.format(sigkind, innterterm)

    def __preprocess(self):
        '''
        this function drop columns whos valuse are all zero
        '''
        #self.data = self.data.loc[:,(self.data!=0).any()]
        self.data = self.data.dropna()

    def __initTarget(self):
        data = self.data.query('process==1').set_index('gauge')
        self.gauge = np.array(data.index)
        self.tonesgn = np.array(data['tone_sgn'])
        self.pebcd = data['PEBCD']
        self.wcd = data['wafer_CD']
        self.target = np.array((self.wcd - self.pebcd)*self.tonesgn)
        self.wt = np.array(data['cost_wt'])

    def getTermSig(self, termname, sigma=0, truncation=1, kinds=['vL', 'vR']):
        '''Note:
        ----
        termname an internal term name, like ap1, bp0
        sigma in nm unit
        '''
        try:
            termname = self.term_to_inner[termname]
        except KeyError:
            raise KeyError('term {} is no in pre-defined list: {}'.fromat(termname, self.term_to_inner.keys))
        if termname in ['ap', 'bp']:
            truncation = snaptogrid(truncation, self.truncrange)
            idx = self.truncrange.tolist().index(truncation)
            termname = '{}{}'.format(termname, idx)

        if termname in ['A', 'Mav', 'Slope']:
            data = self.data.query('process==1')
        else:
            sigma = sigma / 1000.0  # convert to um
            sigmaname = self.__getSigmaName(termname)
            sigmaSeries = self.data[sigmaname].unique()
            sigma = snaptogrid(sigma, sigmaSeries)
            #data = self.data.query('{k}=={v}'.format(k=sigmaname, v=sigma))
            data = self.data.loc[(self.data[sigmaname]-sigma).abs()<1e-7,:]
        data = data.set_index('gauge')
        data = data[['{}_{}'.format(x, termname) for x in kinds]]
        return data

    def getModelSigs(self, model=None, crossorder=1):
        ''' get the features of each term that ready to fit
        Notes:
        -----
        model
            a dict containing all the term informations
            The key is term name, the value is sigma and truncation
            {'AG2':(70, 0), 'Ap':(56, 1.1)
            Default is pure optical model {'A':(0,0)}
        '''
        if model is None:
            model = {'A':(0,0)}
        termlist = model.keys()
        signalsL = pd.DataFrame(index=self.gauge, columns=termlist)
        signalsR = pd.DataFrame(index=self.gauge, columns=termlist)
        for term in termlist:
            sigma, truncation = model[term]
            signal = self.getTermSig(term, sigma, truncation)
            signalsL[term] = signal.filter(regex=r'vL').iloc[:,0]
            signalsR[term] = signal.filter(regex=r'vR').iloc[:,0]
        self.signalsL = signalsL
        self.signalsR = signalsR
        signalsL = PolynomialFeatures(degree=crossorder, include_bias=False).fit_transform(signalsL)
        signalsR = PolynomialFeatures(degree=crossorder, include_bias=False).fit_transform(signalsR)
        self.features = signalsL + signalsR

    def fit(self, model=None, crossorder=1, method='linear', wt=None):
        self.model = model
        self.getModelSigs(model, crossorder)
        if wt is None:
            wt = self.wt
        self.regresssor.fit(self.features, self.target, sample_weight=wt)
        self.result = self.regresssor.predict(self.features)
        self.error = self.result * self.tonesgn + self.pebcd - self.wcd
        coeffs = self.regresssor.coef_
        return RMS(self.error, wt), coeffs


if __name__ == '__main__':
    curpath=os.path.dirname(os.path.realpath(__file__))
    datapath = os.path.join(curpath, 'data\\v0', 'gauge_params_signals.txt')
    s = Solver(datapath, regressor='linear')
    model = {'AG1':   (9.21, 0),
             'AG2':   (72.2, 0),
             'Ap':    (102.02, 1.7),
             'Bp':    (168, 1.7),
             'Am':    (153.73, 2),
             'Bn':    (62.84, 0.5),
             'MG1':   (164.22, 0),
             'Slope': (0,0)}
    rms = s.fit(model)
    print 'Solver rms: {}'.format(rms)
    '''
    startime = timeit.default_timer()
    for i in range(100):
        rms = s.fit(model)
    print 'elapsed time: {}'.format(timeit.default_timer()-startime)
    '''
