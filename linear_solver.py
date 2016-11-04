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
def Range(series):
    return series.max() - series.min()

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
        self.truncrange = np.linspace(0.2, 2.4, 32)

    def __getSigmaName(self, innerterm):
        if re.match(r'^[ab]p\d+$', innerterm):
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

        gaugecolumns = data.columns.tolist().index('process')
        gaugecolumns = data.columns.tolist()[:gaugecolumns]
        self._gauge = data[gaugecolumns]

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
        outmodel = self.packModelCoeff(coeffs)
        rms, calresult, groupstat = self.specErrors()
        return rms, outmodel, calresult, groupstat

    def specErrors(self):
        self._gauge['error'] = self.error
        validgauge = self._gauge.query('cost_wt>0')

        rms = RMS(validgauge['error'], validgauge['cost_wt'])

        grouprms = validgauge.groupby('group').agg(
                    lambda x: RMS(x['error'], x['cost_wt']))['error']
        groupstat = validgauge.groupby('group')['error'].describe().unstack()
        groupstat['rms'] = grouprms
        # groupstat['type'] = 

        typerms = validgauge.groupby('type').agg(
                  lambda x: RMS(x['error'], x['cost_wt']))['error']
        rms1d = typerms.loc['1D']
        rms2d = typerms.loc['2D']

        sigma3 = validgauge['error'].std() * 3

        anchorname = re.compile('anchor', re.IGNORECASE)
        anchor = validgauge.filter(regex=anchorname, axis=0)
        if anchor.empty:
            anchor = validgauge.filter(regex=anchorname, axis=0)
        if anchor.empty:
            anchor = np.nan
        else:
            anchor = anchor['error'][anchor['error'].abs().argmin()]
        range1d = Range(validgauge.query('type=="1D"')['error'])
        range2d = Range(validgauge.query('type=="2D"')['error'])
        unweightrms = RMS(validgauge['error'], np.ones(validgauge.shape[0]))
        total = Range(self._gauge.query('cost_wt>=0').error)

        spec = validgauge.filter(regex=re.compile('^spec$', re.IGNORECASE), axis=1)
        if not spec.empty:
            inspec = (validgauge['error'] < spec) & (validgauge['error'] > -1*spec)
        else:
            spec = validgauge.filter(regex=re.compile(r'^range_(min|max)$', re.IGNORECASE), axis=1)
            if not spec.empty:
                spec.columns = spec.columns.str.lower()
                inspec = (validgauge['error'] < spec['range_max']) & (validgauge['error'] > spec['range_min'])
            else:
                inspec = None
        if inspec is not None:
            validgauge['inspec'] = inspec
            ratio = inspec.sum()/float(inspec.size)
            g1d = validgauge.query('type=="1D"')
            ratio1d = g1d['inspec'].sum()/float(g1d['indpec'].size)
            g2d = validgauge.query('type=="2D"')
            ratio2d = g2d['inspec'].sum()/float(g2d['indpec'].size)
        else:
            ratio = np.nan
            ratio1d = np.nan
            ratio2d = np.nan

        calresult = {'3sigma':sigma3, 'total':total, 'Anchor_error':anchor,
                '2D_range':range2d, '1D_range':range1d, 'Unweighted_RMS':unweightrms,
                '1D_rms':rms1d, '2D_rms':rms2d, 'in_spec_ratio':ratio,
                '1D_in_spec_ratio':ratio1d, '2D_in_spec_ratio':ratio2d,
                }
        return rms, calresult, groupstat

    def packModelCoeff(self, coeffs):
        inmodel = self.model
        outmodel = {}
        for i, k in enumerate(inmodel.keys()):
            outmodel[k] = inmodel[k] + (coeffs[i],)
        return outmodel

if __name__ == '__main__':
    curpath=os.path.dirname(os.path.realpath(__file__))
    datapath = os.path.join(curpath, 'data\\v1', 'gauge_params_signals.txt')
    s = Solver(datapath, regressor='linear')
    model = {'AG1':   (9.21, 0),
             'AG2':   (72.2, 0),
             'Ap':    (102.02, 1.7),
             'Bp':    (168, 1.7),
             'Am':    (153.73, 2),
             'Bn':    (62.84, 0.5),
             'MG1':   (164.22, 0),
             'Slope': (0,0)}
    rms, outmodel, calresult, groupstat = s.fit(model)
    print 'Solver specErrors: {}'.format(calresult)
    print outmodel
    print groupstat
    '''
    startime = timeit.default_timer()
    for i in range(100):
        rms = s.fit(model)
    print 'elapsed time: {}'.format(timeit.default_timer()-startime)
    '''
