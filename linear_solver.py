# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:34:51 2016

@author: hshi
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.optimize import leastsq, minimize

def snaptogrid(v, dataSeries, ret='value'):
    '''Given a number, find the colsese value or its index in a series
    Args:
    ----
    dataSeries
        Type: pandas.Series
    ret: control the returned type
        Type: String
        Value: 'value'|'index'
    '''
    idx = np.abs(dataSeries - v).argmin()
    if ret == 'value':
        return dataSeries[idx]
    else:
        return idx

def RMS(error, wt):
    return np.sqrt(np.sum((np.power(error, 2) * wt))/np.sum(wt))
def Range(series):
    return series.max() - series.min()

class Solver(object):
    def __init__(self, signalpath, modelkind='ADI'):
        self.modelkind = modelkind
        self.__getdata(signalpath)
        self.__defineTerms()
        self.regressor = linear_model.LinearRegression()

    def __getdata(self, signalpath):
        self.gauge = pd.read_csv(os.path.join(signalpath, 'gauge.txt'), sep='\t', index_col=0).set_index('gauge')
        self.params = pd.read_csv(os.path.join(signalpath, 'parameters.txt'), sep='\t', index_col=0).set_index('process')
        self.signals = {}
        for name in os.listdir(os.path.join(signalpath, 'signals')):
            try:
                pid = long(re.match(r'signal(\d+)\.txt', name).groups()[0])
            except AttributeError:
                continue
            self.signals[pid] = pd.read_csv(os.path.join(signalpath, 'signals', name), index_col=0, sep='\t').set_index('gauge')

        self.__preprocess()
        if self.modelkind is "HOD":
            pebcdpath = os.path.join(signalpath, 'pebcd.txt')
            pebcd = pd.read_csv(pebcdpath).set_index('gauge').iloc[:, 0]
            self.gauge['pebcd'] = pebcd
            self.HODtarget = np.array((self.gauge[self.WCDNAME] - self.gauge['pebcd'])*self.gauge['tone_sgn'])

    def __preprocess(self):
        self.gauge = self.gauge.reindex(self.signals[self.signals.keys()[0]].index)
        self.gauge = self.gauge.query('cost_wt>=0')
        for k, v in self.signals.iteritems():
            self.signals[k] = v.reindex(self.gauge.index)

        # find the column names used in this programe
        columns = self.gauge.columns.tolist()
        for col in columns:
            if re.match(r'^wafer_?cd$', col, re.IGNORECASE):
                self.WCDNAME = col
            if re.match(r'^group(name)?$', col, re.IGNORECASE):
                self.GROUPNAME = col

    def __defineTerms(self):
        self.ptdterms = ['Mav', 'A', 'AG1', 'AG2', 'Ap', 'Bp','Am', 'Bn', 'MG1', 'MG2', 'Slope']
        self.ptdterms_inner = ['Mav', 'A', 'AG1','MG1', 'Slope']
        self.ntdterms = ['CDO1', 'DE1', 'MGS1', 'MGS']
        self.term_to_inner = {}
        for term in self.ptdterms+self.ntdterms:
            if term in (self.ptdterms_inner + self.ntdterms):
                self.term_to_inner[term] = term
            elif term == 'AG2':
                self.term_to_inner[term] = 'AG1'
            elif term == 'MG2':
                self.term_to_inner[term] = 'MG2'
            elif term in ['Ap', 'Am']:
                self.term_to_inner[term] = 'ap'
            elif term in ['Bp', 'Bn']:
                self.term_to_inner[term] = 'bp'
        self.trunclist_ap = self.__getTruncationlist('ap')
        self.trunclist_bp = self.__getTruncationlist('bp')

    def __getSigmaName(self, innerterm):
        if re.match(r'^[ab]p\d+$', innerterm):
            return 'sigma_{}'.format(innerterm)
        else:
            return 'sigma{}'.format(innerterm)
    def __getTruncTermByb0(self, b0, kind='ap'):
        if kind == 'ap':
            idx = snaptogrid(b0, self.trunclist_ap['ratio'], ret='index')
            return self.trunclist_ap.loc[idx, 'term']
        elif kind == 'bp':
            idx = snaptogrid(b0, self.trunclist_bp['ratio'], ret='index')
            return self.trunclist_bp.loc[idx, 'term']
        else:
            raise AttributeError('no {} term found in truncation term, specify ap or bp'.format(kind))
    def __getSignalName(self, innterterm, sigkind):
        return '{}_{}'.format(sigkind, innterterm)

    def __getTruncationlist(self, kind='ap'):
        trunc = self.params.filter(regex=r'b0_{}\d+'.format(kind)).iloc[0].reset_index()
        trunc.columns = ['term', 'ratio']
        trunc['term'] = trunc['term'].str.extract(r'b0_({}\d+)'.format(kind))
        trunc = trunc[['term','ratio']]
        return trunc

    def getTermSig(self, termname, sigma=0, truncation=1, kinds=['vL', 'vR', 'sL', 'sR']):
        '''Note:
        ----
        termname an internal term name, like ap1, bp0
        sigma in nm unit
        '''
        try:
            termname = self.term_to_inner[termname]
        except KeyError:
            raise KeyError('term {} is no in pre-defined list: {}'.format(termname, self.term_to_inner.keys))
        if termname in ['ap', 'bp']:
            termname = self.__getTruncTermByb0(truncation, kind=termname)

        if termname in ['A', 'Mav', 'Slope']:
            data = self.signals[self.signals.keys()[0]]
        else:
            sigma = sigma / 1000.0  # convert to um
            sigmaname = self.__getSigmaName(termname)
            sigmaSeries = self.params[sigmaname]
            idx = snaptogrid(sigma, sigmaSeries, ret='index')
            data = self.signals[idx]
            # print '{} is at processid {}'.format(sigmaname, idx)
        data = data[['{}_{}'.format(x, termname) for x in kinds]]
        return data

    def getModelSigs(self, model=None):
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
        vL = pd.DataFrame(index=self.gauge.index, columns=termlist)
        vR = pd.DataFrame(index=self.gauge.index, columns=termlist)
        sL = pd.DataFrame(index=self.gauge.index, columns=termlist)
        sR = pd.DataFrame(index=self.gauge.index, columns=termlist)
        for term in termlist:
            sigma, truncation = model[term]
            signal = self.getTermSig(term, sigma, truncation)
            vL[term] = signal.filter(regex=r'vL').iloc[:,0]
            vR[term] = signal.filter(regex=r'vR').iloc[:,0]
            sL[term] = signal.filter(regex=r'sL').iloc[:,0]
            sR[term] = signal.filter(regex=r'sR').iloc[:,0]
        self.vLs = vL
        self.vRs = vR
        self.sLs = sL
        self.sRs = sR

    def fit(self, model=None, method='linear', wt=None, target='AG1'):
        ''' Common interface to fit the model to data. It will
            fetch signals -> fit the model -> return residule errors and results
        Args:
        ----
            model: define the terms and parameters
                type: dict
            method: option for solver type
                type: String
                value: 'linear'|'nonlinear'|'HOD1'
                for ADI linear solver, ADI nonlinear solver and HOD1 linear solver respectively
            wt: Optional, interface to modify the cost weight
                type: pandas.Series, index is the gauge name
                default value: the original weight in gauge file
            target: for ADI solvers only, coeff of the specified term will be forced to 1
                type: String
                value: term names in model
        Return:
        ----
            [0] a dict containing all model error related results
            [1] a pandas.Series containing coefficient of each terms
            [2] the threshold value, for ADI model only
        '''
        self.model = model
        self.getModelSigs(model)
        if wt is None:
            self.wt = self.gauge.cost_wt
        else:
            self.wt = wt
        if self.modelkind=='ADI' and method == 'linear':
            self.__linearSolver(target=target)
            self.error = self.__calError(self.vLs, self.vRs, self.sLs, self.sRs, self.coeff, self.thres)
        elif self.modelkind=='ADI' and method == 'nonlinear':
            self.__linearSolver(target=target)
            self.coeff = self.__nonlinearSolver()
            self.error = self.__calError(self.vLs, self.vRs, self.sLs, self.sRs, self.coeff, self.thres)
        elif self.modelkind=='HOD' and method == 'HOD1':
            self.__HOD1Solver()
        else:
            raise UserWarning('please input a validate method linear|nonlinear for ADI, HOD1 for HOD')
        outmodel = self.saveModelForm(self.coeff.tolist(), self.thres)
        rms, calresult, groupstat = self.specErrors()
        return rms, outmodel, calresult, groupstat

    def saveModelForm(self, coeffs, thres=None):
        inmodel = self.model
        outmodel = {}
        for i, k in enumerate(inmodel.keys()):
            outmodel[k] = inmodel[k] + (coeffs[i],)
        if (thres is not None) and (thres != np.nan):
            outmodel['thres'] = thres
        return outmodel

    def __linearSolver(self, target='AG1'):
        ''' linear solver for ADI, provide an approximate and fast solve to ADI model,
        also used to obtain the initial coefficients and thres for nonlinear solver
        '''
        thresl = self.vLs.multiply(self.wt, axis='index').sum(axis='index') / self.wt.sum()
        thresr = self.vRs.multiply(self.wt, axis='index').sum(axis='index') / self.wt.sum()
        thresi = (thresl + thresr) / 2.0
        signalsA = self.getTermSig('A')
        xterml = self.vLs.sub(thresi).div(signalsA['sL_A'], axis='index')
        xtermr = self.vRs.sub(thresi).div(signalsA['sR_A'], axis='index')
        xterm = xterml - xtermr
        x = xterm.filter(items=xterm.columns.drop(target))
        y = -1 * xterm[target]
        self.regressor.fit(x, y, sample_weight=self.wt)
        self.coeff = pd.Series(self.regressor.coef_, index=x.columns)
        self.coeff[y.name] = 1
        thresl = (self.coeff * thresl).sum()
        thresr = (self.coeff * thresr).sum()
        self.thres = (thresl + thresr) / 2.0

    def __HOD1Solver(self):
        '''  for HOD model only, linear solve w/o cross terms
        '''
        features = self.vLs + self.vRs
        self.regressor.fit(features, self.HODtarget, sample_weight=self.wt)
        res = pd.Series(self.regressor.predict(features), index=self.gauge.index)
        self.error = res * self.gauge['tone_sgn'] + self.gauge['pebcd'] - self.gauge[self.WCDNAME]
        self.coeff = pd.Series(self.regressor.coef_, index=features.columns)
        self.thres = np.nan

    def __nonlinearSolver(self):
        ''' Nonlinear solver for ADI model, more accurate than linear solver
        '''
        def costfun(p, x, y):
            vLs = self.vLs.reindex(x)
            vRs = self.vRs.reindex(x)
            sLs = self.sLs.reindex(x)
            sRs = self.sRs.reindex(x)
            error = self.__calError(vLs, vRs, sLs, sRs, p, self.thres)
            wt = self.wt.reindex(x)
            wt = (wt / wt.sum()).pow(0.5)
            return error * wt
        self.res = leastsq(costfun, self.coeff, args=(self.gauge.index, 0), full_output=True)
        return pd.Series(self.res[0], index=self.coeff.index)

    def __calError(self, vLs, vRs, sLs, sRs, coeff, thres):
        ''' calculate the model error for ADI model, provide the signals, coeffs and threshold
        '''
        errorl = (vLs.multiply(coeff).sum(axis=1) - thres) / sLs.multiply(coeff).sum(axis=1)
        errorr = (vRs.multiply(coeff).sum(axis=1) - thres) / sRs.multiply(coeff).sum(axis=1)
        return errorl - errorr

    def specErrors(self):
        self.gauge['error'] = self.error
        validgauge = self.gauge.query('cost_wt>0')
        validgauge['cost_wt'] = self.wt

        rms = RMS(validgauge['error'], validgauge['cost_wt'])

        grouprms = validgauge.groupby(self.GROUPNAME).agg(
                    lambda x: RMS(x['error'], x['cost_wt']))['error']
        groupstat = validgauge.groupby(self.GROUPNAME)['error'].describe().unstack()
        groupstat['rms'] = grouprms

        typerms = validgauge.groupby('type').agg(
                  lambda x: RMS(x['error'], self.wt))['error']
        try:
            rms1d = typerms.loc['1D']
        except KeyError:
            rms1d = np.nan
        try:
            rms2d = typerms.loc['2D']
        except KeyError:
            rms2d = np.nan

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
        total = Range(self.gauge.query('cost_wt>=0').error)

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
            try:
                ratio1d = g1d['inspec'].sum()/float(g1d['inspec'].size)
            except ZeroDivisionError:
                ratio1d = np.nan
            g2d = validgauge.query('type=="2D"')
            try:
                ratio2d = g2d['inspec'].sum()/float(g2d['inspec'].size)
            except:
                ratio2d = np.nan
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


if __name__ == '__main__':
    import timeit

    curpath=os.path.dirname(os.path.realpath(__file__))
    '''
    datapath = os.path.join(curpath, 'data', 'HOD')
    s = Solver(datapath, modelkind='HOD')
    model = {'AG1':   (9.21, 0),
             'AG2':   (72.2, 0),
             'Ap':    (102.02, 1.7),
             'Bp':    (168, 1.7),
             'Am':    (153.73, 2),
             'Bn':    (62.84, 0.5),
             'MG1':   (164.22, 0),
             'Slope': (0,0)}

    '''
    # datapath = os.path.join(curpath, 'data', 'all')
    datapath = os.path.join(curpath, 'data', 'ADI', 'all')
    s = Solver(datapath, modelkind='ADI')
    model = {'AG1':   (13.65079365, 0),
             'Bp':    (54.09448819, 1.761290319),
             'Am':    (96.92913386, 2.4),
             'Bn':    (64.80314961, 0.554838707),
             'Slope': (0, 0),
             'CDO1':  (55.43307087, 0),
             'DE1':   (82.20472441, 0),
             'MGS1':  (183.0708661, 0),
             'MGS':   (63.46456693, 0),
             }

    start = timeit.default_timer()
    #result = s.fit(model, method='HOD1')
    rms, outmodel, calresult, groupstat = s.fit(model, method='nonlinear')
    end = timeit.default_timer()
    print 'elapsed time: {}'.format(end - start)