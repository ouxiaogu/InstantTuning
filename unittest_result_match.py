# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:39:48 2016

@author: peyang
"""
import pandas as pd
import os, os.path
import re
import matplotlib.style
import numpy as np
from linear_solver import RMS

def chk_model(jobparm, signalpath=os.path.join(os.getcwd(), r"data\ADI\all")):
    from linear_solver import Solver
    s = Solver(signalpath, modelkind='ADI')
    rms, outmodel, calresult, groupstat = s.fit(jobparm, method='nonlinear')
    return rms, outmodel, calresult, groupstat

def get_jobparm_from_Series(series):
    # load model from a df, which contains all the model forms
    from genetic_algo import gen_term_pairs, covert_job_parm
    # find the column names used in this programe
    colnames = series.index.tolist()
    useterms = []
    for col in colnames:
        if re.match(r'^c[_A-Z]{1}\w*$', col): # cAp or c_ap
            termName = re.match(r'^c_?(\w+)$', col).group(1) # 1st parenthesized subgroup
            useterms.append(termName)
    modelform = series.to_dict()
    termpairs = gen_term_pairs(useterms, modelform)
    modelform = series.to_dict()
    jobparm = covert_job_parm(modelform, termpairs)
    return jobparm

def group_stat(gauges):
    wtlabel    = [col for col in gauges.columns if col.lower() == 'cost_wt'][0]
    errlabel   = [col for col in gauges.columns if 'error' in col.lower()][0]
    grouplabel = [col for col in gauges.columns if 'group' in col.lower()][0]
    validgauges = gauges.query('cost_wt>0')
    grouprms = validgauges.groupby(grouplabel).agg(
                lambda x: RMS(x[errlabel], x[wtlabel]))[errlabel]
    groupstat = validgauges.groupby(grouplabel)[errlabel].describe().unstack()
    groupstat['rms'] = grouprms
    return groupstat

if __name__ == '__main__':
    cwd = os.getcwd()
    datapath = os.path.join(cwd, r"data\ADI\all")

    # instant tuning best model
    inmodelfile = os.path.join(datapath, r'best_instant_model.xlsx')
    df = pd.read_excel(inmodelfile)
    chkjobparm = get_jobparm_from_Series(df.ix[0, :]) # model with processid=0
    rms, outmodel, calresult, groupstat = chk_model(chkjobparm)
    print rms, outmodel

    # TFlex best model result
    tflexgaugefile = os.path.join(datapath, r'best_tflex_gauge_results.xlsx')
    tflexgauges = pd.read_excel(tflexgaugefile)
    validgauges = tflexgauges.query('cost_wt>0')
    wtlabel    = [col for col in validgauges.columns if col.lower() == 'cost_wt'][0]
    errlabel   = [col for col in validgauges.columns if 'error' in col.lower()][0]
    tflexgroupstat = group_stat(validgauges)
    tflexrms = RMS(validgauges[errlabel], validgauges[wtlabel])

    # comparison plot
    matplotlib.style.use('ggplot')
    instgrouprms  = groupstat.ix[:, 'rms']
    tflexgrouprms = tflexgroupstat.ix[:, 'rms']
    dfgrouprms = pd.DataFrame(dict(inst = instgrouprms, tflex = tflexgrouprms))
    ax = dfgrouprms.plot.bar()
    for p in ax.patches:
        ax.annotate(str(np.round(p.get_height(), 3)), (p.get_x() * 1.005, p.get_height() * 1.005))
    title = 'RMS {} ({})'.format(np.round(rms, 3), np.round(tflexrms, 3))
    ax.set_title(title, fontdict=None, loc='center')