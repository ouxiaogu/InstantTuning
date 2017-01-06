# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:11:38 2016

@author: hshi
"""
import re
import numpy as np

def sigma2qtr(sigma):
    pixel = 14.0
    qtr = np.ceil(sigma/pixel *1.8)
    return max([9, qtr])

class RSTtemplate(object):
    def __init__(self, PTDterms=None, NTDterms=None, basetruncterms=None):
        if PTDterms is None:
            PTDterms = ["Mav","A","AG1","MG1","Slope"]
        if NTDterms is None:
            NTDterms = ["CDI1", "CDO1", "CSO", "DE1", "MGS1", "MGS", "MM3", "MM4"]
        if basetruncterms is None:
            basetruncterms = ['ap', 'bp']
        self.nosigmaterms = ['Mav', 'Slope', 'A']
        self.terms2d = ["CDI1", "CDO1", "CSO"]

        self.PTDterms = PTDterms
        self.NTDterms = NTDterms
        self.basetruncterms = basetruncterms
        self.truncterms = []
        self.ranges = {}

    def setTruncations(self, truncmin=0.2, truncmax=2.4, N=32):
        self.truncterms = []
        truncs = np.linspace(truncmin, truncmax, N)
        for i in range(N):
            for term in self.basetruncterms:
                termname = '{}{}'.format(term, i)
                self.truncterms.append(termname)
                self.ranges['b0_{}'.format(termname)] = truncs[i]

    def populateSigma(self, sigmas=None):
        if sigmas is None:
            sigmas = {}
        defaultrange = sigmas.get('default', [])
        temp = []
        for term in self.PTDterms + self.NTDterms + self.truncterms:
            if term not in self.nosigmaterms:
                trange = sigmas.get(term, defaultrange)
                for i,v in enumerate(trange):
                    if i>= len(temp):
                        temp.append({})
                    temp[i][term] = v
        #return temp
        template ='''
jobpath = DM.get_jobpath()
luapath ="/h/data/dummydb/calibrate/job1/lua/resist.lua"
'''
        templatenames = []
        for i, t in enumerate(temp):
            name = 'template{}'.format(i)
            for term in self.PTDterms + self.NTDterms:
                if term not in self.nosigmaterms:
                    self.ranges['sigma{}'.format(term)] = t.get(term, 0)
            for term in self.truncterms:
                self.ranges['sigma_{}'.format(term)] = t.get(term, 0)
            template += self.buildTemplate(name=name)
            templatenames.append(name)
        template = '--{}\n'.format(','.join(self.truncterms + self.NTDterms)) + template
        template = '--{}\n'.format(','.join(templatenames)) + template
        return template

    def buildTemplate(self, name='template', Type='fem'):
        template = '''
{name} = {{ 
    modeltype = "{Type}",
    role = "",
    templatename = "{name}",
    useterms = {{{ptdterms}}},
    {sigmas}
    lua= jobpath..luapath
}}'''
        ptdterms = []
        for term in self.PTDterms:
            if term == 'Slope':
                ptdterms.append('"Slope"')
            else:
                ptdterms.append('"c{}"'.format(term))
        ptdterms = ','.join(ptdterms)
        
        sigmas = []
        for term in self.PTDterms:
            if term not in self.nosigmaterms:
                sigmaname = 'sigma{}'.format(term)
                sigmas.append('{}={},'.format(sigmaname, self.ranges.get(sigmaname, 0)))
        for term in self.truncterms:
            b0name = 'b0_{}'.format(term)
            sigmas.append('{}={},'.format(b0name, self.ranges.get(b0name, 0)))
        for term in self.truncterms:
            sigmaname = 'sigma_{}'.format(term)
            sigma = self.ranges.get(sigmaname, 0)
            sigmas.append('{}={},'.format(sigmaname, sigma))
            qtrname = 'qtrFiltSize_{}'.format(term)
            sigmas.append('{}={},'.format(qtrname, sigma2qtr(float(sigma)*1000)))
        for term in self.NTDterms:
            sigmaname = 'sigma{}'.format(term)
            sigmas.append('{}={},'.format(sigmaname, self.ranges.get(sigmaname, 0)))
            qtrname = 'qtrFiltSize{}'.format(term)
            sigmas.append('{}={},'.format(qtrname, sigma2qtr(float(sigma)*1000)))
        sigmas = '\n\t'.join(sigmas)
        return template.format(name=name, Type=Type,
                               ptdterms=ptdterms, sigmas=sigmas)

    def buildRSTlua(self):
        template = '''AppInit=function()
    local attrtypes=TERM.get_component_attrtypes()
    NL=attrtypes[1]
    CF=attrtypes[2]
    SM=attrtypes[3]
end
TermFunc=function(algo)
	{}
end
'''
        addnormterm = 'TERM.add_term(algo, "c{term}*{term}(sigma{term},qtrFiltSize{term})")'
        add2dterm = 'TERM.add_term(algo, "c{term}*{term}(sigma2D{term}=0.01,sigma{term},qtrFiltSize2D{term}=5,qtrFiltSize{term})")'
        addtruncterm = 'TERM.add_term(algo, "c_{term}*Bp(sigma_{term},qtrFiltSize_{term},b0_{term})")'
        temp = []
        for term in self.truncterms:
            temp.append(addtruncterm.format(term=term))
        for term in self.NTDterms:
            if term in self.terms2d:
                temp.append(add2dterm.format(term=term))
            else:
                temp.append(addnormterm.format(term=term))
        self.RSTlua = template.format('\n\t'.join(temp))

if __name__ == '__main__':
    a = RSTtemplate(NTDterms=['CSO', 'DE1', 'MGS1', 'MGS', 'MM3', 'MM4'])
    a.setTruncations(truncmin=0.2, truncmax=2.4, N=32)
    a.buildRSTlua()
    sigmas = {'AG1':np.linspace(0,0.020,64).tolist()+np.linspace(0.050,0.200,64).tolist(),
              #'MG1':np.linspace(0.030,0.200,128).tolist(),
              #'MGS1':np.linspace(0.030,0.200,128).tolist(),
              'default':np.linspace(0.030,0.200,128).tolist()}
    b = a.populateSigma(sigmas)

    import os
    workfoder = r'C:\Users\hshi\Documents\JIRA\Termsignal\Pengcheng'
    with open(os.path.join(workfoder, 'resist_TSMC.lua'), 'w') as f:
        f.write(a.RSTlua)
    with open(os.path.join(workfoder, 'defineTemplate_TSMC.lua'), 'w') as f:
        f.write(b)
    
