from linear_solver_olddata import isXterm, splitXterm
import re

class Model(object):
    """docstring for Model"""
    def __init__(self, useterms, xterms):
        super(Model, self).__init__()
        self.useterms = useterms
        self.xterms   = xterms

    def gen_term_pairs(self, parmNames):
        ''' generate a dict, key is used term, value is tuples contain sigma, truncation
            Args:
            ----
                useterms: list, define the useterms
                parmNames: iterable, a list or a dict whose keys are model parameter name
            Return:
            ----
                [0] basetermpairs, dict like {'A': ('', ''), 'Bp': ('sigmaBp', 'b0_ratio'), ...}
                [1] termpairs, dict like basetermpairs and cross term pairs {'A': ('', ''), 'DE1*Slope': ({'DE1': ('sigmaDE1', ''), 'Slope': ('', '')}, )
                [2] searchparms, unique list, contains the name of all the searchable parameters
        '''
        xterms = self.xterms
        allterms = self.useterms + xterms.keys()
        baseterms = [t for t in allterms if t not in xterms.keys()]

        termpairs = {}
        searchparms = []

        noneSigmaTerm = {"A", "Mav", "Slope"}
        fixedTruncTerms = {"Ap": "b0_ratio", "Bp": "b0_ratio", "Am": "b0m_ratio", "Bn": "b0n_ratio"}
        for curTerm in baseterms:
            curSigma = ''
            curTruncRatio = ''
            if curTerm not in noneSigmaTerm:
                if curTerm in fixedTruncTerms.keys():
                    curSigma = r'sigma{}'.format(curTerm)
                    curTruncRatio = fixedTruncTerms[curTerm]
                else:
                    for parmname in parmNames:
                        if re.match(r'^sigma_?{}$'.format(curTerm), parmname, re.I) or re.match(r'^sigma_?{}_[\S]+$'.format(curTerm), parmname, re.I):
                            curSigma = parmname
                        elif re.match(r'^b0_?{}$'.format(curTerm), parmname, re.I):
                            curTruncRatio = parmname
            termpairs[curTerm] = (curSigma, curTruncRatio)
            for curParm in (curSigma, curTruncRatio):
                if curParm != '': searchparms.append(curParm)
        misSigmaTerms = [t for t in termpairs.keys() if (t not in noneSigmaTerm) and (termpairs[t][0] == '')]
        misTruncTerms = [t for t in termpairs.keys() if (t in fixedTruncTerms) and (termpairs[t][1] == '')]
        if len(misSigmaTerms) != 0:
            raise IOError('Cannot find sigma for these basic terms: {}'.format(misSigmaTerms))
        if len(misTruncTerms) != 0:
            raise IOError('Cannot find sigma for these basic terms: {}'.format(misTruncTerms))
        self.basetermpairs = termpairs

        for xterm, xTermCpnt in xterms.items():
            terms = splitXterm(xterm)
            termpairs[xterm] = (xTermCpnt, )
            parms = [xTermCpnt[t] for t in terms]
            for parm in parms:
                for curParm in parm:
                    if curParm != '':
                        searchparms.append(curParm)
                        if curParm not in parmNames:
                            raise KeyError('Cannot find the components for xterm {}'.format(xterm))
        self.termpairs   = termpairs
        self.searchparms = list(set(searchparms))
        return termpairs

    def load_variables(self, jobparm):
        '''get all the search-able parameters into model
        termpairs: all the resist terms and its components, PRS is not supported yet
        jobparm: table of search param name and value
        return a table of search param name and (sigmaVal, truncation level), e.g., 'Ap': 30, 1.2
        '''
        termpairs = self.termpairs
        xterms = [t for t in termpairs.keys() if isXterm(t)]
        baseterms = [t for t in termpairs.keys() if t not in xterms]
        jobinput = {}
        for curTerm in baseterms:
            curSigmaName, curTruncRatioName = termpairs[curTerm]
            curSigmaVal = 0.
            curTruncRatioVal = 0.
            if curSigmaName != "":
                assert(curSigmaName in jobparm.keys())
                curSigmaVal = jobparm[curSigmaName]
            if curTruncRatioName != "":
                assert(curTruncRatioName in jobparm.keys())
                curTruncRatioVal = jobparm[curTruncRatioName]
            jobinput[curTerm] = (curSigmaVal, curTruncRatioVal)
        for xterm in xterms:
            terms = splitXterm(xterm)
            xTermCpnt = termpairs[xterm]
            xtermvals = {t: (jobparm.get(xTermCpnt[0][t][0], 0), jobparm.get(xTermCpnt[0][t][1], 0))   for t in terms}
            jobinput[xterm] = (xtermvals, )
        self.input = jobinput
        return jobinput
