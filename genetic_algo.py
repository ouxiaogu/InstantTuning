   # -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 14:14:24 2016

@author: peyang

Genetic algorithm in python
    1. Initialize the search-able parameters into GA seeds by random algorithm
    2. Get the RMS value of current generation of chromosomes by Linear solver
    3. Fitness selection include two steps:
        1). elite selection: the elites only will mutate
        2). mates selection: select the individuals into the mate pool, and
        crossover pair by pair, 3 selection methods are implemented:
        rms_ranking, linear_ranking, tournament
    4. Crossover and mutate to reproduce the next generation of chromosomes
    5. Repeat 2-4 until met stop condition
"""

from numpy import random
import math
import copy
import re
import pandas as pd
import numpy as np
import os, os.path
from linear_solver import Solver
from selectors import rms_ranking, linear_ranking, tournament

DEBUG = 1

class GeneticAlgorithm(object):
    """GA class"""
    def __init__(self, chromNum, eliteRate, muteRate, accuracy, stageNum,
                rmsThres, signalpath, selectmethod='rms_ranking', save=True):
        super(GeneticAlgorithm, self).__init__()
        self.chromNum = chromNum
        self.eliteRate = eliteRate
        self.muteRate = muteRate
        self.accuracy = accuracy
        self.stageNum = stageNum
        self.rmsThres = rmsThres
        self.signalpath = signalpath
        self.lm = Solver(signalpath, regressor='linear')
        self.selectmethod = selectmethod
        self.save = save

    def run(self, parmMin, parmMax, parmSamples, termpairs):
        # perform optimization based on GA

        # initialize 1st generation
        random.seed(0)
        stageIdx = 1
        jobparms, newChroms = self.init(parmMin, parmMax, parmSamples, termpairs)
        if DEBUG:
            # self.print_parm_samples()
            print "parmSamples = %s\nparmNames = %s\nparmBound = %s" % (str(parmSamples), str(self.parmNames), str(self.parmBound))
        self.population = Population(wakerAmount=self.chromNum)

        # optimization iterations
        while not self.check_stop(stageIdx):
            newRmses, models = self.optimize(jobparms)
            wakerGen = Generation(newChroms)
            wakerGen.set_gen_property(stageIdx, models, newRmses)
            self.population.append_gen(wakerGen)
            if DEBUG:
                print  "\nGA print, stage %s:\nminRMS = %f\nparmVals = %s\nrmses = %s\n" % (stageIdx, self.population.get_min_rms(), str(jobparms), str(newRmses))
            jobparms, newChroms = self.population.perform_ga_operations(self.eliteRate, self.muteRate, self.parmMin, self.parmMax, self.parmSamples, self.parmNames, self.parmBound, stageIdx, self.selectmethod) # self member variable parmMin~parmbound are lists
            stageIdx += 1

        if self.save:
            self.population.save_model_results(path = os.path.dirname(self.signalpath), sheetname=self.selectmethod)
        pass

    def init(self, parmMin, parmMax, parmSamples, termpairs):
        # initialize the seeds for GA
        self.termpairs = termpairs

        # change the dictionary to list
        parmNames = []
        if isinstance(parmMin, dict):
            parmNames = [kk for kk, vv in parmMin.items()]
            parmMin = [vv for kk, vv in parmMin.items()]
        if isinstance(parmMax, dict):
            parmMax = [vv for kk, vv in parmMax.items()]
        if isinstance(parmSamples, dict):
            parmSamples = [vv for kk, vv in parmSamples.items()]

        # calculate parmBound
        parmBound = self.calculate_boundary(parmSamples)

        # initialize a generation by parmMin, parmMax, parmSamples, parmBound
        chromNum = self.chromNum
        chromosomes = []
        jobparms = [] # all the chromosome vals
        basechrom = Chromosome()
        for ii in range(chromNum):
            chm = basechrom.clone_chrom(nochildren=True)
            chm.rand_init_chrom(parmSamples, parmBound)

            jobparm = chm.back_parm(parmMin, parmMax, parmSamples, parmBound)
            jobparmtable = {parmNames[jj]: jobparm[jj] for jj in range(len(parmNames))}

            chromosomes.append(chm)
            jobparms.append(jobparmtable)

        # stack the list format parmValue into class
        self.parmMin = parmMin
        self.parmMax = parmMax
        self.parmNames = parmNames
        self.parmSamples = parmSamples
        self.parmBound = parmBound
        return jobparms, chromosomes

    def append_chrom(self, parmVals):
        pass

    def calculate_boundary(self, parmSamples):
        # evaluate parm's length in chromosome, determine every parm's boundary
        parmBound = []
        for i in range(len(parmSamples)):
            assert(parmSamples[i] > 1e-7)
            assert('int' in str(type(parmSamples[i])))
            numSamples = parmSamples[i]
            binLength = math.log(numSamples, 2) # idx starts from 0
            binLength = int(math.floor(binLength)+1)
            parmBound.append(binLength)
        return parmBound

    def check_stop(self, stageIdx):
        if self.population.get_min_rms() <= self.rmsThres:
            return True
        if stageIdx > self.stageNum:
            return True
        return False

    def optimize(self, jobparms):
        # fit a model, calculate rms based on current jobparms
        # possible option is add spec-ratio into cost function
        termpairs = self.termpairs
        models = []
        rmses = []
        for i, jobparm in enumerate(jobparms):
            jobinput = covert_job_parm(jobparm, termpairs)
            rms, model = self.lm.fit(jobinput)
            if DEBUG:
                # print "Chromosome %d:\njobparm = %s\njobinput = %s\nrms = %f" % (i, str(jobparm), str(jobinput), rms)
                pass
            rmses.append(rms)
            models.append(model)
        return rmses, models

    def print_parm_samples(self):
        parmNum = len(self.parmNames)
        parmMin = self.parmMin
        parmMax = self.parmMax
        parmSamples = self.parmSamples
        for ii in range(parmNum):
            sampleNum = parmSamples[ii]
            curParmSamples = [parmMin[ii] + 1.0*jj/(sampleNum - 1)*(parmMax[ii] - parmMin[ii]) for jj in range(sampleNum)]
            print "%s: %s" % (parmNames[ii], str(curParmSamples))

class Chromosome(object):
    """Chromosome: based on the parmSamples
        ----
        The chromosome parameters:
            parmChrom:  chrome binary code list
            parmNum:    number of parameters in chromosome

            genIdx:     the generation N.O. of current chrom
            chromIdx:   the chrom N.O. of current chrom
            strChrom:   string format binary code
            model:      {term: sigma, truncation ratio, coefficient}
            rms:        rms of current model
            parentID:   '{$fatherGenIdx}_{$fatherChromIdx}_{$MotherGenIdx}_{$MotherChromIdx}'
            childIDs:   a list, ['{$child1_genIdx}_{$child1_chromIdx}', ...]
    """
    def __init__(self):
        super(Chromosome, self).__init__()
        self.init_chrom_property()

    def init_chrom_property(self, parmChrom=None, parmNum=0,
                                parentID='-1_-1_-1_-1', childIDs=None,
                                genIdx=0, chromIdx=0, model={}, rms=0.,
                                ):
        '''two reasons for this init_chrom_property method:
            1. A whole picture for chm variables
            2. keep consistence for clone_chrom at anytime
        '''
        if parmChrom is None:
            self.parmChrom = []
        else:
            self.parmChrom = parmChrom
        self.parmNum = parmNum
        self.genIdx = genIdx
        self.chromIdx = chromIdx
        self.model = model
        self.rms = rms
        self.parentID = parentID
        if childIDs is None:
            self.childIDs = []
        else:
            self.childIDs = childIDs
        self.str_chrom()

    def set_chrom_result(self, genIdx=0, chromIdx=0, model={}, rms=0.,):
        self.genIdx = genIdx
        self.chromIdx = chromIdx
        self.model = model
        self.rms = rms
        self.str_chrom()

    def rand_init_chrom(self, parmSamples, parmBound):
        """randomly initialize chromosome by bin length and max range of the parameters"""
        parmChrom = []
        for i in range(len(parmBound)):
            curParmVal = random.randint(0, parmSamples[i]) # np.random, low (inclusive) to high (exclusive).
            subChrom = self.dec2bin(curParmVal, parmBound[i])
            parmChrom += subChrom
        self.parmChrom = parmChrom # chromosome sequence
        self.parmNum = len(parmSamples)

    def dec2bin(self, num, length):
        binChar = bin(num)
        binChar = list(binChar[2:]) # 15 -> '0b1111'
        if len(binChar) < length:
            binChar = ['0' for ii in range(length - len(binChar))] + binChar
        binVal = [int(aa) for aa in binChar]
        return binVal

    def clone_chrom(self, nochildren=False):
        newChrom = Chromosome()
        newChrom.parmChrom = copy.deepcopy(self.parmChrom)
        newChrom.parmNum = self.parmNum
        newChrom.genIdx = self.genIdx
        newChrom.chromIdx = self.chromIdx
        newChrom.model = copy.deepcopy(self.model)
        newChrom.rms = self.rms
        newChrom.parentID = self.parentID
        if nochildren == True:
            newChrom.childIDs = []
        else:
            newChrom.childIDs = copy.deepcopy(self.childIDs)
        return newChrom

    def back_parm(self, parmMin, parmMax, parmSamples, parmBound):
        parmChrom = self.parmChrom
        parmVal = []
        acculumatedLen = 0
        for ii, curBinLen in enumerate(parmBound):
            subChrom = parmChrom[acculumatedLen:(acculumatedLen + curBinLen)]
            acculumatedLen += curBinLen
            binVal = 0 # binary number
            for jj, curBite in enumerate(subChrom):
                binVal += curBite*2**(curBinLen - jj - 1)
            curParmVal = parmMin[ii] + 1. * binVal / (parmSamples[ii] - 1) * (parmMax[ii] - parmMin[ii]) # real model parm val
            parmVal.append(curParmVal)
        return parmVal

    def crossover(self, mother, parmIdx, parmBound):
        # crossover self and mother chromosome, at the parm boundary positions
        assert(self.parmNum == mother.parmNum)
        assert(parmIdx < self.parmNum)
        pos = 0
        for ii in range(parmIdx):
            pos += parmBound[ii]
        chmTemp = self.clone_chrom(nochildren=True)
        self.parmChrom = self.parmChrom[:pos] + mother.parmChrom[pos:]
        mother.parmChrom = mother.parmChrom[:pos] + chmTemp.parmChrom[pos:]

    def mutate(self, parmIndexes, parmSamples, parmBound):
        # mutate within itself, by the unit of parm
        # switch parm i into some other value
        parmNum = self.parmNum
        assert(isinstance(parmIndexes, list))
        assert(all([ii < parmNum for ii in parmIndexes]))
        acculumated_len = lambda x,y: int(np.sum(x[0:y]))  # python index [0->y)
        for ii in parmIndexes:
            curParmVal = random.randint(0, parmSamples[ii]) # np.random, low (inclusive) to high (exclusive).
            startPos = acculumated_len(parmBound, ii)
            self.parmChrom[startPos:startPos+parmBound[ii]] = self.dec2bin(curParmVal, parmBound[ii])

    def str_chrom(self):
        strChrom = ""
        for ii in self.parmChrom:
            strChrom += str(ii)
        self.strChrom = strChrom
        return strChrom

class Generation(object):
    """generation: keep all the chromosomes and rmses for 1 generation"""
    def __init__(self, chromosomes=None):
        super(Generation, self).__init__()
        if chromosomes is None:
            self.chromosomes = []
        else:
            self.chromosomes = chromosomes

    def set_gen_property(self, stageIdx, models, rmses):
        for i, chm in enumerate(self.chromosomes):
            chm.set_chrom_result(stageIdx, i, models[i], rmses[i])

class Population(object):
    '''
    New data structure for population, data stored as a class, contain multiple
     fitness selection
    -----
    Parameters:
        chromosomes:
            A list, containing the whole history of entire family
            one individual instance is a chromosome
        wakerAmount: current environment only sustain a constant waker number
    ----
    Methods:
        ----
        Recording Methods:
            get_genealogy
            add_generation
    '''
    def __init__(self, wakerAmount, chromosomes=None):
        super(Population, self).__init__()
        self.wakerAmount = wakerAmount
        self.matepoolNum = wakerAmount
        if chromosomes is None:
            self.chromosomes = []
        else:
            self.chromosomes = chromosomes
        self.totalIndividuals = len(self.chromosomes)
        self.singlepool = range(self.totalIndividuals)

    def append_gen(self, gen):
        # append a generation into current population
        self.chromosomes += gen.chromosomes[:]
        self.totalIndividuals = len(self.chromosomes)
        self.singlepool = range(self.totalIndividuals)
        self.matepoolNum = self.wakerAmount # reinit the matepool size

    def get_min_rms(self):
        rmses = [chm.rms for chm in self.chromosomes]
        if len(rmses) == 0:
            return np.nan
        return np.min(rmses)

    def elite_selection(self, eliteRate=0):
        '''mutation will be performed on the chromosomes who are selected
        into elite pool, but not crossover'''
        assert(0<=eliteRate<1)
        self.elitepool = []
        eliteNum = int(np.ceil(self.wakerAmount*eliteRate))
        rmses = [chm.rms for chm in self.chromosomes]
        eliteIndexes = np.argsort(rmses)[:eliteNum] # model of top N

        self.matepoolNum -= eliteNum
        self.rmses = rmses
        self.elitepool = eliteIndexes[:]
        self.singlepool = list(np.delete(self.singlepool, eliteIndexes))

    def mate_selection(self, method=""):
        '''Both crossover & mutation will be performed on the chromosomes
        who are selected into mate pool'''
        methods = ['rms_ranking', 'linear_ranking', 'tournament']
        assert(method in methods)
        self.matepool = []

        if method == 'rms_ranking':
            self.matepool = rms_ranking(self.rmses, self.singlepool, self.matepoolNum)
        elif method == 'linear_ranking':
            self.matepool = linear_ranking(self.rmses, self.singlepool, self.matepoolNum, sp=1.5)
        elif method == 'tournament':
            self.matepool = tournament(self.rmses, self.singlepool, self.matepoolNum, tourSize=2)

    def perform_ga_operations(self, eliteRate, muteRate, parmMin, parmMax,
        parmSamples, parmNames, parmBound, stageIdx, selectmethod='rms_ranking'):
        '''
        1. elite selection
        2. matepool selection and perform crossover from tail to head
        3. wakerpool = elitepool + matepool, mutate on the waker pool
        '''
        # selection
        self.elite_selection(eliteRate)
        self.mate_selection(selectmethod)

        elitePoolChroms = [self.chromosomes[ii] for ii in self.elitepool]
        matePoolChroms = [self.chromosomes[ii] for ii in self.matepool]
        wakerPoolChroms = [] # wakers or survivors
        for chm in elitePoolChroms:
            wakerPoolChroms.append(chm.clone_chrom(nochildren=True))
        for chm in matePoolChroms:
            wakerPoolChroms.append(chm.clone_chrom(nochildren=True))
        eliteNum = len(elitePoolChroms)
        mateNum = len(matePoolChroms)
        wakerNum = len(wakerPoolChroms)
        if DEBUG:
            print "Stage %d\nElite pool: %s\nMate pool: %s\n" % (stageIdx, str(self.elitepool), str(self.matepool))

        # elite
        if mateNum%2 != 0:
            elitepool = np.append(self.elitepool, self.matepool[0]) # the last single at mate actually is elite
        else:
            elitepool = self.elitepool[:]
        if DEBUG:
            print "Stage %d\nElite pool: %s\nMate pool: %s\n" % (stageIdx, elitepool, str(self.matepool))

        for ii, parentIdx in enumerate(elitepool):
            wakerPoolChroms[ii].parentID =  '{}_{}_{}_{}'.format(parentIdx/wakerNum+1, parentIdx%wakerNum, parentIdx/wakerNum+1, parentIdx%wakerNum)
            self.chromosomes[parentIdx].childIDs.append('{}_{}'.format(stageIdx+1, ii))
            # ypcLog, hard-coded testing
            print "parentIdx %d, childIDs = %s" % (parentIdx, str(self.chromosomes[parentIdx].childIDs))

        # crossover
        parmNum = len(parmSamples) ## perform GA operator on unit of parm
        chromIdx = wakerNum - 1
        while (chromIdx > eliteNum):
            print "chromIdx = %d" % chromIdx

            mateIdx = chromIdx - eliteNum
            fatherIdx = self.matepool[mateIdx]
            motherIdx = self.matepool[mateIdx-1]
            parmIdx = random.randint(0, parmNum) # np.random, low (inclusive) to high (exclusive).

            wakerPoolChroms[chromIdx].crossover(wakerPoolChroms[chromIdx-1], parmIdx, parmBound)

            for ii in [chromIdx, chromIdx-1]:
                wakerPoolChroms[ii].parentID = '{}_{}_{}_{}'.format(fatherIdx/wakerNum+1, fatherIdx%wakerNum, motherIdx/wakerNum, motherIdx%wakerNum)
                self.chromosomes[fatherIdx].childIDs.append('{}_{}'.format(stageIdx+1, ii))
                self.chromosomes[motherIdx].childIDs.append('{}_{}'.format(stageIdx+1, ii))

            chromIdx -= 2

        # mutation
        for ii in range(wakerNum):
            muteCri = ii * muteRate # binary
            parmIndexes = [pp for pp in range(parmNum) if random.random() < muteCri] #np, half-open interval [0.0, 1.0).
            wakerPoolChroms[ii].mutate(parmIndexes, parmSamples, parmBound)
            if DEBUG:
                print 'chrom %d: %s' % (ii, wakerPoolChroms[ii].str_chrom())

        # translation: chromosome to job param
        jobparms = []
        for chm in wakerPoolChroms:
            jobparm = chm.back_parm(parmMin, parmMax, parmSamples, parmBound)
            jobparmtable = {parmNames[jj]: jobparm[jj] for jj in range(len(parmNames))}
            jobparms.append(jobparmtable)
        return jobparms, wakerPoolChroms

    def save_model_results(self, path=os.getcwd(), sheetname='sheet1'):
        keys = ['genIdx', 'chromIdx', 'strChrom', 'rms', 'model', 'parentID', 'childIDs'] # hard-coded
        fixedTruncTerms = {"Ap": "b0_ratio", "Bp": "b0_ratio", "Am": "b0m_ratio", "Bn": "b0n_ratio"}# hard-coded
        df = pd.DataFrame()
        for chm in self.chromosomes:
            dictChrom = chm.__dict__
            result = {}
            coeffs = []
            sigmas = []
            ratios = []
            for key in keys:
                if key in dictChrom.keys():
                    if key == 'model':
                        for termname, termvals in dictChrom['model'].items():
                            sigma, truncRatio, coeff = termvals
                            result['c{}'.format(termname)] = coeff
                            coeffs.append('c{}'.format(termname))
                            if sigma != 0:
                                result['sigma{}'.format(termname)] = sigma
                                sigmas.append('sigma{}'.format(termname))
                            ratioName = ''
                            if truncRatio != 0:
                                if termname in fixedTruncTerms.keys():
                                    ratioName = fixedTruncTerms[termname]
                                else:
                                    ratioName = 'b0_{}'.format(termname)
                            if (ratioName != '') and (ratioName not in result.keys()):
                                result[ratioName] = truncRatio
                                ratios.append(ratioName)
                    elif key == 'childIDs':
                        result[key] = str(dictChrom[key])
                    else:
                        result[key] = dictChrom[key]
            # http://stackoverflow.com/questions/17839973/construct-pandas-dataframe-from-values-in-variables
            curdf = pd.DataFrame(result, index=[0])
            df = df.append(curdf)
        df.index = range(len(df))
        keys = keys[:] + ratios[:] + coeffs[:] + sigmas[:]
        keys.remove('model')
        df = df[keys]
        df.to_excel(os.path.join(path, 'model_results.xlsx'),sheet_name=sheetname)

def gen_term_pairs(selectedTerms, parmNames):
    # return a table of search param name and (sigmaVal, truncation level), e.g., 'Ap': sigmaAp, b0_ratio
    termpairs = {}
    noneSigmaTerm = {"A", "Mav", "Slope"}

    sigmaParms = [aa for aa in parmNames if re.match(r'sigma[\S]*', aa) and (not re.match(r'sigma2D[\S]*', aa))]
    truncRatioParms = [aa for aa in parmNames if re.match(r'b[0-9]*[-_AaBbPp0-9]*', aa)]

    fixedTruncTerms = {"Ap": "b0_ratio", "Bp": "b0_ratio", "Am": "b0m_ratio", "Bn": "b0n_ratio"}

    for curTerm in selectedTerms:
        curSigma = ""
        curTruncRatio = ""
        if curTerm not in noneSigmaTerm:
            if curTerm in fixedTruncTerms.keys():
                curSigma = r'sigma{}'.format(curTerm)
                curTruncRatio = fixedTruncTerms[curTerm]
            else:
                for sigmaName in sigmaParms:
                    if curTerm in sigmaName:
                        curSigma = sigmaName
                        break
                for truncRatio in truncRatioParms:
                    if curTerm in truncRatio:
                        curTruncRatio = truncRatio
                        break
        termpairs[curTerm] = (curSigma, curTruncRatio)
    return termpairs

def covert_job_parm(jobparm, termpairs):
    '''
    termpairs: all the resist terms and its components, PRS is not supported yet
    jobparm: table of search param name and value
    return a table of search param name and (sigmaVal, truncation level), e.g., 'Ap': 30, 1.2
    '''
    jobinput = {}
    for curTerm, val in termpairs.items():
        curSigmaName, curTruncRatioName = val
        curSigmaVal = 0.
        curTruncRatioVal = 0.
        if curSigmaName != "":
            assert(curSigmaName in jobparm.keys())
            curSigmaVal = jobparm[curSigmaName]
        if curTruncRatioName != "":
            assert(curTruncRatioName in jobparm.keys())
            curTruncRatioVal = jobparm[curTruncRatioName]
        jobinput[curTerm] = (curSigmaVal, curTruncRatioVal)
    return jobinput

if __name__ == "__main__":
    cwd = os.getcwd()
    datapth = os.path.join(cwd, "data\\v0")
    infile = os.path.join(datapth, "GA_init_setting_v0.xlsx")
    df = pd.read_excel(infile)

    parmMin = df[['parmName', 'parmMin']].set_index('parmName').to_dict().values()[0]
    parmMax = df[['parmName', 'parmMax']].set_index('parmName').to_dict().values()[0]
    parmSamples = df[['parmName', 'parmSamples']].set_index('parmName').to_dict().values()[0]
    parmNames = parmSamples.keys()

    useterms = ['AG1', 'AG2', 'Ap', 'Bp', 'Am', 'Bn', 'MG1', 'Slope']
    termpairs = gen_term_pairs(useterms, parmNames)
    print "used terms and their parms: %s" % str(termpairs)

    signalfile = os.path.join(datapth, 'gauge_params_signals.txt')

    for method in ['tournament']: # 'rms_ranking', 'linear_ranking', 'tournament'
    # for method in ['rms_ranking', 'linear_ranking', 'tournament']:
        ga = GeneticAlgorithm(chromNum=3, eliteRate=0.02, muteRate=0.08, accuracy=0.8,
                        stageNum=3, rmsThres = 0.3, signalpath=signalfile, selectmethod=method)
        ga.run(parmMin, parmMax, parmSamples, termpairs)