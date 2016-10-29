   # -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 14:14:24 2016

@author: peyang

Genetic algorithm in python
    1. Initialize the search-able parameters into GA seeds by random algorithm
    2. Get the RMS value of current generation of chromosomes by Linear solver
    3. Fitness selection by RMS, only best chromNum models survive
    4. GA operation, crossover and mutate to reproduce the next generation of chromosomes
    5. Repeat 2-4 until met stop condition

Need to integration:
    1. linear solver in python
        i). derive the signals of current GA seeds
        ii). get the Rmses & coefficients by python linear solver
    2. Search-parameter-specific steps, i.e., the accuracy in python GA should be a list
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
    def __init__(self, chromNum, muteRate, accuracy,
                        stageNum, rmsThres, signalpath):
        super(GeneticAlgorithm, self).__init__()
        self.chromNum = chromNum
        self.muteRate = muteRate
        self.accuracy = accuracy
        self.stageNum = stageNum
        self.rmsThres = rmsThres
        self.signalpath = signalpath
        self.lm = Solver(signalpath, regressor='linear')

    def run(self, parmMin, parmMax, parmSamples, termpairs):
        # perform optimization based on GA

        # initialize 1st generation
        stageIdx = 1
        jobparms, initGen = self.init(parmMin, parmMax, parmSamples, termpairs)
        if DEBUG:
            # self.print_parm_samples()
            print "parmSamples = %s\nparmNames = %s\nparmBound = %s" % (str(parmSamples), str(self.parmNames), str(self.parmBound))
        newRmses, models = self.optimize(jobparms)
        initGen.set_chrom_result(stageIdx, models, newRmses)
        self.population = Population(wakerAmount=self.chromNum)
        self.population.chromosomes += initGen.chromosomes[:]

        # optimization iterations
        while not self.check_stop(stageIdx):
            if DEBUG:
                print  "\nGA print, stage %s:\nminRMS = %f\nparmVals = %s\nrmses = %s\n" % (stageIdx, self.population.minRMS, str(jobparms), str(newRmses))

            self.population.fitness_selection(method='')

            stageIdx += 1
            jobparms, newChroms = self.population.perform_ga_operations(self.muteRate, self.parmMin, self.parmMax, self.parmSamples, self.parmNames, self.parmBound) # self memeber variable parmMin~parmbound are lists
            newRmses, models = self.optimize(jobparms)
            self.population.survivorGen.set_chrom_result(stageIdx, models, newRmses)
            self.population.chromosomes += self.population.survivorGen.chromosomes[:]
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
        random.seed(0)
        for ii in range(chromNum):
            chm = Chromosome()
            chm = chm.rand_init_chrom(parmSamples, parmBound)

            jobparm = chm.back_parm(parmMin, parmMax, parmSamples, parmBound)
            jobparmtable = {parmNames[jj]: jobparm[jj] for jj in range(len(parmNames))}

            chromosomes.append(chm)
            jobparms.append(jobparmtable)

        initGen = Generation(chromosomes)

        # stack the list format parmValue into class
        self.parmMin = parmMin
        self.parmMax = parmMax
        self.parmBound = parmBound
        self.parmNames = parmNames
        self.parmSamples = parmSamples
        return jobparms, initGen

    def append_chrom(self, parmVals):
        pass

    def calculate_boundary(self, parmSamples):
        # evaluate parm's length in chromosome, determine every parm's boundary
        parmBound = []
        for i in range(len(parmSamples)):
            assert(parmSamples[i] > 1e-7)
            assert('int' in str(type(parmSamples[i])))
            numSamples = parmSamples[i]
            binLength = math.log(numSamples-1, 2) # idx starts from 0
            binLength = int(math.floor(binLength)+1)     # binary
            parmBound.append(binLength)
        return parmBound

    def check_stop(self, stageIdx):
        if self.population.cal_min_rms() <= self.rmsThres:
            # print stageIdx, 'rms', self.population.minRMS, self.rmsThres
            return True
        if stageIdx > self.stageNum:
            return True
        return False

    def optimize(self, jobparms):
        # fit a model, calculate rms based on current jobparms
        # possible option is add spec-ratio into cost function
        termpairs = self.termpairs
        models = []
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

    def init_chrom_property(self, parmChrom=[], parmNum=0,
                                parentID='-1_-1_-1_-1', childIDs=[],
                                genIdx=0, chromIdx=0, model={}, rms=0.,
                                ):
        '''two reasons for this init_chrom_property method:
            1. A whole picture for chm variables
            2. keep consistence for clone_chrom at anytime
        '''
        self.parmChrom = parmChrom
        self.parmNum = parmNum
        self.genIdx = genIdx
        self.chromIdx = chromIdx
        self.model = model
        self.rms = rms
        self.parentID = parentID
        self.childIDs = childIDs

    def set_chrom_result(self, genIdx=0, chromIdx=0, model={}, rms=0.,):
        self.genIdx = genIdx
        self.chromIdx = chromIdx
        self.model = model
        self.rms = rms

    def rand_init_chrom(self, parmSamples, parmBound):
        """randomly initialize chromosome by bin length and max range of the parameters"""
        parmChrom = []
        for i in range(len(parmBound)):
            curParmVal = random.randint(0, parmSamples[i]) # np.random, low (inclusive) to high (exclusive).
            subChrom = self.dec2bin(curParmVal, parmBound[i])
            parmChrom += subChrom
        self.parmChrom = parmChrom # chromosome sequence
        self.parmNum = len(parmSamples)
        return self

    def dec2bin(self, num, length):
        binChar = bin(num)
        binChar = list(binChar[2:]) # 15 -> '0b1111'
        if len(binChar) < length:
            binChar = ['0' for ii in range(length - len(binChar))] + binChar
        binVal = [int(aa) for aa in binChar]
        return binVal

    def clone_chrom(self):
        newChrom = Chromosome()
        newChrom.parmChrom = copy.deepcopy(self.parmChrom)
        newChrom.parmNum = self.parmNum
        newChrom.genIdx = self.genIdx
        newChrom.chromIdx = self.chromIdx
        newChrom.model = copy.deepcopy(self.model)
        newChrom.rms = self.rms
        newChrom.parentID = self.parentID
        newChrom.childIDs = copy.deepcopy(childIDs)
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
        chmTemp = self.clone_chrom()
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
    def __init__(self, chromosomes=[]):
        super(Generation, self).__init__()
        self.chromosomes = chromosomes

    def set_gen_property(self, stageIdx, models, rmses):
        for i, chm in enumerate(self.chromosomes):
            chm.set_chrom_result(stageIdx, i, models[i], rmses[i])

    def empty_gen(self):
        if len(self.chromosomes) > 0:
            return False
        else:
            return True

    def clone_gen(self):
        newChroms = [ii.clone_chrom() for ii in self.chromosomes]
        newGen = Generation(newChroms)
        return newGen

    def sort_chroms(self):
        chromNum = len(self.chromosomes)
        for ii in range(0, chromNum-1):
            for jj in range(ii, chromNum):
                if self.chromosomes[ii].rms > self.chromosomes[jj].rms:
                    temp = self.chromosomes[ii].clone_chrom()
                    self.chromosomes[ii] = self.chromosomes[jj].clone_chrom()
                    self.chromosomes[jj] = temp
        pass

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
    def __init__(self, wakerAmount, chromosomes=[]):
        super(Population, self).__init__()
        self.wakerAmount = wakerAmount
        self.matepoolNum = wakerAmount
        self.chromosomes = chromosomes
        self.totalIndividuals = len(self.chromosomes)
        self.singlepool = range(self.totalIndividuals)

    def append_gen(self, gen):
        # append a generation into current population
        self.chromosomes += gen.chromosomes[:]
        self.totalIndividuals = len(self.chromosomes)
        self.singlepool = range(self.totalIndividuals)

    def elite_selection(self, eliteRate=0):
        assert(0<=eliteRate<1)
        eliteNum = int(np.ceil(self.totalIndividuals*eliteRate))
        rmses = [chm.rms for chm in self.chromosomes]
        eliteIndexes = np.argsort(rmses)[:eliteNum] # model of top N

        self.matepoolNum -= len(elites)
        self.rmses = rmses
        self.elitepool = eliteIndexes[:]
        self.singlepool = list(np.delete(self.singlepool, eliteIndexes))

    def matepool_selection(self, method=""):
        '''
        fitness selection is similar with merge sort
        1st generation survivors always are the initGen
        2nd and further generation survivors
            - useChildGen=True, use child generation to perform GA operation
            - useChildGen=False, then use Best N models to perform GA generation
        '''
        methods = ['rms_ranking', 'linear_ranking', 'tournament']
        assert(method in methods)
        self.matepool = []

        if method == 'rms_ranking':
            self.matepool = rms_ranking(self.rmses, self.singlepool, self.matepoolNum)
        elif method == 'linear_ranking':
            self.matepool = linear_ranking(self.rmses, self.singlepool, self.matepoolNum, sp=1.5)
        elif method == 'tournament':
            self.matepool = tournament(self.rmses, self.singlepool, self.matepoolNum, tourSize=2)

    def perform_ga_operations(self, eliteRate, muteRate, parmMin, parmMax, parmSamples, parmNames, parmBound, selMethod='rms_ranking'):
        '''
        1. elite selection
        2. matepool selection and perform crossover from tail to head
        3. wakerpool = elite + matepool, mutate on the waker pool
        '''
        # selection
        self.elite_selection(eliteRate)
        self.matepool_selection(selMethod)
        matePoolChroms = [self.chromosomes[ii] for ii in self.matepool]

        # crossover
        childrenChroms = matePoolChroms[:]
        parmNum = len(parmSamples) ## perform GA operator on unit of parm
        chromNum = len(childrenChroms)
        chromIdx = chromNum - 1
        while (chromIdx > 1):
            parmIdx = random.randint(0, parmNum) # np.random, low (inclusive) to high (exclusive).
            childrenChroms[chromIdx].crossover(childrenChroms[chromIdx-1], parmIdx, parmBound)
            chromIdx -= 2
        --- 08.29 10/28/2016
        # mutation
        for ii in range(chromNum):
            muteCri = ii * muteRate # binary
            parmIndexes = [pp for pp in range(parmNum) if random.random() < muteCri] #np, half-open interval [0.0, 1.0).
            childGen.chromosomes[ii].mutate(parmIndexes, parmSamples, parmBound)
            if DEBUG:
                print 'chrom %d: %s' % (ii, childGen.chromosomes[ii].str_chrom())

        # translate the new parmChrom into job parameters
        # newChroms = self.survivorGen.chromosomes ## list assignment don't work
        jobparms = []
        for curChrom in childGen.chromosomes:
            jobparm = curChrom.back_parm(parmMin, parmMax, parmSamples, parmBound)
            jobparmtable = {parmNames[jj]: jobparm[jj] for jj in range(len(parmNames))}
            jobparms.append(jobparmtable)
        return jobparms, childGen.chromosomes

def gen_term_pairs(selectedTerms, parmNames):
    # return a table of search param name and (sigmaVal, truncation level), e.g., 'Ap': sigmaAp, b0_ratio
    termpairs = {}
    noneSigmaTerm = {"A", "Mav", "Slope"}

    sigmaParms = [aa for aa in parmNames if re.match(r'sigma[\S]*', aa) and (not re.match(r'sigma2D[\S]*', aa))]
    truncRatioParms = [aa for aa in parmNames if re.match(r'b[0-9]*[-_AaBbPp0-9]*', aa)]

    fixedTermPairs = {"Ap": "b0_ratio", "Bp": "b0_ratio", "Am": "b0m_ratio", "Bn": "b0n_ratio"}

    for curTerm in selectedTerms:
        curSigma = ""
        curTruncRatio = ""
        if curTerm not in noneSigmaTerm:
            if curTerm in fixedTermPairs.keys():
                curSigma = r'sigma{}'.format(curTerm)
                curTruncRatio = fixedTermPairs[curTerm]
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
    ga = GeneticAlgorithm(chromNum=5, muteRate=0.08, accuracy=0.8,
                        stageNum=3, rmsThres = 0.3, signalpath=signalfile)
    ga.run(parmMin, parmMax, parmSamples, termpairs)