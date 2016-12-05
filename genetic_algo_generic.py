   # -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 14:14:24 2016

@author: peyang

Instant tuning tool-Genetic algorithm implementation
    1. Initialize the search-able parameters into GA seeds by random algorithm,
    the GA search space and sampling grids are deliberately designed, so all the
    chromosomes can be translated into pre-collected term signals.
    2. Get the RMS value of current generation of chromosomes by Linear solver
    3. GA operations (#population = N)
        1). elite selection: the elites only will mutate (#elites = M)
        2). mates selection: select individuals into the mate pool(size of mate
        pool is N-M), 3 selection methods are implemented: rms_ranking,
        linear_ranking, tournament
        3). crossover in the mate pool pair by pair to reproduce next generation
        4). mutate the crossover-ed offspring and the elites, get all the next
        generation of chromosomes
    4. Repeat 2-3 until met stop condition
"""

from numpy import random
import math
import copy
import re
import pandas as pd
import numpy as np
import os, os.path
from linear_solver_olddata import Solver
from selectors import rms_ranking, linear_ranking, tournament
from model import Model

DEBUG = 0

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
        self.lm = Solver(signalpath,  modelkind='ADI')
        self.selectmethod = selectmethod
        self.save = save

    def run(self, parmMin, parmMax, parmSamples, model):
        # perform optimization based on GA

        # initialize 1st generation
        random.seed(0)
        stageIdx = 1
        jobparms, newChroms = self.init(parmMin, parmMax, parmSamples)
        if DEBUG:
            # self.print_parm_samples()
            print "parmSamples = %s\nparmNames = %s\nparmBound = %s" % (str(parmSamples), str(self.parmNames), str(self.parmBound))
        self.population = Population(wakerAmount=self.chromNum)

        # optimization iterations
        while not self.check_stop(stageIdx):
            self.wakerGen = Generation(newChroms)
            self.optimize(model, jobparms, stageIdx)
            self.population.append_gen(self.wakerGen)
            if self.save:
                self.population.save_model_results(jobparms, path = self.signalpath, postfix=self.selectmethod)
            jobparms, newChroms = self.population.perform_ga_operations(self.eliteRate, self.muteRate, self.parmMin, self.parmMax, self.parmSamples, self.parmNames, self.parmBound, stageIdx, self.selectmethod) # self member variable parmMin~parmbound are lists
            stageIdx += 1

    def init(self, parmMin, parmMax, parmSamples):
        # initialize the seeds for GA

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

    def optimize(self, model, jobparms, stageIdx):
        # fit a model, calculate rms based on current jobparms
        # possible option is add spec-ratio into cost function

        for i, jobparm in enumerate(jobparms):
            jobinput = model.load_variables(jobparm)
            rms, outmodel, calresult, groupstat = self.lm.fit(jobinput, method='nonlinear')
            self.wakerGen.chromosomes[i].set_chrom_result(stageIdx, i, rms, outmodel, calresult, groupstat)
            if DEBUG:
                print "Chromosome %d:\njobparm = %s\njobinput = %s\nrms = %f" % (i, str(jobparm), str(jobinput), rms)
                pass

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

            parentID:   '{$fatherGenIdx}_{$fatherChromIdx}_{$MotherGenIdx}_{$MotherChromIdx}'
            childIDs:   a list, ['{$child1_genIdx}_{$child1_chromIdx}', ...]

            rms:        rms of current model
            model:      {term: sigma, truncation ratio, coefficient}
            calresult:  {'3sigma':sigma3, 'total':total, 'Anchor_error':anchor,
                        '2D_range':range2d, '1D_range':range1d, 'Unweighted_RMS':..}
            groupstat:  DataFrame, statistic info for groups, rms, type, min, max

    """
    def __init__(self):
        super(Chromosome, self).__init__()
        self.init_chrom_property()

    def init_chrom_property(self, parmChrom=None, parmNum=0, parentID='-1_-1_-1_-1', childIDs=None):
        # only need to initialize the intrinsic property and list members
        if parmChrom is None:
            parmChrom = []
        self.parmChrom = parmChrom
        self.parmNum = parmNum
        self.str_chrom()
        self.parentID = parentID
        if childIDs is None:
            childIDs = []
        self.childIDs = childIDs

    def set_chrom_result(self, genIdx=0, chromIdx=0, rms=0., model=None, calresult=None, groupstat=None):
        self.genIdx = genIdx
        self.chromIdx = chromIdx
        self.rms = rms
        if model is None:
            model = {}
        self.model = model
        if calresult is None:
            calresult = {}
        self.calresult = calresult
        if groupstat is None:
            self.groupstat = pd.DataFrame()
        else:
            self.groupstat = groupstat
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
        self.str_chrom()

    def dec2bin(self, num, length):
        binChar = bin(num)
        binChar = list(binChar[2:]) # 15 -> '0b1111'
        if len(binChar) < length:
            binChar = ['0' for ii in range(length - len(binChar))] + binChar
        binVal = [int(aa) for aa in binChar]
        return binVal

    def clone_chrom(self, nochildren=False):
        # only need to clone the intrinsic information of a chromosome
        newChrom = Chromosome()
        newChrom.parmChrom = copy.deepcopy(self.parmChrom)
        newChrom.parmNum = self.parmNum
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
            chromosomes = []
        self.chromosomes = chromosomes

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
    '''
    def __init__(self, wakerAmount, chromosomes=None):
        super(Population, self).__init__()
        self.wakerAmount = wakerAmount
        self.matepoolNum = wakerAmount
        if chromosomes is None:
            chromosomes = []
        self.chromosomes = chromosomes

    def append_gen(self, gen):
        # append a generation into current population
        self.chromosomes += gen.chromosomes[:]
        self.matepoolNum = self.wakerAmount # reinit the matepool size

    def get_min_rms(self):
        rmses = [chm.rms for chm in self.chromosomes]
        if len(rmses) == 0:
            return np.nan
        return np.min(rmses)

    def elite_selection(self, eliteRate=0):
        '''mutation will be performed on the chromosomes who are selected
        into elite pool, but not crossover;
        All the individuals in elite and single pool are unique to each other
        '''
        assert(0<=eliteRate<1)
        eliteNum = int(np.ceil(self.wakerAmount*eliteRate))
        rmses = [chm.rms for chm in self.chromosomes]
        unique_rmses, unique_indices = np.unique(rmses, return_index=True)
        eliteIndexes = np.argsort(rmses)[:eliteNum] # model of top N
        eliteIndexes = [unique_indices[ii] for ii in eliteIndexes]

        self.matepoolNum -= eliteNum
        self.rmses = rmses
        self.elitepool = eliteIndexes[:]
        self.singlepool = list(np.delete(unique_indices, eliteIndexes))

    def mate_selection(self, method=""):
        '''Both crossover & mutation will be performed on the chromosomes
        who are selected into mate pool'''
        methods = ['rms_ranking', 'linear_ranking', 'tournament']
        assert(method in methods)
        self.matepool = []

        if method == 'rms_ranking':
            self.matepool = rms_ranking(self.rmses, self.singlepool, self.matepoolNum)
        elif method == 'linear_ranking':
            self.matepool = linear_ranking(self.rmses, self.singlepool, self.matepoolNum, sp=1.8)
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

        # crossover
        parmNum = len(parmSamples) ## perform GA operator on unit of parm
        chromIdx = wakerNum - 1
        while (chromIdx > eliteNum):
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
        sp = 5
        muteRatioFunc = lambda x: 7 - sp + 2.*(sp-2)*x/(wakerNum-1) #fittest first
        muteRatios = [muteRatioFunc(ii) for ii in range(wakerNum)]
        # print muteRatios
        for ii in range(wakerNum):
            muteCri = muteRatios[ii] * muteRate # ypcChange, ensure some mutation
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

    def save_model_results(self, jobparms, path=os.getcwd(), postfix=''):
        calresultfile = 'model_results'
        groupstatfile = 'group_results'
        if postfix != '':
            calresultfile += '_{}'.format(postfix)
            groupstatfile += '_{}'.format(postfix)
        calresultfile = os.path.join(path, calresultfile+'.xlsx')
        groupstatfile = os.path.join(path, groupstatfile+'.xlsx')
        if len(self.chromosomes) > self.wakerAmount:
            calresults = pd.read_excel(calresultfile)
            groupstats = pd.read_excel(groupstatfile)
        else:
            calresults = pd.DataFrame()
            groupstats = pd.DataFrame()
        stagecalresults = pd.DataFrame()
        stagegroupstats = pd.DataFrame()

        chromKeys = ['genIdx', 'chromIdx', 'strChrom', 'rms',  'parentID', 'childIDs', 'model', 'calresult', 'groupstat'] # hard-coded

        for ii in range(len(self.chromosomes) - self.wakerAmount, len(self.chromosomes)):
            dictChrom = self.chromosomes[ii].__dict__
            modelresult = {}
            for key in chromKeys:
                if key in dictChrom.keys():
                    if key == 'model':
                        for termname, termvals in dictChrom['model'].items():
                            if termname == 'thres':
                                modelresult['thres'] = termvals
                            else:
                                coeff = termvals[-1]
                                modelresult['c{}'.format(termname)] = coeff
                        for curParm, curVal in jobparms[ii%(self.wakerAmount)].items():
                            modelresult[curParm] = curVal
                    elif key == 'childIDs':
                        modelresult[key] = str(dictChrom[key])
                    elif key == 'calresult':
                        for kk, vv in dictChrom[key].items():
                            modelresult[kk] = vv
                    elif key == 'groupstat':
                        groupstat = dictChrom[key]
                        groupstat.ix[:, 'ProcessID'] = ii
                        stagegroupstats = stagegroupstats.append(groupstat)
                    else:
                        modelresult[key] = dictChrom[key]
            # http://stackoverflow.com/questions/17839973/construct-pandas-dataframe-from-values-in-variables
            curcalresults = pd.DataFrame(modelresult, index=[0])
            stagecalresults = stagecalresults.append(curcalresults)
        stagegroupstats.ix[:, 'group'] = stagegroupstats.index
        stagegroupstats.set_index(keys = 'ProcessID', inplace=True)
        keys = [kk for kk in chromKeys if kk not in ['model', 'calresult', 'groupstat']]
        rstkeys = [kk for kk in sorted(modelresult.keys()) if kk not in keys]
        keys = keys + rstkeys
        stagecalresults = stagecalresults[keys]
        calresults = calresults.append(stagecalresults)
        groupstats = groupstats.append(stagegroupstats)
        calresults.ix[:, 'childIDs'] = [str(chm.childIDs) for chm in self.chromosomes]
        calresults.index = range(len(calresults))
        calresults.to_excel(calresultfile)
        groupstats.to_excel(groupstatfile)

if __name__ == "__main__":
    cwd = os.getcwd()
    datapath = os.path.join(cwd, "data\\ADI\\all")
    infile = os.path.join(datapath, "GA_init_setting.xlsx")
    df = pd.read_excel(infile)

    parmMin = df[['parmName', 'parmMin']].set_index('parmName').to_dict().values()[0]
    parmMax = df[['parmName', 'parmMax']].set_index('parmName').to_dict().values()[0]
    parmSamples = df[['parmName', 'parmSamples']].set_index('parmName').to_dict().values()[0]
    parmNames = parmSamples.keys()

    useterms = ['AG1', 'Bp', 'Am', 'Bn', 'Slope', 'DE1', 'CDO1', 'MGS1', 'MGS']
    crossterms = {'Bp*Bn':     {'Bp': ('sigma_Bp', 'b0_Bp'), 'Bn': ('sigma_Bn', 'b0_Bn')},
                  'DE1*DE1':   {'DE1': ('sigma_DE1', '') },
                  'Slope*Bn':  {'Slope': ('', ''), 'Bn': ('sigma_Bn', 'b0_Bn')},
    }
    model = Model(useterms, crossterms)
    termpairs = model.gen_term_pairs(parmNames) # all models share the same template structure
    print "used terms and their parms: %s" % str(termpairs)

    import timeit
    start = timeit.default_timer()
    for method in ['rms_ranking']: # 'rms_ranking', 'linear_ranking', 'tournament'
    # for method in ['linear_ranking', 'tournament', 'rms_ranking']: # 'rms_ranking',
        ga = GeneticAlgorithm(chromNum=50, eliteRate=0.02, muteRate=0.08, accuracy=0.8,
                        stageNum=40, rmsThres = 0.3, signalpath=datapath, selectmethod=method)
        ga.run(parmMin, parmMax, parmSamples, model)
    end = timeit.default_timer()
    print 'elapsed time: {}'.format(end - start)