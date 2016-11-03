# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:22:34 2016

@author: peyang

----
Fitness Selection function:
    1. rms ranking
    2. linear ranking
    3. tournament
    4. roulette wheel selection
----
Note:
    tagetSize: target size of the mate pool
"""
import numpy as np
from numpy import random

def rms_ranking(rmses, singlepool, tagetSize):
    assert(len(singlepool) >= tagetSize)
    singlermses = [rmses[ii] for ii in singlepool]
    ascendingIndexes  = np.argsort(singlermses)
    selected = [singlepool[ii] for ii in ascendingIndexes[:tagetSize]]
    return list(selected)

def linear_ranking(rmses, singlepool, tagetSize, sp=1.5):
    '''
    Linear ranking selection without replacement.
    In rank-based fitness assignment, the population is sorted by fitness.
    Consider 'N' as the number of individuals in the population, 'Pos' the
    position of an individual in this population (least fit individual
    has Pos=0, the fittest individual Pos=N-1) and 'SP' the selective pressure.

    ----
    parameters:
        tagetSize:  int, target number for a full mate pool
        sp:         selection pressure, prob of selecting best individual
                    compared to the average selection prob

    ----
    fitness function:
        fitness = (2 - sp) + 2.*(sp-1)*Pos/(N-1)
    '''
    assert(len(singlepool) >= tagetSize)
    singleNum = len(singlepool)
    singlermses = [rmses[ii] for ii in singlepool]
    ascendingIndexes  = np.argsort(singlermses)
    descendingIndexes = ascendingIndexes[::-1]

    sp = 1.5 # hard-coded, 1<sp<2, selection pressure
    fitness = lambda x: 2 - sp + 2.*(sp-1)*x/(singleNum-1)

    props = [0. for ii in range(singleNum)]
    for ii, jj in enumerate(descendingIndexes):
        props[jj] = fitness(ii) # the probability in loc jj is by a portion of ii
    sumFitness = np.sum(props)
    props = [1.*x/sumFitness for x in props]

    draw = random.choice(singlepool, tagetSize, p=props, replace=False)
    return list(draw)

def tournament(rmses, singlepool, tagetSize, tourSize = 2):
    '''
    Tournament selection without replacement.
    In tournament selection a number Tour of individuals is chosen randomly
    from the population and the best individual from this group is selected
    as parent. This process is repeated until the mate pool is full.

    Note
    ----
        singlepool: list, the indexes of current available single pool
        tagetSize:  int, target number for a full mate pool
        tourSize:   int, how many candidates are chosen into tournament
    '''
    assert(len(singlepool) >= tagetSize)
    if tagetSize == 0:
        return []
    if len(singlepool) < tourSize:
        return singlepool
    draw = random.choice(singlepool, tourSize, replace=False)
    drawRmses = [rmses[ii] for ii in draw]
    winneNo = np.argsort(drawRmses)[0]
    winner = draw[winneNo]

    # append winnerIdx into mate pool, and delete it from single pool
    try:
        winnerIdx = singlepool.index(winner)
        singlepool = list(np.delete(singlepool, winnerIdx))
    except ValueError:
        raise ValueError('The winner {} is not in pre-defined list: {}'.fromat(winner, singlepool))
    return [winner] + tournament(rmses, singlepool, tagetSize-1, tourSize)[:]

if __name__ == '__main__':
    random.seed(0)
    rmses = [round(random.random(), 3) for i in range(10)]
    singleNum = len(rmses)

    print 'rms ranking ...'
    for i in range(11):
        print rms_ranking(rmses, range(singleNum), i)

    print '\nlinear ranking ...'
    for i in range(11):
        random.seed(0)
        print linear_ranking(rmses, range(singleNum), i, 1.5)

    print '\ntournament ranking ...'
    for i in range(11):
        random.seed(0)
        print tournament(rmses, range(singleNum), i, 2)