### Instant tuning tool-Genetic algorithm implementation

    1. Initialize the search-able parameters into GA seeds by random algorithm,
    the GA search space and sampling grid is deliberately designed, so all the
    chromosomes can be translated into pre-collected term signals.
    2. Get the RMS value of current generation of chromosomes by Linear solver
    3. GA operations (\#population = N)
        1). elite selection: the elites only will mutate (\#elites = M)
        2). mates selection: select individuals into the mate pool(size of mate
        pool is N-M), 3 selection methods are implemented: rms_ranking,
        linear_ranking, tournament
        3). crossover in the mate pool pair by pair to reproduce next generation
        4). mutate the crossover-ed offspring and the elites, get all the next
        generation of chromosomes
    5. Repeat 2-3 until met stop condition

### Scope

- ADI, HOD0, NXE, 
- EEB, HOD1, HOD2
- PRS, 

It requires to develop different signal extraction flow and different linear solver for the different products.

### Als for this task

1. Implement GA as the optimizer for the instant tuning tool
2. Assuming the python Linear solver already have matched or comparable results with TFlex linear solver, then make sure GA deliver a better model result than TFlex simplex
3. Explore the best practice flow for the instant tuning flow-GA approach 

By 11.14.2016, the 1st A.I is already finished, for the 2nd AI, our first step is to match the result of an real ADI NTD case.
