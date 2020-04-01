#!/usr/bin/env python
# coding: utf-8

# # Symbolic Regression on Sin-Gordan data

# In[41]:

import geppy as gep
from deap import creator, base, tools
import numpy as np
import random

import operator 
import math
import datetime

import os
import pandas as pd

#doublecheck the data is there
print(os.listdir("./../../../data_gen/clean/data/."))

# read in the data to pandas
singData = pd.read_csv("./../../../data_gen/clean/data/sin_gordan_data.csv",  encoding='utf-8')      

# In[46]:

msk = np.random.rand(len(singData)) < 0.8
train = singData[msk]
holdout = singData[~msk]

holdout.describe()
train.describe()

u          = train.u.values
ux         = train.ux.values
u2x        = train.u2x.values
u3x        = train.u3x.values
u4x        = train.u4x.values
u5x        = train.u5x.values


ut         = train.u2t.values  # this is our target, now mapped to Y

# In[56]:


# as a test I'm going to try and accelerate the fitness function
from numba import jit

@jit
def evaluate(individual):
    """Evalute the fitness of an individual: MSE (mean squared error)"""
    func = toolbox.compile(individual)
    
    # below call the individual as a function over the inputs
    
    # Yp = np.array(list(map(func, X)))
    Yp = np.array(list(map(func,  u,ux,u2x,u3x,u4x,u5x))) 
    
    # return the MSE as we are evaluating on it anyway - then the stats are more fun to watch...
    return np.mean((ut - Yp) ** 2),         


# ### [optional] Enable the linear scaling technique. It is hard for GP to determine real constants, which are important in regression problems. Thus, we can (implicitly) ask GP to evolve the shape (form) of the model and we help GP to determine constans by applying the simple least squares method (LSM).

# In[57]:


from numba import jit

@jit
def evaluate_ls(individual):
    """
    First apply linear scaling (ls) to the individual 
    and then evaluate its fitness: MSE (mean squared error)
    """
    func = toolbox.compile(individual)
    Yp = np.array(list(map(func,u,ux,u2x,u3x,u4x,u5x))) 
    
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, ut, rcond=-1)   
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(ut),   # MSE
    
    # regarding the above special cases, the optimal linear scaling w.r.t LSM is just the mean of true target values
    individual.a = 0
    individual.b = np.mean(ut)
    return np.mean((ut - individual.b) ** 2),


# In[51]:

pset = gep.PrimitiveSet('Main', input_names=['u','ux','u2x','u3x','u4x','u5x'])

h         = 3          # head length t = h(n-1) + 1
n_genes   = 2    # number of genes in a chromosome
r         = 20         # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique

# size of population and number of generations
n_pop   = 100
n_gen   = 500
champs  = 3

# Generally, more complicated problems require a larger head length and longer chromosomes formed with more genes. **The most important is that we should use the `GeneDc` class for genes to make use of the GEP-RNC algorithm.**

# In[52]:# In[50]:


def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_function(math.sin, 1)        # I tested adding my own functions
pset.add_function(math.cos, 1)
pset.add_function(math.tan, 1)
pset.add_rnc_terminal()


#%%
# # Create the individual and population
# Our objective is to **minimize** the MSE (mean squared error) for data fitting.
# ## Define the indiviudal class, a subclass of *gep.Chromosome*

creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10)   
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
# , linker=operator.mul)
#, linker=operator.add
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)



# In[58]:

if enable_ls:
    toolbox.register('evaluate', evaluate_ls)
else:
    toolbox.register('evaluate', evaluate)

# In[59]:


toolbox.register('select', tools.selTournament, tournsize=3)

# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

# 2. Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)

# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

# In[60]:
# Stastics 

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# In[61]:

pop = toolbox.population(n=n_pop) # 
hof = tools.HallOfFame(champs)   # only record the best three individuals ever found in all generations


startDT = datetime.datetime.now()
print (str(startDT))

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                          stats=stats, hall_of_fame=hof, verbose=True)


print ("Evolution times were:\n\nStarted:\t", startDT, "\nEnded:   \t", str(datetime.datetime.now()))

print(hof[0])

best_ind = hof[0]

print(best_ind)

symplified_best = gep.simplify(best_ind)
print(symplified_best)
if enable_ls:
    print(best_ind.a, best_ind.b)
    symplified_best = best_ind.a * symplified_best + best_ind.b
    print(symplified_best )
    
    
key= '''
Using GEP to predict the PDE u2t = u2x - sin(u)

Our symbolic regression process found the following equation offers our best prediction:

'''

print('\n', key,'\t', str(symplified_best))


# In[70]:

#predPE =  0.99*holdout.u2x - 0.99*np.sin(holdout.u) 
#from sklearn.metrics import mean_squared_error, r2_score
#test_mse =  mean_squared_error(holdout.u2t, predPE)
#test_r2  = r2_score(holdout.u2t, predPE)
#print("Mean squared error:", mean_squared_error(holdout.u2t, predPE))
#print("R2 score :", r2_score(holdout.u2t, predPE))

#%%
#
#
#path = 'results/sing_limb.txt'
#
#if os.path.exists(path):
#    os.remove(path)
#    
#file=open(path, "w")
#
#file.writelines("%s \n%s %s \n%s %s\n%s %s\n%s %s\n%s %s \n%s %s %s \n%s %s\n%s %s\n%s %s\n%s %s\n%s \n" % ('settings', 
#      'head =',      h,  
#      '#genes =',    n_genes,    
#      'len of RNC =',r,          
#      '# of pop =',  n_pop,      
#      '# of gen =',  n_gen,
#      'best indices =', best_ind.a, best_ind.b,
#        'Target model =','u2x - sin(u)',
#        'best model =',str(symplified_best),
#        'Test_MSE = ', test_mse,
#        'Test_R2 = ',  test_r2,
#          log))
#
#file.close()

# In[68]:
#
## we want to use symbol labels instead of words in the tree graph
#rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan'}  
#gep.export_expression_tree(best_ind, rename_labels, 'results/sing_limb_tree.pdf')

# In[76]:


#from matplotlib import pyplot
#pyplot.rcParams['figure.figsize'] = [20, 5]
#plotlen=200
#pyplot.plot(predPE.head(plotlen))       # predictions are in blue
#pyplot.plot(holdout.u2t.head(plotlen)) # actual values are in orange
#pyplot.show()
#
#best_ind = hof[0]
#for gene in best_ind:
#    print(gene.kexpression)

# In[68]:
#
## we want to use symbol labels instead of words in the tree graph
#rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan'}  
#gep.export_expression_tree(best_ind, rename_labels, 'results/sing_limb_tree_0.pdf')
#
## In[67]:
#
#
##output the top 3 champs
#champs = 3
#for i in range(champs):
#    ind = hof[i]
#    symplified_model = gep.simplify(ind)
#
#    print('\nSymplified best individual {}: '.format(i))
#    print(symplified_model)
#    print("raw indivudal:")
#    print(hof[i])
#%%