import casadi as ca
import numpy as np

'''
    DaeBuilder is a class in CasADi intended to facilitate the modeling complex 
    dynamical system for use with optimal control algorithms.
    
    1. 对DAE的每步构造
    2. 符号重构造DAE
    3. 构造CasADi.Function
    
'''

