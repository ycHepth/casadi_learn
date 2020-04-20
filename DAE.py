import casadi as ca
import numpy as np

'''
    DaeBuilder is a class in CasADi intended to facilitate the modeling complex 
    dynamical system for use with optimal control algorithms.
    
    1. 对DAE的每步构造
    2. 符号重构造DAE
    3. 构造CasADi.Function
    
    数学形式：
        输入表达式：
            t -- time
            c -- constant
            p -- independent parameters
            d -- dependent parameters
            x -- 微分变量x(显式ODE)
            s -- 微分变量s(隐式ODE)
            sdot -- time derivates of s
            z -- 代数变量
            q -- 二次化变量
            w -- 局部变量（?) --> 优化问题的决策变量
            y -- 输出变量
        输出表达式：
            ddef -- 计算d的显示表达式
            wdef -- 计算w的显示表达式
            ode  -- 显式ode: x_dot = ode(...)
            dae  -- 隐式ode: dae(...) = 0
            alg  -- 代数方程
            quad -- 二次方程
            ydef -- 计算y的显示表达式
'''

# construct instance
'''
    h_dot = v
    v_dot = (u-a*v**2)/m - g
    m_dot = -b*u**2
'''

dae = ca.DaeBuilder()

a = dae.add_p('a')
b = dae.add_p('b')
u = dae.add_u('u')
h = dae.add_x('h')
v = dae.add_x('v')
m = dae.add_x('x')

hdot = v
# 这里没有对g做定义！
vdot = (u-a*v**2)/m - g
mdot = -b*u**2

dae.add_ode('hdot',hdot)
dae.add_ode('vdot',vdot)
dae.add_ode('mdot',mdot)

dae.set_start('h',0)
dae.set_start('v',0)
dae.set_start('m',1)

# 设置单位
dae.set_unit('h','m')
dae.set_unit('v','m/s')
dae.set_unit('m','kg')

f = dae.create('f',['x','u','p'],['ode'])
f = dae.create('f',['x','u','p'],['jac_ode_x'])

# 线性组合
dae.add_lc('gamma',['ode'])
# hessian矩阵
hes = dae.create('hes',['x','u','p','lam_ode'],['hes_gamma_x_x'])

