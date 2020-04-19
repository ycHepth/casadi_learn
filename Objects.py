import casadi as ca
import numpy as np

'''
    use CasADi to create function objects
    in C++ refered to <functors>
    
    f = function_name(name,arguments,...,[opts])
    
    通过传入输入表达式和输出表达式的列表，生成一个函数对象
'''

# x = ca.SX.sym('x',2)
# y = ca.SX.sym('y')
# f = ca.Function('f',[x,y],\
#                 [x,ca.sin(y)*x])
#
# # that is : (x,y) -> (x,sin(y)x) 的映射
# # 另外用MX也可以实现函数对象的生成：
# x = ca.MX.sym('x',2)
# y = ca.MX.sym('y')
# f = ca.Function('f',[x,y],\
#                 [x,ca.sin(y)*x],\
#                 ['x','y'],['r','q']) #新增的两项是对输入/输出的命名： x->r, y->q

'''
    Part1: 函数对象的调用
'''

# r0,q0 = f(1.1,3.3)
# print('r0:',r0)
# print('q0' ,q0)
# # 对于参数的输入需要按序，
# # 对于变量名+变量值的组合，导致引入了字典，in C++: std::map<std::string,MatrixType>
# res = f(x=1.1,y=3.3)
# print('res: ', res)
#
# #通过self.call()得到函数结果
# arg = [1.1,3.3]
# arg = {'x':1.1,'y':3.3}
# res = f.call(arg)
# print('res: ', res)

'''
    Part2: MX ==> SX
    通过MX定义的函数对象只包含了内置的基本操作，拓展性有限；转化为SX使函数对象的操作更广泛
    sx_function = mx_function.expand()
'''

'''
    Part3: 非线性求根问题
    g0(z,x1,x2,...) = 0    (1)
    g1(z,x1,x2,...) = y1
            ...
    gm(z,x1,x2,...) = ym
    
    式（1）由隐函数导出了z关于x1,x2,...的函数。
    通过CasADi导出形式
    G:{z_guess,x1,x2,...,xn} -> {z,y1,y2,...,yn}
    
    --------------------------------------------
    z = ca.SX.sym('x',nz)
    x = ca.SX.sym('z',nx)
    g0 = {expression of x,z}
    g1 = {expression of x,z}
    g  = ca.Function（'g',[z,x],[g0,g1])
    G  = rootfinder('G','newton',g)
    --------------------------------------------
'''

'''
    Part4: 初值问题和敏感性分析
    Consider DAE:
        dx = z + p
        0  = z cos(z) - x
    敏感性分析见例程代码<sensitivities_analysis>
'''

# Integrator
# x = ca.SX.sym('x')
# z = ca.SX.sym('z')
# p = ca.SX.sym('p')
# dae = {'x':x,'z':z,'p':p,'ode':z+p,'alg':z*ca.cos(z)-x}
# F = ca.integrator('F','idas',dae)
#
# r = F(x0 = 0,z0 = 0, p=0.1)
# print('r: ',r['xf'])


'''
        NLP = nonlinear programming
                
        min_x   f(x,p)
        s.t.    x_lb <=    x    <= x_ub
                g_lb <=  g(x,p) <= g_ub
        
        NLP的求解需要参数：{p,lbx,ubx,lbg,ubg} + guess for primal-dual solution {x0, lam_x0, lam_g0}
        NLP求解器在CasADi中不可微
        
        IPOPT = primal-dual interior point method for optimal programming
        
        一般来说，NLP solver需要给出jacobian of constraint and hessian of lagrangian function 
        
        e.g.:
        consider system:
            min_{x,y,z}     x**2+100*z**2
            s.t.            z+(1-x)**2-y = 0
'''

# x = ca.SX.sym('x')
# y = ca.SX.sym('y')
# z = ca.SX.sym('z')
# nlp = {'x':ca.vertcat(x,y,z),'f':x**2+100*z**2,'g':z+(1-x)**2-y}
# S = ca.nlpsol('S','ipopt',nlp)
# r = S(x0=[2.5,3.,0.75],lbg=0,ubg=0)
# x_opt= r['x']
# print('x_opt: ',x_opt)

'''
    QP：CasADi提供了两种求解方法= 顶层接口 + 底层接口
    顶层接口的和NLP的形式类似，但是要求以下几点：
        1. 问题形式类似NLP
        2. Objective function必须是！！！凸二次函数（关于x）
        3. constraint function必须是x的线性函数
    如果objective function不是凸的，那么可能不能找到解/或者解不唯一

         
    底层接口是为了解决形如(SDP?)：
        min_x       (1/2)x.T*H*x + g^Tx
        s.t.    x_lb <=    x    <= x_ub
                a_lb <=   Ax    <= a_ub 
                (这里的AX反映的是约束）
                
                    
    consider system:
        min_{x,y}   x**2+y**2
        s.t.        x+y-10 >= 0
'''

#high-level method

# x = ca.SX.sym('x')
# y = ca.SX.sym('y')
# qp = {'x':ca.vertcat(x,y),'f':x**2+y**2,'g':x+y-10}
# S = ca.qpsol('S','qpoases',qp)
# # 这里由于期望的解是唯一的，所以初始guess不那么重要
# r = S(lbg=0)
# x_opt = r['x']
# print('x_opt: ',x_opt)

#low-level method

# 这一部分是对问题的范式转换

H = 2*ca.DM.eye(2)
A = ca.DM.ones(1,2)
g = ca.DM.zeros(2)
lba = 10

# 由于运用的是DM（数值形式），对于高效计算需要转换成稀疏模式（sparsity pattern)
qp = {}
qp['h'] = H.sparsity()
qp['a'] = A.sparsity()
S = ca.conic('S','qpoases',qp)

r = S(h=H,g=g,a=A,lba=lba)
x_opt = r['x']
print('x_opt: ',x_opt)