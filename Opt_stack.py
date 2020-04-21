import casadi as ca
import numpy as np

'''
    min_{x,y} (y-x**2)**2
    s.t.      x**2+y**2=1
              x + y >= 1
'''

opti = ca.Opti()

x = opti.variable()
y = opti.variable()

#opti.set_value(x,3) 赋值是必要的，否则都是做符号建模

opti.minimize( (y - x**2)**2)

# 约束可以一起添加
# opti.subject_to([x**2+y**2==1,x+y>=1])
# 对于上下限约束
# opti.subject_to(opti.bounded(0,x,1))
# 对于向量的约束，是按列进行约束条件附加的

opti.subject_to(x**2+y**2==1)
opti.subject_to(x+y>=1)

opti.solver('ipopt')

# p_opts = {'expand':True}
# s_opts = {'max_iter':100}
# opti.solver('ipopt',p_opts,s_opts)

# # opti_stack支持对参数的初始估计,缺省值是0
# opti.set_initial(x,2)
# opti.set_initial(10*x[0],2)


sol = opti.solve()

sol.value(sol['x'])
sol.value(sol['y'])

# 对于变量的数值值，通过value方法获取

print(x)
print(y)

'''
    当求解器不能得到最优值（不收敛）时，通过
        opti.debug.value(x)
    获得非收敛的解
    
    对于求解过程的绘图，通过debug模式获得：
        opti.callback(lambda i: plot(opti.debug.value(x))
        
'''