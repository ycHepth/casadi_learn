import casadi as ca
import numpy as np

# 生成C-code可以通过关于<Function>实例调用<generate>成员函数
#
x = ca.MX.sym('x')
y = ca.MX.sym('y')
# f = ca.Function('f',[x,y],[x,ca.sin(y)*x],['x','y'],['r','q'])
# f.generate('gen.c')

# 可以对一个生成的C文件进行多个CasADi函数的包含
# f = ca.Function('f',[x],[ca.sin(x)])
# g = ca.Function('g',[x],[ca.cos(x)])
# C = ca.CodeGenerator('gen.c')
# C.add(f)
# C.add(g)
# C.generate()

