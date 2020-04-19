import casadi as ca
import math
import numpy as np

'''
    Part 1: SX structure
'''
# # 创建一个符号标量,名称为 “x”
# x = ca.SX.sym('x')
#
# # y is a symbol vertor with 5X1
# y = ca.SX.sym('y',5)
#
# # z is a symbol matrix with 4x2
# z = ca.SX.sym('z',4,2)
#
# # ！！！SX.sym 返回的是一个SX类的实例
#
# # f = ca.MX.sym('f')
# f = x**2+10
# f = ca.sqrt(f)
# # print('f:',f)
#
# # casadi支持不先对于符号量做声明
# B1 = ca.SX.zeros(4,5)
# B2 = ca.SX(4,5)
# # print(B1)
# # print(B2)
# # 这里B1和B2是有区别的： B1生成的是稠密矩阵，所有元素是实际的0；B2生成的是稀疏矩阵，元素是<结构>0
# '''
#     Structural zeros refer to zero responses by those subjects whose count response will always be zero,
#     in contrast to random (or sampling) zeros that occur to subjects whose count response can be greater than zero,
#     but appear to be zero due to sampling variability.
# '''
# # 通常意义下，没有进一步调用方法生成的SX实例都是系数矩阵
# # 对于形参是标量类型的SX，生成的是标量元素的SX实例
#
# # 创建具有指定数值的SX矩阵实例
# C = ca.SX([[1,2],[3,4]])
# print(C)

# 对于C++下的casadi SX对象，要注意到对命名空间和头文件的引用
'''
    #include <casadi/casadi.hpp>
    using namespace casadi;
    int main(){
        SX x = SX::sym("x");
        SX y = SX::sym("y",5);
        SX z = SX::sym("z",4,2)
        SX f = pow(x,2) + 10;
        f = sqrt(f)
        std::cout << "f: " << f << std::endl;
        return 0;
    }
'''


'''
    Part 2: DM structure
'''

# 与SX区别在于:SX是符号化对象，MX是数值化对象，也就是说再SX中的符号值在MX中都被替换为数值值
#   DM对象不适合密集性计算（因为MX存储了数值内容，容易产生对内存的需求)

# C = ca.DM(2,3)
# print(C) # 如果直接print,得到的结果是对于参数的描述
# C_dense = ca.DM(2,3).full()
# print(C_dense)
# print(np.array(C))
# C_dense == np.array(C)
# # 也就是说由于full()的调用，
# # 使得DM实例转化为了array类型，对于DM对象的直接调用和SX对象类似，得到的带有结构0的稀疏矩阵

'''
    Part3: MX structure
'''

# # consider SX
# x = ca.SX.sym('x',2,2)
# y = ca.SX.sym('y')
# f = 3*x + y
# print(f)
# print(f.shape)
#
# # MX is more general matrix expression than SX
# x = ca.MX.sym('x',2,2)
# y = ca.MX.sym('y')
# f = 3*x +y
# print(f)
# print(f.shape)

# 与SX不同的是，对于元素的操作没有严格到单元或二元映射，而是通用的多个稀疏矩阵的输入和输出
# 简单说来就是形成了更有通用性的模型建立，对于操作的复杂度降低了
# 因此对于较大的矩阵，MX能够获得更高的效率

# MX 支持对于元素的获取和设置，但不推荐。

# x = ca.MX.sym('x',2)
# A = ca.MX(2,2)
# A[0,0] = x[0]
# A[1,1] = x[0] + x[1]
# print("A: ", A)

# 对于SX和MX的混合计算是不可实现的。

'''
    Part 4: sparsity class
'''
# CasADi的矩阵存储是通过列向量压缩实现的（和MATLAB一样）
# sparsity class是指矩阵存储的稀疏模式 = 行列维度 + 两个向量（分别对应行和列的非零元素位置）
# sparsity class的作用是生成非标准型的矩阵

# print(ca.SX.sym('x',ca.Sparsity.lower(3)))

# M = ca.SX([[3,7],[4,5]])
# print(M[0,:])
# M[0,:] = 1
# print(M)

# # 1.单元素操作
# M = ca.diag(ca.SX([3,4,5,6]))
# print(M)
# print(M[0,0],M[1,0],M[-1,-1]) # 这里index = -1是指最后的元素（右下角元素）

# # 2.剪裁操作
# print(M[:,1])           # in C++: M(Slice(),1)
# print(M[1:,1:4:2])      # in C++: M(Slice(1,-1),Slice(1,4,2))

# # 3.列表操作
# M = ca.SX([[3,7,8,9],[4,5,6,1]])
# print(M)
# print(M[0,[0,3]],M[[5,-6]])     # 注意到这里第二项的5是按列从上到下找到的，但-6是从末端向前算的

'''
    Part 5: 算数操作
'''
# x = ca.SX.sym('x')
# y = ca.SX.sym('y',2,2)
# print(ca.sin(y)-x)
#
# # x = ca.MX.sym('x')
# # y = ca.MX.sym('y',2,2)
# # print(ca.sin(y)-x)
# # 1.数乘和矩阵乘法
# # 对于C++和python，*运算是对于每个元素做的，对于矩阵间的计算需要调用方法 mtimes(A,B)
# print(y*y,ca.mtimes(y,y))
#
# # 2.转置
# print(y.T)
# # 对于C++，调用的方法是A.T()
#
# # 3.reshape
# x = ca.SX.eye(4)
# print(ca.reshape(x,2,8))
#
# # 4.矩阵（向量）拼接
# x = ca.SX.sym('x',5)
# y = ca.SX.sym('y',5)
#
# print(ca.vertcat(x,y))# 垂直拼接
# print(ca.horzcat(x,y))# 水平拼接（因为存储方式就是列堆叠，所以更具有效率）

# # 5.水平/垂直剪裁 = 裁剪为两部分
# x = ca.SX.sym('x',5,3)
# print(x)
# w = ca.horzsplit(x,[0,1,3]) #[...]中的是offset，第一项必须是0，最后一项必须是列数
# print(w[0],w[1])
#
# w = ca.vertsplit(x,[0,3,5])
# print(w[0],w[1])

# # 6.内积
# # <A,B> := tr(A,B)
# x = ca.SX.sym('x',2,2)
# print(ca.dot(x,x))

'''
    Part 6: 属性
'''
# 适用于矩阵或稀疏模式
# y = ca.SX.sym('y',10,1)
# print(y.shape)

# .size1() => 行数
# .size2() => 列数
# .numel() => 元素个数 == .size1() * .size2()
# .sparisity() => 稀疏模式

'''
    Part 7: 微分计算
'''
# 微分计算可以从前向和反向分别计算，分别对应得到jacobian vector和jacobian-transposed vector

#1. jacobian
A = ca.SX.sym('A',3,2)
x = ca.SX.sym('x',2)
print('A:',A,' x: ',x,'Ax: ',ca.mtimes(A,x))
print('J:',ca.jacobian(ca.mtimes(A,x),x))

print(ca.dot(A,A))
print(ca.gradient(ca.dot(A,A),A))

[H,g] = ca.hessian(ca.dot(x,x),x) #hessian（）会同时返回梯度和hessian矩阵
print('H：', H)
print('g:',g)

v = ca.SX.sym('v',2)
f = ca.mtimes(A,x)

print(ca.jtimes(f,x,v)) # jtimes = jacobian function * vector

