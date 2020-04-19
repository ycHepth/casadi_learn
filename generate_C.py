import casadi as ca
import numpy as np
import os
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

f = ca.Function('f',[x],[ca.sin(x)])
# opts = dict(main=True,mex=True)   # mex是对于matlab平台能调用的衍生C语言 main是对于linux环境的 command line
opts = dict(with_header = True)     # used for linking to C/C++ application
f.generate('gen.c',opts)

'''
    对于生成的c文件，主要用在以下用途：
        1. c_code 编译为DDL
        2. 命令行调用
        3. link to C/C++ application
        
    #动态库的生成： gcc -fPIC -shared gen.c -o gen.so
'''

# 这里需要首先在cmd下用gcc编译c_code为动态库文件
# 通过casadi.external()引用动态库获得函数对象

# f = ca.external('f','./gen.so')
# print(f(3.14))

# 使用CasADi的Importer类及时执行编译, 不需要调用系统命令行（跳过了.so文件的生成）
# C = ca.Importer('gen.c','clang')
# f = ca.external('f',C)
# print(f(3.14))

'''
    API of the generated code:
        assume the name of access function'name is <fname>
    1.  Reference counting
        void fname_incref(void)
        void fname_decref(void)
        
    2.  函数输入输出个数统计
        int fname_n_in(void)
        int fname_n_out(void)
        
    3.  函数输入输出的名称统计
        const char* fname_name_in(void)
        const char* fname_name_out(void)
        
    4.  输入输出的稀疏模式
        (返回值是一个指针，指向常值整型。前两个数值是行和列数，第三个数值是bool型：
        1 -- dense;
        0 -- 每一列的非零偏移
        const int* fname_sparsity_in(void)
        const int* fname_sparsity_out(void)
        
    5. 内存对象
        当多个函数需要调用同一个函数而避免冲突时，需用工作在不同的内存对象上。
        （比如线程处理）
        void* fname_alloc_mem(void)
            分配内存对象（关于fname)
        int fname_int_mem(void* mem)
            初始化内存对象：0 -- succeed
        int fname_free_mem(void* mem)
            释放内存对象： 0 -- succeed
            
    6.  工作向量 (?)
        int fname_work(int* sz_arg, int* sz_res, int* sz_iw, int* sz_w)
        
    7.  数值评估
        int fname(const double** arg, double** res, int* iw, double* w, void* mem)
            
'''
