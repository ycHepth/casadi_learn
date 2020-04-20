import casadi as ca
import numpy as np

'''
    Van der pol系统
        min_{x,u} \int_{t = 0, T} (x0**2+x1**2+u**2)dt
        s.t.:
                x0dot = (1-x1**2)*x0-x1+u
                x1d0t = x0
                -1.0 <= u <= 1.0
                x1   >= -0.25
        x0(0) = 0
        x1(0) = 1
        T     = 10
'''

#1. direct_single_shooting
# 对NLP问题的控制量做离散化处理
# Ref: Numerical Optimal Control(Slide)

def direct_single_shooting(opt):
    T = 10
    N = 20

    x1 = ca.MX.sym('x1')
    x2 = ca.MX.sym('x2')
    x  = ca.vertcat(x1,x2)
    u  = ca.MX.sym('u')

    xdot = ca.vertcat((1-x2**2)*x1-x2+u,x1)

    L  = x1**2+x2**2+u**2

    if opt==False:
        # using CVodes instead of RK4
        dae = {'x':x,'p':u,'ode':xdot,'quad':L}
        opts = {'tf':T/N}   #这里是做了离散化，N是控制区间
        F = ca.integrator('F','cvodes',dae,opts)
    else:
        #using fixed step Runge-kutta 4 integrator
        # ref: https: // blog.csdn.net / xiaokun19870825 / article / details / 78763739
        M = 4
        DT = T/M/N
        f = ca.Function('f',[x,u],[xdot,L])
        X0 = ca.MX.sym('X0',2)
        U = ca.MX.sym('U')
        X = X0
        Q = 0
        for j in range(M):
            k1,k1_q = f(X,U)
            k2,k2_q = f(X+DT/2*k1,U)
            k3,k3_q = f(X+DT/2*k2,U)
            k4,k4_q = f(X+DT*k3,U)
            X = X+DT/6*(k1 + 2*k2+ 2*k3 + k4)
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        F = ca.Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

    # Evaluate at a test point
    # Fk = F(x0=[0.2,0.3],p=0.4)
    # print(Fk['xf'])
    # print(Fk['qf'])

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # Formulate the NLP
    # 这时u是离散控制量
    Xk = ca.MX([0, 1])
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_' + str(k))
        w += [Uk]   # w是局部变量，这里就是u
        # 至于下列的约束为什么都是List类型： 因为是离散控制量，目标函数是L在区间上的积分，所以对每个分割内进行计算并求和
        lbw += [-1]
        ubw += [1]
        w0 += [0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk = Fk['xf']
        J=J+Fk['qf'] # 通过累加操作来逼近积分，又因为用了RK4，所以积分的精度得到了保证。

        # Add inequality constraint
        g += [Xk[0]]
        lbg += [-.25]
        ubg += [ca.inf]


    # Create an NLP solver
    # 这里x指向的是控制量！！！
    # 因为目标函数所求的是最小的u,x
    # 而x是由u给出的，得到了u就可以根据动态系统得到x
    prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']

    # Plot the solution
    u_opt = w_opt #这里的操作验证了上述注释的内容
    x_opt = [[0, 1]]
    for k in range(N):
        Fk = F(x0=x_opt[-1], p=u_opt[k])
        x_opt += [Fk['xf'].full()]  #.full()转化为np.array
    x1_opt = [r[0] for r in x_opt]
    x2_opt = [r[1] for r in x_opt]

    tgrid = [T/N*k for k in range(N+1)]

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x1_opt, '--')
    plt.plot(tgrid, x2_opt, '-')
    plt.step(tgrid, ca.vertcat(ca.DM.nan(1), u_opt), '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2','u'])
    plt.grid()
    plt.show()


# direct_single_shooting(False)

def direct_multi_shooting(opt):
    # control and state nodes start value in NLP
    # 与single_shooting相似，但是把状态点放在了特定的shooting nodes上，并作为NLP的决策变量：
    # 注意到在single_shooting方法中，NLP的决策变量唯一且为控制量。
    # multi方法往往是优于single方法的，因为将问题的维度提高了，从而提高了问题的收敛性
    T = 10
    N = 20
    x1 = ca.MX.sym("x1")
    x2 = ca.MX.sym("x2")
    x = ca.vertcat(x1,x2)
    u = ca.MX.sym('u')

    xdot = ca.vertcat((1-x2**2)*x1-x2+u,x1)

    L = x1**2+x2**2+u**2

    if opt == False:
        dea = {'x':x,'p':u,'ode':xdot,'quad':L}
        opts = {'tf': T/N}
        F = ca.integrator('F','cvodes',dea,opts)
    else:
        M = 4
        DT = T/N/M
        f = ca.Function('f',[x,u],[xdot,L])
        X0 = ca.MX.sym('x0',2) # 这里的X0是initial
        U = ca.MX.sym('U')
        X = X0
        Q = 0   # Q 是 L 的逼近斜率
        for i in range(M):
            k1,k1_q = f(X,U)
            k2,k2_q = f(X+DT/2*k1,U)
            k3,k3_q = f(X+DT/2*k2,U)
            k4,k4_q = f(X+DT*k3,U)
            X = X+DT/6*(k1+2*k2+2*k3+k4)
            Q = Q+DT/6*(k1_q+2*k2_q+2*k3_q+k4_q)
        F = ca.Function('F',[X0,U],[X,Q],['x0','p'],['xf','qf'])

        Fk = F(x0 = [0.2,0.3],p=0.4)
        print(Fk['xf'])
        print(Fk['qf'])

    w = []
    w0= []
    lbw=[]
    ubw=[]
    J = 0
    g = []
    lbg = []
    ubg = []

    # 初始变量是[0,1]

    Xk = ca.MX.sym('X0',2)
    w += [Xk] #这里开始和single有区别：（single中为 w += [Uk])
    lbw += [0,1]
    ubw += [0,1]
    w0 += [0,1]

    for k in range(N):
        Uk = ca.MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-1]
        ubw += [1]
        w0 += [0]

        Fk = F(x0=Xk,p=Uk)
        Xk_end = Fk['xf']
        J = J + Fk['qf']

        Xk = ca.MX.sym('X_' + str(k+1),2)
        w += [Xk]
        lbw += [-0.25,-ca.inf]
        ubw += [ca.inf,ca.inf]
        w0 += [0,0]

        #这里添加了等式约束，在single中添加的是等式约束
        g += [Xk_end-Xk]
        lbg += [0,0]
        ubg += [0,0]

    prob = {'f':J,'x':ca.vertcat(*w),'g':ca.vertcat(*g)}
    solver= ca.nlpsol('solver','ipopt',prob)

    sol = solver(x0 = w0,lbx = lbw, ubx = ubw, lbg = lbg, ubg= ubg)
    w_opt = sol['x'].full().flatten()

    x1_opt = w_opt[0::3]
    x2_opt = w_opt[1::3]
    u_opt  = w_opt[2::3]

    tgrid = [T/N*k for k in range(N+1)]

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x1_opt, '--')
    plt.plot(tgrid, x2_opt, '-')
    plt.step(tgrid, ca.vertcat(ca.DM.nan(1), u_opt), '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2','u'])
    plt.grid()
    plt.show()


# direct_multi_shooting(False)

def direct_collocation(opt):
    # 对控制量和主变量都做离散化，不再需要对离散时间动态的构造
    # 用了多项式插值的方法参数化整个状态轨迹,从而实现匹配

    d = 3 # degree of interpolating polynomial
    tau_root = np.append(0,ca.collocation_points(d,'legendre'))

    #匹配方程的参数矩阵
    C = np.zeros((d+1,d+1))
    #连续方程的参数矩阵
    D = np.zeros(d+1)
    #二次方程的参数矩阵
    B = np.zeros(d+1)

    for j in range(d+1):
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p*= np.poly1d([1,-tau_root[r]]) / (tau_root[j] - tau_root[r])

        D[j] = p(1.0)

        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        pint = np.polyint(p)
        B[j] = pint(1.0)

    T = 10

    x1 = ca.MX.sym('x1')
    x2 = ca.MX.sym('x2')
    x = ca.vertcat(x1,x2)
    u = ca.MX.sym('u')

    xdot = ca.vertcat((1 - x2 ** 2) * x1 - x2 + u, x1)
    L = x1 ** 2 + x2 ** 2 + u ** 2

    # 连续动态的建模
    f = ca.Function('f',[x,u],[xdot,L],['x','u'],['xdot','L'])

    # 离散化控制量
    N = 20
    h = T/N

    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    x_plot = []
    u_plot = []

    #这一步是与multi方法一致的
    Xk = ca.MX.sym('X0',2)
    w.append(Xk)
    # 醉了，前面用append不好吗？明明优雅很多
    lbw.append([0,1])
    ubw.append([0,1])
    w0.append([0,1])
    x_plot.append(Xk)

    for k in range(N):
        Uk = ca.MX.sym('U_'+str(k))
        w.append(Uk)
        lbw.append([-1])
        ubw.append([1])
        w0.append([0])
        u_plot.append(Uk)

        Xc = []
        for j in  range(d):
            Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j),2)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-0.25,-np.inf])
            ubw.append([np.inf,np.inf])
            w0.append([0,0])

        Xk_end = D[0]*Xk
        for j in range(1,d+1):
            xp = C[0,j]*Xk
            for r in range(d): xp = xp+C[r+1,j]*Xc[r]

            fj,qj = f(Xc[j-1],Uk)
            g.append(h*fj-xp)
            lbg.append([0,0])
            ubg.append([0,0])

            Xk_end = Xk_end + D[j]*Xc[j-1]

            J = J + B[j]*qj*h

        Xk = ca.MX.sym('X_'+str(k+1),2)
        w.append(Xk)
        lbw.append([-0.25,-np.inf])
        ubw.append([np.inf,np.inf])
        w0.append([0,0])
        x_plot.append(Xk)

        g.append(Xk_end-Xk)
        lbg.append([0,0])
        ubg.append([0,0])

    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    x_plot = ca.horzcat(*x_plot)
    u_plot = ca.horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    prob = {'f':J,'x':w,'g':g}
    solver = ca.nlpsol('solver','ipopt',prob)

    trajectories = ca.Function('trajectories',[w],[x_plot,u_plot],['w'],['x','u'])

    sol = solver(x0 = w0, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)
    x_opt,u_opt = trajectories(sol['x'])
    x_opt = x_opt.full() # to numpy array
    u_opt = u_opt.full()

    tgrid = [T / N * k for k in range(N + 1)]

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x_opt[0], '--')
    plt.plot(tgrid, x_opt[1], '-')
    plt.step(tgrid, np.append(np.nan,u_opt[0]), '-.')
    plt.xlabel('t')
    plt.legend(['x1', 'x2', 'u'])
    plt.grid()
    plt.show()


direct_collocation(True)
