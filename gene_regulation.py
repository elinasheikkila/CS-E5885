""" CS-E5885; Modeling Biological Networks
Identification of a gene regulatory network from gene expression time-course data """

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use('MacOSX')
plt.rcParams.update({'text.usetex': True})

""" Step 1: Original data """

N = 5

swi5 = np.array([0.076, 0.0186, 0.009, 0.0117, 0.0088, 0.0095, 0.0075, 0.007, 0.0081, 0.0057,
                 0.0052, 0.0093, 0.0055, 0.006, 0.0069, 0.0093, 0.009, 0.0129, 0.0022, 0.0018])
cbf1 = np.array([0.0419, 0.0365, 0.0514, 0.0473, 0.0482, 0.0546, 0.0648, 0.0552, 0.0497, 0.0352,
                 0.0358, 0.0338, 0.0309, 0.0232, 0.0191, 0.019, 0.0176, 0.0105, 0.0081, 0.0072])
gal4 = np.array([0.0207, 0.0122, 0.0073, 0.0079, 0.0084, 0.01, 0.0096, 0.0107, 0.0113, 0.0116,
                 0.0073, 0.0075, 0.0082, 0.0078, 0.0089, 0.0104, 0.0114, 0.01, 0.0086, 0.0078])
gal80 = np.array([0.0225, 0.0175, 0.0165, 0.0147, 0.0145, 0.0144, 0.0106, 0.0119, 0.0104, 0.0142,
                  0.0084, 0.0097, 0.0088, 0.0087, 0.0086, 0.011, 0.0124, 0.0093, 0.0079, 0.0103])
ash1 = np.array([0.1033, 0.0462, 0.0439, 0.0371, 0.0475, 0.0468, 0.0347, 0.0247, 0.0269, 0.019,
                 0.0134, 0.0148, 0.0101, 0.0088, 0.008, 0.009, 0.0113, 0.0154, 0.003, 0.0012])

t_exp = np.linspace(0, 190, 20)
gene_names = ['SWI5', 'CBF1', 'GAL4', 'GAL80', 'ASH1']
Y = np.reshape(np.concatenate((swi5, cbf1, gal4, gal80, ash1)), (5, 20))

""" Step 2: ODE model and gradient descent for parameter value estimation """


def ode_system(y, t, p, c):
    c = np.reshape(c, (N, N))
    dy_dt = np.zeros(N)
    dx = np.zeros(N)
    if t//10 < 19:
        dx[:] = (- Y[:, int(t//10)] + Y[:, int(t//10 + 1)]) / 10
    for i in range(N):
        di_dt = 0
        for j in range(N):
            if i == j:
                di_dt += 0
            elif c[i, j] == 1:
                if (p[1, i*N+j] + y[j]) != 0:
                    di_dt += (p[0, i*N+j] * p[1, i*N+j] * dx[j]) / ((p[1, i*N+j] + y[j]) ** 2)
            elif c[i, j] == -1:
                if (p[1, i*N+j] + y[j]) != 0:
                    di_dt -= (p[0, i*N+j] * dx[j]) / ((p[1, i*N+j] + y[j]) ** 2)
        dy_dt[i] = di_dt + p[2, i]*y[i] + p[2, i+N]
    return dy_dt


def ode_extended(y, t, p, c):
    c = np.reshape(c, (N, N))
    Jx = np.zeros((N, N))
    Jk = np.zeros((N, 3*N*N))
    dx = np.zeros(N)
    if t//10 < 19:
        dx[:] = (- Y[:, int(t//10)] + Y[:, int(t//10 + 1)]) / 10
    for i in range(N):
        Jk[i, 2*N*N+i] += y[i]
        Jk[i, 2*N*N+N+i] += 1
        for j in range(N):
            if i == j:
                Jx[i, j] += p[2, i]
            elif c[i, j] == 1:
                Jx[i, j] -= 2*p[0, i*N+j]*p[1, i*N+j]*dx[j] / (p[1, i*N+j] + y[j])**3
                Jk[i, i*N*2+j*2] += (p[1, i*N+j] * dx[j]) / (p[1, i*N+j] + y[j])**2
                Jk[i, i*N*2+j*2+1] += (p[0, i*N+j] * dx[j] / (p[1, i*N+j] + y[j])**2) * (1 - 2*p[1, i*N+j]/(p[1, i*N+j] + y[j]))
            elif c[i, j] == -1:
                Jx[i, j] += 2*p[0, i*N+j]*dx[j] / (p[1, i*N+j] + y[j])**3
                Jk[i, i*N*2+j*2] -= dx[j] / (p[1, i*N+j] + y[j])**2
                Jk[i, i*N*2+j*2+1] += (2 * p[0, i*N+j] * dx[j]) / ((p[1, i*N+j] + y[j])**3)
    s = np.reshape(y[N:], (N, 3*N*N))
    dy_dt = ode_system(y[0:N], t, p, c)
    ds_dt = np.matmul(Jx, s) + Jk
    return np.concatenate((dy_dt, ds_dt.flatten()))


def target_function(p, c, start, stop, penalty):
    out = sp.integrate.solve_ivp(lambda t, y: ode_system(y=y, t=t, p=p, c=c), y0=Y[:, start], t_span=[0, max(t_exp)], t_eval=t_exp[start:stop+1])
    L = np.sum(np.linalg.norm(Y[:, start:start+len(out.t)] - out.y) ** 2)
    return L + penalty * np.sum(np.abs(p[0:2, :]))


def target_grad(p, c, start, stop):
    s0 = np.zeros((N, 3*N*N))
    y0 = np.concatenate((Y[:, start], s0.flatten()))
    out = sp.integrate.solve_ivp(lambda t, y: ode_extended(y=y, t=t, p=p, c=c), y0=y0, t_span=[0, max(t_exp)], t_eval=t_exp[start:stop+1])
    dL_dk = np.zeros(3*N*N)
    for i in range(len(out.t)):
        s = np.reshape(out.y[N:, i], (N, 3*N*N))
        sum_term = -2 * np.matmul(s.T, (Y[:, i] - out.y[0:N, i]))
        dL_dk = dL_dk + sum_term.flatten()
    return dL_dk


def line_search(p, c, grad, start, stop, penalty):
    b = 1 / 2
    tau = 1 / 2
    r = 0.1
    max_iter = 20
    m = np.dot(grad.flatten(), -grad.flatten())
    for j in range(max_iter):
        if (target_function(p, c, start, stop, penalty) - target_function(p - r * grad, c, start, stop, penalty)) >= (-r) * b * m:
            break
        else:
            r = tau * r
    return r


def grad_descent(p0, c, n_iter, start, stop, penalty):
    eps = 10**(-7)
    P_path = np.zeros((3, N*N, n_iter + 1))
    P_path[:, :, 0] = p0
    i_last = 0
    for i in range(n_iter):
        p = np.copy(P_path[:, :, i])
        grad = target_grad(p, c, start, stop)
        grad1 = grad[np.linspace(0, 48, 25, dtype=int)]
        grad2 = grad[np.linspace(1, 49, 25, dtype=int)]
        grad3 = grad[50:]
        grad = np.stack((grad1, grad2, grad3))
        P_path[:, :, i + 1] = p - grad * line_search(p, c, grad, start, stop, penalty)
        for j in range(N*N):
            if c[j] == 0:
                P_path[0, j, i + 1] = 0
                P_path[1, j, i + 1] = 0
            if P_path[0, j, i + 1] < 0:
                P_path[0, j, i + 1] = 0
            if P_path[1, j, i + 1] < 0:
                P_path[1, j, i + 1] = 0
            if j < 5 and P_path[2, j, i + 1] > 0:
                P_path[2, j, i + 1] = 0
            if j >= 5 and P_path[2, j, i + 1] < 0:
                P_path[2, j, i + 1] = 0
        new_score = target_function(P_path[:, :, i + 1], c, start, stop, penalty)
        old_score = target_function(P_path[:, :, i], c, start, stop, penalty)
        if (new_score > old_score) or (np.abs(new_score - old_score) <= eps):
            break
        i_last += 1
        # plot_intermediate(p, c, start, stop)
    return P_path, i_last


""" Step 3: Sequential Floating Forward Search, SFFS, for network structure inference """


def sfs_step(i_p, i_c, start, stop, penalty):
    print('sfs_step')
    init_val = 10
    c_p = i_p
    c_c = i_c
    idx_z = np.argwhere(i_c == 0)
    score = 0
    for i in range(N*N - np.count_nonzero(i_c)):
        n_p = np.copy(i_p)
        n_c_up = np.copy(i_c)
        n_c_down = np.copy(i_c)
        n_p[0, idx_z[i]] = init_val
        n_p[1, idx_z[i]] = init_val
        n_c_up[idx_z[i]] = 1
        n_c_down[idx_z[i]] = -1
        P_up, i_last_up = grad_descent(n_p, n_c_up, 5, start, stop, penalty)
        P_down, i_last_down = grad_descent(n_p, n_c_down, 5, start, stop, penalty)
        n_score_up = target_function(P_up[:, :, i_last_up], n_c_up, start, stop, penalty)
        n_score_down = target_function(P_down[:, :, i_last_down], n_c_down, start, stop, penalty)
        if (n_score_up < n_score_down) and (n_score_up < score) or score == 0:
            c_p = P_up[:, :, i_last_up]
            c_c = n_c_up
            score = n_score_up
        elif (n_score_down < n_score_up) and (n_score_down < score) or score == 0:
            c_p = P_down[:, :, i_last_down]
            c_c = n_c_down
            score = n_score_down
    return c_p, c_c


def sbs_step(i_p, i_c, start, stop, penalty):
    print('sbs_step')
    score = 0
    c_p = i_p
    c_c = i_c
    idx_nz = np.nonzero(i_c)
    for i in range(len(idx_nz[0])):
        n_p = np.copy(i_p)
        n_c = np.copy(i_c)
        j = idx_nz[0][i]
        n_p[0, j] = 0
        n_p[1, j] = 0
        n_c[j] = 0
        P, i_last = grad_descent(n_p, n_c, 5, start, stop, penalty)
        new_score = target_function(P[:, :, i_last], n_c, start, stop, penalty)
        if new_score < score or score == 0:
            c_p = P[:, :, i_last]
            c_c = n_c
            score = new_score
    return c_p, c_c


def sffs(start, stop, penalty):
    print('SFFS started for interval {}-{} min'.format(t_exp[start], t_exp[stop]))
    c_p = np.zeros((3, N*N))
    c_p[2, 0:5] = -0.001
    c_p[2, 5:10] = 0.001
    c_c = np.zeros(N * N)
    f_p, f_c = c_p, c_c
    d_min = 0.01
    subset_scores = np.zeros(N * N + 1)
    converged = False
    count = 0
    while not converged:
        c_p, c_c = sfs_step(c_p, c_c, start, stop, penalty)
        k = np.count_nonzero(c_c)
        print('Edge added. {} edges with score {}'. format(k, target_function(c_p, c_c, start, stop, penalty)))
        stop_loop = False
        while not stop_loop and k > 0:
            n_p, n_c = sbs_step(c_p, c_c, start, stop, penalty)
            score = target_function(n_p, n_c, start, stop, penalty)
            if (score < subset_scores[k-1]) and (np.abs(score - subset_scores[k-1]) > d_min) or \
                    (subset_scores[k-1] == 0):
                if len(np.argwhere(subset_scores != 0)) == 0:
                    f_p, f_c = n_p, n_c
                elif score < np.min(subset_scores[np.argwhere(subset_scores != 0)]):
                    f_p, f_c = n_p, n_c
                subset_scores[k-1] = score
                c_p, c_c = n_p, n_c
                k = k - 1
                print('Edge removed. {} edges with score {}'.format(k, score))
            else:
                stop_loop = True
        if k == N*N:
            count += 1
        if count == 2:
            converged = True
            print('SFFS converged.')
    return f_p, f_c


""" Step 4: Plotting and results """


def plot_original():
    f1 = plt.figure(num=1, figsize=(12, 6))
    for idx_1 in range(N):
        plt.plot(t_exp, Y[idx_1, :])
    plt.figure(1)
    plt.legend(gene_names)
    plt.grid()
    plt.xticks(t_exp)
    plt.xlim(0, t_exp[-1])
    plt.xlabel('t (min)')
    plt.ylabel('gene expression')
    plt.title('Original data')
    f1.savefig('Original_data.png')


def plot_intermediate(p, c, start, stop):
    out = sp.integrate.solve_ivp(lambda t, y: ode_system(y=y, t=t, p=p, c=c), y0=Y[:, start], t_span=[0, max(t_exp)],
                                 t_eval=t_exp[start:stop + 1])
    plt.figure(figsize=(12, 6))
    color = ['red', 'blue', 'black', 'green', 'yellow']
    for j in range(N):
        plt.plot(t_exp[start:stop + 1], Y[j, start:stop + 1], label=gene_names[j], color=color[j])
        plt.plot(t_exp[start:start + len(out.t)], out.y[j, :], ls='--', label=gene_names[j] + ' estimated', color=color[j])
    plt.grid()
    plt.legend()
    plt.xticks(t_exp)
    plt.xlim(0, t_exp[-1])
    plt.xlabel('t (min)')
    plt.ylabel('gene expression')
    plt.title('Original data and results')
    plt.show(block=True)


def plot_results(all_f_p, all_f_c, starts, stops):
    f2 = plt.figure(figsize=(12, 6))
    y0 = Y[:, starts[0]]
    for i in range(len(all_f_c[:, 0])):
        out = sp.integrate.solve_ivp(lambda t, y: ode_system(y=y, t=t, p=all_f_p[:, :, i], c=all_f_c[i, :]),
                                     y0=y0, t_span=[t_exp[starts[i]], t_exp[stops[i]]], t_eval=t_exp[starts[i]:stops[i]+1])
        for j in range(N):
            plt.plot(t_exp[starts[i]:stops[i]+1], Y[j, starts[i]:stops[i]+1], color='black')
            plt.plot(t_exp[starts[i]:starts[i]+len(out.t)], out.y[j, :], ls='--', color='blue')
        y0 = out.y[:, -1]
    plt.grid()
    plt.xticks(t_exp)
    plt.xlim(0, t_exp[-1])
    plt.xlabel('t (min)')
    plt.ylabel('gene expression')
    plt.title('Original data and results')
    f2.savefig('Results.png')


def fpr_tpr(c):
    correct_c = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0])
    Pos = 5
    Neg = 3
    multiplied = correct_c * c
    TP = np.count_nonzero(np.where(multiplied > 0))
    TPR = TP / (Pos + Neg)
    FPR = (np.count_nonzero(c) - TP) / (N * N - (Pos + Neg))
    return FPR, TPR


def plot_roc(all_f_c):
    sum_matrix = np.sum(all_f_c, axis=0)
    TPR = np.zeros(N*N)
    FPR = np.zeros(N*N)
    annotations = []
    for i in range(N*N):
        sum_matrix_mod = np.copy(sum_matrix)
        edges = np.zeros(N*N)
        for j in range(i):
            i_max = np.argmax(np.abs(sum_matrix_mod))
            edges[i_max] = sum_matrix_mod[i_max]
            sum_matrix_mod[i_max] = 0
        FPR[i], TPR[i] = fpr_tpr(edges)
        annotations.append('{}'.format(i))
        print('Connectivity with threshold {}'.format(i))
        print(np.reshape(edges, (N, N)))
    f3, ax = plt.subplots(figsize=(12, 6))
    plt.plot(FPR, TPR, color='green')
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='black', ls='--')
    for xi, yi, text in zip(FPR, TPR, annotations):
        ax.annotate(text, xy=(xi, yi), xycoords='data', xytext=(1.5, 1.5), textcoords='offset points')
    plt.title('ROC curve for different edge count values')
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    f3.savefig('ROC.png')


def plot_roc_penalty(start, stop, penalty_sequence):
    TPR = np.zeros(len(penalty_sequence))
    FPR = np.zeros(len(penalty_sequence))
    annotations = []
    for i in range(len(penalty_sequence)):
        p, c = sffs(start, stop, penalty_sequence[i])
        FPR[i], TPR[i] = fpr_tpr(c)
        annotations.append('{}'.format(penalty_sequence[i]))
    f4, ax = plt.subplots(figsize=(12, 6))
    plt.plot(FPR, TPR, color='green')
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='black', ls='--')
    for xi, yi, text in zip(FPR, TPR, annotations):
        ax.annotate(text, xy=(xi, yi), xycoords='data', xytext=(1.5, 1.5), textcoords='offset points')
    plt.title('ROC curve for different regularization penalty values')
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    f4.savefig('ROC_penalty.png')


""" Alternative approach 1: Correlation """


def correlate(n, start, stop, penalty):
    beta = 12
    init_val = 10
    p = np.zeros((3, N*N))
    p[2, 0:5] = -0.001
    p[2, 5:10] = 0.001
    c = np.zeros(N*N)
    c_matrix = np.corrcoef(Y[:, start:stop+1], rowvar=True)
    adjacency = np.abs(0.5 + 0.5*c_matrix)**beta
    np.fill_diagonal(adjacency, 0)
    adjacency_mod = np.copy(adjacency.flatten())
    for i in range(n):
        i_max1 = np.argmax(np.abs(adjacency_mod))
        adjacency_mod[i_max1] = 0
        i_max2 = np.argmax(np.abs(adjacency_mod))
        n_c1 = np.copy(c)
        n_c2 = np.copy(c)
        n_p1 = np.copy(p)
        n_p2 = np.copy(p)
        if adjacency_mod[i_max2] < 0:
            n_c1[i_max1] = -1
            n_c2[i_max2] = -1
        else:
            n_c1[i_max1] = 1
            n_c2[i_max2] = 1
        adjacency_mod[i_max2] = 0
        n_p1[0:2, i_max1] = init_val
        n_p2[0:2, i_max2] = init_val
        P1, i_last1 = grad_descent(n_p1, n_c1, 5, start, stop, penalty)
        score1 = target_function(P1[:, :, i_last1], n_c1, start, stop, penalty)
        P2, i_last2 = grad_descent(n_p2, n_c2, 5, start, stop, penalty)
        score2 = target_function(P2[:, :, i_last2], n_c2, start, stop, penalty)
        if score1 <= score2:
            c = n_c1
            p = P1[:, :, i_last1]
        else:
            c = n_c2
            p = P2[:, :, i_last2]
    return p, c


def sfs_cor(start, stop, penalty):
    print('SFS with correlation metrics started for interval {}-{} min'.format(t_exp[start], t_exp[stop]))
    best_score = 0
    f_p, f_c = correlate(0, start, stop, penalty)
    for i in range(11):
        c_p, c_c = correlate(i, start, stop, penalty)
        P, i_last = grad_descent(c_p, c_c, 5, start, stop, penalty)
        c_p = P[:, :, i_last]
        score = target_function(c_p, c_c, start, stop, penalty)
        if (score < best_score) or best_score == 0:
            f_p, f_c = c_p, c_c
            best_score = score
    print('SFS with correlation metrics completed.')
    return f_p, f_c


def main():

    n_intervals = 1  # n_intervals = 19
    starts = [0]  # starts = np.linspace(0, 18, 19, dtype=int)
    stops = [19]  # stops = np.linspace(1, 19, 19, dtype=int)
    all_f_p = np.zeros((3, (N*N), n_intervals))
    all_f_c = np.zeros((n_intervals, N*N))
    score_sum = 0
    penalty = [0.001]
    penalty_sequence = np.linspace(0.00001, 0.0001, 10)

    plot_original()
    if n_intervals != 1:
        print('Piecewise iteration over time with {} intervals started.'.format(n_intervals))
        for i in range(n_intervals):
            print('Interval {}/{} started'.format(i+1, n_intervals))
            st = time.time()
            f_p, f_c = sffs(starts[i], stops[i], penalty[0])
            # f_p, f_c = sfs_cor(starts[i], stops[i], penalty)
            et = time.time()
            print('Interval {}/{} completed \n'.format(i+1, n_intervals))
            print('Elapsed wall time: {}'.format(et - st))
            print('Edge count: {}'.format(np.count_nonzero(f_c)))
            # print('Parameter values:\n{}'.format(f_p))
            print('Network structure:\n{}\n'.format(np.reshape(f_c, (N, N))))
            score = target_function(f_p, f_c, starts[i], stops[i], penalty[0])
            score_sum += score
            print('Score: {}'.format(score))
            print('Total score so far: {}\n'.format(score_sum))
            all_f_p[:, :, i] = f_p
            all_f_c[i, :] = f_c
            print('Piecewise iteration over time completed.')
            print('Total score: {}'.format(score_sum))
            plot_results(all_f_p, all_f_c, starts, stops)
            # plot_roc(all_f_c)

    if n_intervals == 1:
        f_p, f_c = sffs(starts[0], stops[0], penalty[0])
        plot_results(f_p, f_c, starts, stops)
        # plot_roc_penalty(starts[0], stops[0], penalty_sequence)


main()
