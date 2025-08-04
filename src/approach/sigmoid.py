import numpy as np


# イテレーション数の最大値
MAX_ITER = 100 # 仮


def sigmoid_train(dec_values: np.array) -> tuple:
    """
    評価用データの decision_function() の値（dec_values）を用いて、シグモイド関数の fitting を行う。
    """
    # Minimal step taken in line search.
    min_step = 1e-5
    # For numerically strict PD of Hessian.
    sigma = 1e-12
    eps = 1e-5
    
    # データの数(=l)
    num_data = dec_values.shape[0]
    prior0 = np.sum(dec_values < 0)
    prior1 = np.sum(dec_values >= 0)
    assert (prior0 + prior1) == num_data
    
    # Initial Points.
    A = 0.0
    B = np.log((prior0 + 1.0) / (prior1 + 1.0))
    print(F'{num_data=}, {prior0=}, {prior1=}, {A=}, {B=}')
    
    hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
    loTarget = 1.0 / (prior0 + 2.0)
    print(F'{hiTarget=}, {loTarget=}')
    
    # Initial Fun Value.
    fval = 0.0
    t = []
    
    for dec in dec_values:
        if dec >= 0:
            target = hiTarget
        else:
            target = loTarget
        t.append(target)
        fApB = dec * A + B
        if fApB >= 0:
            fval += target * fApB + np.log1p(np.exp(-fApB))
        else:
            fval += (target - 1.0) * fApB + np.log1p(np.exp(fApB))
    print(F'{fval=}')
    
    for iter in range(MAX_ITER):
        # Update Gradient and Hessian. (use H' = H + sigma I)
        # numerically ensures strict PD.
        h11 = sigma
        h22 = sigma
        h21 = 0.0
        g1 = 0.0
        g2 = 0.0
        
        for i, dec in enumerate(dec_values):
            fApB = dec * A + B
            if fApB >= 0:
                p = np.exp(-fApB) / (1.0 + np.exp(-fApB))
                q = 1.0 / (1.0 + np.exp(-fApB))
            else:
                p = 1.0 / (1.0 + np.exp(fApB))
                q = np.exp(fApB) / (1.0 + np.exp(fApB))
            d2 = p * q
            h11 += dec * dec * d2
            h22 += d2
            h21 += dec * d2
            d1 = t[i] - p
            g1 += dec * d1
            g2 += d1
        
        # Stopping Criteria.
        #print(F'{g1=}, {g2=}')
        if (np.fabs(g1) < eps) and (np.fabs(g2) < eps):
            break
        
        # Finding Newton direction: -inv(H')*g
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB
        #print(F'{det=}, {dA=}, {dB=}, {gd=}')
        
        # Line Search.
        stepsize = 1.0
        while stepsize >= min_step:
            newA = A + stepsize * dA
            newB = B + stepsize * dB
            
            # New function value.
            newf = 0.0
            for i, dec in enumerate(dec_values):
                fApB = dec * newA + newB
                if fApB >= 0:
                    newf += t[i] * fApB + np.log1p(np.exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB + np.log1p(np.exp(fApB))
            
            # Check sufficient decrease.
            delta = (fval - newf) / fval
            if delta > 1e-16:
                A = newA
                B = newB
                fval = newf
                break
            else:
                stepsize /= 2.0
            
        if stepsize < min_step:
            print('Line search fails in two-class probability estimates.')
            break
    
    print(F'sigmoid_train: {fval=}, {sigma=}, {A=}, {B=}')
    if iter >= MAX_ITER:
        print('Reaching maximal iterations in two-class probability estimates.')
    
    return A, B


def sigmoid_predict(decision_value: float, A: float, B: float) -> float:
    """
    データ x の decision_function() の値（decision_value）から、データ x が該当タスクに所属する確率を返す。
    """
    fApB = decision_value * A + B
    if fApB >= 0:
        return np.exp(-fApB) / (1 + np.exp(-fApB))
    else:
        return 1 / (1 + np.exp(fApB))


if __name__ == '__main__':
    # TEST
    # 平均0、分散1（標準偏差1）の正規分布（標準正規分布）に従う乱数
    temp_dec_values = np.random.randn(100)
    a, b = sigmoid_train(temp_dec_values)
    proba_0 = sigmoid_predict(0, A=a, B=b)
    print(F'TEST DONE. {proba_0 = }')
