import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from sklearn.preprocessing import StandardScaler


# ==================================================
# Helper: global scaling
# ==================================================

def scale_environments(env_dict):

    scaler = StandardScaler()

    all_data = np.vstack(list(env_dict.values()))
    scaler.fit(all_data)

    scaled = {}

    for k, X in env_dict.items():
        scaled[k] = scaler.transform(X)

    return scaled


# ==================================================
# DO-TEARS
# ==================================================

class TEST_DOTEARS:

    def __init__(
        self,
        env_dict,
        interventions,
        lambda1=0.1,
        max_iter=100,
        h_tol=1e-8,
        rho_max=1e8,
        w_threshold=0.3,
        scale=True
    ):

        if scale:
            env_dict = scale_environments(env_dict)

        self.data = env_dict
        self.interventions = interventions

        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

        self.p = list(env_dict.values())[0].shape[1]

        self.V_inverse = self._estimate_exogenous_variances()


    # --------------------------------------------------
    # Estimate noise variances
    # --------------------------------------------------

    def _estimate_exogenous_variances(self):

        all_data = np.vstack(list(self.data.values()))

        variances = all_data.var(axis=0)

        variances = np.maximum(variances, 1e-6)

        return np.diag(1 / variances)


    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    def _loss(self, W):

        V_inv = self.V_inverse
        total_samples = sum(X.shape[0] for X in self.data.values())

        loss = 0
        G = np.zeros_like(W)

        for env, X in self.data.items():

            targets = self.interventions.get(env, [])

            mask = np.ones_like(W)

            for t in targets:
                mask[:, t] = 0

            W_env = W * mask

            R = X - X @ W_env

            env_loss = 0.5 / X.shape[0] * ((R @ V_inv) ** 2).sum()

            weight = X.shape[0] / total_samples

            loss += weight * env_loss

            G += weight * (-1 / X.shape[0] * X.T @ R @ V_inv)

        return loss, G


    # --------------------------------------------------
    # Acyclicity constraint
    # --------------------------------------------------

    def _h(self, W):

        M = W * W
        M = np.clip(M, 0, 10)
        E = slin.expm(M)

        h = np.trace(E) - self.p

        G = E.T * W * 2

        return h, G


    # --------------------------------------------------
    # Convert doubled variables
    # --------------------------------------------------

    def _adj(self, w):

        d = self.p

        return (w[:d*d] - w[d*d:]).reshape(d, d)


    # --------------------------------------------------
    # Objective
    # --------------------------------------------------

    def _func(self, w):

        W = self._adj(w)

        loss, G_loss = self._loss(W)

        h, G_h = self._h(W)

        obj = loss + 0.5*self.rho*h*h + self.alpha*h + self.lambda1*w.sum()

        G = G_loss + (self.rho*h + self.alpha)*G_h

        g_obj = np.concatenate((G + self.lambda1, -G + self.lambda1))

        return obj, g_obj


    # --------------------------------------------------
    # Fit
    # --------------------------------------------------

    def fit(self):

        d = self.p

        w_est = np.zeros(2*d*d)

        self.rho = 1
        self.alpha = 0

        h = np.inf

        bounds = [(0,0) if i==j else (0,None)
                  for _ in range(2)
                  for i in range(d)
                  for j in range(d)]

        for _ in range(self.max_iter):

            while self.rho < self.rho_max:

                sol = sopt.minimize(
                    self._func,
                    w_est,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds
                )

                w_new = sol.x

                h_new, _ = self._h(self._adj(w_new))

                if h_new > 0.25*h:
                    self.rho *= 10
                else:
                    break

            w_est = w_new
            h = h_new

            self.alpha += self.rho*h

            if h <= self.h_tol or self.rho >= self.rho_max:
                break

        W = self._adj(w_est)

        W[np.abs(W) < self.w_threshold] = 0

        return W
    

class TEST_DOTEARS2:

    def __init__(
        self,
        env_dict,
        interventions,
        lambda1=0.1,
        max_iter=100,
        h_tol=1e-8,
        rho_max=1e8,
        w_threshold=0.3,
        scale=True
    ):

        if scale:
            env_dict = scale_environments(env_dict)

        self.data = env_dict
        self.interventions = interventions

        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

        self.p = list(env_dict.values())[0].shape[1]

        self.V_inverse = self._estimate_exogenous_variances()


    # --------------------------------------------------
    # Estimate noise variances
    # --------------------------------------------------

    def _estimate_exogenous_variances(self): 
        p = self.p 
        variances = np.zeros(p) 
        for env, X in self.data.items(): 
            targets = self.interventions.get(env, []) 
            
            for t in targets: 
                variances[t] = X[:, t].var() 
                
        variances = np.maximum(variances, 1e-5) 
        return np.diag(1 / variances)


    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    def _loss(self, W):

        V_inv = self.V_inverse
        total_samples = sum(X.shape[0] for X in self.data.values())

        loss = 0
        G = np.zeros_like(W)

        for env, X in self.data.items():

            targets = self.interventions.get(env, [])

            mask = np.ones_like(W)

            for t in targets:
                mask[:, t] = 0

            W_env = W * mask

            R = X - X @ W_env

            env_loss = 0.5 / X.shape[0] * ((R @ V_inv) ** 2).sum()

            weight = X.shape[0] / total_samples

            loss += weight * env_loss

            G += weight * (-1 / X.shape[0] * X.T @ R @ V_inv)

        return loss, G


    # --------------------------------------------------
    # Acyclicity constraint
    # --------------------------------------------------

    def _h(self, W): 
        
        E = slin.expm(W * W) 

        h = np.trace(E) - self.p 

        G = E.T * W * 2 

        return h, G


    # --------------------------------------------------
    # Convert doubled variables
    # --------------------------------------------------

    def _adj(self, w):

        d = self.p

        return (w[:d*d] - w[d*d:]).reshape(d, d)


    # --------------------------------------------------
    # Objective
    # --------------------------------------------------

    def _func(self, w):

        W = self._adj(w)

        loss, G_loss = self._loss(W)

        h, G_h = self._h(W)

        obj = loss + 0.5*self.rho*h*h + self.alpha*h + self.lambda1*w.sum()

        G = G_loss + (self.rho*h + self.alpha)*G_h

        g_obj = np.concatenate((G + self.lambda1, -G + self.lambda1))

        return obj, g_obj


    # --------------------------------------------------
    # Fit
    # --------------------------------------------------

    def fit(self):

        d = self.p

        w_est = np.zeros(2*d*d)

        self.rho = 1
        self.alpha = 0

        h = np.inf

        bounds = [(0,0) if i==j else (0,None)
                  for _ in range(2)
                  for i in range(d)
                  for j in range(d)]

        for _ in range(self.max_iter):

            while self.rho < self.rho_max:

                sol = sopt.minimize(
                    self._func,
                    w_est,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds
                )

                w_new = sol.x

                h_new, _ = self._h(self._adj(w_new))

                if h_new > 0.25*h:
                    self.rho *= 10
                else:
                    break

            w_est = w_new
            h = h_new

            self.alpha += self.rho*h

            if h <= self.h_tol or self.rho >= self.rho_max:
                break

        W = self._adj(w_est)

        W[np.abs(W) < self.w_threshold] = 0

        return W
    
