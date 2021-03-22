import numpy as np
from collections import OrderedDict
from itertools import combinations, chain
from numpy.linalg.linalg import qr
from scipy.sparse.linalg import svds
from scipy.linalg import pinv
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import randomized_svd
import numexpr as ne
import sys
import time
import multiprocessing
from sklearn.utils.extmath import randomized_svd
# from pathos.multiprocessing import ProcessingPoll as Pool
import pathos

# print(sys.path)
sys.path.append('./dPCA');
from dPCA import dPCA
from dPCA import nan_shuffle


class EE_dPCA(BaseEstimator):
    def __init__(self, labels=None, join=None, n_components=10, regularizer=None, copy=True, n_iter=0, hyper_lamb=None, hyper_tau=None, rho = None):

        # create labels from alphabet if not provided
        if isinstance(labels, str):
            self.labels = labels
        elif isinstance(labels, int):
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            labels = alphabet[:labels]
        else:
            raise TypeError(
                'Wrong type for labels. Please either set labels to the number of variables or provide the axis labels as a single string of characters (like "ts" for time and stimulus)')

        self.join = join
        self.regularizer = 0 if regularizer == None else regularizer
        self.opt_regularizer_flag = regularizer == 'auto'
        self.n_components = n_components
        self.copy = copy
        self.marginalizations = self._get_parameter_combinations()
        self.n_iter = n_iter
        
        self.cross_flag = 'auto'

        self.hyper_lamb = 0 if hyper_lamb == None else hyper_lamb
        self.hyper_tau = 0 if hyper_tau == None else hyper_tau
        self.rho = rho
        self.start = self.end = self.total_time = self.cross_val_time = 0
        self.explained_variance_ratio_ = None


        # set debug mode, 0 = no reports, 1 = warnings, 2 = warnings & progress, >2 = everything
        self.debug = 2
        self.protect = None
        self.n_tri = 3

        if regularizer == 'auto':
            print("""You chose to determine the regularization parameter automatically. This can
                    take substantial time and grows linearly with the number of crossvalidation
                    folds. The latter can be set by changing self.n_trials (default = 3). Similarly,
                    use self.protect to set the list of axes that are not supposed to get to get shuffled
                    (e.g. upon splitting the data into test- and training, time-points should always
                    be drawn from the same trial, i.e. self.protect = ['t']). This can significantly
                    speed up the code.""")

            self.n_trials = 3
            self.protect = None

    def fit(self, X, trialX=None):
        self._fit(X, trialX=trialX)
        return self

    def fit_transform(self, X, trialX=None):
        self.start = time.time()
        self._fit(X, trialX=trialX)
        return self.transform(X)

    def _get_parameter_combinations(self, join=True):
        # subsets = () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)"
        subsets = list(
            chain.from_iterable(combinations(list(range(len(self.labels))), r) for r in range(len(self.labels))))

        # delete empty set & add (0,1,2)
        del subsets[0]
        subsets.append(list(range(len(self.labels))))

        # create dictionary
        pcombs = OrderedDict()
        for subset in subsets:
            key = ''.join([self.labels[i] for i in subset])
            pcombs[key] = set(subset)

        # condense dict if not None
        if isinstance(self.join, dict) and join:
            for key, combs in self.join.items():
                tmp = [pcombs[comb] for comb in combs]

                for comb in combs:
                    del pcombs[comb]

                pcombs[key] = tmp

        return pcombs

    def _marginalize(self, X, save_memory=False):

        def mmean(X, axes, expand=False):

            Z = X.copy()

            for ax in np.sort(axes)[::-1]:
                Z = np.mean(Z, ax)

                if expand == True:
                    Z = np.expand_dims(Z, ax)

            return Z

        def dense_marg(Y, mYs):

            tmp = np.zeros_like(Y)
            for key in list(mYs.keys()):
                mYs[key] = (tmp + mYs[key]).reshape((Y.shape[0], -1))

            return mYs

        Xres = X.copy()  # residual of data

        # center data
        Xres -= np.mean(Xres.reshape((Xres.shape[0], -1)), -1).reshape((Xres.shape[0],) + (len(Xres.shape) - 1) * (1,))

        # init dict with marginals
        Xmargs = OrderedDict()

        # get parameter combinations
        pcombs = self._get_parameter_combinations(join=False)

        # subtract the mean
        S = list(pcombs.values())[-1]  # full set of indices

        if save_memory:
            for key, phi in pcombs.items():
                S_without_phi = list(S - phi)

                # compute marginalization and save
                Xmargs[key] = mmean(Xres, np.array(S_without_phi) + 1, expand=True)

                # subtract the marginalization from the data
                Xres -= Xmargs[key]
        else:
            # efficient precomputation of means
            pre_mean = {}

            for key, phi in pcombs.items():
                if len(key) == 1:
                    pre_mean[key] = mmean(Xres, np.array(list(phi)) + 1, expand=True)
                else:
                    pre_mean[key] = mmean(pre_mean[key[:-1]], np.array([list(phi)[-1]]) + 1, expand=True)

            # compute marginalizations
            for key, phi in pcombs.items():
                key_without_phi = ''.join(filter(lambda ch: ch not in key, self.labels))
                # self.labels.translate(None, key)

                # build local dictionary for numexpr
                X = pre_mean[key_without_phi] if len(key_without_phi) > 0 else Xres

                if len(key) > 1:
                    subsets = list(chain.from_iterable(combinations(key, r) for r in range(1, len(key))))
                    subsets = [''.join(subset) for subset in subsets]
                    local_dict = {subset: Xmargs[subset] for subset in subsets}
                    local_dict['X'] = X

                    Xmargs[key] = ne.evaluate('X - ' + ' - '.join(subsets), local_dict=local_dict)
                else:
                    Xmargs[key] = X

        # condense dict if not None
        if isinstance(self.join, dict):
            for key, combs in self.join.items():
                Xshape = np.ones(len(self.labels) + 1, dtype='int')
                for comb in combs:
                    sh = np.array(Xmargs[comb].shape)
                    Xshape[(sh - 1).nonzero()] = sh[(sh - 1).nonzero()]

                tmp = np.zeros(Xshape)

                for comb in combs:
                    tmp += Xmargs[comb]
                    del Xmargs[comb]

                Xmargs[key] = tmp

        Xmargs = dense_marg(X, Xmargs)

        return Xmargs

    def _optimize_regularization(self,X,trialX,center=True,lams='auto'):
        
        # center data
        if center:
            X = X - np.mean(X.reshape((X.shape[0],-1)),1).reshape((X.shape[0],)\
                  + len(self.labels)*(1,))

        # compute variance of data
        varX = np.sum(X**2)

        # test different inits and regularization parameters
        if lams == 'auto':
            N = 45
            lams = np.logspace(0,N,num=N, base=1.4, endpoint=False)*1e-7

        
        # compute crossvalidated score over n_trials repetitions
        scores = self.crossval_score(lams,X,trialX,mean=False)

        # take mean over total scores
        totalscore = np.mean(np.sum(np.dstack([scores[key] for key in list(scores.keys())]),-1),0)

        # Raise warning if optimal lambda lies at boundaries
        if np.argmin(totalscore) == 0 or np.argmin(totalscore) == len(totalscore) - 1:
            if self.debug > 0:
                print("Warning: Optimal regularization parameter lies at the \
                       boundary of the search interval. Please provide \
                       different search list (key: lams).")

        # set minimum as new lambda
        self.regularizer = lams[np.argmin(totalscore)]

        if self.debug > 1:
            print('Optimized regularization, optimal lambda = ', self.regularizer)
            print('Regularization will be fixed; to compute the optimal \
                   parameter again on the next fit, please \
                   set opt_regularizer_flag to True.')

            self.opt_regularizer_flag = False

    def crossval_score(self,lams,X,trialX,means=True):
        
        # placeholder for scores
        scores = np.zeros((self.n_trials,len(lams))) if mean else {key : np.zeros((self.n_trials,len(lams))) for key in list(self.marginalizations.keys())}

        # compute number of samples in each condition
        N_samples = self._get_n_samples(trialX,protect=self.protect)

        for trial in range(self.n_trials):
            # print('hhh')
            print("Starting trial ", trial + 1, "/", self.n_trials)

            # perform split into training and test trials
            trainX, validX = self.train_test_split(X,trialX,N_samples=N_samples)

            # compute marginalization of test and validation data
            trainmXs, validmXs = self._marginalize(trainX), self._marginalize(validX)

            # compute crossvalidation score for every regularization parameter
            for k, lam in enumerate(lams):
                # fit dpca model
                self.regularizer = lam
                self._fit(trainX,mXs=trainmXs,optimize=False)

                # compute crossvalidation score
                if mean:
                    scores[trial,k] = self._score(validX,validmXs)
                else:
                    tmp = self._score(validX,validmXs,mean=False)
                    for key in list(self.marginalizations.keys()):
                        scores[key][trial,k] = tmp[key]

        return scores
    
    def cross_validation(self,X,trialX,center=True, lams='auto',taus='auto'):

        temp_1 = time.time()        
        if center:
            X = X - np.mean(X.reshape((X.shape[0],-1)),1).reshape((X.shape[0],)\
                  + len(self.labels)*(1,))
        print('in _optimzie', type(trialX))

        # compute variance of data
        varX = np.sum(X**2)

        # test different inits and regularization parameters
        if lams == 'auto':

            print('start cross-val to get best lams')
            N = 10
            # lams = np.logspace(0,N,num=N, base=1.4, endpoint=False)*1e-5
            lams = np.logspace(0,30,num=N, base=1.6, endpoint=False)*1e-5

        if taus == 'auto':
            N = 10
            # taus = np.logspace(0,N,num=N, base=1.4, endpoint=False)*1e-5
            taus = np.logspace(0,30,num=N, base=1.6, endpoint=False)*1e-5


        # compute crossvalidated score over n_trials repetitions
        scores = self.cal_score(lams,taus,X,trialX,mean=False)
        # print('scores',scores)
        # print('np_stack',np.stack([scores[key] for key in list(scores.keys())]))

        # take mean over total scores
        # totalscore = np.mean(np.sum(np.stack([scores[key] for key in list(scores.keys())],axis=-1),-1),0)
        totalscore = np.sum(np.stack([scores[key] for key in list(scores.keys())]),-1)
        # print('total_score:',totalscore)
        # Raise warning if optimal lambda lies at boundaries
        if np.argmin(totalscore) == 0 or np.argmin(totalscore) == len(totalscore) - 1:
            if self.debug > 0:
                print("Warning: Optimal regularization parameter lies at the \
                       boundary of the search interval. Please provide \
                       different search list (key: lams).")
                       
        pos = np.unravel_index(np.argmin(totalscore),totalscore.shape)
        # set minimum as new lambda
        self.hyper_lamb = lams[pos[0]]
        self.hyper_tau = taus[pos[1]]

        print('best lambda,tau = ',self.hyper_lamb,self.hyper_tau)

        if self.debug > 1:
            print('Optimized hyperparameter, optimal lambda = ', self.hyper_lamb,'optimal tau = ',self.hyper_tau)

            self.cross_flag = False
        temp_2 = time.time()
        self.cross_val_time = temp_2 - temp_1

    
    def cal_score(self,lams,taus,X,trialX,mean=True):
        # placeholder for scores
        scores = np.zeros((len(lams),len(taus))) if mean else {key : np.zeros((len(lams),len(taus))) for key in list(self.marginalizations.keys())}
        XmXs = self._marginalize(X)
        for k, lam in enumerate(lams):
            for p, tau in enumerate(taus):
                # fit dpca model
                self.hyper_lamb = lam
                self.hyper_tau = tau
                self._fit(X,mXs=XmXs,optimize=False,cross=False)
                # compute crossvalidation score
                if mean:
                    scores[k, p] = self._score(X,XmXs)
                else:
                    tmp = self._score(X,XmXs,mean=False)
                    for key in list(self.marginalizations.keys()):
                        scores[key][k,p] = tmp[key]
        '''scores = np.zeros((self.n_tri,len(lams),len(taus))) if mean else {key : np.zeros((self.n_tri,len(lams),len(taus))) for key in list(self.marginalizations.keys())}
        

        # compute number of samples in each condition
        N_samples = self._get_n_samples(trialX,protect=self.protect)

        for trial in range(self.n_tri):

            # print('hhh')
            print("Starting trial ", trial + 1, "/", self.n_tri)

            # print('N-samples',N_samples)

            # perform split into training and test trials
            trainX, validX = self.train_test_split(X,trialX,N_samples=N_samples)

            # compute marginalization of test and validation data
            trainmXs, validmXs = self._marginalize(trainX), self._marginalize(validX)

            # compute crossvalidation score for every regularization parameter
            for k, lam in enumerate(lams):
                for p, tau in enumerate(taus):
                    # fit dpca model
                    self.hyper_lamb = lam
                    self.hyper_tau = tau
                    self._fit(trainX,mXs=trainmXs,optimize=False,cross=False)

                    # compute crossvalidation score
                    if mean:
                        scores[trial,k, p] = self._score(validX,validmXs)
                    else:
                        tmp = self._score(validX,validmXs,mean=False)
                        for key in list(self.marginalizations.keys()):
                            scores[key][trial,k,p] = tmp[key]'''

        return scores

    def _score(self,X,mXs,mean=True):
        
        n_features = X.shape[0]
        X = X.reshape((n_features,-1))

        error = {key: 0 for key in list(mXs.keys())}
        PDY  = {key : np.dot(self.F[key],np.dot(self.D[key].T,X)) for key in list(mXs.keys())}
        trPD = {key : np.sum(self.F[key]*self.D[key],1) for key in list(mXs.keys())}
        # print('F.shape',F['s'].shape)
        for key in list(mXs.keys()):
            error[key] = np.sum((mXs[key] - PDY[key] + trPD[key][:,None]*X)**2)

        return error if not mean else np.sum(list(error.values()))

    def softhreshold(self, X, lamb):
        S_lamb = np.zeros(X.shape)
        S_lamb[X > lamb] = X[X > lamb] - lamb
        S_lamb[X < -lamb] = X[X < -lamb] + lamb
        # print(S_lamb)
        return np.array(S_lamb)

    def prox_nuclear_norm(self, W, theta_hat, lamb):
        # print('W.shape:',W.shape)
        # print('theta_hat.shape:',theta_hat.shape)
        neg_lamb = -lamb
        # print(neg_lamb)
        result = theta_hat
        result[W > lamb + theta_hat] = theta_hat[W > lamb + theta_hat] + lamb
        result[W < neg_lamb + theta_hat] = theta_hat[W < neg_lamb + theta_hat] - lamb
        return result

    def prox_3(self, W, tau):
        # U, sigma, VT = randomized_svd(W, num_components, n_iter=self.n_iter, random_state=np.random.randint(10e5))
        U, sigma, VT = np.linalg.svd(W)

        Sigma = np.zeros((U.shape[1],U.shape[1]))
        row, col = np.diag_indices_from(Sigma)
        Sigma[row,col] = sigma

        # print('U.shape[1]',U.shape[1],'Sigma.shape[0]:',Sigma.shape[0])
        # print('prox_3-Sigma:',Sigma)
        S_tau = self.softhreshold(Sigma, tau)
        # print('prox_3-S_tau:', S_tau)
        temp1 = np.dot(U, S_tau)
        prox_W = np.dot(temp1, VT)
        return prox_W

    def prox_4(self, W, tau, theta_hat):
        U, sigma, VT = np.linalg.svd(W - theta_hat)

        Sigma = np.zeros((U.shape[1],U.shape[1]))
        row, col = np.diag_indices_from(Sigma)
        Sigma[row,col] = sigma

        # print('U.shape[1]',U.shape[1],'Sigma.shape[0]:',Sigma.shape[0])

        S_tau = self.softhreshold(Sigma, tau)
        # print('Sigma:',Sigma)
        # print('S_tau:', S_tau)
        temp1 = np.dot(U, S_tau)
        temp2 = np.dot(temp1, VT)

        if np.max(sigma) > tau:
            # print('max(sigma)', np.max(sigma))
            result = temp2 + theta_hat
        else:
            result = theta_hat

        return result

    def pca(self, mat,k):
        e_vals,e_vecs = np.linalg.eig(mat)
        sorted_indices = np.argsort(e_vals)
        return e_vals[sorted_indices[:-k-1:-1]],e_vecs[:,sorted_indices[:-k-1:-1]]
    
    def cal_W(self,W_i,ru,a,W,a_i):
        return W_i + ru * (2 * a - W - a_i)

    def EE_dpca(self, X, mXs, pinvX, lamb, tau, T, ru):

        n_features = X.shape[0]
        # print('X',X)
        rX = X.reshape((n_features,-1))
        # print('rX.shape:',rX.shape)
        pinvX = pinv(rX) if pinvX is None else pinvX


        D, F = {}, {}

        for key in list(mXs.keys()):
            print('key',key)
            mX = mXs[key].reshape((n_features, -1))  # called X_phi in paper
            # mX = mXs[key].reshape((-1, n_features))
            # print('mX.shape',mX.shape)
            # print('pinvX.shape',pinvX.shape)
            theta_hat = np.dot(mX,pinvX)
            # print('theta_hat',theta_hat)
            # print('mX',mX)

  
            # initialize W_1, W_2, W_3, W_4, W = DF
            W = W_1 = W_2 = W_3 = W_4 = theta_hat
            # Pool = multiprocessing.Pool()
            for i in range(T):

                # with pathos.multiprocessing.ProcessingPool(4) as pool:
                # with multiprocessing.Pool(4) as pool:

                a_1 = self.softhreshold(W_1, 4 * lamb)
                # print(type(lamb))
                # a_1 = pool.starmap(self.softhreshold,zip(W_1,[4 * lamb]))

                a_2 = self.prox_nuclear_norm(W_2, theta_hat, lamb)
                # a_2 = pool.map(self.prox_nuclear_norm, W_2, theta_hat, lamb)

                a_3 = self.prox_3(W_3, 4 * tau)
                # a_3 = pool.map(self.prox_3,W_3, 4 * tau)
                a_4 = self.prox_4(W_4, tau, theta_hat)
                # a_4 = pool.map(self.prox_4,W_4, tau, theta_hat)

                a = (a_1 + a_2 + a_3 + a_4) / 4
                # with pathos.multiprocessing.ProcessingPool(4) as pool:

                '''W_1 = pool.map(self.cal_W,W_1,ru,a,W,a_1)
                W_2 = pool.map(self.cal_W,W_2,ru,a,W,a_2)
                W_3 = pool.map(self.cal_W,W_3,ru,a,W,a_3)
                W_4 = pool.map(self.cal_W,W_4,ru,a,W,a_4)'''

                W_1 = W_1 + ru * (2 * a - W - a_1)
                W_2 = W_2 + ru * (2 * a - W - a_2)
                W_3 = W_3 + ru * (2 * a - W - a_3)
                W_4 = W_4 + ru * (2 * a - W - a_4)
                W = W + ru * (a - W)


            if isinstance(self.n_components,dict):
                # U,s,V = randomized_svd(np.dot(W,rX), n_components=self.n_components[key],random_state=np.random.randint(10e5))
                U,s,V = np.linalg.svd(np.dot(W,rX))
                print(U.shape)
            else:
                # U,s,V = randomized_svd(np.dot(W,rX), n_components=self.n_components,random_state=np.random.randint(10e5))
                U,s,V = np.linalg.svd(np.dot(W,rX))
            

            F[key] = U
            D[key] = np.dot(U.T,W).T

            # print('D shape', D[key].shape)
            # print('W-DF=',np.linalg.norm(W-np.dot(D[key],F[key])))

            # pca好像也不太靠谱
            '''p = theta_hat.shape[0]
            if isinstance(self.n_components, dict):
                q = self.n_components[key]
            else:
                q = self.n_components

            vals,vecs = self.pca(W,q)
            # print('vals',vals)
            F[key] = np.real(vecs.T)
            # print('F.shape',F[key].shape)
            Lambda = np.diag(np.real(vals))
            D[key] = np.dot(F[key].T, Lambda)
            print('W-DF=', np.linalg.norm(W-np.dot(D[key],F[key])))'''

            # print(W)
            # print(D[key])
            # print('after PCA, D.shape,F.shape',D[key].shape, F[key].shape)


            

            # QR分解8太行，维数不对qwq
            '''D[key], F[key] = np.linalg.qr(W)
            print('after QR,',D[key].shape,F[key].shape)'''
            ''' D[key] = np.random.rand(p, q)
            F[key] = np.random.rand(q, p)
        

            mu = 1

            iteration = 10
            for i in range(iteration):
                print('iter_i:',i)
                temp1 = np.dot(W, F[key].T)
                # print(q)
                # print('F[key]', F[key].shape)
                # print('dot.shape',np.dot(F[key], F[key].T).shape)
                temp2 = np.linalg.inv(np.dot(F[key], F[key].T) + mu * np.identity(q))
                D[key] = np.dot(temp1, temp2)
                temp3 = np.linalg.inv(np.dot(D[key].T, D[key]) + mu * np.identity(q))
                temp4 = np.dot(temp3, D[key].T)
                F[key] = np.dot(temp4, W)
            print(D[key].shape)
'''
        return D, F

    def _add_regularization(self,Y,mYs,lam,SVD=None,pre_reg=False):
        """ Prepares the data matrix and its marginalizations for the randomized_dpca solver (see paper)."""
        n_features = Y.shape[0]

        if not pre_reg:
            regY = np.hstack([Y.reshape((n_features,-1)),lam*np.eye(n_features)])
        else:
            regY = Y
            regY[:,-n_features:] = lam*eye(n_features)

        if not pre_reg:
            regmYs = OrderedDict()

            for key in list(mYs.keys()):
                regmYs[key] = np.hstack([mYs[key],np.zeros((n_features,n_features))])
        else:
            regmYs = mYs

        if SVD is not None:
            U,s,V = SVD

            M = ((s**2 + lam**2)**-1)[:,None]*U.T
            pregY = np.dot(np.vstack([V.T*s[None,:],lam*U]),M)
        else:
            pregY = np.dot(regY.reshape((n_features,-1)).T,np.linalg.inv(np.dot(Y.reshape((n_features,-1)),Y.reshape((n_features,-1)).T) + lam**2*np.eye(n_features)))

        return regY, regmYs, pregY

    def _zero_mean(self, X):
        """ Subtracts the mean from each observable """
        return X - np.mean(X.reshape((X.shape[0], -1)), 1).reshape((X.shape[0],) + (len(X.shape) - 1) * (1,))

    def _roll_back(self, X, axes, invert=False):
        ''' Rolls all axis in list crossval_protect to the end (or inverts if invert=True) '''
        rX = X
        axes = np.sort(axes)

        if invert:
            for ax in reversed(axes):
                rX = np.rollaxis(rX, -1, start=ax)
        else:
            for ax in axes:
                rX = np.rollaxis(rX, ax, start=len(X.shape))

        return rX

    def _get_n_samples(self, trialX, protect=None):
        """ Computes the number of samples for each parameter combinations (except along protect) """
        n_unprotect = len(trialX.shape) - len(protect) - 1 if protect is not None else len(trialX.shape) - 1
        n_protect = len(protect) if protect is not None else 0

        return trialX.shape[0] - np.sum(np.isnan(trialX[(np.s_[:],) + (np.s_[:],) * n_unprotect + (0,) * n_protect]), 0)

    def _check_protected(self, X, protect):
        ''' Checks if protect == None or, alternatively, if all protected axis are at the end '''
        if protect is None:
            protected = True
        else:
            # convert label in index
            protect = [self.labels.index(ax) for ax in protect]
            if set(protect) == set(np.arange(len(self.labels) - len(protect), len(self.labels))):
                protected = True
            else:
                protected = False
                print(
                    'Not all protected axis are at the end! While the algorithm will still work, the performance of the shuffling algorithm will substantially decrease due to unavoidable copies.')

        return protected

    def train_test_split(self, X, trialX, N_samples=None, sample_ax=0):
        def flat2d(A):
            ''' Flattens all but the first axis of an ndarray, returns view. '''
            return A.reshape((A.shape[0], -1))

        protect = self.protect

        n_samples = trialX.shape[-1]  # number of samples
        n_unprotect = len(X.shape) - len(protect) if protect is not None else len(X.shape)
        n_protect = len(protect) if protect is not None else 0

        if sample_ax != 0:
            raise NotImplemented('The sample axis needs to come first.')

        # test if all protected axes lie at the end
        protected = self._check_protected(trialX, protect)

        # reorder matrix to protect certain axis (for speedup)
        if ~protected:
            # turn crossval_protect into index listX
            # axes = [self.labels.index(ax) + 2 for ax in protect]
            axes = [self.labels.index(ax) + 2 for ax in protect]

            # reorder matrix
            trialX = self._roll_back(trialX, axes)
            X = np.squeeze(self._roll_back(X[None, ...], axes))

        # compute number of samples in each condition
        if N_samples is None:
            N_samples = self._get_n_samples(trialX, protect=self.protect)
        print('N_samples', N_samples.shape)
        # get random indices
        idx = (np.random.rand(*N_samples.shape) * N_samples).astype(int)

        # select values
        blindX = np.empty(trialX.shape[1:])
        print('blindX.shape', blindX.shape)

        # iterate over multi_index
        it = np.nditer(np.empty(N_samples.shape), flags=['multi_index'])
        print('n_protect',n_protect)

        while not it.finished:
            # print('trailX',trialX.shape)
            # print('np.s_[:]',np.s_.shape)
            # print('idx',idx[it.multi_index],'it',it.multi_index)
            # print('new_blindX',blindX[it.multi_index + (np.s_[:],) * n_protect].shape)
            # print('new_trialX.shape',trialX[(idx[it.multi_index],) + it.multi_index + (np.s_[:],) * n_protect].shape)
            blindX[it.multi_index + (np.s_[:],) * n_protect] = trialX[(idx[it.multi_index],) + it.multi_index + (np.s_[:],) * n_protect]
            it.iternext()

        # compute trainX
        trainX = (X * (N_samples / (N_samples - 1))[(np.s_[:],) * n_unprotect + (None,) * n_protect] - blindX /
                  (N_samples - 1)[(np.s_[:],) * n_unprotect + (None,) * n_protect])

        # inverse rolled axis in blindX
        if ~protected:
            blindX = self._roll_back(blindX[..., None], axes, invert=True)[..., 0]
            trainX = self._roll_back(trainX[..., None], axes, invert=True)[..., 0]

        # remean datasets (both equally)
        trainX -= np.mean(flat2d(trainX), 1)[(np.s_[:],) + (None,) * (len(X.shape) - 1)]
        blindX -= np.mean(flat2d(blindX), 1)[(np.s_[:],) + (None,) * (len(X.shape) - 1)]

        return trainX, blindX

    def shuffle_labels(self, trialX):
        # import shuffling algorithm from cython source
        protect = self.protect

        # test if all protected axes lie at the end
        protected = self._check_protected(trialX, protect)

        # reorder matrix to protect certain axis (for speedup)
        if ~protected:
            # turn crossval_protect into index list
            axes = [self.labels.index(ax) + 2 for ax in protect]

            # reorder matrix
            trialX = self._roll_back(trialX, axes)

        # reshape all non-protect axis into one vector
        original_shape = trialX.shape
        trialX = trialX.reshape((-1,) + trialX.shape[-len(protect):])

        # reshape all protected axis into one
        original_shape_protected = trialX.shape
        trialX = trialX.reshape((trialX.shape[0], -1))

        # shuffle within non-protected axis
        nan_shuffle.shuffle2D(trialX)

        # inverse reshaping of protected axis
        trialX = trialX.reshape(original_shape_protected)

        # inverse reshaping & sample axis
        trialX = trialX.reshape(original_shape)
        # trialX = np.rollaxis(trialX,0,len(original_shape))

        # inverse rolled axis in trialX
        if protected:
            trialX = self._roll_back(trialX, axes, invert=True)

        return trialX

    def _fit(self, X, trialX=None, mXs=None, center=True, SVD=None, optimize=True,cross=True,hypers=None):

        def flat2d(A):
            return A.reshape((A.shape[0], -1))
        # print('X',X)
        # X = check_array(X)

        n_features = X.shape[0]
        print('original X.shape',X.shape)
        # center data
        if center:
            X = X - np.mean(flat2d(X), 1).reshape((n_features,) + len(self.labels) * (1,))

        # marginalize data
        if mXs is None:
            mXs = self._marginalize(X)
            # print('original mX.shape', mXs.shape)

        if self.cross_flag and cross:
            self.cross_validation(X,trialX)

        if self.opt_regularizer_flag and optimize:
            if self.debug > 0:
                print("OMG,Start optimizing regularization.")

            if trialX is None:
                raise ValueError('To optimize the regularization parameter, the trial-by-trial data trialX needs to be provided.')

            self._optimize_regularization(X,trialX)

        # add regularization
        if self.regularizer > 0:
            regX, regmXs, pregX = self._add_regularization(X,mXs,self.regularizer*np.sum(X**2),SVD=SVD)
        else:
            regX, regmXs, pregX = X, mXs, pinv(X.reshape((n_features,-1)))
            # regX, regmXs, pregX = X, mXs, pinv(X.reshape((-1,n_features)))
            # print('pregX',pregX)
        
        # time_1 = time.time()

        print('lambda',self.hyper_lamb, 'tau',self.hyper_tau)

        # time_2 = time.time()

        # cross_val_time = time_2 - time_1


        # lambda = 0.1, tau = 0.1, T = 10, ru = 1
        self.D, self.F = self.EE_dpca(regX, regmXs, pinvX = pregX, lamb = self.hyper_lamb, tau = self.hyper_tau, T = 100, ru = self.rho)

    def transform(self, X, marginalization=None):
        X = self._zero_mean(X)
        total_variance = np.sum((X - np.mean(X)) ** 2)

        def marginal_variances(marginal):
            D, Xr = self.D[marginal], X.reshape((X.shape[0], -1))
            return [np.sum(np.dot(D[:, k], Xr) ** 2) / total_variance for k in range(D.shape[1])]

        if marginalization is not None:
            D, Xr = self.D[marginalization], X.reshape((X.shape[0], -1))
            # print('D.shape', D.shape)
            X_transformed = np.dot(D.T, Xr).reshape((D.shape[1],) + X.shape[1:])
            self.explained_variance_ratio_ = {marginalization: marginal_variances(marginalization)}
        else:
            X_transformed = {}
            self.explained_variance_ratio_ = {}
            for key in list(self.marginalizations.keys()):
                X_transformed[key] = np.dot(self.D[key].T, X.reshape((X.shape[0], -1))).reshape(
                    (self.D[key].shape[1],) + X.shape[1:])
                self.explained_variance_ratio_[key] = marginal_variances(key)
                # print('D.shape',key, D[key].shape)


        self.end = time.time()
        self.total_time = self.end - self.start - self.cross_val_time

        return X_transformed

    def inverse_transform(self, X, marginalization):
        X = self._zero_mean(X)
        X_transformed = np.dot(self.P[marginalization], X.reshape((X.shape[0], -1))).reshape(
            (self.P[marginalization].shape[0],) + X.shape[1:])

        return X_transformed

    def reconstruct(self, X, marginalization):
        return self.inverse_transform(self.transform(X, marginalization), marginalization)
