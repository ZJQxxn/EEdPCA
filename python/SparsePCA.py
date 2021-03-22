"""Matrix factorization with Sparse PCA"""
# Author: Vlad Niculae, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

from __future__ import print_function
import sys
sys.path.append('./')

import numpy as np
from collections import OrderedDict
from itertools import combinations, chain
from scipy.sparse.linalg import svds
from scipy.linalg import pinv

import pywt

from sklearn.base import BaseEstimator
from sklearn.utils.extmath import randomized_svd
import numexpr as ne
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from dPCA import nan_shuffle
import time


class SparsePCA(BaseEstimator):


    def __init__(self, labels=None, join=None, n_components=10, copy=True, max_iter=10, threshold_val=1.5):
        if isinstance(labels, str):
            self.labels = labels
        elif isinstance(labels, int):
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            labels = alphabet[:labels]
        else:
            raise TypeError(
                'Wrong type for labels. Please either set labels to the number of variables or provide the axis labels as a single string of characters (like "ts" for time and stimulus)')

        self.n_components = n_components
        self.max_iter=max_iter
        self.threshold_val=threshold_val
        self.join = join
        self.copy = copy
        self.marginalizations = self._get_parameter_combinations()
        self.start = self.end = self.total_time = self.cross_val_time = 0


    def fit(self, X, trialX=None):

        self._fit(X,trialX=trialX)
        # print('hhh')
        return self

    def fit_transform(self, X, trialX=None):
        
        self.start = time.time()
        self._fit(X,trialX=trialX)

        return self.transform(X)

    def _get_parameter_combinations(self,join=True):
        
        # subsets = () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)"
        subsets = list(chain.from_iterable(combinations(list(range(len(self.labels))), r) for r in range(len(self.labels))))

        # delete empty set & add (0,1,2)
        del subsets[0]
        subsets.append(list(range(len(self.labels))))

        # create dictionary
        pcombs = OrderedDict()
        for subset in subsets:
            key = ''.join([self.labels[i] for i in subset])
            pcombs[key] = set(subset)

        # condense dict if not None
        if isinstance(self.join,dict) and join:
            for key, combs in self.join.items():
                tmp = [pcombs[comb] for comb in combs]

                for comb in combs:
                    del pcombs[comb]

                pcombs[key] = tmp
        # print(pcombs['s'])
        return pcombs

    def _marginalize(self,X,save_memory=False):
        

        def mmean(X,axes,expand=False):
            
            Z = X.copy()

            for ax in np.sort(axes)[::-1]:
                Z = np.mean(Z,ax)

                if expand == True:
                    Z = np.expand_dims(Z,ax)

            return Z

        def dense_marg(Y,mYs):
            
            tmp = np.zeros_like(Y)
            for key in list(mYs.keys()):

                mYs[key] = (tmp + mYs[key]).reshape((Y.shape[0],-1))

            return mYs

        Xres = X.copy()      # residual of data

        # center data
        Xres -= np.mean(Xres.reshape((Xres.shape[0],-1)),-1).reshape((Xres.shape[0],) + (len(Xres.shape)-1)*(1,))

        # init dict with marginals
        Xmargs = OrderedDict()

        # get parameter combinations
        pcombs = self._get_parameter_combinations(join=False)

        # subtract the mean
        S = list(pcombs.values())[-1]    # full set of indices

        if save_memory:
            for key, phi in pcombs.items():
                S_without_phi = list(S - phi)

                # compute marginalization and save
                Xmargs[key] = mmean(Xres,np.array(S_without_phi)+1,expand=True)

                # subtract the marginalization from the data
                Xres -= Xmargs[key]
        else:
            # efficient precomputation of means
            pre_mean = {}

            for key, phi in pcombs.items():
                if len(key) == 1:
                    pre_mean[key] = mmean(Xres,np.array(list(phi))+1,expand=True)
                else:
                    pre_mean[key] = mmean(pre_mean[key[:-1]],np.array([list(phi)[-1]])+1,expand=True)

            # compute marginalizations
            for key, phi in pcombs.items():
                key_without_phi = ''.join(filter(lambda ch: ch not in key, self.labels))
                # self.labels.translate(None, key)

                # build local dictionary for numexpr
                X = pre_mean[key_without_phi] if len(key_without_phi) > 0 else Xres

                if len(key) > 1:
                    subsets = list(chain.from_iterable(combinations(key, r) for r in range(1,len(key))))
                    subsets = [''.join(subset) for subset in subsets]
                    local_dict = {subset : Xmargs[subset] for subset in subsets}
                    local_dict['X'] = X

                    Xmargs[key] = ne.evaluate('X - ' + ' - '.join(subsets),local_dict=local_dict)
                else:
                    Xmargs[key] = X

        # condense dict if not None
        if isinstance(self.join,dict):
            for key, combs in self.join.items():
                Xshape = np.ones(len(self.labels)+1,dtype='int')
                for comb in combs:
                    sh = np.array(Xmargs[comb].shape)
                    Xshape[(sh-1).nonzero()] = sh[(sh-1).nonzero()]

                tmp = np.zeros(Xshape)

                for comb in combs:
                    tmp += Xmargs[comb]
                    del Xmargs[comb]

                Xmargs[key] = tmp

        Xmargs = dense_marg(X,Xmargs)

        # print(Xmargs['s'])
        return Xmargs


    def _sparsepca(self,X,mXs):
        
        # print('X',X)
        n_features = X.shape[0]
        rX = X.reshape((n_features,-1))
        # print('rX.shape[1]:',rX.shape[1])
        U_total,V_total,W_total = {}, {}

        for key in list(mXs.keys()):

            mX = mXs[key].reshape((n_features, -1))
            U, s, V = np.linalg.svd(mX, full_matrices=True)  
            cnt = 0
            U_total[key] = U
            W_total[key] = V.T
            def normalize(vector):
                norm=np.linalg.norm(vector)
                if norm>0:
                    return vector/norm
                else:
                    return vector
            print("starting iterations...")
            while True:
            
                V_total[key] = pywt.threshold(np.dot(U[:self.n_components],mX), self.threshold_val)
                U_total[key] = np.dot(V_total[key],mX.T)
                U_total[key] = np.array([normalize(u_i) for u_i in U_total[key]])
                if cnt%2==0:
                    print("{} out of {} iterations".format(cnt,self.max_iter))
                cnt += 1
                if cnt == self.max_iter:
                    V[key] = np.array([normalize(v_i) for v_i in V[key]])
                    break

        return U_total, V_total, W_total



    def _fit(self, X, trialX=None, mXs=None, center=True, SVD=None, optimize=True):
        

        def flat2d(A):
            ''' Flattens all but the first axis of an ndarray, returns view. '''
            return A.reshape((A.shape[0],-1))

        # X = check_array(X)

        n_features = X.shape[0]

        # print('X',X)
        # center data
        if center:
            X = X - np.mean(flat2d(X),1).reshape((n_features,) + len(self.labels)*(1,))
        # print(X)
        # marginalize data
        if mXs is None:
            mXs = self._marginalize(X)
            # print(mXs)
            
        # compute closed-form solution
        self.U, self.V, self.W  = self._sparsepca(X,mXs)


    def _zero_mean(self,X):
        """ Subtracts the mean from each observable """
        return X - np.mean(X.reshape((X.shape[0],-1)),1).reshape((X.shape[0],) + (len(X.shape)-1)*(1,))

    def _roll_back(self,X,axes,invert=False):
        ''' Rolls all axis in list crossval_protect to the end (or inverts if invert=True) '''
        rX = X
        axes = np.sort(axes)

        if invert:
            for ax in reversed(axes):
                rX = np.rollaxis(rX,-1,start=ax)
        else:
            for ax in axes:
                rX = np.rollaxis(rX,ax,start=len(X.shape))

        return rX

    def _get_n_samples(self,trialX,protect=None):
        print('get_n_samples:',type(trialX))
        """ Computes the number of samples for each parameter combinations (except along protect) """
        n_unprotect = len(trialX.shape) - len(protect) - 1 if protect is not None else len(trialX.shape) - 1
        n_protect   = len(protect) if protect is not None else 0

        return trialX.shape[0] - np.sum(np.isnan(trialX[(np.s_[:],) + (np.s_[:],)*n_unprotect + (0,)*n_protect]),0)

    def _check_protected(self,X,protect):
        ''' Checks if protect == None or, alternatively, if all protected axis are at the end '''
        if protect is None:
            protected = True
        else:
            # convert label in index
            protect = [self.labels.index(ax) for ax in protect]
            if set(protect) == set(np.arange(len(self.labels)-len(protect),len(self.labels))):
                protected = True
            else:
                protected = False
                print('Not all protected axis are at the end! While the algorithm will still work, the performance of the shuffling algorithm will substantially decrease due to unavoidable copies.')

        return protected

    def train_test_split(self,X,trialX,N_samples=None,sample_ax=0):
        
        def flat2d(A):
            ''' Flattens all but the first axis of an ndarray, returns view. '''
            return A.reshape((A.shape[0],-1))

        protect = self.protect

        n_samples   = trialX.shape[-1]                       # number of samples
        n_unprotect = len(X.shape) - len(protect) if protect is not None else len(X.shape)
        n_protect   = len(protect) if protect is not None else 0

        if sample_ax != 0:
            raise NotImplemented('The sample axis needs to come first.')

        # test if all protected axes lie at the end
        protected = self._check_protected(trialX,protect)

        # reorder matrix to protect certain axis (for speedup)
        if ~protected:
            # turn crossval_protect into index listX
            axes = [self.labels.index(ax) + 2 for ax in protect]

            # reorder matrix
            trialX = self._roll_back(trialX,axes)
            X = np.squeeze(self._roll_back(X[None,...],axes))

        # compute number of samples in each condition
        if N_samples is None:
            N_samples = self._get_n_samples(trialX,protect=self.protect)

        # get random indices
        idx = (np.random.rand(*N_samples.shape)*N_samples).astype(int)

        # select values
        blindX = np.empty(trialX.shape[1:])

        # iterate over multi_index
        it = np.nditer(np.empty(N_samples.shape), flags=['multi_index'])

        while not it.finished:
            blindX[it.multi_index + (np.s_[:],)*n_protect] = trialX[(idx[it.multi_index],) + it.multi_index + (np.s_[:],)*n_protect]
            it.iternext()

        # compute trainX
        trainX = (X*(N_samples/(N_samples-1))[(np.s_[:],)*n_unprotect + (None,)*n_protect] - blindX/(N_samples-1)[(np.s_[:],)*n_unprotect + (None,)*n_protect])

        # inverse rolled axis in blindX
        if ~protected:
            blindX = self._roll_back(blindX[...,None],axes,invert=True)[...,0]
            trainX = self._roll_back(trainX[...,None],axes,invert=True)[...,0]

        # remean datasets (both equally)
        trainX -= np.mean(flat2d(trainX),1)[(np.s_[:],) + (None,)*(len(X.shape)-1)]
        blindX -= np.mean(flat2d(blindX),1)[(np.s_[:],) + (None,)*(len(X.shape)-1)]

        return trainX, blindX

    def shuffle_labels(self,trialX):
        
        

        # import shuffling algorithm from cython source
        protect = self.protect

        # test if all protected axes lie at the end
        protected = self._check_protected(trialX,protect)

        # reorder matrix to protect certain axis (for speedup)
        if ~protected:
            # turn crossval_protect into index list
            axes = [self.labels.index(ax) + 2 for ax in protect]

            # reorder matrix
            trialX = self._roll_back(trialX,axes)

        # reshape all non-protect axis into one vector
        original_shape = trialX.shape
        trialX = trialX.reshape((-1,) + trialX.shape[-len(protect):])

        # reshape all protected axis into one
        original_shape_protected = trialX.shape
        trialX = trialX.reshape((trialX.shape[0],-1))

        # shuffle within non-protected axis
        nan_shuffle.shuffle2D(trialX)

        # inverse reshaping of protected axis
        trialX = trialX.reshape(original_shape_protected)

        # inverse reshaping & sample axis
        trialX = trialX.reshape(original_shape)
        #trialX = np.rollaxis(trialX,0,len(original_shape))

        # inverse rolled axis in trialX
        if protected:
            trialX = self._roll_back(trialX,axes,invert=True)

        return trialX

    def significance_analysis(self,X,trialX,n_shuffles=100,n_splits=100,n_consecutive=1,axis=None,full=False):
        
        def compute_mean_score(X,trialX,n_splits):
            K = 1 if axis is None else X.shape[-1]

            if type(self.n_components) == int:
                scores = {key : np.empty((self.n_components,K)) for key in keys}
            else:
                scores = {key : np.empty((self.n_components[key],K)) for key in keys}

            for shuffle in range(n_splits):
                print('.', end=' ')

                # do train-validation split
                trainX, validX = self.train_test_split(X,trialX)

                # fit a dPCA model to training data & transform validation data
                trainZ = self.fit_transform(trainX)
                validZ = self.transform(validX)

                # reshape data to match Cython input
                for key in keys:
                    ncomps = self.n_components if type(self.n_components) == int else self.n_components[key]

                    # mean over all axis not in key
                    axset = self.marginalizations[key]
                    axset = axset if type(axset) == set else set.union(*axset)
                    axes = set(range(len(X.shape)-1)) - axset
                    for ax in axes:
                        trainZ[key] = np.mean(trainZ[key],axis=ax+1)
                        validZ[key] = np.mean(validZ[key],axis=ax+1)

                    # reshape
                    if len(X.shape)-2 in axset and axis is not None:
                        trainZ[key] = trainZ[key].reshape((ncomps,-1,K))
                        validZ[key] = validZ[key].reshape((ncomps,-1,K))
                    else:
                        trainZ[key] = trainZ[key].reshape((ncomps,-1,1))
                        validZ[key] = validZ[key].reshape((ncomps,-1,1))

                # compute classification score
                for key in keys:
                    ncomps = self.n_components if type(self.n_components) == int else self.n_components[key]
                    for comp in range(ncomps):
                        scores[key][comp] = nan_shuffle.classification(trainZ[key][comp],validZ[key][comp])

            return scores

        if self.opt_regularizer_flag:
            print("Regularization not optimized yet; start optimization now.")
            self._optimize_regularization(X,trialX)

        keys = list(self.marginalizations.keys())
        keys.remove(self.labels[-1])

        # shuffling is in-place, so we need to copy the data
        trialX = trialX.copy()

        # compute score of original data
        print("Compute score of data: ", end=' ')
        true_score = compute_mean_score(X,trialX,n_splits)
        print("Finished.")

        # data collection
        scores = {key : [] for key in keys}

        # iterate over shuffles
        for it in range(n_shuffles):
            print("\rCompute score of shuffled data: ", str(it), "/", str(n_shuffles), end=' ')

            # shuffle labels
            self.shuffle_labels(trialX)

            # mean trial-by-trial data
            X = np.nanmean(trialX,axis=0)

            score = compute_mean_score(X,trialX,n_splits)

            for key in keys:
                scores[key].append(score[key])

        # binary mask, if data score is above maximum shuffled score make true
        masks = {}
        for key in keys:
            maxscore = np.amax(np.dstack(scores[key]),-1)
            masks[key] = true_score[key] > maxscore

        if n_consecutive > 1:
            for key in keys:
                mask = masks[key]

                for k in range(mask.shape[0]):
                    masks[key][k,:] = nan_shuffle.denoise_mask(masks[key][k].astype(np.int32),n_consecutive)

        if full:
            return masks, true_score, scores
        else:
            return masks

    def transform(self, X, marginalization=None):
        
        X = self._zero_mean(X)
        total_variance = np.sum((X - np.mean(X))**2)

        def marginal_variances(marginal):
            ''' Computes the relative variance explained of each component
                within a marginalization
            '''
            V, Xr = self.V[marginal], X.reshape((X.shape[0],-1))
            return [np.sum(np.dot(V[:,k], Xr)**2) / total_variance for k in range(V.shape[1])]

        X_transformed = {}
        self.explained_variance_ratio_ = {}
        for key in list(self.marginalizations.keys()):
            # print('D[key]',self.D[key])
            X_transformed[key] = np.dot(self.V[key].T, X.reshape((X.shape[0],-1))).reshape((self.V[key].shape[1],) + X.shape[1:])
            self.explained_variance_ratio_[key] = marginal_variances(key)
            # print(self.explained_variance_ratio_[key])
        
        # print(X_transformed['t'][0,5])
        
        self.end = time.time()
        self.total_time = self.end - self.start

        return X_transformed

    def inverse_transform(self, X, marginalization):
        
        X = self._zero_mean(X)
        X_transformed = np.dot(self.P[marginalization],X.reshape((X.shape[0],-1))).reshape((self.P[marginalization].shape[0],) + X.shape[1:])
 
        return X_transformed

    def reconstruct(self, X, marginalization):
        
        return self.inverse_transform(self.transform(X,marginalization),marginalization)



