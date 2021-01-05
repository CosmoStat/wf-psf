import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import time
import logging
import multiprocessing as mp
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from functools import partial


######################
# PLOTTING FUNCTIONS #
######################
def plot_func(flatarr, wind=False, savepath='', cmap='gist_stern'):
    sqrtN = int(np.sqrt(flatarr.shape[0]))
    if not wind:
        plt.imshow(flatarr.reshape(sqrtN,sqrtN), cmap=cmap, 
                   interpolation='Nearest')
    else:
        vmin, vmax = wind
        plt.imshow(flatarr.reshape(sqrtN,sqrtN), cmap=cmap, 
                   interpolation='Nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def plot_func_1D(data, title='', ymargin=(.1,.3), common=False):
    if len(data.shape)==1:
        grid_size = data.shape[0]
        xs = np.array(range(grid_size))-.5
        plt.bar(xs,data)
        plt.xlim([-.5,grid_size-.5])
        plt.ylim(np.min(data)-ymargin[0],np.max(data)+ymargin[1])
        plt.xticks(range(grid_size))
        plt.title(title)
        plt.xticks([])
        plt.show()
        plt.close()
    else:
        grid_size = data.shape[1]
        xs = np.array(range(grid_size))-.5
        for j,bump in enumerate(data):
            plt.bar(xs,bump)
            if not common:
                print j
                plt.xlim([-.5,grid_size-.5])
                plt.ylim(np.min(data)-ymargin[0],np.max(data)+ymargin[1])
                plt.xticks(range(grid_size))
                plt.title(title)
                plt.xticks([])
                plt.show()
                plt.close()
        if common:
            plt.xlim([-.5,grid_size-.5])
            plt.ylim(np.min(data)-ymargin[0],np.max(data)+ymargin[1])
            plt.xticks(range(grid_size))
            plt.title(title)
            plt.xticks([])
            plt.show()
            plt.close()

def plot_func_2D(im, wind=False):
    if not wind:
        plt.imshow(im, cmap='gist_stern', 
                   interpolation='Nearest')
    else:
        vmin, vmax = wind
        plt.imshow(im, cmap='gist_stern', 
                   interpolation='Nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


#############
# UTILITIES #
#############
def EuclidCost(Nr, Nc, divmed=False, timeit=False, trunc=False, maxtol=745.13, truncval=745.13):
    if timeit:
        start = time.time()
    N = Nr * Nc
    C = np.zeros((N,N))
    for k1 in range(N):
        for k2 in range(k1):
            r1, r2 = int(float(k1) / Nc)+1, int(float(k2) / Nc)+1
            c1, c2 = k1%Nc + 1, k2%Nc + 1
            C[k1, k2] = (r1-r2)**2 + (c1-c2)**2
            C[k2, k1] = C[k1, k2]
    if timeit:
        print 'cost matrix computed in '+str(time.time()-start)+'s.'
    if divmed:
        C /= np.median(C)
    if trunc:
        C[C>maxtol] = truncval
    return C
    
def EuclidCostRGBHist(Nbins, divmed=False, timeit=False, trunc=False, 
                      dtype='float64', maxtol=745.13, truncval=745.13):
    if timeit:
        start = time.time()
    N = Nbins**3
    idx=0
    bins = np.empty((N, 3))
    for i in range(Nbins):
        for j in range(Nbins):
            for k in range(Nbins):
                bins[idx,:] = [i,j,k]
                idx += 1
    C = np.zeros((N,N)).astype(dtype)
    for i in range(N):
        for j in range(i+1):
            C[i,j] = np.linalg.norm(bins[i]-bins[j])**2
            if i != j:
                C[j,i] = C[i,j]
    if timeit:
        print 'cost matrix computed in '+str(time.time()-start)+'s.'
    if divmed:
        C /= np.median(C)
    if trunc:
        C[C>maxtol] = truncval
    return C
    
def nab_L2(p,q):
    return p-q
        
def unwrap_rep(dicweights, datashape):
    n,p = datashape
    Ys, w = dicweights[:p,:], dicweights[p:,:]
    return Ys, w
    
def alphatolbda(alpha):
    return (np.exp(alpha).T / np.sum(np.exp(alpha), axis=1)).T
    
        

    
    
##############
### THEANO ###
##############
# define Theano variables
Datapoint = T.vector('Datapoint')
Cost = T.matrix('Cost')
Gamma = T.scalar('Gamma')
Ker = T.exp(-Cost/Gamma)
n_iter = T.iscalar('n_iter')
Tau = T.scalar('Tau')
Rho = T.scalar('Rho')

# variable change (for simplex constraint)
def varchange(newvar):
    return T.exp(newvar)/T.sum(T.exp(newvar))

# define weights and dictionary
Newvar_lbda = T.vector('Newvar_lbda')
Newvar_D = T.matrix('Newvar_D')
lbda = varchange(Newvar_lbda)
D, D_varchange_updates = theano.scan(varchange, sequences=[Newvar_D.T])
D = D.T

theano_varchange = theano.function([Newvar_D], D)

## Regular version ##
# Sinkhorn barycenter iteration
def sinkhorn_step(a,b,p,D,lbda,Ker,Tau):
    newa = D/T.dot(Ker,b)
    a = a**Tau * newa**(1.-Tau)
    #a = Tau*a + (1-Tau)*newa
    p = T.prod(T.dot(Ker.T,a)**lbda, axis=1)
    newb = p.dimshuffle(0,'x')/T.dot(Ker.T,a)
    b = b**Tau * newb**(1.-Tau)
    #b = Tau*b + (1-Tau)*newb
    return a,b,p

# Unbalanced Sinkhorn barycenter iteration
def unbal_sinkhorn_step(a,b,p,D,lbda,Ker,Tau,Rho):
    newa = (D/T.dot(Ker,b))**(Rho / (Rho+Gamma))
    a = a**Tau * newa**(1.-Tau)
    #p = T.prod(T.dot(Ker.T,a)**lbda, axis=1)
    p = T.sum(T.dot(Ker.T,a)**(Gamma/(Gamma+Rho))*lbda, axis=1)**((Rho+Gamma)/Gamma)
    newb = p.dimshuffle(0,'x')/T.dot(Ker.T,a)
    newb = newb**(Rho / (Rho+Gamma))
    b = b**Tau * newb**(1.-Tau)
    return a,b,p

# Sinkhorn algorithm
result, updates = theano.scan(sinkhorn_step, outputs_info=[T.ones_like(D),
                              T.ones_like(D), T.ones_like(D[:,0])], 
                              non_sequences=[D,lbda,Ker,Tau], n_steps=n_iter)

# Unbalanced Sinkhorn algorithm
unbal_result, unbal_updates = theano.scan(unbal_sinkhorn_step, outputs_info=[T.ones_like(D),
                              T.ones_like(D), T.ones_like(D[:,0])], 
                              non_sequences=[D,lbda,Ker,Tau,Rho], n_steps=n_iter)

# keep only the final barycenter
bary = result[2][-1]
unbal_bary = unbal_result[2][-1]
unbal_bary = unbal_bary / T.sum(unbal_bary)

# Theano barycenter function
Theano_wass_bary = theano.function([D,lbda,Gamma,Cost,n_iter, theano.In(Tau,value=0)], bary)

Loss = 1./2*(Datapoint-bary).norm(2)**2
KLLoss = T.sum(bary*T.log(bary/Datapoint - bary + Datapoint))
L1 = (Datapoint-bary).norm(1)

Grads = T.grad(Loss, [D,lbda])
KLGrads = T.grad(KLLoss, [D,lbda])
L1Grads = T.grad(L1, [D,lbda])

Theano_wass_grad = theano.function([Datapoint, D, lbda, Gamma, Cost, n_iter, theano.In(Tau,value=0)], 
                                    outputs=[Loss]+Grads, updates=updates)
Theano_KL_grad = theano.function([Datapoint, D, lbda, Gamma, Cost, n_iter, theano.In(Tau,value=0)], 
                                    outputs=[KLLoss]+KLGrads, updates=updates)
Theano_L1_grad = theano.function([Datapoint, D, lbda, Gamma, Cost, n_iter, theano.In(Tau,value=0)], 
                                    outputs=[L1]+L1Grads, updates=updates)

# Unbalanced Theano barycenter function
unbal_wass_bary = theano.function([D,lbda,Gamma,Cost,n_iter, Rho, theano.In(Tau,value=0)], unbal_bary)

unbal_Loss = 1./2*(Datapoint-unbal_bary).norm(2)**2
unbal_KLLoss = T.sum(unbal_bary*T.log(unbal_bary/Datapoint - unbal_bary + Datapoint))
unbal_L1 = (Datapoint-unbal_bary).norm(1)

unbal_Grads = T.grad(unbal_Loss, [D,lbda])
unbal_KLGrads = T.grad(unbal_KLLoss, [D,lbda])
unbal_L1Grads = T.grad(unbal_L1, [D,lbda])

unbal_wass_grad = theano.function([Datapoint, D, lbda, Gamma, Cost, n_iter, Rho,
                                   theano.In(Tau,value=0)], 
                                    outputs=[unbal_Loss]+unbal_Grads, updates=unbal_updates)
unbal_KL_grad = theano.function([Datapoint, D, lbda, Gamma, Cost, n_iter, Rho,
                                   theano.In(Tau,value=0)], 
                                    outputs=[unbal_KLLoss]+unbal_KLGrads, updates=unbal_updates)
unbal_L1_grad = theano.function([Datapoint, D, lbda, Gamma, Cost, n_iter, Rho,
                                   theano.In(Tau,value=0)], 
                                    outputs=[unbal_L1]+unbal_L1Grads, updates=unbal_updates)
                                    
# compute grad after change of variable
varchange_Grads = T.grad(Loss, [Newvar_D,Newvar_lbda])
varchange_KLGrads = T.grad(KLLoss, [Newvar_D,Newvar_lbda])
varchange_L1Grads = T.grad(L1, [Newvar_D,Newvar_lbda])
                                        
varchange_Theano_wass_grad = theano.function([Datapoint, Newvar_D, Newvar_lbda, Gamma, Cost, n_iter, 
                                              theano.In(Tau,value=0)],
                                              outputs=[Loss]+varchange_Grads,
                                              updates=updates)
                                        
varchange_Theano_KL_grad = theano.function([Datapoint, Newvar_D, Newvar_lbda, Gamma, Cost, n_iter, 
                                              theano.In(Tau,value=0)],
                                              outputs=[KLLoss]+varchange_KLGrads,
                                              updates=updates)
                                        
varchange_Theano_L1_grad = theano.function([Datapoint, Newvar_D, Newvar_lbda, Gamma, Cost, n_iter, 
                                              theano.In(Tau,value=0)],
                                              outputs=[L1]+varchange_L1Grads,
                                              updates=updates)

# compute grad after change of variable (unbalanced)
unbal_varchange_Grads = T.grad(unbal_Loss, [Newvar_D,Newvar_lbda])
unbal_varchange_KLGrads = T.grad(unbal_KLLoss, [Newvar_D,Newvar_lbda])
unbal_varchange_L1Grads = T.grad(unbal_L1, [Newvar_D,Newvar_lbda])

unbal_varchange_wass_grad = theano.function([Datapoint, Newvar_D, Newvar_lbda, Gamma, Cost, n_iter, Rho,
                                              theano.In(Tau,value=0)], 
                                              outputs=[unbal_Loss]+unbal_varchange_Grads,
                                              updates=unbal_updates)

unbal_varchange_KL_grad = theano.function([Datapoint, Newvar_D, Newvar_lbda, Gamma, Cost, n_iter, Rho,
                                              theano.In(Tau,value=0)], 
                                              outputs=[unbal_KLLoss]+unbal_varchange_KLGrads,
                                              updates=unbal_updates)

unbal_varchange_L1_grad = theano.function([Datapoint, Newvar_D, Newvar_lbda, Gamma, Cost, n_iter, Rho,
                                              theano.In(Tau,value=0)], 
                                              outputs=[unbal_L1]+unbal_varchange_L1Grads,
                                              updates=unbal_updates)

# multiprocessing wrapper
def mp_Theano_grad(Xw, Ys, gamma, C, n_iter_sinkhorn, tau, rho, unbalanced, loss):
    datapoint, wi = Xw
    if unbalanced:
        if loss=='L2':
            return unbal_varchange_wass_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
        elif loss=='L1':
            return unbal_varchange_L1_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
        elif loss=='KL':
            return unbal_varchange_KL_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
    else:
        if loss=='L2':
            return varchange_Theano_wass_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, tau)
        elif loss=='L1':
            return varchange_Theano_L1_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, tau)
        elif loss=='KL':
            return varchange_Theano_KL_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, tau)
            


# L-BFGS wrapper
def LBFGSFunc(dicweights, X, gamma, C, n_components, n_iter_sinkhorn=20, 
              tau=0, rho=float('inf'), varscale=100, 
              unbalanced=False, loss='L2', n_process=4, 
              Verbose=False, savepath='', logpath='', checkplots=False):
    start = time.time()
    n, p = X.shape
    dicweights = dicweights.reshape(n+p,n_components)
    Ys, w = unwrap_rep(dicweights, (n,p))
    
    if n_process>1:
        pool = mp.Pool(n_process)
        mp_grads = partial(mp_Theano_grad, Ys=Ys, gamma=gamma, C=C,
                                n_iter_sinkhorn=n_iter_sinkhorn, tau=tau, rho=rho,
                                unbalanced=unbalanced, loss=loss)
        Xw = zip(X,w)
        res = pool.map(mp_grads, Xw)
        err = 0 
        fullgrad = np.zeros((dicweights.shape))
        for i, (this_err, grad, graw) in enumerate(res):
            err += this_err
            fullgrad[:p,:] += grad/n
            fullgrad[p+i,:] = varscale*graw
        pool.close()
        pool.join()
    else:
        err = 0
        fullgrad = np.zeros((dicweights.shape))
        for i,(datapoint,wi) in enumerate(zip(X,w)):
            if unbalanced:
                if loss=='L2':
                    this_err, grad, graw = unbal_varchange_wass_grad(datapoint, 
                                          Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
                elif loss=='L1':
                    this_err, grad, graw = unbal_varchange_L1_grad(datapoint, 
                                          Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
                elif loss=='KL':
                    this_err, grad, graw = unbal_varchange_KL_grad(datapoint, 
                                          Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
            else:
                if loss=='L2':
                    this_err, grad, graw = varchange_Theano_wass_grad(datapoint, 
                                          Ys, wi, gamma, C, n_iter_sinkhorn, tau)
                elif loss=='L1':
                    this_err, grad, graw = varchange_Theano_L1_grad(datapoint, 
                                          Ys, wi, gamma, C, n_iter_sinkhorn, tau)
                elif loss=='KL':
                    this_err, grad, graw = varchange_Theano_KL_grad(datapoint, 
                                          Ys, wi, gamma, C, n_iter_sinkhorn, tau)
            err += this_err
            fullgrad[:p,:] += grad/n
            fullgrad[p+i,:] = varscale*graw
    if Verbose:
        info = 'Current Error: {} - Duration: {} - Time: {}:{}'.format(
                    err, time.time()-start, time.localtime()[3], time.localtime()[4])
        if logpath:
            logging.info(info)
        else:
            print info
    if savepath and LBFGSFunc.besterr>err:
        LBFGSFunc.besterr = err
        np.save(savepath+'dicweights.npy',dicweights)
        if checkplots:
            for i,yi in enumerate(alphatolbda(Ys.T)):
                plot_func(yi, savepath=savepath+'atom_{}.png'.format(i))
    return err, fullgrad.flatten()
LBFGSFunc.besterr = 1e10
    
def LBFGSDescent(X, n_components, gamma, n_iter_sinkhorn=20, C=None,
                tau=0, rho=float('inf'), varscale=100, unbalanced=False, loss='L2',
                feat_0=None, feat_init="random", wgt_0=None, wgt_init="uniform",
                n_process=1, Verbose=False, savepath='', logpath='', checkplots=False,
                factr=1e7, pgtol=1e-05, maxiter=15000):
    """Compute Wasserstein dictionary and weights.

        Parameters
        ----------
        X : np.ndarray
            Input data to learn the representation on.
            
        n_components : int
            Number of desired atoms in the representation.
            
        gamma : float
            Entropic penalty parameter.
            
        n_iter_sinkhorn : int
            Number of 'Sinkhorn' iterations for barycenter computation.
            
        C : np.ndarray
            Cost function for the ground metric. If None, assumes data are square images
            flattened in np.flatten() convention and computes naturel Euclidean pixel distances.
            
        tau : float (<=0)
            Heavyball parameter (see section 4.3. in Schmitz et al 2017). Default is 0, i.e. no
            acceleration.
            
        rho : float (>=0)
            Unbalanced KL trade-off parameter (see section 4.4. in Schmitz et al 2017): 
            the smaller, the further from balanced OT we can stray. Ignored if unbalanced=False.
            
        varscale : float
            Trade-off hyperparameter between dictionary and weight optimization (see 
            end of section 3.1. in Schmitz et al 2017).
            
        unbalanced : bool
            If True, learn Unbalanced Wasserstein dictionary (with KL divergence for marginals).
            See section 4.4. in Schmitz et al 2017.
            
        loss : string
            Which fitting loss to use for the data similarity term. Implemented options are
            'L2' (default), 'L1' and 'KL'.
            
        feat_0 : np.ndarray or None
            Initialization of the dictionary. If None (default), initialize it as prescribed
            in feat_init.
            
        feat_init : str
            How to initialize dictionary atoms. Implemented options are:
                - 'sampled': initialize each atom as a randomly drawn datapoint.
                - 'uniform': initialize each atom as a uniform histogram.
                - 'random' (default): intialize each atom as white gaussian noise.
            Ignored if feat_0 is given.
            
        wgt_0 : np.ndarray or None
            Initialization of the weights. If None (default), initialize it as prescribed
            in wgt_init.
            
        wgt_init : str
            How to initialize the weights. Implemented options are:
                - 'uniform': give constant equal weight to each atom.
                - 'random' (default): draw random initial weights.
            Ignored if wgt_0 is given.
            
        n_process : int (deprecated)
            If higher than 1, run several parallel batches. Not recommended as Theano
            already parallelizes computation (in a much better way).
        
        Verbose : bool
            Whether to print (or save to log) information as the learning goes on.
            
        savepath : str
            Path to folder where results will be saved. If empty (default), results are not 
            written to disk.
            
        logpath : str
            Path to folder where logfile should be written. If empty (default), info is printed 
            to stdout instead. Ignored if Verbose=False.
            
        checkplots : bool
            If true, plots of atoms are saved (and overwritten each time). Assumes squared
            images in np.flatten() convention.
            
        factr, pgtol, maxiter:
            SciPy L-BFGS parameters. See: 
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
        

        Returns
        -------
        (D,lbda) : tuple
            Learned dictionary and weights.
            
        f : float
            Final error achieved.
            
        dic : dictionary
            L-BFGS output dictionary.
    """
    if logpath:
        logging.basicConfig(filename=logpath, level=logging.DEBUG)
        info = '\n\n##### N_ITER_SINK: {}\t TAU: {}\t VARSCALE: {}\t RHO: {} #####\nsavepath: {}\n\n'.format(
        n_iter_sinkhorn, tau, varscale,rho,savepath)
        logging.info(info)
    n, p = X.shape
    if C is None:
        C = EuclidCost(int(np.sqrt(p)), int(np.sqrt(p)), divmed=False, timeit=True)
    # INITIALIZATION
    if feat_0 is None:
        if feat_init == "sampled":
            Ys = np.empty((n_components, p))
            Ys[:] = X[np.random.randint(0, n, n_components), :]
        elif feat_init == "uniform":
            Ys = np.ones((n_components, p)) / p
        elif feat_init == "random":
            Ys = np.random.rand(n_components, p)
            Ys = (Ys.T / np.sum(Ys, axis = 1)).T
    else:
        Ys = feat_0
    if wgt_0 is None:
        if wgt_init == "uniform":
            w = np.ones((n, n_components)) / n_components
        elif wgt_init == "random":
            w = np.random.rand(n, n_components)
            w = (w.T / np.sum(w, axis=1)).T
    else:
        w = wgt_0
    dicw0 = np.log(np.vstack((Ys.T,w))).flatten()
    args = (X, gamma, C, n_components, n_iter_sinkhorn, tau, rho, varscale, 
            unbalanced, loss, n_process, Verbose, savepath, logpath, checkplots)
    x, f, dic = lbfgs(LBFGSFunc, dicw0, args=args, factr=factr, pgtol=pgtol, maxiter=maxiter)
    print dic
    print 'FINAL ERROR:\t{}'.format(f)
    return unwrap_rep(x.reshape(n+p,n_components), (n,p)), f, dic

