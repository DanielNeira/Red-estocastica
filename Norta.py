import numpy as np
import pandas as pd
from scipy.stats import norm, beta, expon, uniform, gamma, erlang, poisson

class Norta(object):
    """
    This class contains the NORmal To Anything (NORTA) algorithm to produce vector with correlations in the dimensions.
    
    Init:
    matrix M (mxn), wich has n-dimensions and m-observations per dimension.
    *args Marginal functions from which the dimension come from.
    If needed **kwargs will contain the parameter (by name) of the Marginal functions
    
    Methods:
    generate: Once initialize the parameters and functions, given the n-observation to generate, the method
    creates a matrix of mxn elementes, with identical Marginal distributions as the original described and with 
    the same correlation.
    """
    def __init__(self):
        self.contenedor = []
        self.fit_contenedor = []
    
    def set_marginal(self, *args, **kwargs):
        if args[0] == 'normal':
            if ('mu' in kwargs) and ('sigma' in kwargs):
                self.contenedor.append({'function':norm,'mu':kwargs.get('mu'),'sigma':kwargs.get('sigma')})
                print("Normal function successfully added")
            else:
                print("Error when adding function to Norta")
                print("Remember that to add a normal function as arg and kwargs: set_marginal('normal',mu=,sigma=)")
        elif args[0] == 'beta':
            if len(kwargs) <= 2:
                if ('a' in kwargs) and ('b' in kwargs):
                    self.contenedor.append({'function':beta,'mu':kwargs.get('mu'),'sigma':kwargs.get('sigma')})
                    print("Beta function successfully added")
                else:
                    print("Error when adding function to Norta")
                    print("Remember that to add a beta function as arg and kwargs: set_marginal('beta',a=,b=)")
            else:
                if ('a' in kwargs) and ('b' in kwargs) and ('loc' in kwargs) and ('scale' in kwargs):
                    self.contenedor.append({'function':beta,'a':kwargs.get('a'),'b':kwargs.get('b'),
                                            'loc':kwargs.get('loc'),'scale':kwargs.get('scale') })
                    print("Beta function successfully added")
                else:
                    print("Error when adding function to Norta")
                    print("Remember that to add a beta function as arg and kwargs: set_marginal('beta',a=,b=,loc=,scale=)")
        elif args[0] == 'expon':
            if ('lambda' in kwargs):
                self.contenedor.append({'function':expon,'lambda':kwargs.get('lambda')})
                print("Exponencial function successfully added")
            else:
                print("Error when adding function to Norta")
                print("Remember that to add a exponencial function as arg and kwargs: set_marginal('expon',lambda=)")
        elif args[0] == 'uniform':
            if ('a' in kwargs) and ('b' in kwargs):
                self.contenedor.append({'function':uniform,'a':kwargs.get('a'),'b':kwargs.get('b')})
                print("Uniform function successfully added")
            else:
                print("Error when adding function to Norta")
                print("Remember that to add a uniform function as arg and kwargs: set_marginal('uniform',a=,b=")
        elif args[0] == 'gamma':
            if ('mu' in kwargs):
                self.contenedor.append({'function':gamma,'a':kwargs.get('a')})
                print("Gamma function successfully added")
            else:
                print("Error when adding function to Norta")
                print("Remember that to add a gamma function as arg and kwargs: set_marginal('gamma',a=")
        elif args[0] == 'erlang':
            if ('mu' in kwargs):
                self.contenedor.append({'function':erlang,'mu':kwargs.get('mu')})
                print("Erlang function successfully added")
            else:
                print("Error when adding function to Norta")
                print("Remember that to add a erlang function as arg and kwargs: set_marginal('erlang',mu=")
        elif args[0] == 'poisson':
            if ('mu' in kwargs):
                self.contenedor.append({'function':poisson,'mu':kwargs.get('mu')})
                print("Poisson function successfully added")
            else:
                print("Error when adding function to Norta")
                print("Remember that to add a poisson function as arg and kwargs: set_marginal('poisson',mu=")
        else:
            print("Function does not exist in this implementation")
        
    def delete_marginal(self,*args):
        """
        -1: Elimina último agregado
        'distribution_name': elimina el nombre de la distribución, si existen duplicados, los elimina desde el
        último agregado hasta el primero.
        indice: Elimina la función por indice, partiendo desde 
        """
        if args[0] == -1:
            self.contenedor.pop()
            return print("Function deleted")
        elif isinstance(args[0], (int,float)):
            self.contenedor.pop(args[0])
            return print("Function deleted")
        else:
            for i in range(len(self.contenedor)):
                if str(self.contenedor[len(self.contenedor)-1-i]['function']) == args[0]:
                    self.contenedor.pop(len(self.contenedor)-1-i)
                    return print("Function deleted")
        return print("Function could not be deleted")
    
    def set_data(self,data):
        """
        Recieves for now just a numpy array, that has to be the same dimension as the functions loaded so far
        """
        m,n = np.shape(data)
        if len(self.contenedor) != n:
            raise Exception('Not maching number in functions %d and dimension %d in Matrix' 
                            %(len(self.contenedor),m))
        else:
            self._generate_correlation_matrix(data,porcentaje=0.1, muestra=10000)
            self._generate_L()
    
    def set_corr_matrix(self,Corr,porcentaje=0.1, muestra=10000):
        n = len(self.contenedor)
        self.C = np.empty([n,n])
        np.fill_diagonal(self.C, 1)
        for dim1 in range(n):
            for dim2 in range(n):
                if dim1 < dim2 and dim1 != dim2:
                    W1 = np.random.normal(0, 1, (muestra))
                    W2 = np.random.normal(0, 1, (muestra))
                    rho = Corr[dim1][dim2]
                    if rho < 0:
                        l = -1
                        u = 0
                    else:
                        l = 0
                        u = 1
                    r = rho
                    rho_estimado = self._rho_function(r,W1,W2,muestra,dim1,dim2)
                    while np.absolute(rho_estimado - rho) > porcentaje*np.absolute(rho):
                        if rho_estimado > rho:
                            u = r
                        else:
                            l = r
                        r = (l+u)/2
                        rho_estimado = self._rho_function(r,W1,W2,muestra,dim1,dim2)
                        if r < 1.0e-15:
                            break
                    self.C[dim1,dim2] = r
                    self.C[dim2,dim1] = r
        print("Calculated correlation:\n",self.C)
        self._generate_L()

    def _generate_correlation_matrix(self,data,porcentaje = 0.1, muestra = 10000):
        """
        Genera la matrix de correlaciones para la 
        """
        m,n = np.shape(data)
        self.C = np.empty([n,n])
        np.fill_diagonal(self.C, 1)
        for dim1 in range(n):
            for dim2 in range(n):
                if dim1 < dim2 and dim1 != dim2:
                    W1 = np.random.normal(0, 1, (muestra))
                    W2 = np.random.normal(0, 1, (muestra))
                    rho = np.corrcoef(data[:,dim1],data[:,dim2])[0,1]
                    if rho < 0:
                        l = -1
                        u = 0
                    else:
                        l = 0
                        u = 1
                    r = rho
                    rho_estimado = self._rho_function(r,W1,W2,muestra,dim1,dim2)
                    #print("rho_estimado - rho - r - [l,u]")
                    while np.absolute(rho_estimado - rho) > porcentaje*np.absolute(rho):
                        if rho_estimado > rho:
                            u = r
                        else:
                            l = r
                        r = (l+u)/2
                        #r = (l+u)/np.absolute(rho_estimado - rho) #INTENTO DE CONVERGENCIA
                        #r = (l+u)/1+np.absolute(rho_estimado - rho) #intento 3
                        rho_estimado = self._rho_function(r,W1,W2,muestra,dim1,dim2)
                        if r < 1.0e-15:
                        #if r < 1.0e-7:
                            #r = 0
                            break
                        #print(rho_estimado,"-",rho,"-",r,"-","[%f,%f]" %(l,u),np.absolute(rho_estimado - rho))
                    self.C[dim1,dim2] = r
                    self.C[dim2,dim1] = r
        print("Calculated correlataion:\n",self.C)
        
    def _rho_function(self,r,W1,W2,m,dim1,dim2):
        z = np.empty([2,m])
        z[0,:] = np.copy(W1)
        z[1,:] = r*W1 + (np.sqrt(1-np.power(r,2)))*W2
        x = np.empty([2,m])
        x[0,:] = self._return_coresponding_distribution(dim1,z[0,:])
        x[1,:] = self._return_coresponding_distribution(dim2,z[1,:])
        numerador = np.sum((x[0,:]-np.mean(x[0,:]))*(x[1,:]-np.mean(x[1,:])))
        denominador = np.sqrt( np.sum(np.power(x[0,:]-np.mean(x[0,:]),2))*
                          np.sum(np.power(x[1,:]-np.mean(x[1,:]),2)) )
        rho_estimado = numerador/denominador
        return rho_estimado
        
    def set_and_fit_data(self,data,show=False):
        """
        Recibe la data, calcula cual es la distribución que mejor queda para cada dimensión, y luego genera el
        modelo.
        Se recibe un dataframe de pandas
        
        ---Todavia en construcción---
        """
        if isinstance(data, pd.DataFrame):
            for column in data:
                best_dist, best_fit_params = self.best_fit_distribution(data[column], bins=200, ax=None)
                print("For dimension %s best distribution is %s" %(column,best_dist.name))
                self._set_marginal_from_data(best_dist,best_fit_params)
            data = data.values
            self._generate_correlation_matrix(data,porcentaje=0.1, muestra=10000)
            self._generate_L()
        else:
            print("This method only recieve a Data frame as argument")
    
    def _set_marginal_from_data(self, distribution, params):
        self.fit_contenedor.append({'function':distribution,'params':params})
            
    def best_fit_distribution(self,data, bins=200, ax=None):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Distributions to check
        DISTRIBUTIONS = [norm, beta, expon, uniform, gamma, erlang, poisson]

        # Best holders
        best_distribution = norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                        end
                    except Exception:
                        pass

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                pass
        return (best_distribution, best_params)
    
    def _generate_L(self):
        self.L = np.linalg.cholesky(self.C)
    
    def generate_sample(self, m=1):
        if len(self.fit_contenedor)==0:
            n = len(self.contenedor)
        else:
            n = len(self.fit_contenedor)
        W = np.random.normal(0, 1, (n,m))
        print(W)
        Z = np.dot(self.L,W)
        if n==1:
            Z = Z.reshape((-1, 1)) 
        else:
            Z = Z.T
        X = np.empty([m, n])
        print(Z)
        for dimension in range(n):
            X[:,dimension] = self._return_coresponding_distribution(dimension,Z[:,dimension])
        return X

    def _return_coresponding_distribution(self,dimension,data):
        if len(self.fit_contenedor) == 0:
            for func in [self.contenedor[dimension]['function']]:
                if func == norm:
                    return  func.ppf(q=norm.cdf(data),loc=self.contenedor[dimension]['mu'],
                                      scale=self.contenedor[dimension]['sigma'])
                elif func == beta:
                    if len(self.contenedor[dimension]) <= 3:
                        return func.ppf(q=norm.cdf(data),a=self.contenedor[dimension]['a'],
                                          b=self.contenedor[dimension]['b'])
                    else:
                        return func.ppf(q=norm.cdf(data),a=self.contenedor[dimension]['a'],
                                          b=self.contenedor[dimension]['b'],
                                       loc=self.contenedor[dimension]['loc'],
                                        scale=self.contenedor[dimension]['scale'])
                elif func == expon:
                    return func.ppf(q=norm.cdf(data),scale=1/(self.contenedor[dimension]['lambda']))
                elif func == uniform:
                    return func.ppf(q=norm.cdf(data),loc=self.contenedor[dimension]['a'],
                                      scale=self.contenedor[dimension]['b'] - 
                                    self.contenedor[dimension]['a'])
                elif func == gamma:
                    return func.ppf(q=norm.cdf(data),a=self.contenedor[dimension]['a'])
                elif func == erlang:
                    return func.ppf(q=norm.cdf(data),a=self.contenedor[dimension]['a'])
                elif func == poisson:
                    return func.ppf(q=norm.cdf(data),mu=self.contenedor[dimension]['mu'])
                else:
                    print("A error has ocurred") 
        else:
            for func in [self.fit_contenedor[dimension]['function']]:
                arg = self.fit_contenedor[dimension]['params'][:-2]
                loc = self.fit_contenedor[dimension]['params'][-2]
                scale = self.fit_contenedor[dimension]['params'][-1]
                return func.ppf(q=norm.cdf(data), *arg, loc=loc, 
                         scale=scale) if arg else func.ppf(q=norm.cdf(data)
                                                           , loc=loc, scale=scale)