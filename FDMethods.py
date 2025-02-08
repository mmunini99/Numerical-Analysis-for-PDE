import numpy as np


class first_order_first_derivative_FiniteDiff(object):

    def __init__(self, n, h):

        self.n = n
        self.h = h

    def __matrix__(self):
        return np.zeros((self.n+1,self.n+1))
    
    def FD(self):
        store = self.__matrix__()
        for i in range(0,self.n, 1):
            store[i,i] = -1
            store[i,i+1] = 1

        store = store / self.h
            
        return store


    def BD(self):
        store = self.__matrix__()
        for i in range(1,self.n+1,1):
            store[i,i-1] = -1
            store[i,i] = 1

        store = store / self.h

        return store


    def CD(self):
        store = self.__matrix__()
        for i in range(1,self.n,1):
            store[i,i-1] = -1
            store[i,i+1] = 1

        store = store / (2*self.h)

        return store



class second_order_first_derivative_FiniteDiff(object):

    def __init__(self, n, h):

        self.n = n
        self.h = h

    def __matrix__(self):
        return np.zeros((self.n+1,self.n+1))
    
    def FD(self):
        store = self.__matrix__()
        for i in range(0,self.n-1, 1):
            store[i,i] = -3
            store[i,i+1] = 4
            store[i,i+2] = -1

        store = store / (2*self.h)
            
        return store


    def BD(self):
        store = self.__matrix__()
        for i in range(2,self.n+1,1):
            store[i,i-2] = 1
            store[i,i-1] = -4
            store[i,i] = 3

        store = store / (2*self.h)

        return store
    

class second_derivative_FiniteDiff(object):

    def __init__(self, n, h):

        self.n = n
        self.h = h

    def __matrix__(self):
        return np.zeros((self.n+1,self.n+1))
    
    def FD(self):
        store = self.__matrix__()
        for i in range(0,self.n-1, 1):
            store[i,i] = 1
            store[i,i+1] = -2
            store[i,i+2] = 1

        store = store / (self.h**2)
            
        return store


    def BD(self):
        store = self.__matrix__()
        for i in range(2,self.n+1,1):
            store[i,i-2] = 1
            store[i,i-1] = -2
            store[i,i] = 1

        store = store / (self.h**2)

        return store


    def CD(self):
        store = self.__matrix__()
        for i in range(1,self.n,1):
            store[i,i-1] = 1
            store[i,i] = -2
            store[i,i+1] = 1

        store = store / (self.h**2)

        return store
