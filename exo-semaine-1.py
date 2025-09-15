# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
# ---

# %%
##### Les rayures 

import numpy as np

def rayures(n):
    lignes, colonnes = np.indices((n, n))
    return colonnes % 2

print(rayures(5))

# %%
###Le damier

import numpy as np

def checkers(n, upletf = True):
    lignes, colonnes = np.indices((n,n))
    M = lignes + colonnes 
    return (M + 1) % 2

print(checkers(5))

# %%
###Le super damoier par blocs 

import numpy as np

def block_checkers(n, k) :
    lignes, colonnes = np.indices((n*k, n*k))
    lignes2 = lignes // k 
    colonnes2 = colonnes // k 
    return (lignes2 + colonnes2) % 2

print(block_checkers(6, 3))    

# %%
###L'escalier

import numpy as np

@np.vectorize
def aux(X, n):
    if X <= n :
        return X
    else : 
        return 2*n - X
    
def stairs(n) :
    I, J = np.indices(((2*n+1),2*n+1))
    return aux(I, n) + aux(J, n)


print(stairs(4)) 

# %%
