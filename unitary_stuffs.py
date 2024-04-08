import numpy as np
#parametrization taken from https://arxiv.org/pdf/1103.3408.pdf
#This code constructs a unitary matrix from a dxd matrix of parameters

#computes the adjoint of a matrix
pi = np.pi
def adj(x):
  return np.transpose(np.conjugate(x))

def exp_of_projection(d,N,theta):
  exp_P=np.identity(d,dtype=complex)
  exp_P[N,N]=np.complex(np.cos(theta),np.sin(theta))
  return exp_P

def exp_of_sigma(d,M,N,theta):
  exp_S=np.identity(d, dtype=complex)
  exp_S[M,M]=np.complex(np.cos(theta),0)
  exp_S[N,N]=np.complex(np.cos(theta),0)
  exp_S[M,N]=np.complex(0,np.sin(theta))
  exp_S[N,M]=np.complex(0,np.sin(theta))
  return exp_S

#Lambda should be d by d matrix
def construct_unitary(d,Lambda):
  prod_of_exp_of_P=np.identity(d,dtype=complex)
  for k in range(d):
    prod_of_exp_of_P=np.matmul(prod_of_exp_of_P,exp_of_projection(d,k,Lambda[k,k]))

  U=np.identity(d,dtype=complex)

  for M in range(d-1):
    for N in range(M+1,d):
      U=np.matmul(U,np.matmul(exp_of_projection(d,N,Lambda[N,M]),exp_of_sigma(d,M,N,Lambda[M,N])))

  U=np.matmul(U,prod_of_exp_of_P)

  return U
