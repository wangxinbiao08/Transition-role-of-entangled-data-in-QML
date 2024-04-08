import pennylane as qml
from pennylane import numpy as np
import numpy as np
from scipy import linalg

import sys
sys.path.append('.')
import unitary_stuffs as u
from parser_argument import get_args

args = get_args()
pi = np.pi
class haar_distribution:
  """ 
  A class used to represent the Haar distribution on the unitary group U(d).
  
  ...
  
  Attributes
  ----------
  
  dimension: int
    The dimension of the space on which the unitaries act. 
    
  Methods
  -------
  density(arr: Lambda) --> float
    The unnormalized Haar density function. Lambda is expected to be a d by d matrix of parameters between 0 and 2pi.
    The unnormalized density is intended for use in the Metropolis Hastings sampling as the addition of a normalization
    constant would only increase computation complexity. This is supplied so that the user may employ a sampling method 
    other than Metropolis Hastings.
    
  normalized_density(arr: Lambda) --> float
    THe normalized Haar density function. Lambda is expected to be a d by d matrix of parameters between 0 and 2pi.
    
  generate_sample(int: size, float: burn_in) --> arr
    Generates a sample of approximately Haar distributed uniatry matrices.  Size is the size of the sample produced.
    burn_in is the percent of initial states discarded in the markov chain produced by Metropolis Hastings. burn_in 
    is expected to be between 0 and 1. A common choice is .2. This will discard the first 20% percent of the markov 
    chain the markov chain. A higher value of burn_in will produced more accurate results, but will greatly increase
    the computational complexity.
    
  a(arr: v, arr: w) --> float
    The acceptance probability needed to choose states in the markov chain generated in Metropolis Hastings. The user
    does not need to access this method directly.
    
  proposal(int: d) --> arr
    Proposes the next state in the markov chain in Metropolis Hastings. The user does not need to access this method
    directly.
    
  coin(float: p) --> int
    A simple binomial coin. Used in the Metropolis Hastings algorithm. The user does not need to access this method 
    directly.
  """
  def __init__(self,dimension):
    self.dimension = dimension #Dimension of the vector space on which the unitaries act
  #unnormalized haar density. 
  #Lambda is a dxd matrix of values between 0 and 2pi
  def density(self,Lambda):
    d = self.dimension
    value = 1
    for M in range(d-1):
      for N in range(M+1,d):
        value = value*np.sin(Lambda[M,N])*(np.cos(Lambda[M,N]))**(2*(N-M)-1)
    return value
  
  #normalized haar density
  def normalized_density(self,Lambda):
    d = self.dimension
    numerator = (2*pi)**(d*(d+1)*.5)
    denominator = 1
    for m in range(1,d):
        for n in range(m+1,d+1):
            denominator = denominator*2*(n-m)
    
    normalization_constant = numerator/denominator
    return self.density(Lambda)/normalization_constant
  #generates a sample of approximately uniformly distributed unitaries 
  def generate_sample(self,size, burn_in):
    #burn_in is expected to be a positive number less than 1. Usual is around .2. 
    iterations = int(size/(1-burn_in))
    d = self.dimension
    Lambdas = []
    initial_Lambda = self.proposal(d)
    Lambdas.append(initial_Lambda)

    for N in range(iterations):
      w = self.proposal(d)
      alpha = self.a(Lambdas[N],w)
      flip = self.coin(alpha)

      if flip == 1:
        Lambdas.append(w)
      else:
        Lambdas.append(Lambdas[N])

    
    Lambda_Sample = Lambdas[-size:]
    Unitary_Sample = []
    for L in Lambda_Sample:
      Unitary_Sample.append(u.construct_unitary(d,L))
    
    return Unitary_Sample
    
  #### probability helper methods for MCMC ####
  
  #acceptance probability for metropolis hastings algorithm
  def a(self,v,w):
    d = self.dimension
    M = [1,self.density(w)/self.density(v)]
    return min(M)
   
  @staticmethod
  def coin(p):
    return np.random.binomial(1,p)
    
  @staticmethod
  #uniform proposal method for metropolis hastings algorithm
  def proposal(d):
    P = np.zeros((d,d))
    for M in range(d):
        for N in range(d):
            if M<N:
                P[M,N]=np.random.uniform()*pi/2
            elif M==N:
                P[N,M]=np.random.uniform()*pi*2
            else:
                P[M,N]=np.random.uniform()*pi
  
    return P


def generate_haar_unitary(N=1000, seed=42):
    """Generate complex uniform unitary matrix.
    Returns
    -------
    - X : array of shape (M, N) with orthonormal columns
    """
    alpha = 1
    M = int(alpha * N)
    np.random.seed(seed)
    gaussian_matrix = (1./np.sqrt(2.))*(np.random.normal(0, 1., (M, M)) + 1j*np.random.normal(0, 1., (M,M)))
    U, R = linalg.qr(gaussian_matrix)
    #Then we multiply on the right by the phases of the diagonal of R to really get the Haar measure
    D = np.diagonal(R)
    Lambda = D / np.abs(D)
    U = np.multiply(U, Lambda)
    #Then we take its n first columns (scaling of TRAMP needs E[A_{mu i}^2] of order 1/N)
    A = U[:, 0:N]
    return A
  
  
""" gererate unitart from a quantum circuit  """
def generate_sub_block(para_, loc=None):
    d=2
    
    ###--- note that we change the sub-block structure of the hea circuit to CNOT-RotU3-CNOT ---###
    # if args.q2_gate=='cnot':
    #     qml.CNOT(wires=[loc, loc+1])
    ###--- note that we change the sub-block structure of the hea circuit to CNOT-RotU3-CNOT ---###
    
    for i in range(d):        
        qml.RX(para_[3*i], wires=loc+i)
        qml.RY(para_[3*i+1], wires=loc+i)
        qml.RX(para_[3*i+2], wires=loc+i)
    qml.CNOT(wires=[loc, loc+1])

        
def generate_layer(para_):
    count = 0
    for i in range(args.n_qubits//2):
        generate_sub_block(para_[count*6: (count+1)*6], loc=2*i)
        count = count + 1
    
    for i in range((args.n_qubits-1)//2):
        generate_sub_block(para_[count*6: (count+1)*6], loc=2*i+1)
        count = count + 1
        
# dev = qml.device("default.qubit", wires=args.n_qubits)
# @qml.qnode(dev)
def generate_circuit_block(para_, num_blocks):       
    ##  para_node record thes number of parameters in each layer  ##
    for j in range(num_blocks):
        generate_layer(para_[j*(args.n_qubits-1)*6: (j+1)*(args.n_qubits-1)*6])
        

def generate_circuit_matrix(para_, num_blocks):
    """ calculating the unitary matrix don't require the device and measurement """
    circuit = lambda para: generate_circuit_block(para, num_blocks=num_blocks)
    get_matrix = qml.matrix(circuit)(para_)  
    return get_matrix


""" for test """
g_num_blocks = 1
np.random.seed(42)
random_para = np.random.uniform(0, 2*np.pi, g_num_blocks*(args.n_qubits-1)*6)
random_circuit_U = generate_circuit_matrix(random_para, g_num_blocks)
r_c_t_U = random_circuit_U.T.conjugate()

end = 'end'


  

