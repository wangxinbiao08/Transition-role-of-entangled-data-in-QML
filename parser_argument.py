import argparse
from argparse import Namespace
def get_args():
    parser = argparse.ArgumentParser("NFL_Observable")

    # model and data type
    parser.add_argument('--Ob_type', type=str, default='basis_0', help='the type of employed observable')
    parser.add_argument('--pauli_str_pos', type=str, default='sum', help='the front half is the pauli string and the rear half is the position of string')
    parser.add_argument('--data_type', type=str, default='mixed_state', help='the type of used data')
    parser.add_argument('--train_manner', type=str, default='one_stage_UV', help='training manner opt: one_stage, two_stage')
    parser.add_argument('--test_dist', type=str, default='haar_n', help='test distribution opt: haar_n, haar_1')

    # schmidt rank
    parser.add_argument('--schmidt_rank', type=int, default=1, help='the schmidt rank of entangled states')

    # hyper-parameter
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
    parser.add_argument('--num_train', type=int, default=2, help='num of training data, set <50 for W_states')
    parser.add_argument('--num_test', type=int, default=1000, help='num of test data  if >1000 refer to test over all haar state') 
    parser.add_argument('--epoch_num', type=int, default=2000, help='num of training epochs')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--arch', type=str, default='', help='which architecture to use')
    parser.add_argument('--noise', type=float, default=0.2, help='depolarization rate')
    
    # circuit
    parser.add_argument('--n_qubits', type=int, default=10, help='number of qubits')
    parser.add_argument('--num_blocks', type=int, default=20, help='number of blocks')
    parser.add_argument('--IsFiniteShot', type=str, default='True', help='noisy setting of noiseless setting')
    parser.add_argument('--n_shots', type=int, default=10, help='number of measurements')

    # seed 
    parser.add_argument('--seed_para', type=int, default=0, help='random seed for generated para')
    parser.add_argument('--seed_u', type=int, default=0, help='random seed for generated haar unitary')
    parser.add_argument('--seed_data', type=int, default=0, help='random seed for generated haar state')

    
    # optimization
    parser.add_argument('--init_method', type=str, default='unif', help='the distribution of initial params, gaussian or uniform')
    parser.add_argument('--opt_method', type=str, default='ADAM', help='the optimization algorithm 0:ADAM 1:SGD 2:QNG')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')   ## since 2021_12_21, the learning rate is changed to be 0.02
    parser.add_argument('--lr_decay', type=str, default='False', help='decide whether adopt learning rate decay strategy') 
    parser.add_argument('--batch_size', type=int, default=1, help='batch size = 1 for vqe, <=4 for N_tangled state')

    parser.add_argument('--lamb', type=float, default=0.6, help='hyper-parameter of coordinate descent algorithm')

    args = parser.parse_args()

    return args


