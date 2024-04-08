import pennylane as qml
from pennylane import numpy as np
import sys
import qiskit
import os
import random
from tqdm import tqdm
import time

from parser_argument import get_args

from generate_haar_unitary import generate_haar_unitary
args = get_args()


def main(num_train, sr, n_shots, seed_u, seed_data):
    n_qubits = args.n_qubits
    data_size = num_train
    schmidt_rank = sr
    test_dist = args.test_dist
    data_type = args.data_type

    print(f'n_qubits: {n_qubits} || data_size: {data_size} || schmidt_rank: {schmidt_rank} || nshots: {n_shots}')
    
    def save():
        path = os.path.abspath('.')
        path_result = os.path.join(path, "FiniteShotResult")
        if not os.path.exists(path_result):
            os.mkdir(path_result)
        path_n_data = os.path.join(path_result, f'{data_type}_{n_qubits}')
        if not os.path.exists(path_n_data):
            os.mkdir(path_n_data)
        path_N_r_s = os.path.join(path_n_data, f'N{data_size}_r{schmidt_rank}_s{n_shots}')
        if not os.path.exists(path_N_r_s):
            os.mkdir(path_N_r_s)
        path_seed = os.path.join(path_N_r_s, f'seed_u{seed_u}_seed_d{seed_data}')
        if not os.path.exists(path_seed):
            os.mkdir(path_seed)
            
        return path_seed

    def general_UOU_set(n_qubits, seed_u):
        one_divide_eps = 2
        # Set the dimensionality of the sphere
        dim = 2 ** n_qubits

        # Generate the indices for each point
        indices = np.meshgrid(*[np.arange(one_divide_eps)] * dim, indexing="ij")

        # Convert the indices to coordinates
        coordinates = np.array(indices).reshape((dim, -1)).T / (one_divide_eps - 1) * 2 

        # Normalize the coordinates to lie on the sphere
        norms = np.linalg.norm(coordinates, axis=1)
        coordinates /= norms[:, None]
        uou_list = []
        for i in range(1, len(coordinates)):
            uou = np.outer(coordinates[i], np.conjugate(coordinates[i]))
            uou_list.append(uou.numpy())
        target_uou_index = np.random.choice(range(len(uou_list)), 1, replace=False)
        print("target_uou_index: ", target_uou_index)
        target_uou = uou_list[target_uou_index[0]]
        return uou_list, target_uou, target_uou_index
    
    def general_entangled_haar_state(data_size, n_qubits, schmidt_rank, seed):
        dataset_ls_x = []
        d = 2**n_qubits
        np.random.seed(seed)
        if data_type == 'general_orth':
            data_train_all = generate_haar_unitary(N=2**n_qubits, seed=seed+10000)
        for i in range(data_size):
            schmidt_coeff_tempt = qiskit.quantum_info.random_statevector(schmidt_rank, seed+520*i)
            schmidt_coeff_tempt = schmidt_coeff_tempt.data
            mixed_data_tempt = np.zeros((2**n_qubits, 2**n_qubits))
            if data_type != 'general_orth':
                data_train_all = generate_haar_unitary(N=2**n_qubits, seed=seed+10000+i)
            for r in range(schmidt_rank):
                mixed_data_tempt = (np.abs(schmidt_coeff_tempt[r])**2) * np.outer(data_train_all[(schmidt_rank*i+r)%d], np.conjugate(data_train_all[(schmidt_rank*i+r)%d])) + mixed_data_tempt
            dataset_ls_x.append(mixed_data_tempt.numpy())    
        return dataset_ls_x
    
    def UOU_Set(n_qubits, seed_u):
        """generate the set of target operators uou
            and select one as the target operator"""
        uou_list = []
        for i in range(2**n_qubits):
            uou_tempt = np.zeros(2**n_qubits)
            uou_tempt[i] = 1
            uou = np.outer(uou_tempt, np.conjugate(uou_tempt))
            uou_list.append(uou.numpy())
        np.random.seed(seed_u)
        target_uou_index = np.random.choice(range(len(uou_list)), 1, replace=False)
        print("target_uou_index: ", target_uou_index)
        target_uou = uou_list[target_uou_index[0]]
        return uou_list, target_uou, target_uou_index


    def special_entangled_haar_state(data_size, n_qubits, schmidt_rank, seed):
        """generate the special partial traced entangled states
        where in the quantum system, the state is the mixed states 
        of the linear combination of the natural computational basis"""
        dataset_ls_x, non_zero_index_ls = [], []
        np.random.seed(seed)
        seed_data = np.random.choice(range(1000,10000), data_size, replace=False)
        if data_type == 'orth':
            non_zero_index_ls = np.random.choice(2**n_qubits, schmidt_rank*data_size, replace=False).numpy()
        if data_type != 'orth':
            for i in range(data_size):
                non_zero_index = np.random.choice(2**n_qubits, schmidt_rank, replace=False)
                non_zero_index_ls.extend([non_zero_index[j].numpy() for j in range(len(non_zero_index))])
        count = 0
       # print("data_non_zero_index: ", non_zero_index_ls)
        if non_zero_index_ls[0]  == 1:
            end = "end"
        for i in range(data_size):
            schmidt_coeff_tempt = qiskit.quantum_info.random_statevector(schmidt_rank, seed+520*i)
            schmidt_coeff_tempt = schmidt_coeff_tempt.data
            mixed_data_tempt = np.zeros((2**n_qubits, 2**n_qubits))
            for r in range(schmidt_rank):
                # data_tempt = qiskit.quantum_info.random_statevector(2**n_qubits, seed_data[i].numpy())   #arguements: (dim, seed)
                # data_tempt = data_tempt.data
                data_tempt = np.zeros(2**n_qubits)
                data_tempt[non_zero_index_ls[count]] = 1
                count += 1
                mixed_data_tempt = (np.abs(schmidt_coeff_tempt[r])**2) * np.outer(data_tempt, np.conjugate(data_tempt)) + mixed_data_tempt
            dataset_ls_x.append(mixed_data_tempt.numpy())    
        return dataset_ls_x, non_zero_index_ls

    def circuit_ob(input, Ob, n_qubits):
        qml.QubitDensityMatrix(input, wires=[i for i in range(n_qubits)])
        return qml.sample(qml.Hermitian(Ob, wires=[i for i in range(n_qubits)]))

    dev = qml.device('default.mixed', wires=n_qubits, shots=n_shots)
    circuit = qml.QNode(circuit_ob, dev)

    # dataset = special_entangled_haar_state(data_size, n_qubits, schmidt_rank, seed_data)
    # uou_list, target_uou, target_uou_index = UOU_Set(n_qubits, seed_u)

    def sample_estimate(dataset, Ob, n_qubits):
        sample_mean_ls, sample_std_ls = [], []
        for j in range(len(dataset)):
            sample_mean = circuit(dataset[j], Ob, n_qubits).mean()
            sample_std = circuit(dataset[j], Ob, n_qubits).std()
            sample_mean_ls.append(sample_mean) # sample_mean.numpy()
            sample_std_ls.append(sample_std) # sample_std.numpy()
        return sample_mean_ls, sample_std_ls

    # sample_mean_ls, sample_std_ls = sample_estimate(dataset, target_uou, n_qubits)

    def all_uou_dataset(uou_list, dataset):
        """ Input:
                uou_list: the set of all candidate operator UOU
                dataset: the generated training entangled data with schmidt rank r
            Output:
                uou_out_ls: the list of the trace of the product of 
                all quantum states in the dataset and the candidate operator UOU in uou_list"""
        uou_out_ls = []
        for j in range(len(uou_list)):
            dataset_out_ls = []
            uou_tempt = uou_list[j]
            for k in range(len(dataset)):
                data_tempt = dataset[k]
                uou_dataset_out = np.abs(np.trace(uou_tempt @ data_tempt))
                dataset_out_ls.append(uou_dataset_out)
            uou_out_ls.append(dataset_out_ls)
        return uou_out_ls

    # uou_out_ls = all_uou_dataset(uou_list, dataset)
                
    def choose_target_operator(sample_mean_ls, sample_std_ls, uou_out_ls):
        """ Output:
                min_error_index: the index of the selected opertor in the uou_ls with
                                the minimum error """
        error_ls = []
        for j in range(len(uou_out_ls)):
            ground_uou_tempt = uou_out_ls[j]
            error = np.linalg.norm(np.array(ground_uou_tempt) - np.array(sample_mean_ls))
            error_ls.append(error)
        min_error_index = np.where(error_ls == np.min(error_ls))[0]
        if len(min_error_index) == 1:
            return min_error_index[0], np.min(error_ls)
        elif len(min_error_index) > 1:
            # std_min_error_index = [sample_std_ls[k] for k in min_error_index]
            # min_std_index = np.where(std_min_error_index==np.min(std_min_error_index))[0]
            # min_error_std_index = min_error_index[min_std_index]
            # return min_error_std_index
            return min_error_index[0], np.min(error_ls)
        
    def swap_circuit(n_qubits, idx):
        for i in range(2*n_qubits):
            qml.Identity(wires=i)
        qml.SWAP(wires=[idx, n_qubits+idx]) 
        
    def get_swap_matrix(n_qubits, idx):
        get_matrix = qml.matrix(swap_circuit)(n_qubits, idx)
        return get_matrix  

    def prediction(target_uou, estimate_uou):
        d = 2**n_qubits
        UOU = target_uou
        VOV = estimate_uou
        if test_dist == 'haar_n':
            test_error = 2 * np.abs(1 - np.trace(VOV @ UOU)) / (d*(d+1))
        elif test_dist == 'haar_1':
            test_error = np.abs(np.trace(np.kron(UOU-VOV, UOU-VOV) + np.kron(UOU-VOV, UOU-VOV) @ get_swap_matrix(n_qubits))) / (6**n_qubits)
        return test_error


    if data_type in ['orth', 'mixed_state']:
        dataset, non_zero_index_ls = special_entangled_haar_state(data_size, n_qubits, schmidt_rank, seed_data)
        uou_list, target_uou, target_uou_index = UOU_Set(n_qubits, seed_u)
    elif data_type in ['general_orth', 'general_mixed_state']:
        dataset = general_entangled_haar_state(data_size, n_qubits, schmidt_rank, seed_data)
        uou_list, target_uou, target_uou_index = general_UOU_set(n_qubits, seed_u)
    sample_mean_ls, sample_std_ls = sample_estimate(dataset, target_uou, n_qubits)
    uou_out_ls = all_uou_dataset(uou_list, dataset)
    choose_target_operator_index, loss = choose_target_operator(sample_mean_ls, sample_std_ls, uou_out_ls)
    estimate_uou = uou_list[choose_target_operator_index]
    prediction_error = prediction(target_uou, estimate_uou)

    print("prediction error: ", prediction_error)
    path = save()
    with open(os.path.join(path, f'test_error.txt'), 'w+') as e:
        np.savetxt(e, [prediction_error]) 
    with open(os.path.join(path, f'target_index.txt'), 'w+') as a:
        np.savetxt(a, target_uou_index)
    with open(os.path.join(path, f'loss.txt'), 'w+') as f:
        np.savetxt(f, [loss]) 
    
    # with open(os.path.join(path, f'data_index.txt'), 'w+') as b:
    #         np.savetxt(b, non_zero_index_ls) 
    # end = "end"

n_shots = args.n_shots
seed_u = args.seed_u
seed_data = args.seed_data
if args.n_qubits>=4:
    data_size_ls = [2**i for i in range(args.n_qubits+1)]  # range(4, 2**args.n_qubits+4, 2**args.n_qubits//16) # range(1,2**args.n_qubits+1)
else:
    data_size_ls = [i for i in range(1, args.n_qubits+1)]
for seed_u in tqdm(range(2), desc='seed_u'):
    for seed_data in range(5):
        for num_train in data_size_ls:
            if args.data_type in ['mixed_state', 'general_mixed_state'] and args.n_qubits<4:
                schmidt_rank_ls = [i for i in range(1, args.n_qubits+1)]  #  [i for i in range(1, 2**args.n_qubits+1, 2**args.n_qubits//16)]
            if args.data_type in ['mixed_state', 'general_mixed_state'] and args.n_qubits>=4:
                schmidt_rank_ls = [2**i for i in range(args.n_qubits+1)]
            elif args.data_type in ['orth', 'general_orth']:
                schmidt_rank_ls = [i for i in range(1, 2**args.n_qubits//num_train+1)]  
            for sr in data_size_ls:
                time_start = time.time()
                for n_shots in [500000, 100000, 50000, 10000, 5000, 1000, 500, 100]: #[10, 100,500,1000,5000,10000,20000]:     #[10, 100, 300, 500, 800, 1000, 2000, 5000, 10000, 20000, 40000]:
                    main(num_train, sr, n_shots, seed_u, seed_data)
                time_end = time.time()
                print("time for ergodic the whole n_shots_ls: ", time_end-time_start)
end = 'end' 
""" this version is used for MOBA which supports parallizing computation for multiple seeds """
# for sr in schmidt_rank_ls:
#     for n_shots in [10, 100, 300, 500, 800, 1000, 2000, 5000, 10000]:
#         if args.data_type in ['mixed_state', 'general_mixed_state']:
#             data_size_ls = [i for i in range(1, 2**args.n_qubits+1)]
#         elif args.data_type in ['orth', 'general_orth']:
#             data_size_ls = [i for i in range(1, 2**args.n_qubits//sr+1)]  
#         for num_train in data_size_ls:
#             main(num_train, sr, n_shots, seed_u, seed_data)
                                

# for seed_u in range(1):
#     for seed_data in range(10, 100):
#         sr = 1
#         num_train = 1
#         main(num_train, sr, seed_u, seed_data)
        
# num_train = 2
# sr = 1
# for seed_u in range(5):
#     for seed_data in range(4):
#         main(num_train, sr, seed_u, seed_data)
        

