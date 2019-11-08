import numpy as np

if __name__ == "__main__":
    # the state of N qubits is a complex vector of dimension 2^N
    n_qubits = 2
    dimension = 2 ** n_qubits
    state = [1j, 0, 0, 0]

    def print_probs(state, n_qubits):
        # the elements of this state are squared to calculate outcome probabilities
        for bitstring in range(n_qubits ** 2):
            probability = np.abs(state[bitstring]) ** 2

            print("Bitstring", format(bitstring, "0" + str(n_qubits) + "b"), " has probability ", probability)
        print()

    print_probs(state, n_qubits)
    # an example with a "superposition" over outcomes
    print_probs([0, -1j / np.sqrt(2), 0, 1 / np.sqrt(2)], n_qubits)

    # evolution is then given by a unitary matrix
    identity = np.array([[1, 0], [0, 1]]) # identity on one qubit
    flip = np.array([[0, 1], [1, 0]]) # a flip or X-gate on one qubits
    flip_first = np.kron(flip, identity) # tensor products make this a two qubit operation

    new_state = flip_first@state
    print_probs(new_state, n_qubits)

    flip_second = np.kron(identity, flip)
    print_probs(new_state, n_qubits)

    # if we start in the state with all qubits in zero
    # then we can take a shortcut to get the probabilities of any particular bitstring
    all_zeros = [1] + [0]*(dimension-1)
    bs = np.random.choice(range(dimension))
    assert (flip_second@all_zeros)[bs] == flip_second[bs, 0]
