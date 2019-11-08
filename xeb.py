# xeb.py
#
# Authors M. Sohaib Alam & Will Zeng

import matplotlib.pyplot as plt
from typing import List, Callable
import numpy as np
import functools
import itertools


def random_unitary(dim: int) -> np.ndarray:
    # follows the algorithm in https://arxiv.org/pdf/math-ph/0609050.pdf
    # returns a unitary of size dim x dim
    Z = np.array([np.random.normal(0, 1) + np.random.normal(0, 1) * 1j for _ in range(dim ** 2)]).reshape(dim, dim)
    Q, R = np.linalg.qr(Z)
    diag = np.diagonal(R)
    lamb = np.diag(diag) / np.absolute(diag)
    unitary = np.matmul(Q, lamb)

    # this condition asserts that the matrix is unitary
    assert np.allclose(unitary.conj().T @ unitary, np.eye(dim))

    return unitary


def simulate_probability(unitary: np.ndarray, bitstring:int) -> float:
    # simulates the probability of measuring bitstring when evolving from the ground state
    # according to the quantum program given unitary
    return np.abs(unitary[bitstring, 0])**2


def quantum_sample_probability(n_qubits: int, trials: int) -> List:
    # returns the probabilities of a randomly chosen bistring outcome over "trials" number of different
    # random quantum programs on n_qubits

    dimension = 2**n_qubits
    # picks a random bitstring as labelled by the integers 1 to 2**n_qubits
    bitstring = np.random.choice(dimension)

    # keeps track of the probability of sampling the (randomly) chosen bitstring
    probs_bitstring = []
    # simulate the execution of many Haar-random quantum programs
    for _ in range(trials):
        unitary = random_unitary(dimension)
        prob = simulate_probability(unitary, bitstring)
        probs_bitstring.append(prob)

    return probs_bitstring


def random_bitstring_probs(n_qubits: int, n_programs: int) -> List:
    dim = 2**n_qubits
    # keep track of probability of sampling (randomly) chosen bitstring
    probs_bitstring = []
    # simulate many Haar-random circuits
    for _ in range(n_programs):
        unitary = random_unitary(dim)
        bitstring = np.random.choice(dim, p=[np.abs(unitary[b,0])**2 for b in range(dim)])
        prob = np.abs(unitary[bitstring,0])**2
        probs_bitstring.append(prob)
    return probs_bitstring


def fidelity_xeb(n_qubits: int, trials: int, n_samples: int, sampler: Callable[[np.ndarray, int], float]) -> float:
    dim = 2**n_qubits
    # keep track of the ideal simulated probabilities
    ideal_probs = []
    # loop over the random programs
    for _ in range(trials):
        unitary = random_unitary(dim)
        sample_probs = [sampler(unitary, bb) for bb in range(dim)]
        samples = np.random.choice(dim, size=n_samples, p=sample_probs)
        for sample in samples:
            ideal_prob = simulate_probability(unitary, sample)
            ideal_probs.append(ideal_prob)

    return dim*np.mean(ideal_probs) - 1


def fidelity_xeb_noisy(n_qubits: int, trials: int, n_samples: int, prob_no_error: float) -> float:
    dim = 2**n_qubits

    # keep track of ideal output probabilities
    ideal_probs = []
    
    # identify 1q Pauli operators
    sI = np.array([[1, 0], [0, 1]])
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0, -1]])
    paulis = [sI, sX, sY, sZ]

    # identify depolarizing operators over n-qubit space
    depolarizing_ops = []
    for x in itertools.product(paulis, repeat=n_qubits):
        op = functools.reduce(lambda a, b: np.kron(a, b), x)
        depolarizing_ops.append(op)

    # loop over random programs
    for _ in range(trials):
        unitary = random_unitary(dim)

        # sample an operator according to specified probabilities
        all_ops = [unitary] + depolarizing_ops
        probabilities = [prob_no_error] + len(depolarizing_ops)*[(1-prob_no_error)/len(depolarizing_ops)]
        op_idx = np.random.choice(len(all_ops), p=probabilities)
        op = all_ops[op_idx]

        # draw samples from the resultant state
        sample_probs = [simulate_probability(op, bb) for bb in range(dim)]
        samples = np.random.choice(dim, size=n_samples, p=sample_probs)

        # collect ideal sampling probability for these samples
        for sample in samples:
            ideal_prob = simulate_probability(unitary, sample)
            ideal_probs.append(ideal_prob)

    # calculate and return the fidelity of the XEB
    return dim*np.mean(ideal_probs) - 1


if __name__ == "__main__":
    # empirical Porter-Thomas distribution
    n_qubits = 4
    porter_thomas = quantum_sample_probability(n_qubits, 10_000)

    # theoretical Porter-Thomas distribution
    dim = 2**n_qubits
    xspace = np.linspace(0.0, 1.0, 100)
    yspace = dim * np.exp(-dim*xspace)

    # plot both empirical and theoretical calculations
    plt.figure(figsize=(9, 6))
    plt.hist(porter_thomas, bins=50, density=True, label='Empirical Distribution')
    plt.plot(xspace, yspace, label='Theoretical Porter-Thomas Distribution')
    # plot the uniform distribution for reference
    plt.axvline(x=1/dim, linestyle='dotted', color='r', label='Uniform Distribution')

    plt.xlabel("Probability p")
    plt.ylabel("Probability that the random bistring occurs with probability p")
    plt.legend(loc='best')
    plt.show()

    # empirical distribution of random bitstring probabilities
    rand_bb_probs = random_bitstring_probs(n_qubits, 10_000)
    yspace = xspace*(dim**2)*np.exp(-dim*xspace)

    # plot both empirical and theoretical calculations
    plt.figure(figsize=(9, 6))
    plt.hist(rand_bb_probs, bins=50, density=True, label='Empirical')
    plt.plot(xspace, yspace, label='Theoretical')
    # plot the uniform distribution for reference
    plt.axvline(x=1/dim, linestyle='dotted', color='r', label='Uniform Distribution')

    plt.xlabel("Probability p")
    plt.ylabel("Probability that the random bistring occurs with probability p")
    plt.legend(loc='best')
    plt.show()

    # sample f_xeb for an ideal processor using the same parameters as in the Google paper
    f_xeb = fidelity_xeb(n_qubits=n_qubits, trials=10, n_samples=10**5, sampler=simulate_probability)
    print("Empirical FXEB: ", f_xeb)
    print("Theoretical FXEB: ", dim*(2/(dim+1)) - 1)
    print('\n')

    # sample f_xeb for a uniform distribution sampler
    def unif_dist(unitary, bitstring):
        return 1/dim # all bitstrings have the same probability
    unif_xeb = fidelity_xeb(n_qubits=n_qubits, trials=10, n_samples=10**5, sampler=unif_dist)
    print("Empirical FXEB of a uniform sampler: ", unif_xeb)
    print("Theoretical FXEB of a uniform sampler", 0.0)
    print('\n')

    # run the noisy experiment
    p = 0.7
    noisy_xeb = fidelity_xeb_noisy(n_qubits=6, trials=10**3, n_samples=10, prob_no_error=p)
    print("Empirical FXEB of a noisy simulation: ", noisy_xeb)
    print("Theoretical FXEB of a noisy simulation: ", p*(dim-1)/(dim+1))
