# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:33:35 2024

@author: dpope
"""

#!/usr/bin/env python
# coding: utf-8

# ## Background
# 
# [Variational quantum algorithms](https://arxiv.org/abs/2012.09265) are promising candidate hybrid-algorithms for observing the utility of quantum computation on noisy near-term devices. Variational algorithms are characterized by the use of a classical optimization algorithm to iteratively update a parameterized trial solution, or "ansatz". Chief among these methods is the Variational Quantum Eigensolver (VQE) that aims to solve for the ground state of a given Hamiltonian represented as a linear combination of Pauli terms, with an ansatz circuit where the number of parameters to optimize over is polynomial in the number of qubits.  Given that size of the full solution vector is exponential in the number of qubits, successful minimization using VQE requires, in general, additional problem specific information to define the structure of the ansatz circuit.
# 
# Executing a VQE algorithm requires the following 3 components:
# 
# 1.  Hamiltonian and ansatz (problem specification)
# 2.  Qiskit Runtime estimator
# 3.  Classical optimizer
# 
# Although the Hamiltonian and ansatz require domain specific knowledge to construct, these details are immaterial to the Runtime, and we can execute a wide class of VQE problems in the same manner.
# 

# ## Setup
# 
# Here we import the tools needed for a VQE experiment.
# ***
#***


# General imports
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt




# runtime imports
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator

# To run on hardware, select the backend with the fewest number of jobs in the queue
service = QiskitRuntimeService(channel="ibm_quantum")


backend = service.get_backend("ibmq_qasm_simulator")





# ## Step 1: Map classical inputs to a quantum problem
# 
# Here we define the problem instance for our VQE algorithm. Although the problem in question can come from a variety of domains, the form for execution through Qiskit Runtime is the same. Qiskit provides a convenience class for expressing Hamiltonians in Pauli form, and a collection of widely used ansatz circuits in the [`qiskit.circuit.library`](https://docs.quantum-computing.ibm.com/api/qiskit/circuit_library).
# 
# Here, our example Hamiltonian is derived from a quantum chemistry problem
# 


'''
define three parameters of the model:
n = number of qubits/spins
J = coupling strength between neighbouring qubits/spins
g = transverse energy coefficient, depends on the strength of the transverse magnetic field
'''

n = 2
J = -1.
g = 1

#lists all the h_z values that we go through
h_z_list = [-1.0,-0.5,-0.1,0.1,0.5,1.0]


tolerance = 0.25

#returns a list that represents the Hamiltonian
def build_Ising_hamiltonian(n,J,h_x,h_z):

    hamiltonianList = []        

    #Add spin-spin interaction terms to Hamiltonian
    for i in range(0,n-1):
        
        #initialize currentPauliWord as an empty string
        currentPauliWord = ""
        
        for j in range(n):
       
            if j == i or j == i+1:
                #add a Z term to currentPauliWord
                currentPauliWord = currentPauliWord + "Z"
        
            else:
                #add an identity term to currentPauliWord
                currentPauliWord = currentPauliWord + "I"
        
        #Note that (currentPauliWord,J) is a tuple or, more specifically, a 2-tuple
        spinInteractionTerm = (currentPauliWord,J)
        print("spinInteractionTerm=",spinInteractionTerm)
        hamiltonianList.append(spinInteractionTerm)    

    print("hamiltonianList=",hamiltonianList)

        
    #add transverse B field terms to Hamiltonian    
    for i in range(0,n):
        currentPauliWord = ""
        
        for j in range(n):
       
            if j == i:
                #add an X term to currentPauliWord
                currentPauliWord = currentPauliWord + "X"
        
            else:
                #add an identity term to currentPauliWord
                currentPauliWord = currentPauliWord + "I"
        
        #change transverse terms to be physical by having -h
        #when spin is aligned with B field, energy is lower
        transverseTerm = (currentPauliWord,h_x)

        hamiltonianList.append(transverseTerm)        


    #add longitudinal B field terms to Hamiltonian    
    for i in range(0,n):
        currentPauliWord = ""
        
        for j in range(n):
       
            if j == i:
                #add an Z term to currentPauliWord
                currentPauliWord = currentPauliWord + "Z"
        
            else:
                #add an identity term to currentPauliWord
                currentPauliWord = currentPauliWord + "I"
        
       
        transverseTerm = (currentPauliWord,h_z)

        hamiltonianList.append(transverseTerm) 


    return hamiltonianList



hamiltonians = []

for i in range(len(h_z_list)):
    hamiltonians.append(SparsePauliOp.from_list(build_Ising_hamiltonian(n,J,g,h_z_list[i])))


# Our choice of ansatz is the `EfficientSU2` that, by default, linearly entangles qubits, making it ideal for quantum hardware with limited connectivity.
# 


#Note: ansatz is the same for all hamiltonians
ansatz = EfficientSU2(hamiltonians[0].num_qubits)
ansatz.decompose().draw("mpl", style="iqp")


# From the previous figure we see that our ansatz circuit is defined by a vector of parameters, $\theta_{i}$, with the total number given by:
num_params = ansatz.num_parameters

#create the magnetization operator: 1/N \sigma_{i} Z_{i}
magnetizationList = []


for i in range(n):
    operatorString = ""
    
    for j in range(n):
        if j == i:
            operatorString += "Z"
        else:
            operatorString += "I"

    magnetizationList.append((operatorString,1/n))



M = SparsePauliOp.from_list(magnetizationList)

# ## Step 2: Optimize problem for quantum execution.
# 

# To reduce the total job execution time, Qiskit Runtime V2 primitives only accept circuits (ansatz) and observables (Hamiltonian) that conforms to the instructions and connectivity supported by the target system (referred to as instruction set architecture (ISA) circuits and observables, respectively).
# 

# ### ISA Circuit
# 

# We can schedule a series of [`qiskit.transpiler`](https://docs.quantum-computing.ibm.com/api/qiskit/transpiler) passes to optimize our circuit for a selected backend and make it compatible with the instruction set architecture (ISA) of the backend. This can be easily done using a preset pass manager from `qiskit.transpiler` and its `optimization_level` parameter.
# 
# *   [`optimization_level`](https://docs.quantum-computing.ibm.com/api/qiskit/transpiler_preset#preset-pass-manager-generation): The lowest optimization level just does the bare minimum needed to get the circuit running on the device; it maps the circuit qubits to the device qubits and adds swap gates to allow all 2-qubit operations. The highest optimization level is much smarter and uses lots of tricks to reduce the overall gate count. Since multi-qubit gates have high error rates and qubits decohere over time, the shorter circuits should give better results.
# 



from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=2)

ansatz_isa = pm.run(ansatz)

ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")
ansatz_isa.draw(output="mpl", filename="isa_ansatz_circuit.png")


# ### ISA Observable
# 

# Similarly, we need to transform the Hamiltonian to make it backend compatible before running jobs with [`Runtime Estimator V2`](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2#run). We can perform the transformation using the `apply_layout` the method of `SparsePauliOp` object.
# 

hamiltonians_isa = [i.apply_layout(layout=ansatz_isa.layout) for i in hamiltonians]

M_isa = M.apply_layout(layout=ansatz_isa.layout)


# ## Step 3: Execute using Qiskit Primitives.
# 
# Like many classical optimization problems, the solution to a VQE problem can be formulated as minimization of a scalar cost function.  By definition, VQE looks to find the ground state solution to a Hamiltonian by optimizing the ansatz circuit parameters to minimize the expectation value (energy) of the Hamiltonian.  With the Qiskit Runtime [`Estimator`](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2) directly taking a Hamiltonian and parameterized ansatz, and returning the necessary energy, the cost function for a VQE instance is quite simple.
# 
# Note that the `run()` method of [Qiskit Runtime `EstimatorV2`](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2)  takes an iterable of `primitive unified blocs (PUBs)`. Each PUB is an iterable in the format `(circuit, observables, parameter_values: Optional, precision: Optional)`.
# 



def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    return energy




#Creating a callback function. 
# Callback functions are a common way for users to obtain additional information about the status of an iterative algorithm.  The standard SciPy callback routine allows for returning only the interim vector at each iteration.  However, it is possible to do much more than this.  Here, we show how to use a mutable object, such as a dictionary, to store the current vector at each iteration, for example in case we need to restart the routine due to failure, and also return the current iteration number and the average time per iteration.
 
def build_callback(ansatz, hamiltonian, estimator, callback_dict,h_z_value_index):    
    """Return callback function that uses Estimator instance,
    and stores intermediate values into a dictionary.

    Parameters:
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        callback_dict (dict): Mutable dict for storing values
        h_value_counter denotes the current index of h. It's between 0 and n-1. 

    Returns:
        Callable: Callback function object
    """

    def callback(current_vector):
        """Callback function storing previous solution vector,
        computing the intermediate cost value, and displaying number
        of completed iterations and average time per iteration.

        Values are stored in pre-defined 'callback_dict' dictionary.

        Parameters:
            current_vector (ndarray): Current vector of parameters
                                      returned by optimizer
        """
        # Keep track of the number of iterations
        callback_dict["iters"][h_z_value_index] += 1
 
        # Set the prev_vector to the latest one
        callback_dict["prev_vector"][h_z_value_index] = current_vector

        # Compute the value of the cost function at the current vector
        # This adds an additional function evaluation
        #Note that we put hamiltonian and current_vector within lists as EstimatorV2.run() requires this data type
        #for its arguments.
        pub = (ansatz, [hamiltonian], [current_vector])
        result = estimator.run(pubs=[pub]).result()
        current_cost = result[0].data.evs[0]
        callback_dict["cost_history"][h_z_value_index].append(current_cost)

        #print out every 25^th minimum energy estimate        
        if callback_dict["iters"][h_z_value_index] % 25 == 0:
        
            # Print to screen on single line
            print(
                "Iters. done: {} [Current cost: {}]".format(callback_dict["iters"][h_z_value_index], current_cost),
                #end="\r",
                end="\n",
                flush=True,
                )

    return callback

callback_dict = {
    "prev_vector": [None]*len(h_z_list),
    "iters": [0]*len(h_z_list),
    "cost_history": [[] for i in range(len(h_z_list))],
}

# We can now use a classical optimizer of our choice to minimize the cost function. Here, we use the [COBYLA routine from SciPy through the `minimize` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). Note that when running on real quantum hardware, the choice of optimizer is important, as not all optimizers handle noisy cost function landscapes equally well.
# 
# To begin the routine, we specify a random initial set of parameters:
# 
x0 = 2 * np.pi * np.random.random(num_params)

# Because we are sending a large number of jobs that we would like to execute together, we use a [`Session`](https://docs.quantum-computing.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.Session) to execute all the generated circuits in one block.  Here `args` is the standard SciPy way to supply the additional parameters needed by the cost function.
# 

# To run on local simulator:
#   1. Use the Estimator from qiskit.primitives instead.
#   2. Remove the Session context manager below.
with Session(backend=backend) as session:    
    print("session.details()=",session.details())
    
    #initialize res as a list of n zeroes
    res=[0]*len(h_z_list)

    #This variable stores all the final results from estimator
    pub_result=[0]*len(h_z_list)
    
    for i in range(len(h_z_list)):
        
        estimator = Estimator(session=session)
        estimator.options.default_shots = 5_000

        callback = build_callback(ansatz_isa, hamiltonians_isa[i], estimator, callback_dict,i)

        res[i] = minimize(
            cost_func,
            x0,
            args=(ansatz_isa, hamiltonians_isa[i], estimator),
            method="cobyla",
            tol = tolerance,
            callback=callback,
        )
    
        M_estimator = Estimator(session=session)
        M_estimator.options.default_shots = 5_000

        #best IBM documentation on Estimator:
        #https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2
    
        job = M_estimator.run([(ansatz_isa, M_isa, [res[i].x])])
        pub_result[i] = job.result()[0]
    
        print(f"h_z value = {h_z_list[i]}")
        print("i=",i)
        print("***<M>***")
        print(f"pub_result[i].data.evs= {pub_result[i].data.evs}")
     
for i in range(len(h_z_list)):
    print(f"i={i} | res[i].x={res[i].x}")

data_evs_list=[0.0]*len(h_z_list)

for i in range(len(h_z_list)):
    data_evs_list[i] = abs(pub_result[i].data.evs)

print("data_evs_list=",data_evs_list)

fig, ax = plt.subplots()
ax.plot(h_z_list, data_evs_list,marker='x')
ax.set_xlabel("h_z")
ax.set_ylabel("|<M>|")
plt.draw()

#create a graph with a smooth curve of best fit
#subplots creates a wrapper (?) for creating multiple plots together.
#In this case, we just use the default value of 1 row & 1 column of plot(s)
fig2, ax2 = plt.subplots()

ax2.scatter(h_z_list, data_evs_list,marker='x')


poly_coeffs = np.polyfit(h_z_list, data_evs_list,5)
y = np.polyval(poly_coeffs,h_z_list)
polyline = np.linspace(h_z_list[0], h_z_list[-1], 50)
yy = np.polyval(poly_coeffs,polyline)

ax2.plot(polyline,yy)

ax2.set_xlabel("h_z")
ax2.set_ylabel("|<M>|")
plt.draw()


# ## Step 4: Post-process, return result in classical format.
# 

# If the procedure terminates correctly, then the `prev_vector` and `iters` values in our `callback_dict` dictionary should be equal to the solution vector and total number of function evaluations, respectively.  This is easy to verify:
# 
 
for i in range(len(h_z_list)):
    print(f"i={i}")
    callback_dict["iters"][i] == res[i].nfev


# We can also now view the progress towards convergence as monitored by the cost history at each iteration:
# 


for i in range(len(h_z_list)):
    fig, ax = plt.subplots()
    ax.plot(range(callback_dict["iters"][i]), callback_dict["cost_history"][i])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    plt.draw()