

# ## Random circuit generation using only the Clifford + T set of gates

# In[4]:


# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for generating random circuits."""

import numpy as np
import random
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import (IGate, U1Gate, U2Gate, U3Gate, XGate,
                                                   YGate, ZGate, HGate, SGate, SdgGate, TGate,
                                                   TdgGate, RXGate, RYGate, RZGate, CXGate,
                                                   CYGate, CZGate, CHGate, CRZGate, CU1Gate,
                                                   CU3Gate, SwapGate, RZZGate,
                                                   CCXGate, CSwapGate)
from qiskit.circuit.exceptions import CircuitError
from qiskit.util import deprecate_arguments


@deprecate_arguments({'n_qubits': 'num_qubits'})
def random_circuit(num_qubits, depth, HGate_prob, max_operands=2, measure=False,
                   conditional=False, reset=False, seed=None,
                   *, n_qubits=None):  # pylint:disable=unused-argument
    """Generate random circuit of arbitrary size and form.
    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.extensions`. For example:
    .. jupyter-execute::
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(2, 2, measure=True)
        circ.draw(output='mpl')
    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum operands of each gate (between 1 and 3)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)
        n_qubits (int): deprecated, use num_qubits instead
    Returns:
        QuantumCircuit: constructed circuit
    Raises:
        CircuitError: when invalid options given
    """
    if max_operands < 1 or max_operands > 2:
        raise CircuitError("max_operands must be between 1 and 2")

    one_q_ops = [SGate, TGate, HGate]
    two_q_ops = [CXGate]
    
    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(num_qubits)
    
    #qbits random initialization
    #random_circuit.initial_state = []
    #for i in range(num_qubits):
     #   b = random.randint(0,1)
      #  if b == 1:
       #     qc.x(i)
        #    random_circuit.initial_state.insert(0,1)
        #else:
         #   random_circuit.initial_state.insert(0,0)
    #qc.barrier()

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)
    
    n_of_HGates=0
    n_of_gates=0
    random_circuit.Us=[]                                         #PROBABILITY OF CHOOSING H GATE 
    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # U will be a list with the following syntax: U = [gate, qbit on which the gate acts]
        # in the case of the CXGate: U = [Cx, control qbit, target qbit]
        U=[]
        # choose either 1 or 2 qbits for the operation
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = random.choices([1, max_possible_operands], weights=[1-(1-HGate_prob)/3,(1-HGate_prob)/3], k=1) #Weighted distribution
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands[0]]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]
            
            if num_operands[0] == 1:
                n_of_gates += 1
                operation = random.choices(one_q_ops, weights=[((1-HGate_prob)/3)/(1-(HGate_prob/3)),\
                                                               ((1-HGate_prob)/3)/(1-(HGate_prob/3)),\
                                                                HGate_prob/(1-(HGate_prob/3))], k=1)[0]
                if (operation == HGate):
                    n_of_HGates += 1
                    U.append(['H', operands[0]])
                if (operation == SGate):
                    U.append(['S', operands[0]])
                if (operation == TGate):
                    U.append(['T', operands[0]])
                               
            elif num_operands[0] == 2:
                n_of_gates += 1
                operation = rng.choice(two_q_ops)
                U.append(['CX', operands[0], operands[1]])
                           
            num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)
            
            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)

            qc.append(op, register_operands)          
        random_circuit.Us.append(U)
        random_circuit.ratio_HGates = (n_of_HGates/n_of_gates)*100
    if measure:
        qc.measure(qr, cr)
    return qc