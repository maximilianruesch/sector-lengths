
README

#################################################
What is this repository for?

Contains my code for research in quantum information science. You might be interested in some nice functionalities in

qgeo.py   (main library of useful functions)
Qk_sdp.py (sdp to determine if state is a groundstate of local Hamiltonian)
sdp_pure_state_mmix_marginals_(iterative_hard_thresholding).py


It contains python code to calculate various things in quantum information science / 
entanglement / information geometry, on (mostly) qubits. Below are some examples

- create states:  ket('01'), P('XXIIZ'), (hyper-) graph states in various ways
- common operations: partial transpose, partial trace, etc.
- many common states implemented. chi4(), GHZ(n), etc.
- random objects: unitaries, pure states, density matrices
- operator decompositions: pauli_coeff(Op)
- various local things, restricted pauli basis, pauli decomposition, weight of operator: weight(op)
- calculate distances, fidelity(rho, sig), entropies, trace_distances etc.
- groundstate, gap, spectrum, etc.
- (hyper-) graph state functionalities, as e.g. partial traces, shrink/delete, state to edges, etc.
- calculate information projections

Most functions have as input pure states or density matrices (=numpy arrays), and are designed to be easy to both read and use.


#################################################
Can I use it?

YES! The code is open source under the BSD 3-Clause License. See LICENSE.txt

#################################################
How do I get set up?

you need python 2.7, and fairly recent versions of numpy, scipy

#################################################
Contribution guidelines:

If you want to contribute, get in touch with us.
In almost all cases, readability over speed.

Thanks to the contributers:
Tristan Kraft
Nikolai Wyderka

#################################################
Who do I talk to?

Felix Huber
felix.huber@uj.edu.pl
https://chaos.if.uj.edu.pl/ZOA/index.php?which=people&lang=en&who=FelixHuber
http://www.physik.uni-siegen.de/tqo/members/formermembers/huber/index.xml?lang=e

Nikolai Wyderka
wyderka@physik.uni-siegen.de
http://www.physik.uni-siegen.de/tqo/members/wyderka/
