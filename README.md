# 2025 Quantum Informaton Hackathon

## Repository structure

```text
quantum2025/
├─ codes/
│  ├─ newbies_1/   # Grover + 3-SAT + hill-climbing configuration recovery
│  ├─ newbies_2/   # OD (origin–destination) distribution + DynamicConfigRecovery + adaptive resampling
│  ├─ newbies_3/   # 3-qubit bit-flip code + code-space projection
│  └─ newbies_4/   # U(1) 2×2 plaquette + gauge-invariant configuration recovery
├─ readings/       # Papers, slides, and notes related to the contest and examples
├─ configuration_recovery.py  # Shared configuration-recovery utilities (planned)
├─ fermion.py                 # Fermionic / many-body utilities (planned)
└─ README.md

```


## Notebook overview
### 1. newbies_1 – Grover + 3-SAT + hill-climbing configuration recovery
#### Concept
A small 3-variable SAT instance is solved with Grover’s algorithm, and the QPU’s noisy shot distribution is post-processed using a simple hill-climbing configuration recovery to extract multiple satisfying assignments.

##### 1.1 3-SAT instance
The SAT formula is defined as a list of clauses:
- Each clause is a tuple of integers.
- Positive i means x_{i-1}, negative -i means ¬x_{i-1}.
clauses = [
    ( 1,  2,  3),   #  x0 ∨  x1 ∨  x2
    (-1,  2,  3),   # ¬x0 ∨  x1 ∨  x2
    ( 1, -2,  3),   #  x0 ∨ ¬x1 ∨  x2
    ( 1,  2, -3),   #  x0 ∨  x1 ∨ ¬x2
    (-1, -2, -3),   # ¬x0 ∨ ¬x1 ∨ ¬x2  ← excludes only 111
]
- An is_satisfied(bitstr: str) -> bool function checks whether a 3-bit string satisfies all clauses.
- Exhaustive search over 000–111 builds all_sols, the list of satisfying bitstrings. These are the states that the Grover oracle should mark with a phase flip.

##### 1.2 Grover oracle and diffusion
- apply_oracle(qc, solutions):
  - For each solution bitstring sol:
    1. Apply X to positions where sol has 0, so that the solver state becomes |111…1⟩.
    2. Implement a multi-controlled Z using H–MCX–H on the last qubit.
    3. Undo the earlier X gates.
  - Repeating this for each solution builds a multi-solution Grover oracle.
  - 
- apply_diffusion(qc):
  - Standard Grover diffusion operator:
    1. Apply H to all qubits
    2. Apply X to all qubits
    3. Apply a multi-controlled Z on |111…1⟩
    4. Undo X, then undo H
  - This acts as a reflection about the average amplitude, amplifying marked states.

- build_grover_for_sat(n, solutions, iterations):
  - Prepares an equal superposition on n qubits via H⊗n
  - Repeats [oracle → diffusion] for the given number of iterations
  - Measures all qubits at the end
    
##### 1.3 Running on ibm_aachen
run_on_aachen(qc, shots):
- Creates QiskitRuntimeService(), selects backend = service.backend("ibm_aachen")
- Uses generate_preset_pass_manager(optimization_level=3, backend=backend) for transpilation
- Samples with a Sampler(mode=backend) primitive for the given shots
- Returns a dictionary of counts: {"bitstring": count, ...}

##### 1.4 Configuration recovery via hill-climbing
Two helpers implement simple hill-climbing over bitstrings:
- hill_climb_from(start, counts):
  - Treats the counts as a score function on bitstrings.
  - Starting from start, iteratively flips one bit at a time, moving to a neighbor only if it has a higher count.
  - Stops when no neighbor improves the score; returns the local maximum.
    
- recover_multiple_solutions(counts, num_sols):
  - Sorts bitstrings by their counts in descending order to form initial candidates.
  - For each candidate, runs hill_climb_from to reach a local maximum.
  - Collects distinct local maxima until num_sols solutions have been found.
    
This yields a list of recovered candidate solutions from a noisy shot distribution, even if the QPU does not place the global maximum probability exactly on each satisfying assignment.



### 2. newbies_2 – OD distribution, DynamicConfigRecovery, and adaptive sampling
#### Concept
A 3×3 origin–destination (OD) demand matrix is embedded as amplitudes on a 2n-qubit register.
The circuit is sampled once on a QPU, and then a DynamicConfigRecovery class plus adaptive reweighting is used to refine a set of “valid” OD pairs using purely classical resampling.

##### 2.1 DynamicConfigRecovery
class DynamicConfigRecovery:
    def __init__(self, n, prob_threshold, max_distance):
        self.n = n
        self.prob_threshold = prob_threshold
        self.max_distance = max_distance
        self.valid_pairs = set()
- n is the number of bits used to encode one index (i or j), so the register has 2n qubits in total.
- valid_pairs stores OD pairs (i, j) that are currently considered “valid”.
The main method is:
def repair_and_update(self, raw_counts):
  1. Convert raw bitstrings into (i, j) counts
  2. If valid_pairs is non-empty, reroute invalid pairs to the
    -    nearest valid pair within manhattan distance ≤ max_distance
  3. Normalize to form a probability distribution
  4. Update valid_pairs to those with probability ≥ prob_threshold
  5. Return (valid_pairs, probs)

- Step 2 implements a simple “repair” step: invalid OD pairs are mapped to their closest currently-valid pair if they are within a bounded manhattan distance.

##### 2.2 Amplitude embedding for OD distribution
def build_od_embedding_circuit(n, distribution):
    qr = QuantumRegister(2*n, 'q')
    cr = ClassicalRegister(2*n, 'c')
    qc = QuantumCircuit(qr, cr)

    dim = 2**(2*n)
    amps = np.zeros(dim, complex)
    for (i, j), p in distribution.items():
        idx = (i << n) | j   # pack (i,j) into one integer index
        amps[idx] = np.sqrt(p)
    amps /= np.linalg.norm(amps)

    qc.initialize(amps, qr)
    qc.measure(qr, cr)
    return qc
- distribution is a classical probability distribution over (i, j).
- The circuit prepares a pure state whose amplitude squared gives this distribution.
  
##### 2.3 QPU and local sampling
- sample_qpu_once(qc, backend, shots):
  - Transpiles the circuit for the backend with optimization level 3.
  - Uses a Session(backend) and Sampler for shots samples.
  - Returns the resulting shot counts as a dictionary.
- sample_local(distribution, shots, n):
  - Draws shots samples from a classical multinomial with probabilities given by distribution.
  - Converts each sampled (i, j) pair into a bitstring of length 2n (i bits then j bits).

##### 2.4 Main loop: QPU once + adaptive refinement
In main():
  1. Define a 3×3 OD demand:
raw_demand = {
    (0,0):5,(0,1):3,(0,2):2,
    (1,0):1,(1,1):4,(1,2):5,
    (2,0):2,(2,1):2,(2,2):6
}
total = sum(raw_demand.values())
od_dist = {p: d/total for p, d in raw_demand.items()}
  2. Set parameters:
    - n = (3-1).bit_length() # 2 bits for indices 0–2
    - prob_threshold, max_distance, and shots
  3. Use build_od_embedding_circuit and sample_qpu_once with backend "ibm_aachen" to obtain raw_counts from a single QPU run.
  4. Initialize DynamicConfigRecovery(n, prob_threshold, max_distance).
  5. For a fixed number of iterations (e.g. 5):
    - Run valid_pairs, probs = dcr.repair_and_update(current_counts)
    - If probs is empty, break.
    - Apply a biasing step:
      - Define gamma = 1 + 0.5 * itr
      - Compute p^gamma for each pair to sharpen the distribution
      - Renormalize the probabilities
    - Draw new current_counts via sample_local(current_dist, shots, n) from this biased distribution.
  6. At the end, print the final dcr.valid_pairs, the set of OD pairs that survived the iterative refinement.
     
Takeaway: This notebook shows a “QPU once + classical iterative refinement” pattern, where an initial quantum sampling is refined purely classically via dynamic configuration recovery and adaptive resampling.

### 3. newbies_3 – 3-qubit bit-flip code and code-space projection
#### Concept
A 3-qubit bit-flip repetition code is used to encode a logical state.
Noisy samples from ibm_aachen are then projected onto the code subspace by mapping each observed bitstring to the nearest codeword in Hamming distance (000 or 111).

##### 3.1 BitFlipConfigRecovery
class BitFlipConfigRecovery:
    def __init__(self, valid_codewords):
        """
        valid_codewords: list of bitstrings, e.g. ['000', '111']
        """
        self.valid = valid_codewords
        self.freq = Counter()

    def repair_and_update(self, raw_counts):
        recovered = Counter()
        # Accumulate raw frequencies
        for bits, cnt in raw_counts.items():
            self.freq[bits] += cnt
        # Project each sample onto its nearest codeword
        for bits, cnt in raw_counts.items():
            best = min(
                self.valid,
                key=lambda v: sum(a != b for a, b in zip(bits, v))
            )
            recovered[best] += cnt
        return recovered
- self.freq keeps track of the full raw distribution over all observed bitstrings.
- recovered records the distribution after projection onto the code space.

##### 3.2 3-qubit bit-flip code circuit
def encode_bit_flip_code():
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qr, cr)

    # Prepare logical |+> on data qubit
    qc.h(qr[0])
    # Encode repetition code
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[2])
    # Measure all
    qc.measure(qr, cr)
    return qc
- The logical information is effectively mirrored on three physical qubits (|0⟩ becomes |000⟩, |1⟩ becomes |111⟩ in the Z basis).
- The code in this notebook prepares a superposition via H on the first qubit and encodes it into the repetition code.
 
##### 3.3 Running and recovery
In main():
  1. Build the encoding circuit qc = encode_bit_flip_code().
  2. Transpile and run it on ibm_aachen using QiskitRuntimeService, Session, and Sampler for, e.g., 2048 shots.
  3. Feed the resulting raw_counts into BitFlipConfigRecovery(['000', '111']).
  4. Build a pandas.DataFrame with columns:
    - bitstring – observed bitstring
    - raw – count in raw_counts
    - recovered – count in the projected distribution
  5. Print the table for comparison.
     
Takeaway: This notebook demonstrates projection of noisy measurement outcomes onto a quantum error-correcting code space (here, the simple 3-qubit repetition code) using Hamming-distance–based configuration recovery.

### 4. newbies_4 – U(1) 2×2 plaquette and gauge-invariant configuration recovery
#### Concept
A minimal U(1) lattice gauge toy model is built on four “link” qubits forming a 2×2 plaquette.
A 4-body plaquette interaction is implemented via a sequence of gates.
Noisy measurement outcomes (with additional artificial measurement noise) are then projected onto the gauge-invariant subspace defined by even-parity bitstrings.
##### 4.1 GaugeConfigRecovery
class GaugeConfigRecovery:
    def __init__(self, valid_configs):
        """
        valid_configs: list of bitstrings satisfying Gauss law (even parity)
        """
        self.valid = valid_configs
        self.freq = Counter()

    def repair_and_update(self, raw_counts):
        recovered = Counter()
        # Accumulate raw frequencies
        for bits, cnt in raw_counts.items():
            self.freq[bits] += cnt
        # Project each sample to the nearest gauge-invariant config
        for bits, cnt in raw_counts.items():
            best = min(
                self.valid,
                key=lambda v: sum(a != b for a, b in zip(bits, v))
            )
            recovered[best] += cnt
        return recovered
- This is structurally similar to the bit-flip code recovery, but now the set of valid configurations is “all 4-bit strings with even parity”.
 
##### 4.2 U(1) 2×2 plaquette circuit
def build_u1_plaquette():
    qr = QuantumRegister(4, 'link')
    cr = ClassicalRegister(4, 'c')
    qc = QuantumCircuit(qr, cr)

    # Initial superposition over link configurations
    qc.h(qr)

    # Plaquette rotation example
    theta = 0.3

    # Implement a Z⊗Z⊗Z⊗Z phase on the four links
    qc.h(qr[2]); qc.h(qr[3])
    qc.cx(qr[0], qr[2])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[2], qr[3])
    qc.rz(2*theta, qr[3])
    qc.cx(qr[2], qr[3])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[0], qr[2])
    qc.h(qr[2]); qc.h(qr[3])

    qc.measure(qr, cr)
    return qc
- The 4-body phase is implemented via standard tricks (Hadamards plus a chain of CNOTs and a single-qubit RZ).

##### 4.3 Measurement noise
def add_measurement_noise(raw_counts, flip_prob=0.1):
    noisy = Counter()
    for bits, cnt in raw_counts.items():
        for _ in range(cnt):
            b = ''.join(
                bit if random.random() > flip_prob
                else ('1' if bit == '0' else '0')
                for bit in bits
            )
            noisy[b] += 1
    return noisy
- Each bit of each shot is independently flipped with probability flip_prob, simulating classical measurement noise.
  
##### 4.4 Gauge-invariant configurations
def generate_valid_configs():
    valid = []
    for i in range(16):
        bits = format(i, '04b')
        if bits.count('1') % 2 == 0:
            valid.append(bits)
    return valid
- The gauge-invariant subspace is chosen as “4-bit states with even parity” (a minimal toy implementation of a Gauss-law constraint).
  
##### 4.5 Main flow
1. Build and run the plaquette circuit on ibm_aachen to obtain raw_counts.
2. Add artificial measurement noise via add_measurement_noise.
3. Generate the set of even-parity configurations and construct GaugeConfigRecovery(valid).
4. Run recovered = dcr.repair_and_update(noisy_counts).
5. Display results in a pandas.DataFrame with columns:
  - config
  - raw (noisy_counts)
  - recovered (after projection onto gauge-invariant space)

Takeaway: This notebook provides a small example of lattice gauge theory + configuration recovery, projecting noisy measurements back into a gauge-invariant (even-parity) subspace.
::contentReference[oaicite:0]{index=0}

