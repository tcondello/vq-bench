"""
Terminal Bench Sandbox Task Definitions for VQ-bench.
Defines 20 realistic multi-step developer code navigation, bug localization, and architecture tasks.
"""

SANDBOX_TASKS = [
    # Category 1: Bug & Error Localization
    {
        "id": "task_01",
        "category": "Bug Localization",
        "prompt": "Find where the HDF5 mutex lock is acquired during flat slice I/O to prevent race conditions.",
        "target_file": "src/bin/vqb/h5.rs",
        "target_symbols": ["H5Dread", "H5Dwrite", "LOCK", "hdf5_sys"],
        "expected_snippet_keywords": ["hdf5_sys::LOCK", "H5Dread", "read_raw"]
    },
    {
        "id": "task_02",
        "category": "Bug Localization",
        "prompt": "Locate the exact assertion that checks codebook code levels are power-of-2 arity and sorted.",
        "target_file": "src/util/codebooks.rs",
        "target_symbols": ["rows_have_pow2_arity_and_are_sorted", "assert_eq"],
        "expected_snippet_keywords": ["pow2", "sorted", "assert"]
    },
    {
        "id": "task_03",
        "category": "Bug Localization",
        "prompt": "Find where the variable-length code store lengths table is appended when row widths disagree.",
        "target_file": "src/bin/vqb/codes.rs",
        "target_symbols": ["CodeStore", "lengths", "variable_length"],
        "expected_snippet_keywords": ["lengths", "stride", "variable"]
    },
    {
        "id": "task_04",
        "category": "Bug Localization",
        "prompt": "Where is the counting global memory allocator defined to capture peak heap during fit and encode?",
        "target_file": "src/bin/vqb/mem.rs",
        "target_symbols": ["CountingAlloc", "GlobalAlloc"],
        "expected_snippet_keywords": ["GlobalAlloc", "peak", "alloc"]
    },
    {
        "id": "task_05",
        "category": "Bug Localization",
        "prompt": "Locate the SIMD Hadamard transform / rotation stage implementation used across quantizers.",
        "target_file": "src/primitives/conditioners/hadamard.rs",
        "target_symbols": ["Hadamard", "apply"],
        "expected_snippet_keywords": ["hadamard", "rotate", "dim"]
    },

    # Category 2: Symbol & Interface Resolution
    {
        "id": "task_06",
        "category": "Symbol Resolution",
        "prompt": "Find the definition of the Quantizer trait and its byte_split size accounting method.",
        "target_file": "src/quantizer.rs",
        "target_symbols": ["Quantizer", "byte_split"],
        "expected_snippet_keywords": ["pub trait Quantizer", "byte_split", "Pipeline"]
    },
    {
        "id": "task_07",
        "category": "Symbol Resolution",
        "prompt": "Find the definition of the Primitive trait and its required lifecycle methods (apply, reconstruct, score).",
        "target_file": "src/primitive.rs",
        "target_symbols": ["Primitive", "apply", "reconstruct", "score"],
        "expected_snippet_keywords": ["pub trait Primitive", "apply", "reconstruct", "score"]
    },
    {
        "id": "task_08",
        "category": "Symbol Resolution",
        "prompt": "Where is the CodeLayout struct defined for packing and unpacking model bits and scalar fields?",
        "target_file": "src/util/coding.rs",
        "target_symbols": ["CodeLayout", "pack", "unpack"],
        "expected_snippet_keywords": ["struct CodeLayout", "pack_scalars", "unpack"]
    },
    {
        "id": "task_09",
        "category": "Symbol Resolution",
        "prompt": "Locate the Split adapter stage that fans out vectors across child pipeline branches.",
        "target_file": "src/splitter.rs",
        "target_symbols": ["Split", "Splitter", "from_factory"],
        "expected_snippet_keywords": ["pub struct Split", "Splitter", "branch"]
    },
    {
        "id": "task_10",
        "category": "Symbol Resolution",
        "prompt": "Where is the Pipeline struct implemented as a chain of Primitive stages?",
        "target_file": "src/pipeline.rs",
        "target_symbols": ["Pipeline", "stages"],
        "expected_snippet_keywords": ["pub struct Pipeline", "stages", "Primitive"]
    },

    # Category 3: Multi-File Architectural Navigation
    {
        "id": "task_11",
        "category": "Architecture Navigation",
        "prompt": "How does the driver in run.rs chunk dataset encoding across the Rayon thread pool?",
        "target_file": "src/bin/vqb/run.rs",
        "target_symbols": ["run_dataset", "encode", "rayon"],
        "expected_snippet_keywords": ["rayon", "chunks", "encode", "par_iter"]
    },
    {
        "id": "task_12",
        "category": "Architecture Navigation",
        "prompt": "Where are raw evaluation scores and reconstructions streamed to the .raw capture file?",
        "target_file": "src/bin/vqb/raw.rs",
        "target_symbols": ["RawCapture", "stream", "round_trips"],
        "expected_snippet_keywords": ["RawCapture", "write", "score"]
    },
    {
        "id": "task_13",
        "category": "Architecture Navigation",
        "prompt": "Find how config expansions generate resolved parameter combinations from JSON arrays in config.rs.",
        "target_file": "src/bin/vqb/config.rs",
        "target_symbols": ["expand", "Config", "ResolvedMethod"],
        "expected_snippet_keywords": ["expand", "cartesian", "sweep"]
    },
    {
        "id": "task_14",
        "category": "Architecture Navigation",
        "prompt": "Locate where Lloyd-Max k-means clustering is implemented in math numerical primitives.",
        "target_file": "src/math/kmeans.rs",
        "target_symbols": ["lloyd_kmeans", "centroids"],
        "expected_snippet_keywords": ["lloyd_kmeans", "centroids", "assign"]
    },
    {
        "id": "task_15",
        "category": "Architecture Navigation",
        "prompt": "Where is the Orthogonal Procrustes problem solved using singular value decomposition (SVD)?",
        "target_file": "src/math/procrustes.rs",
        "target_symbols": ["orthogonal_procrustes", "svd"],
        "expected_snippet_keywords": ["procrustes", "svd", "orthogonal"]
    },

    # Category 4: Quantizer Family Implementations
    {
        "id": "task_16",
        "category": "Quantizers",
        "prompt": "Find the EDEN (Extreme Density Embedding Network) quantizer pipeline construction.",
        "target_file": "src/quantizers/eden.rs",
        "target_symbols": ["Eden", "pipeline"],
        "expected_snippet_keywords": ["Eden", "pipeline", "CastNormal"]
    },
    {
        "id": "task_17",
        "category": "Quantizers",
        "prompt": "Where is the TurboQuant production pipeline chaining CastNormal with 1-bit QJL residual defined?",
        "target_file": "src/quantizers/turboquant_prod.rs",
        "target_symbols": ["TurboQuantProd", "Qjl"],
        "expected_snippet_keywords": ["TurboQuantProd", "Qjl", "residual"]
    },
    {
        "id": "task_18",
        "category": "Quantizers",
        "prompt": "Locate the Product Quantization (PQ) and Optimized Product Quantization (OPQ) implementations.",
        "target_file": "src/quantizers/opq.rs",
        "target_symbols": ["Opq", "Pq", "pipeline"],
        "expected_snippet_keywords": ["Opq", "pipeline", "Split"]
    },
    {
        "id": "task_19",
        "category": "Quantizers",
        "prompt": "Where is the E-RaBitQ (Enhanced RaBitQ) error-compensated quantized inner product defined?",
        "target_file": "src/quantizers/e_rabitq.rs",
        "target_symbols": ["ERaBitQ", "pipeline"],
        "expected_snippet_keywords": ["ERaBitQ", "pipeline", "CastNormal"]
    },
    {
        "id": "task_20",
        "category": "Quantizers",
        "prompt": "Where is the SimHash 1-bit hypercube projection quantizer implemented?",
        "target_file": "src/quantizers/simhash.rs",
        "target_symbols": ["SimHash", "pipeline"],
        "expected_snippet_keywords": ["SimHash", "pipeline", "Sign"]
    }
]
