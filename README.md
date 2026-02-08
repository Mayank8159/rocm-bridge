### 📄 `README.md`

```markdown
# 🚀 ROCm-Bridge: CUDA-to-ROCm Performance Optimizer

ROCm-Bridge is an automated tool designed to bridge the architectural gap between NVIDIA (CUDA) and AMD (ROCm) GPUs. It uses static analysis (Clang AST) and dynamic profiling (rocprof) to identify hardware-specific bottlenecks—like warp vs. wavefront sizing—and suggest optimized code fixes.

---

## 📂 Project Structure

Collaborators should follow this hierarchy when adding new modules or test cases:

```text
rocm-bridge/
├── analyzer/           # Static Analysis (LLVM/Clang)
│   ├── parser.py       # Scans AST for NVIDIA-specific intrinsics/constants
│   └── rules.py        # Logic for Wavefront (64) vs Warp (32) mapping
├── profiler/           # Dynamic Analysis (ROCm Tools)
│   ├── runner.py       # Compiles and executes code via hipcc/rocprof
│   └── collector.py    # Parses CSV/JSON telemetry from the GPU
├── engine/             # The "Brain"
│   ├── recommender.py  # Matches profile data to code transformation rules
│   └── templates/      # Optimized ROCm code snippets for replacement
├── app/                # User Interface
│   └── main.py         # Streamlit dashboard for project demos
├── examples/           # Test Suite
│   ├── cuda_sample.cu  # Input: Legacy NVIDIA-optimized code
│   └── hip_sample.cpp  # Output: Generated AMD-optimized code
├── scripts/            # Automation
│   └── setup_rocm.sh   # Environment configuration scripts
├── requirements.txt    # Python dependencies
└── README.md           # You are here

```

---

## 🛠️ Setup & Installation

### 1. Prerequisites (System Level)

This project requires the **AMD ROCm SDK** and **LLVM** installed on the host system (or WSL2).

* **ROCm:** 6.x or higher
* **Clang/LLVM:** Version 17.x recommended

```powershell
# Windows (via winget)
winget install LLVM.LLVM

```

### 2. Python Environment

We recommend using a virtual environment to manage dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/WSL
pip install -r requirements.txt

```

---

## 🚀 How to Contribute

### Adding a New Analysis Rule

1. Open `analyzer/rules.py`.
2. Define a new detection pattern (e.g., detecting `cudaSharedMemoryConfig`).
3. Add the corresponding ROCm optimization template in `engine/templates/`.

### Running the Dashboard

To view the UI and test the current "Bridge" logic:

```bash
streamlit run app/main.py

```

---

## 📊 Roadmap

* [ ] Implement AST-based `__shfl_sync` to `v_readlane` mapping.
* [ ] Integrate **Omniperf** for deeper memory bottleneck detection.
* [ ] Add automated "Side-by-Side" speedup comparison in the UI.

---

## 📜 License

This project is developed for the **AMD Slingshot Hackathon** under the MIT License.

