# ROCm Bridge - Complete Project Explanation

## 1. What This Project Is
ROCm Bridge is a CUDA-to-ROCm portability assistant prototype.

Its purpose is to help identify:
- CUDA code patterns that are not portable to AMD GPUs.
- Performance anti-patterns that hurt AMD wavefront utilization.
- Practical optimization steps to move from CUDA assumptions to HIP/ROCm-friendly code.

It combines:
- A Streamlit dashboard UI.
- A static analyzer (AST + rules).
- A dynamic profiler layer (real ROCm mode or simulated mode).
- A recommendation engine that correlates static and runtime data.

## 2. High-Level Architecture
Intended pipeline:
1. Ingest CUDA source code.
2. Parse AST and detect static portability/performance issues.
3. Profile runtime behavior (or simulate metrics when ROCm is unavailable).
4. Correlate findings and generate optimization recommendations.
5. Present recommendations and transformed code preview in UI.

Core modules:
- `app/` -> UI and orchestration surface.
- `analyzer/` -> static analysis.
- `profiler/` -> dynamic analysis.
- `engine/` -> recommendation intelligence.

## 3. Repository Structure and Roles
- `app/main.py`
  - Streamlit entrypoint and dashboard.
  - Upload control, fake pipeline status, telemetry charts, and code preview.

- `analyzer/parser.py`
  - `CudaParser` using `libclang` for AST parsing.
  - CUDA parsing flags (`-x cuda`, `-D__CUDACC__`, etc.).
  - Kernel detection heuristics.
  - Runs rule engine and emits analysis report.

- `analyzer/rules.py`
  - Rule framework (`BaseRule`, `AnalysisIssue`, `RuleEngine`).
  - Built-in rules:
    - `ROCM_001` hardcoded warp-size assumption (`32`).
    - `ROCM_002` NVIDIA-specific intrinsics.
    - `ROCM_003` potential shared-memory bank conflicts.
    - `ROCM_004` risky `warpSize` dependency.

- `profiler/runner.py`
  - `ProfilingRunner` for compile/profile execution.
  - Uses `hipcc` and `rocprof` if available.
  - Falls back to simulation mode with randomized metrics.
  - Includes filename-based demo behavior (`opt`, `hip`, `fix` => "good" metrics).

- `profiler/collector.py`
  - `ProfileCollector` parses profiler CSV.
  - Extracts counters like:
    - `VALUUtilization`
    - `WavefrontOccupancy`
    - `LDSBankConflict`
    - `MemUnitStalled`
  - Classifies bottleneck and computes health score.

- `profiler/__init__.py`
  - Exposes convenience API `profile_kernel(...)`.
  - Orchestrates run + collect in one call.

- `engine/recommender.py`
  - `RecommendationEngine` with `Recommendation` dataclass.
  - Correlates static issues with telemetry data.
  - Prioritizes fixes and estimates impact.
  - Produces suggestion details and before/after snippets.

- `engine/templates/optimization_plan.md`
  - Placeholder template for future report rendering.

- `examples/`
  - `cuda_sample.cu`: intentionally problematic sample.
  - `cuda_sample_opt.cu`: improved sample.
  - `hip_sample.cpp`: placeholder output target.

- `include/hip/hip_runtime.h`
  - Mock HIP header for IntelliSense/development convenience.

- `scripts/setup_rocm.sh`
  - Exports ROCm-related environment variables.

- `Dockerfile`
  - Container setup using Python 3.12 slim.
  - Installs clang 17 and sets `LLVM_LIB_PATH`.
  - Runs Streamlit at port `10000`.

- `render.yaml`
  - Render deployment config for web service + keep-alive cron.

- `requirements.txt`
  - Python dependencies (`clang`, `streamlit`, `pandas`, `plotly`, `numpy`, etc.).

## 4. Detailed Runtime Behavior
### 4.1 Current UI Behavior (`app/main.py`)
The current Streamlit app is polished and interactive but mostly demo-driven:
- Metrics and logs are randomized.
- Pipeline progress is simulated.
- Converted code preview is generated text.
- Uploaded file currently does not trigger full backend analysis pipeline wiring.

So the app currently behaves as a demo front-end rather than a fully integrated analyzer/profiler/recommender workflow.

### 4.2 Static Analysis Behavior (`analyzer/`)
What is implemented:
- Real AST parsing setup via `clang.cindex`.
- Rule-based traversal and issue extraction.
- Portability score deduction based on issue severity.

What to note:
- Some checks are heuristic and simplified for prototype/demo use.
- Full production-grade AST rewriting is not implemented yet.

### 4.3 Dynamic Profiling Behavior (`profiler/`)
Two modes:
- Real mode: compile with `hipcc`, profile with `rocprof`.
- Simulation mode: generate realistic-looking CSV metrics for non-ROCm environments.

This design makes the project demo-friendly on machines without AMD hardware.

### 4.4 Recommendation Behavior (`engine/`)
- Correlates rule IDs (e.g., `ROCM_001`) with runtime signals.
- Raises confidence/priority when telemetry confirms issue impact.
- Outputs actionable recommendations and estimated gains.

## 5. End-to-End Data Flow (Intended)
1. User uploads `.cu` file in Streamlit.
2. `CudaParser.analyze(...)` creates AST report and issue list.
3. `profile_kernel(...)` runs profiling and returns telemetry.
4. `engine.generate(static_issues, metrics)` creates prioritized recommendations.
5. UI renders final report, bottlenecks, and optimization plan.

## 6. What Is Complete vs What Is Pending
### Implemented Well
- Clean modular backend structure.
- Rule engine and telemetry analysis primitives.
- Recommendation correlation logic.
- Containerized deployment path.
- Strong demo UX.

### Not Fully Wired Yet
- `app/main.py` is not fully connected to backend modules for true end-to-end execution.
- `engine/templates/optimization_plan.md` is minimal placeholder.
- Automatic source-to-source code transformation is currently illustrative, not a complete compiler rewrite stage.

## 7. Deployment Summary
### Local Run
```bash
git clone https://github.com/Team7SENSITIVE/rocm-bridge.git
cd rocm-bridge
pip install -r requirements.txt
streamlit run app/main.py
```

### Docker/Render
- Build/run defined in `Dockerfile`.
- Render service defined in `render.yaml`.
- Streamlit served on port `10000`.

## 8. Practical One-Line Summary
ROCm Bridge is a well-structured hackathon prototype: the backend modules for static analysis, profiling, and recommendation are present, while the Streamlit app currently focuses on a high-quality simulated demo and still needs full wiring for true end-to-end execution.
