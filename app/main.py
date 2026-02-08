import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import sys
import random

# --- CONFIGURATION ---
st.set_page_config(
    page_title="ROCm Bridge | AMD Enterprise Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INDUSTRY CSS OVERHAUL ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400&family=Inter:wght@400;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
    }

    /* Industry Header */
    .main-header {
        background: linear-gradient(90deg, #ED1C24 0%, #000000 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 8px solid #ED1C24;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Glass Panels */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
    }

    /* Real-time Console Log */
    .console-box {
        background-color: #000000;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #00FF41; /* Matrix Green */
        height: 200px;
        overflow-y: auto;
    }

    /* Enhanced Sidebar */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#161b22, #0d1117);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #ED1C24, #b91d24);
        color: white;
        border: none;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- DYNAMIC MOCK DATA GENERATORS ---
def get_hardware_specs(gpu_model):
    # Base specs are static, but we can randomize "Available Memory" or "Thermals"
    specs = {
        "AMD Instinct MI300X": {"HBM3": "192GB", "CUs": "304", "Peak Perf": "163.4 TFLOPS", "Arch": "CDNA 3"},
        "AMD Instinct MI250": {"HBM2e": "128GB", "CUs": "208", "Peak Perf": "95.7 TFLOPS", "Arch": "CDNA 2"},
        "Radeon RX 7900 XTX": {"GDDR6": "24GB", "CUs": "96", "Peak Perf": "61.3 TFLOPS", "Arch": "RDNA 3"}
    }
    return specs.get(gpu_model, specs["AMD Instinct MI300X"])

def generate_random_metrics():
    """Generates realistic-looking fluctuating metrics."""
    return {
        "wavefront": round(random.uniform(75.0, 99.9), 1),
        "wavefront_delta": round(random.uniform(-2.5, 5.0), 1),
        "memory": round(random.uniform(1.2, 5.3), 1),  # TB/s
        "memory_delta": round(random.uniform(-0.5, 0.8), 2),
        "latency": round(random.uniform(0.8, 12.5), 2), # ms
        "latency_delta": round(random.uniform(-1.5, 0.5), 2)
    }

def generate_console_logs(target_arch):
    """Generates a random sequence of log messages."""
    steps = [
        f"> Target detected: {target_arch}",
        "> Loading hipcc toolchain v6.0.2...",
        "> Analyzing AST for warp-divergence...",
        "> Injecting ROCTX markers...",
        f"> Optimizing LDS bank layout for {random.randint(32, 64)}-bank config...",
        "> Unrolling loops (Factor: " + str(random.choice([2, 4, 8])) + ")...",
        "> Re-compiling with -O3 -ffast-math...",
        "> Binary generation successful."
    ]
    # Randomly select a subset to make it look dynamic
    return "\n".join(random.sample(steps, k=5) + ["> PIPELINE COMPLETE."])

# --- CORE UI COMPONENTS ---

def draw_header():
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin:0; color:white;">ROCm Bridge Enterprise</h1>
        <p style="margin:0; color:#ccc; opacity:0.8;">Automated CUDA-to-HIP Transpilation & Performance Orchestrator</p>
    </div>
    """, unsafe_allow_html=True)

def render_telemetry_dashboard(gpu_model):
    """Renders the metrics with random values."""
    data = generate_random_metrics()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric(
            "Wavefront Efficiency", 
            f"{data['wavefront']}%", 
            f"{data['wavefront_delta']}%", 
            help="AMD Wavefront 64 utilization rate"
        )
    with col2:
        st.metric(
            "Memory Bandwidth", 
            f"{data['memory']} TB/s", 
            f"{data['memory_delta']} TB/s", 
            delta_color="normal"
        )
    with col3:
        st.metric(
            "Kernel Latency", 
            f"{data['latency']} ms", 
            f"{data['latency_delta']} ms", 
            delta_color="inverse"
        )

    st.markdown("### Compute Unit (CU) Load Distribution")
    
    # Randomize Chart Data
    cu_count = 96 if "7900" in gpu_model else (208 if "MI250" in gpu_model else 304)
    # Generate random load pattern (some high, some low to show imbalance)
    load_data = np.random.normal(loc=65, scale=20, size=cu_count)
    load_data = np.clip(load_data, 0, 100) # Clip between 0-100%
    
    df = pd.DataFrame({"CU ID": np.arange(cu_count), "Load %": load_data})
    
    fig = px.bar(
        df, 
        x="CU ID", 
        y="Load %", 
        color="Load %",
        color_continuous_scale="RdBu_r" # Red is high load
    )
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        height=300,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APPLICATION ---

def main():
    draw_header()

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/AMD_Logo.svg/1200px-AMD_Logo.svg.png", width=120)
        st.markdown("### System Configuration")
        target_arch = st.selectbox("Hardware Target", ["AMD Instinct MI300X", "AMD Instinct MI250", "Radeon RX 7900 XTX"])
        mode = st.radio("Optimization Strategy", ["Strict Portability", "Max Throughput", "Energy Efficient"])
        
        st.divider()
        st.markdown(f"**Runtime Status:** <span class='status-active'>ONLINE</span>", unsafe_allow_html=True)
        st.caption(f"Session ID: {random.randint(1000, 9999)}-ROCM")
        
        uploaded_file = st.file_uploader("Upload CUDA Source", type=["cu", "cpp"])

    # Split Main View
    main_col, side_col = st.columns([2, 1])

    with side_col:
        st.markdown("### Hardware Profile")
        specs = get_hardware_specs(target_arch)
        
        # Display Specs in a clean grid
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Arch**\n{specs['Arch']}")
            st.markdown(f"**VRAM**\n{specs.get('HBM3', specs.get('HBM2e', specs.get('GDDR6')))}")
        with c2:
            st.markdown(f"**CUs**\n{specs['CUs']}")
            st.markdown(f"**Perf**\n{specs.get('TFLOPS', specs.get('Peak Perf'))}")
        
        st.divider()
        st.markdown("### Live System Logs")
        
        # Container for dynamic logs
        log_placeholder = st.empty()
        # Default state
        log_placeholder.markdown(
            f'<div class="console-box">> System Ready.<br>> Awaiting input...</div>', 
            unsafe_allow_html=True
        )

    with main_col:
        if uploaded_file:
            if st.button("⚡ EXECUTE TRANSPILATION PIPELINE"):
                # 1. Randomize the Console Log content
                new_logs = generate_console_logs(target_arch)
                log_placeholder.markdown(
                    f'<div class="console-box">{new_logs.replace(chr(10), "<br>")}</div>', 
                    unsafe_allow_html=True
                )
                
                # 2. Simulate Processing Delay with Random Steps
                with st.status("Orchestrating Workflow...", expanded=True) as s:
                    st.write("Parsing Abstract Syntax Tree...")
                    time.sleep(random.uniform(0.3, 0.8))
                    
                    st.write(f"Detected {random.randint(2, 5)} CUDA-specific anti-patterns...")
                    time.sleep(random.uniform(0.3, 0.8))
                    
                    st.write(f"Applying {target_arch} specific optimizations...")
                    time.sleep(random.uniform(0.3, 0.8))
                    
                    s.update(label="Pipeline Execution Successful", state="complete", expanded=False)
                
                # 3. Render Dashboard with FRESH Random Numbers
                render_telemetry_dashboard(target_arch)
                
                # 4. Show Optimized Code Snippet
                st.subheader("Converted Source (Preview)")
                
                # Randomize the preview slightly to make it look alive
                opt_level = random.choice(["O2", "O3", "fast-math"])
                thread_block = random.choice([64, 128, 256])
                
                code_preview = f"""// [ROCm Bridge] Auto-Generated for {target_arch}
// Optimization Strategy: {mode} | Flags: -{opt_level}
#include <hip/hip_runtime.h>

#define WAVEFRONT_SIZE 64 // Optimized for CDNA

__global__ void optimized_kernel(float* A, float* B, float* C, int N) {{
    // Replaced blockDim.x check with Wavefront-agnostic logic
    int idx = hipBlockIdx_x * {thread_block} + hipThreadIdx_x;
    
    // Memory coalescing optimization applied
    if (idx < N) {{
        C[idx] = A[idx] * B[idx]; // FMA instruction fusion
    }}
}}
"""
                st.code(code_preview, language="cpp")
                
                st.success(f"Transpilation complete. Binary size: {random.randint(14, 45)} MB.")
        else:
            # Empty state
            st.info("👆 Please upload a CUDA (.cu) file in the sidebar to begin analysis.")
            
            # Show a "Ghost" chart to make the UI look populated
            st.markdown("### Historical Cluster Load (Last 24h)")
            ghost_data = pd.DataFrame({
                "Time": pd.date_range(start="2024-01-01", periods=24, freq="H"),
                "Cluster Load": np.random.randint(20, 80, 24)
            })
            st.area_chart(ghost_data.set_index("Time"), height=200)

if __name__ == "__main__":
    main()