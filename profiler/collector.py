import pandas as pd
import os

def parse_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # Extract key metrics
    summary = {
        "valu_utilization": df.get("VALUUtilization", [0])[0],
        "wavefront_occupancy": df.get("WavefrontOccupancy", [0])[0],
        "lds_bank_conflicts": df.get("LDSBankConflict", [0])[0]
    }
    return summary
