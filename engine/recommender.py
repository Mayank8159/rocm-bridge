def generate_fix(static_issues, dynamic_metrics=None):
    report = []
    
    for issue in static_issues:
        rec = {
            "line": issue['line'],
            "issue": issue['rule']['message'],
            "suggestion": "Manual Review Required"
        }
        
        # Logic: If we have dynamic data, be smarter
        if dynamic_metrics:
            if issue['rule']['id'] == "AMD_001" and dynamic_metrics['valu_utilization'] < 40:
                rec['suggestion'] = "CRITICAL: Change blockDim.x to 64 to fix 50% VALU idleness."
                rec['auto_fix'] = True
                
        report.append(rec)
        
    return report
