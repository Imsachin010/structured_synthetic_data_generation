# Nested Schema Execution Results

## Overview

- **Total Queries Executed:** 30
- **Successful Outputs:** 22 (73.3%)
- **Failed Outputs:** 8 (26.7%)

## Repair Breakdown

This table breaks down exactly where the LLM stumbled and required a feedback repair loop.

| Repair Layer | Total Repair Attempts | Total Traced Unique Errors |
|--------------|-----------------------|----------------------------|
| **JSON Parse Malformations** | 18 | 18 |
| **Deep Schema Validations** | 7 | 7 |

## Visualizations

### Validation Outcomes
![Success Rate](c:/Users/Imsac/Desktop/Project_X/X3/vlm_dataset_generation/experiments/analysis/success_rate.png)

### Repair Frequencies
![Repair Attempts](c:/Users/Imsac/Desktop/Project_X/X3/vlm_dataset_generation/experiments/analysis/repair_counts.png)
