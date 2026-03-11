# Quantitative evaluation: LLM vs Grid Search comparison

# Global random seed for reproducibility and fair benchmarking.
# Using the same seed ensures that identical simulation inputs
# (e.g., Grid Search testing count=3 and LLM testing count=3)
# yield the exact same cycle time. This eliminates random variance
# as a confounding factor in the comparison.
GLOBAL_SEED = 42
