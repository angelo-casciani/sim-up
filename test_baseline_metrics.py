#!/usr/bin/env python3
"""Test that baseline metrics are correctly extracted from logs."""

import tempfile
import os
from src.simulator import SIMODDiscoverer

# Test with sepsis.xes
work_dir = tempfile.mkdtemp(prefix='simod_baseline_test_')
discoverer = SIMODDiscoverer('nokal/simod:latest', True)

print("=" * 80)
print("Testing baseline metrics extraction from SEPSIS.XES")
print("=" * 80)

model, metrics = discoverer.discover('logs/sepsis.xes', work_dir)

print("\n" + "=" * 80)
print("BASELINE METRICS RESULT:")
print("=" * 80)
print(f"Model keys: {model.keys()}")
print(f"Model: {model}")
print(f"\nMetrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Check if metrics are real (not hardcoded defaults)
if metrics['avg_cycle_time'] == 14400:
    print("\n❌ ERROR: Still using hardcoded default metrics!")
    print("  Expected real metrics from sepsis.xes")
    print(f"  Got: {metrics['avg_cycle_time']}")
else:
    print("\n✅ SUCCESS: Real metrics extracted from log!")
    print(f"  Cycle time: {metrics['avg_cycle_time']:.0f} seconds")

# Cleanup
import shutil
shutil.rmtree(work_dir)

