#!/usr/bin/env python3
"""Dump Routine JSON schema to see if it's suitable for docstrings."""

import json
import sys

# Add project to path
sys.path.insert(0, '/Users/dimavremenko/Desktop/code/bluebox-sdk')

try:
    from bluebox.data_models.routine.routine import Routine

    schema = Routine.model_json_schema()

    print("="*80)
    print("Routine.model_json_schema() output:")
    print("="*80)
    print(json.dumps(schema, indent=2))
    print("\n")
    print("="*80)
    print(f"Total characters: {len(json.dumps(schema))}")
    print(f"Total lines: {len(json.dumps(schema, indent=2).split(chr(10)))}")
    print("="*80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
