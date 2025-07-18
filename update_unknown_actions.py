#!/usr/bin/env python3
"""
Script to update CSV file by replacing 'Unknown' entries with appropriate action labels.
Logic:
1. Find sequences of 'Unknown' entries
2. For each sequence, if followed by a valid action, use that action for the unknowns
3. For remaining unknowns at the end, use the last valid action found
"""

import pandas as pd
import sys
from pathlib import Path

def update_unknown_actions(input_file, output_file=None):
    """
    Update unknown actions in CSV file according to the specified logic.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input)
    """
    print(f"Reading file: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Unknown entries: {sum(df['predicted_action'] == 'Unknown')}")
    
    # Create a copy to work with
    updated_df = df.copy()
    
    # Get all unique valid actions (excluding 'Unknown')
    valid_actions = df[df['predicted_action'] != 'Unknown']['predicted_action'].unique()
    print(f"Valid actions found: {list(valid_actions)}")
    
    # Process the data
    last_valid_action = None
    i = 0
    
    while i < len(updated_df):
        current_action = updated_df.iloc[i]['predicted_action']
        
        if current_action != 'Unknown':
            # Found a valid action, store it
            last_valid_action = current_action
            i += 1
        else:
            # Found unknown, look for the end of unknown sequence
            unknown_start = i
            
            # Find the end of the unknown sequence
            while i < len(updated_df) and updated_df.iloc[i]['predicted_action'] == 'Unknown':
                i += 1
            
            unknown_end = i
            unknown_count = unknown_end - unknown_start
            
            # Determine what action to use for this unknown sequence
            if i < len(updated_df):
                # There's a valid action after the unknowns
                next_action = updated_df.iloc[i]['predicted_action']
                fill_action = next_action
                print(f"Filling {unknown_count} unknowns (frames {unknown_start}-{unknown_end-1}) with '{fill_action}'")
            else:
                # We're at the end of the file, use last valid action
                if last_valid_action is not None:
                    fill_action = last_valid_action
                    print(f"Filling {unknown_count} unknowns at end (frames {unknown_start}-{unknown_end-1}) with last valid action '{fill_action}'")
                else:
                    # No valid action found at all, use a default
                    fill_action = 'Unknown'  # Keep as unknown if no valid action found
                    print(f"Warning: No valid action found, keeping {unknown_count} unknowns as 'Unknown'")
            
            # Fill the unknown sequence
            updated_df.iloc[unknown_start:unknown_end, updated_df.columns.get_loc('predicted_action')] = fill_action
    
    # Set output file
    if output_file is None:
        output_file = input_file
    
    # Save the updated file
    updated_df.to_csv(output_file, index=False)
    
    print(f"Updated file saved to: {output_file}")
    print(f"Unknown entries after processing: {sum(updated_df['predicted_action'] == 'Unknown')}")
    
    # Show action distribution
    action_counts = updated_df['predicted_action'].value_counts()
    print("\nAction distribution after processing:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_unknown_actions.py <input_csv_file> [output_csv_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    try:
        update_unknown_actions(input_file, output_file)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
