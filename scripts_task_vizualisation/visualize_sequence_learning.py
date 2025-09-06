#!/usr/bin/env python3
"""
Sequence Learning Task Visualization

Creates timeline visualization showing learning blocks, dot sequences,
user responses vs correct answers, with synchronized EEG data display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from visualization_utils import (
    create_loader, load_task_data, add_eeg_subplot,
    DEFAULT_SUBJECT, FIGURE_DPI, SEQUENCE_COLORS
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Data display limits
DEFAULT_MAX_BLOCKS = None  # Number of blocks to display, None for all available
DEFAULT_MAX_TIME = None  # Maximum time to display in seconds, None for full recording

# Visual layout parameters
BLOCK_HEIGHT = 0.8  # Visual height of learning block bars
DOT_BAR_HEIGHT = 0.2  # Visual height of dot activation bars
DOT_BAR_OFFSET = 0.1  # Vertical offset for dot bars within blocks
RESPONSE_SYMBOL_OFFSET = 0.3  # Vertical offset for response symbols

def parse_sequence_string(seq_str):
    """Parse sequence string like '6-1-4-5-3-2-3' into list of integers."""
    if not seq_str or seq_str == 'n/a':
        return []
    return [int(x) for x in seq_str.split('-')]

def create_sequence_learning_plot(loader, subject_id, task_variant=None):
    """Create sequence learning visualization."""
    # Try to determine the correct task variant if not specified
    used_variant = None
    if task_variant is None:
        # Try 8target first, then 6target
        for variant in ['seqLearning8target', 'seqLearning6target']:
            try:
                raw, events_df, sampling_rate, actual_duration = load_task_data(loader, subject_id, variant)
                used_variant = variant
                print(f"Using task variant: {variant}")
                break
            except:
                continue
        else:
            raise ValueError(f"No sequence learning data found for subject {subject_id}")
    else:
        raw, events_df, sampling_rate, actual_duration = load_task_data(loader, subject_id, task_variant)
        used_variant = task_variant
    
    # Show complete data by default
    if DEFAULT_MAX_TIME is None:
        display_duration = actual_duration
    else:
        display_duration = min(actual_duration, DEFAULT_MAX_TIME)
    
    # Filter relevant events using actual data
    block_events = events_df[events_df['value'].str.contains('learningBlock_', na=False)]
    dot_events = events_df[events_df['value'].str.contains('dot_no', na=False)]
    
    # Determine number of blocks to display
    max_blocks = len(block_events) if DEFAULT_MAX_BLOCKS is None else min(len(block_events), DEFAULT_MAX_BLOCKS)
    y_positions = [max_blocks - 1 - i for i in range(max_blocks)]
    
    # Create subplot structure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])
    
    # Include task variant in title
    target_count = '8' if '8target' in used_variant else '6'
    fig.suptitle(f'Sequence Learning Task ({target_count} targets) - Subject: {subject_id}', 
                 fontsize=16, fontweight='bold')
    
    # Process learning blocks
    for i, (_, block_event) in enumerate(block_events.iterrows()):
        if i >= max_blocks:
            break
            
        block_start = block_event['onset']
        if block_start > display_duration:
            break
            
        block_num = int(block_event['value'].split('_')[1])
        y_pos = y_positions[i]
        
        # Calculate block end from actual events
        if i + 1 < len(block_events):
            block_end = block_events.iloc[i + 1]['onset']
        else:
            # For last block, find last event and add small buffer
            last_event_time = events_df[events_df['onset'] >= block_start]['onset'].max()
            block_end = last_event_time + 10  # Small buffer for visualization
        
        block_end = min(block_end, display_duration)
        block_duration = block_end - block_start
        
        # Draw block background using transparency, not color
        ax1.barh(y_pos, block_duration, left=block_start, height=BLOCK_HEIGHT,
                color='lightgray', alpha=0.3, edgecolor='black', linewidth=1)
        
        ax1.text(block_start + block_duration/2, y_pos, f'Block {block_num}', 
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Process dots within this block
        block_dots = dot_events[(dot_events['onset'] >= block_start) & 
                               (dot_events['onset'] < block_end)]
        
        # Group dot ON/OFF pairs and draw them
        for j in range(0, len(block_dots), 2):
            if j + 1 < len(block_dots):
                on_event = block_dots.iloc[j]
                off_event = block_dots.iloc[j + 1]
                
                if 'ON' in on_event['value'] and 'OFF' in off_event['value']:
                    dot_start = on_event['onset']
                    dot_duration = off_event['onset'] - dot_start
                    
                    # Extract and display dot number
                    dot_match = re.search(r'dot_no(\d+)_ON', on_event['value'])
                    if dot_match:
                        dot_num = dot_match.group(1)
                        
                        # Draw dot activation
                        ax1.barh(y_pos + DOT_BAR_OFFSET, dot_duration, left=dot_start, height=DOT_BAR_HEIGHT,
                                color='darkblue', alpha=0.8, edgecolor='black', linewidth=0.5)
                        
                        # Always show dot numbers (dots are ~0.6s, always long enough)
                        ax1.text(dot_start + dot_duration/2, y_pos + DOT_BAR_OFFSET + DOT_BAR_HEIGHT/2, dot_num, 
                                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Find user response - it's on the last dot_OFF event with user_answer data
        response_rows = block_dots[block_dots['user_answer'].notna() & (block_dots['user_answer'] != 'n/a')]
        if not response_rows.empty:
            response_row = response_rows.iloc[-1]
            user_seq = parse_sequence_string(response_row['user_answer'])
            correct_seq = parse_sequence_string(response_row['correct_answer'])
            
            if user_seq and correct_seq:
                is_correct = user_seq == correct_seq
                response_color = SEQUENCE_COLORS['correct'] if is_correct else SEQUENCE_COLORS['incorrect']
                response_symbol = '✓' if is_correct else '✗'
                
                # Position response symbol near block end
                ax1.text(block_end - 5, y_pos + RESPONSE_SYMBOL_OFFSET, response_symbol, 
                        ha='center', va='center', fontsize=14, fontweight='bold', color=response_color)
                
                # Show sequence if reasonable length
                if len(user_seq) <= 8:  # Reasonable display limit
                    user_str = '-'.join(map(str, user_seq))
                    ax1.text(block_end - 15, y_pos - 0.3, f'User: {user_str}', 
                            ha='left', va='center', fontsize=7)
    
    # Use same time axis for both plots
    ax1.set_xlim(0, display_duration)
    ax1.set_ylim(-0.5, max_blocks + 0.5)
    ax1.set_title('Sequence Learning Blocks with Dot Presentations and Responses', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Learning Block', fontsize=12)
    ax1.set_yticks(y_positions[:max_blocks])
    ax1.set_yticklabels([f'Block {i+1}' for i in range(max_blocks)])
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    
    # Clear legend with distinct encodings
    legend_elements = [
        mpatches.Patch(color='lightgray', alpha=0.3, label='Learning Block'),
        mpatches.Patch(color='darkblue', alpha=0.8, label='Dot Presentation'),
        mpatches.Patch(color=SEQUENCE_COLORS['correct'], label='Correct Response'),
        mpatches.Patch(color=SEQUENCE_COLORS['incorrect'], label='Incorrect Response')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Add EEG subplot using same time range
    add_eeg_subplot(ax2, raw, events_df, display_duration)
    
    # Add block markers to EEG plot for synchronization
    for i, (_, block_event) in enumerate(block_events.iterrows()):
        if i >= max_blocks:
            break
        block_time = block_event['onset']
        if block_time <= display_duration:
            ax2.axvline(block_time, color='gray', linestyle='-', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('sequence_learning_timeline.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: sequence_learning_timeline.png")

def main():
    """Main execution for standalone use."""
    print("=== Creating Sequence Learning Visualization ===")
    
    loader = create_loader()
    
    try:
        create_sequence_learning_plot(loader, DEFAULT_SUBJECT)
        print("\n✓ Sequence learning visualization completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()