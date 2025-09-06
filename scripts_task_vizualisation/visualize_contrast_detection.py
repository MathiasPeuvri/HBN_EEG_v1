#!/usr/bin/env python3
"""
Contrast Change Detection Task Visualization

Creates trial sequence visualization showing target presentations,
responses, and feedback with synchronized EEG data display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from visualization_utils import (
    create_loader, load_task_data, add_eeg_subplot,
    DEFAULT_SUBJECT, DEFAULT_RUN, DEFAULT_TRIAL_COUNT,
    FIGURE_DPI, CONTRAST_COLORS
)

def create_contrast_detection_plot(loader, subject_id, run_number):
    """Create contrast change detection visualization."""
    # Load data using simplified function
    raw, events_df, sampling_rate, actual_duration = load_task_data(loader, subject_id, 'contrastChangeDetection', run=run_number)
    
    # Filter relevant events
    trial_events = events_df[events_df['value'].isin(['left_target', 'right_target', 'left_buttonPress', 'right_buttonPress'])]
    
    # Create subplot structure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    fig.suptitle(f'Contrast Change Detection Task - Subject: {subject_id}, Run: {run_number}', 
                 fontsize=16, fontweight='bold')
    
    # Process trials (simplified logic)
    trial_count = 0
    x_pos = 0
    trial_spacing = 2.9  # Space between trials
    
    i = 0
    max_trials = DEFAULT_TRIAL_COUNT if DEFAULT_TRIAL_COUNT is not None else len(trial_events)
    while i < len(trial_events) and trial_count < max_trials:
        event = trial_events.iloc[i]
        
        if event['value'] in ['left_target', 'right_target']:
            target_side = 'left' if event['value'] == 'left_target' else 'right'
            
            # Draw target bar
            ax1.bar(x_pos, 1, width=2.4, color=CONTRAST_COLORS[target_side], 
                   alpha=0.7, edgecolor='black', linewidth=1)
            
            # Look for response and add feedback
            if i + 1 < len(trial_events):
                next_event = trial_events.iloc[i + 1]
                if next_event['value'] in ['left_buttonPress', 'right_buttonPress']:
                    response_side = 'left' if next_event['value'] == 'left_buttonPress' else 'right'
                    
                    # Add response indicator
                    ax1.text(x_pos + 1.2, 1.2, '✋', ha='center', va='center',
                            fontsize=16, color=CONTRAST_COLORS[response_side])
                    
                    # Add feedback
                    correct = (target_side == response_side)
                    feedback_symbol = '😊' if correct else '😞'
                    feedback_color = '#32CD32' if correct else '#DC143C'
                    ax1.text(x_pos + 1.2, 1.4, feedback_symbol, ha='center', va='center',
                            fontsize=14, color=feedback_color)
                    
                    i += 1  # Skip the response event
            
            # Add trial label
            ax1.text(x_pos + 1.2, -0.15, f'T{trial_count+1}', ha='center', va='center',
                    fontsize=9, fontweight='bold')
            
            x_pos += trial_spacing
            trial_count += 1
        
        i += 1
    
    # Format task plot
    ax1.set_xlim(-0.5, x_pos)
    ax1.set_ylim(-0.3, 1.8)
    ax1.set_title('Exemplar Contrast Change Detection (CCD) trial sequence', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Trial Events', fontsize=12)
    ax1.set_xticklabels([])
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=CONTRAST_COLORS['right'], label='Right contrast increase'),
        mpatches.Patch(color=CONTRAST_COLORS['left'], label='Left contrast increase'),
        mpatches.Patch(color='#32CD32', label='Positive feedback'),
        mpatches.Patch(color='#DC143C', label='Negative feedback')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Use actual EEG duration for time range
    max_time = min(actual_duration, x_pos)  # Don't exceed EEG duration
    
    # Add EEG subplot using shared function
    add_eeg_subplot(ax2, raw, events_df, max_time)
    
    # Add trial markers to EEG plot
    for i in range(trial_count):
        trial_time = i * trial_spacing
        if trial_time <= max_time:
            ax2.axvline(trial_time, color='gray', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('contrast_change_detection_sequence.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: contrast_change_detection_sequence.png")

def main():
    """Main execution for standalone use."""
    print("=== Creating Contrast Change Detection Visualization ===")
    
    loader = create_loader()
    
    try:
        create_contrast_detection_plot(loader, DEFAULT_SUBJECT, DEFAULT_RUN)
        print("\n✓ Contrast change detection visualization completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()