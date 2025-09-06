#!/usr/bin/env python3
"""
Resting State Task Visualization

Creates timeline visualization showing eye-open/closed states
with synchronized EEG data display.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from visualization_utils import (
    create_loader, load_task_data, add_eeg_subplot,
    DEFAULT_SUBJECT, FIGURE_DPI, MIN_SEGMENT_DURATION, 
    MIN_LABEL_DURATION, RESTING_STATE_COLORS, INSTRUCTION_COLORS
)

def create_resting_state_plot(loader, subject_id):
    """Create resting state visualization."""
    # Load data using simplified function
    raw, events_df, sampling_rate, actual_duration = load_task_data(loader, subject_id, 'RestingState')
    
    # Filter eye events
    eye_events = events_df[events_df['value'].isin(['instructed_toOpenEyes', 'instructed_toCloseEyes'])]
    
    # Determine appropriate time range (use actual EEG duration, not arbitrary extensions)
    max_time = actual_duration
    
    # Create subplot structure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    fig.suptitle(f'Resting State Task - Subject: {subject_id}', fontsize=16, fontweight='bold')
    
    # Create timeline segments
    current_state = 'open'
    start_time = 0
    
    for _, event in eye_events.iterrows():
        if event['onset'] > max_time:
            break
            
        onset = event['onset']
        duration = onset - start_time
        
        # Draw state segment
        if duration > MIN_SEGMENT_DURATION:  # Only meaningful segments
            color = RESTING_STATE_COLORS[current_state]
            label = 'Eyes open' if current_state == 'open' else 'Eyes closed'
            ax1.barh(0.5, duration, left=start_time, height=0.6, 
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            if duration > MIN_LABEL_DURATION:  # Add text for longer segments
                ax1.text(start_time + duration/2, 0.5, label, 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add instruction marker
        color = INSTRUCTION_COLORS['open'] if event['value'] == 'instructed_toOpenEyes' else INSTRUCTION_COLORS['close']
        ax1.axvline(onset, color=color, linestyle='--', alpha=0.8, linewidth=2)
        
        # Update state
        current_state = 'closed' if event['value'] == 'instructed_toCloseEyes' else 'open'
        start_time = onset
    
    # Final segment (if within EEG duration)
    if start_time < max_time:
        final_duration = max_time - start_time
        ax1.barh(0.5, final_duration, left=start_time, height=0.6,
               color=RESTING_STATE_COLORS[current_state], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Format task plot
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(0, 1.3)
    ax1.set_title('Timeline of an exemplar resting-state sequence', fontsize=14, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    
    # Add legend
    red_line = mlines.Line2D([], [], color=INSTRUCTION_COLORS['open'], linestyle='--', 
                           linewidth=2, label='Instructed to open eyes')
    blue_line = mlines.Line2D([], [], color=INSTRUCTION_COLORS['close'], linestyle='--', 
                            linewidth=2, label='Instructed to close eyes')
    ax1.legend(handles=[red_line, blue_line], loc='upper right')
    
    # Add EEG subplot using shared function
    # Prepare event markers for EEG plot
    eye_markers = eye_events.copy()
    eye_markers['type'] = eye_markers['value'].map({
        'instructed_toOpenEyes': 'open',
        'instructed_toCloseEyes': 'close'
    })
    
    add_eeg_subplot(ax2, raw, events_df, max_time, eye_markers)
    
    plt.tight_layout()
    plt.savefig('resting_state_timeline.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: resting_state_timeline.png")

def main():
    """Main execution for standalone use."""
    print("=== Creating Resting State Visualization ===")
    
    loader = create_loader()
    
    try:
        create_resting_state_plot(loader, DEFAULT_SUBJECT)
        print("\n✓ Resting state visualization completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()