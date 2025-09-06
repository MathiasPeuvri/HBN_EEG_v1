#!/usr/bin/env python3
"""
Surround Suppression Task Visualization

Creates timeline visualization showing stimulus presentations with
different conditions and synchronized EEG data display.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from visualization_utils import (
    create_loader, load_task_data, add_eeg_subplot,
    DEFAULT_SUBJECT, DEFAULT_RUN,
    FIGURE_DPI
)

# Task-specific configuration
MAX_STIMULI_TO_DISPLAY = 15  # Limit display for better readability

# Task-specific condition colors
CONDITION_COLORS = {
    1.0: '#4A90E2',  # Bright blue
    2.0: '#F5A623',  # Bright orange  
    3.0: '#7ED321'   # Bright green
}

def create_surround_suppression_plot(loader, subject_id, run_number):
    """Create surround suppression visualization with data-driven approach."""
    # Load data using simplified function
    print(f"Loading data for {subject_id}, run {run_number}...")
    raw, events_df, sampling_rate, actual_duration = load_task_data(loader, subject_id, 'surroundSupp', run=run_number)
    print(f"Data loaded: {actual_duration:.1f}s duration")
    
    # Filter stimulus events
    stim_events = events_df[events_df['value'] == 'stim_ON']
    total_stimuli = len(stim_events)
    
    # Limit display for readability
    if len(stim_events) > MAX_STIMULI_TO_DISPLAY:
        stim_events = stim_events.head(MAX_STIMULI_TO_DISPLAY)
        showing_subset = True
    else:
        showing_subset = False
    
    # Create subplot structure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), height_ratios=[2, 1])
    fig.suptitle(f'Surround Suppression Task - Subject: {subject_id}, Run: {run_number}', 
                 fontsize=16, fontweight='bold')
    
    # Get unique stimulus conditions from data
    unique_conditions = sorted(stim_events['stimulus_cond'].unique())
    
    # Draw stimulus events with two-row layout
    for _, stim in stim_events.iterrows():
        onset = stim['onset']  # Use actual time from events
        duration = float(stim['duration'])
        
        # Get stimulus parameters
        bg = float(stim['background'])
        fg = float(stim['foreground_contrast'])
        cond = float(stim['stimulus_cond'])
        
        # Select color based on condition
        color = CONDITION_COLORS.get(cond, '#808080')
        
        # Row 1: Background presence indicator (thin bar at top)
        if bg == 1.0:
            ax1.barh(0.8, duration, left=onset, height=0.15,
                    color='#333333', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax1.text(onset + duration/2, 0.8, 'BG', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        
        # Row 2: Main stimulus bar with condition color
        ax1.barh(0.4, duration, left=onset, height=0.35,
                color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add foreground contrast value with better positioning
        if fg > 0:
            ax1.text(onset + duration/2, 0.4, f'{fg:.1f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    
    # Limit time window to show just the displayed stimuli
    if len(stim_events) > 0:
        last_stim_end = stim_events.iloc[-1]['onset'] + float(stim_events.iloc[-1]['duration']) + 2
        display_duration = min(last_stim_end, actual_duration)
    else:
        display_duration = actual_duration
    
    # Format task plot using limited duration
    ax1.set_xlim(0, display_duration)
    ax1.set_ylim(0, 1.1)
    
    # Update title based on whether showing subset
    if showing_subset:
        title = f'Surround Suppression Timeline (Showing first {len(stim_events)} of {total_stimuli} stimuli)'
    else:
        title = f'Surround Suppression Timeline ({len(stim_events)} stimuli)'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    ax1.set_yticks([])
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    
    # Add summary info box
    info_text = f'Total: {total_stimuli} stimuli\n'
    if showing_subset:
        info_text += f'Showing: 1-{MAX_STIMULI_TO_DISPLAY}'
    else:
        info_text += f'Showing: All'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Create clean legend
    legend_elements = []
    for cond in unique_conditions:
        color = CONDITION_COLORS.get(cond, '#808080')
        legend_elements.append(mpatches.Patch(color=color, label=f'Condition {int(cond)}'))
    
    legend_elements.extend([
        mpatches.Patch(color='#333333', label='BG = Background present'),
        mpatches.Patch(color='white', label='Number = Foreground contrast')
    ])
    
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    
    # Add EEG subplot using shared function (with limited time range)
    print(f"Adding EEG subplot (0-{display_duration:.1f}s)...")
    add_eeg_subplot(ax2, raw, events_df, display_duration)
    print("EEG subplot added")
    
    # Add stimulus markers to EEG plot
    for _, stim in stim_events.iterrows():
        onset = stim['onset']
        duration = float(stim['duration'])
        cond = float(stim['stimulus_cond'])
        color = CONDITION_COLORS.get(cond, '#808080')
        
        # Add vertical line at stimulus onset
        ax2.axvline(onset, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        # Add duration shading
        ax2.axvspan(onset, onset + duration, color=color, alpha=0.1, zorder=0)
    
    # Format EEG subplot (match display window)
    ax2.set_xlim(0, display_duration)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('surround_suppression_sequence.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: surround_suppression_sequence.png")

def main():
    """Main execution for standalone use."""
    print("=== Creating Surround Suppression Visualization ===")
    
    loader = create_loader()
    
    try:
        create_surround_suppression_plot(loader, DEFAULT_SUBJECT, DEFAULT_RUN)
        print("\n✓ Surround suppression visualization completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()