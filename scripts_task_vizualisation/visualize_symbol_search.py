#!/usr/bin/env python3
"""
Symbol Search Task Visualization

Creates visualization showing trial responses across pages with performance
feedback and response timing, synchronized with EEG data display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from visualization_utils import (
    create_loader, load_task_data, add_eeg_subplot,
    DEFAULT_SUBJECT, FIGURE_DPI, SYMBOL_COLORS
)

def create_symbol_search_plot(loader, subject_id):
    """Create Symbol Search task visualization."""
    # Load data using actual events and EEG
    raw, events_df, sampling_rate, actual_duration = load_task_data(loader, subject_id, 'symbolSearch')
    
    # Filter relevant events
    trial_events = events_df[events_df['value'] == 'trialResponse']
    page_events = events_df[events_df['value'] == 'newPage']
    start_event = events_df[events_df['value'] == 'symbolSearch_start']
    
    if trial_events.empty:
        raise ValueError("No trial response events found in data")
    
    # Create subplot structure 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])
    fig.suptitle(f'Symbol Search Task - Subject: {subject_id}', 
                 fontsize=16, fontweight='bold')
    
    # Organize trials by pages
    pages = []
    current_page_trials = []
    page_starts = page_events['onset'].tolist() if not page_events.empty else [0]
    
    # Group trials into pages based on newPage events
    for _, trial in trial_events.iterrows():
        trial_time = trial['onset']
        
        # Check if we've moved to a new page
        new_page_started = False
        for page_start in page_starts:
            if page_start > trial_time and current_page_trials:
                pages.append(current_page_trials)
                current_page_trials = []
                new_page_started = True
                break
        
        current_page_trials.append(trial)
    
    # Add the last page
    if current_page_trials:
        pages.append(current_page_trials)
    
    # Plot pages and trials
    y_offset = 0
    page_height = 0.8
    trial_height = 0.05
    page_spacing = 0.5
    
    for page_idx, page_trials in enumerate(pages):
        if not page_trials:
            continue
            
        page_start_time = page_trials[0]['onset']
        page_end_time = page_trials[-1]['onset']
        page_duration = page_end_time - page_start_time
        
        # Draw page background
        ax1.barh(y_offset, page_duration, left=page_start_time, height=page_height,
                color=SYMBOL_COLORS['trial_bar'], alpha=0.3, edgecolor='black', linewidth=1)
        
        # Add page label
        ax1.text(page_start_time + page_duration/2, y_offset + page_height/2, 
                f'Page {page_idx + 1}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Draw individual trials as colored dots within the page
        trial_y_base = y_offset + page_height * 0.2
        trial_spacing = page_height * 0.6 / max(1, len(page_trials) - 1) if len(page_trials) > 1 else 0
        
        for trial_idx, trial in enumerate(page_trials):
            trial_time = trial['onset']
            user_answer = trial['user_answer']
            correct_answer = trial['correct_answer']
            
            # Determine if response was correct
            is_correct = (user_answer == correct_answer)
            trial_color = SYMBOL_COLORS['correct'] if is_correct else SYMBOL_COLORS['incorrect']
            
            # Position trial dot within page
            trial_y = trial_y_base + trial_idx * trial_spacing
            
            # Draw trial dot
            ax1.scatter(trial_time, trial_y, s=50, color=trial_color, 
                       alpha=0.8, edgecolor='black', linewidth=0.5, zorder=5)
            
            # Add response indicators for better visibility
            response_symbol = '✓' if is_correct else '✗'
            ax1.text(trial_time, trial_y, response_symbol, 
                    ha='center', va='center', fontsize=6, fontweight='bold', color='white')
        
        # Calculate performance for this page
        correct_count = sum(1 for trial in page_trials if trial['user_answer'] == trial['correct_answer'])
        total_trials = len(page_trials)
        accuracy = correct_count / total_trials * 100
        
        # Show page accuracy - position better to avoid overlap
        ax1.text(page_start_time - 5, y_offset + page_height/2, 
                f'{accuracy:.0f}%', ha='right', va='center', 
                fontsize=10, fontweight='bold')
        
        y_offset += page_height + page_spacing
    
    # Format task plot
    max_time = actual_duration
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(-0.2, y_offset)
    ax1.set_title('Symbol Search Pages and Trial Responses', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pages', fontsize=12)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=SYMBOL_COLORS['trial_bar'], alpha=0.3, label='Page Background'),
        mpatches.Patch(color=SYMBOL_COLORS['correct'], label='Correct Response'),
        mpatches.Patch(color=SYMBOL_COLORS['incorrect'], label='Incorrect Response')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Add task start marker
    if not start_event.empty:
        task_start_time = start_event.iloc[0]['onset']
        if task_start_time <= max_time:
            ax1.axvline(task_start_time, color='green', linestyle='-', alpha=0.8, linewidth=3)
            ax1.text(task_start_time, y_offset - 0.3, 'Task Start', 
                    ha='center', va='top', fontsize=10, fontweight='bold', color='green')
    
    # Add page boundary markers
    for page_start in page_starts[1:]:  # Skip first page start
        if page_start <= max_time:
            ax1.axvline(page_start, color=SYMBOL_COLORS['page_boundary'], 
                       linestyle='--', alpha=0.7, linewidth=2)
    
    # Add EEG subplot
    add_eeg_subplot(ax2, raw, events_df, max_time)
    
    # Add page markers to EEG plot for synchronization
    for page_start in page_starts:
        if page_start <= max_time:
            ax2.axvline(page_start, color=SYMBOL_COLORS['page_boundary'], 
                       linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('symbol_search_timeline.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: symbol_search_timeline.png")

def main():
    """Main execution for standalone use."""
    print("=== Creating Symbol Search Visualization ===")
    
    loader = create_loader()
    
    try:
        create_symbol_search_plot(loader, DEFAULT_SUBJECT)
        print("\n✓ Symbol Search visualization completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()