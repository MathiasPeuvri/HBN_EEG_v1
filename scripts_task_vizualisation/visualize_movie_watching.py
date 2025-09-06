#!/usr/bin/env python3
"""
Movie Watching Task Visualization

Creates timeline visualization showing all four movies with their durations
and synchronized EEG data display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from visualization_utils import (
    create_loader, load_task_data, add_eeg_subplot,
    DEFAULT_SUBJECT, FIGURE_DPI, MOVIE_COLORS
)

def create_movie_watching_plot(loader, subject_id):
    """Create movie watching visualization with 4 separate 2-panel plots for each movie."""
    # Movie names in the order they appear in the task description
    movies = ['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'ThePresent']
    movie_titles = ['Despicable Me', 'Diary of a Wimpy Kid', 'Fun with Fractals', 'The Present']
    
    print("Loading movie data...")
    
    # Create subplot structure: 4 rows, 2 columns per movie (movie info + EEG)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), height_ratios=[1, 1, 1, 1])
    fig.suptitle(f'Movie Watching Task - Subject: {subject_id}', fontsize=16, fontweight='bold')
    
    # Reshape axes for easier indexing: axes[movie_idx][0] = movie panel, axes[movie_idx][1] = EEG panel
    
    for i, movie in enumerate(movies):
        try:
            # Load individual movie data and EEG
            raw, events_df, sampling_rate, duration = load_task_data(loader, subject_id, movie)
            
            # Movie info panel (left, compact)
            ax_movie = axes[i, 0]
            
            # Draw movie bar
            color = MOVIE_COLORS[movie]
            ax_movie.barh(0.5, duration, height=0.6, color=color, alpha=0.7, 
                         edgecolor='black', linewidth=1)
            
            # Add movie title
            ax_movie.text(duration/2, 0.5, movie_titles[i], ha='center', va='center',
                         fontsize=12, fontweight='bold', color='black')
            
            # Add start and stop markers
            video_events = events_df[events_df['value'].isin(['video_start', 'video_stop'])]
            for _, event in video_events.iterrows():
                marker_color = 'green' if event['value'] == 'video_start' else 'red'
                ax_movie.axvline(event['onset'], color=marker_color, linestyle='-', 
                               alpha=0.9, linewidth=3)
                
                # Add label
                label = 'START' if event['value'] == 'video_start' else 'STOP'
                ax_movie.text(event['onset'], 0.9, label, ha='center', va='bottom',
                            fontsize=9, fontweight='bold', color=marker_color)
            
            # Format movie panel
            ax_movie.set_xlim(0, duration + 5)
            ax_movie.set_ylim(0, 1.2)
            ax_movie.set_title(f'{movie_titles[i]} ({duration:.1f}s)', 
                             fontsize=12, fontweight='bold')
            ax_movie.set_yticks([])
            ax_movie.set_xlabel('Time (seconds)', fontsize=10)
            
            # EEG panel (right, expanded)
            ax_eeg = axes[i, 1]
            
            # Add EEG subplot using the individual movie's EEG data
            add_eeg_subplot(ax_eeg, raw, events_df, duration)
            
            # Add video start/stop markers to EEG
            for _, event in video_events.iterrows():
                marker_color = 'green' if event['value'] == 'video_start' else 'red'
                ax_eeg.axvline(event['onset'], color=marker_color, linestyle='--', 
                             alpha=0.7, linewidth=2)
            
            print(f"✓ Processed {movie}: {duration:.1f}s")
            
        except Exception as e:
            print(f"Warning: Could not load {movie}: {e}")
            # Create empty plots for failed movies
            axes[i, 0].text(0.5, 0.5, f'{movie_titles[i]}\nNot available', 
                          ha='center', va='center', transform=axes[i, 0].transAxes,
                          fontsize=12, color='red')
            axes[i, 1].text(0.5, 0.5, 'EEG data\nnot available', 
                          ha='center', va='center', transform=axes[i, 1].transAxes,
                          fontsize=12, color='red')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
    
    # Add overall legend
    start_line = mlines.Line2D([], [], color='green', linewidth=3, label='Video start')
    stop_line = mlines.Line2D([], [], color='red', linewidth=3, label='Video stop')
    fig.legend(handles=[start_line, stop_line], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('movie_watching_timeline.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: movie_watching_timeline.png")

def main():
    """Main execution for standalone use."""
    print("=== Creating Movie Watching Visualization ===")
    
    loader = create_loader()
    
    try:
        create_movie_watching_plot(loader, DEFAULT_SUBJECT)
        print("\n✓ Movie watching visualization completed!")
        print("The visualization shows:")
        print("  - 4 separate 2-panel plots, one for each movie")
        print("  - Left panels: Movie timeline with start/stop markers")
        print("  - Right panels: Individual EEG recordings for each movie")
        print("  - Color-coded movie segments with actual durations")
        print("  - Each movie has its own synchronized EEG data")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()