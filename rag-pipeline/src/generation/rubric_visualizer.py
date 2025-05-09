import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
from src.generation.rubric_types import Rubric, Criterion

class RubricVisualizer:
    def __init__(self, output_dir: str = "output/visualizations"):
        """
        Initialize the rubric visualizer.
        
        Args:
            output_dir: Directory where visualizations will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the style
        plt.style.use('default')  # Use default matplotlib style
        sns.set_theme()  # Use seaborn's default theme
        sns.set_palette("husl")  # Set color palette
        
    def _save_figure(self, filename: str) -> str:
        """
        Save the current figure to a file.
        
        Args:
            filename: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        return filepath
        
    def generate_score_chart(
        self,
        rubric: Rubric,
        scores: Dict[str, Decimal],
        title: Optional[str] = None
    ) -> str:
        """
        Generate a radar chart showing scores for each criterion.
        
        Args:
            rubric: The rubric being evaluated
            scores: Dictionary mapping criterion names to scores
            title: Optional title for the chart
            
        Returns:
            Path to the generated chart
        """
        # Prepare data
        criteria = [c.name for c in rubric.criteria]
        max_scores = [float(c.max_score) for c in rubric.criteria]
        actual_scores = [float(scores.get(c.name, 0)) for c in rubric.criteria]
        
        # Create radar chart
        angles = [n / float(len(criteria)) * 2 * math.pi for n in range(len(criteria))]
        angles += angles[:1]  # Close the loop
        
        max_scores += max_scores[:1]
        actual_scores += actual_scores[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot max scores
        ax.plot(angles, max_scores, 'o-', linewidth=2, label='Maximum Score')
        ax.fill(angles, max_scores, alpha=0.25)
        
        # Plot actual scores
        ax.plot(angles, actual_scores, 'o-', linewidth=2, label='Actual Score')
        ax.fill(angles, actual_scores, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        ax.set_ylim(0, max(max_scores) * 1.1)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        if title:
            plt.title(title)
        else:
            plt.title(f"Score Distribution - {rubric.title}")
            
        return self._save_figure(f"score_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
    def generate_progress_tracking(
        self,
        rubric: Rubric,
        historical_scores: List[Dict[str, Decimal]],
        dates: List[datetime],
        title: Optional[str] = None
    ) -> str:
        """
        Generate a line chart showing progress over time.
        
        Args:
            rubric: The rubric being evaluated
            historical_scores: List of score dictionaries over time
            dates: List of dates corresponding to the scores
            title: Optional title for the chart
            
        Returns:
            Path to the generated chart
        """
        # Prepare data
        df = pd.DataFrame(historical_scores)
        df['date'] = dates
        df.set_index('date', inplace=True)
        
        # Create line plot
        plt.figure(figsize=(12, 6))
        for criterion in rubric.criteria:
            plt.plot(df.index, df[criterion.name], marker='o', label=criterion.name)
            
        # Add maximum score lines
        for criterion in rubric.criteria:
            plt.axhline(y=float(criterion.max_score), color='gray', linestyle='--', alpha=0.5)
            
        # Customize plot
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.title(title or f"Progress Tracking - {rubric.title}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        return self._save_figure(f"progress_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
    def generate_performance_analytics(
        self,
        rubric: Rubric,
        all_scores: List[Dict[str, Decimal]],
        title: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate performance analytics including distribution plots and statistics.
        
        Args:
            rubric: The rubric being evaluated
            all_scores: List of score dictionaries from multiple evaluations
            title: Optional title for the chart
            
        Returns:
            Tuple of (path to generated chart, analytics dictionary)
        """
        # Prepare data
        df = pd.DataFrame(all_scores)
        
        # Calculate statistics
        stats = {}
        for criterion in rubric.criteria:
            criterion_stats = {
                'mean': float(df[criterion.name].mean()),
                'median': float(df[criterion.name].median()),
                'std': float(df[criterion.name].std()),
                'min': float(df[criterion.name].min()),
                'max': float(df[criterion.name].max()),
                'max_possible': float(criterion.max_score)
            }
            stats[criterion.name] = criterion_stats
            
        # Create distribution plots
        fig, axes = plt.subplots(len(rubric.criteria), 1, figsize=(12, 4 * len(rubric.criteria)))
        if len(rubric.criteria) == 1:
            axes = [axes]
            
        for ax, criterion in zip(axes, rubric.criteria):
            # Plot distribution
            sns.histplot(data=df, x=criterion.name, ax=ax, kde=True)
            
            # Add mean and median lines
            ax.axvline(stats[criterion.name]['mean'], color='red', linestyle='--', label='Mean')
            ax.axvline(stats[criterion.name]['median'], color='green', linestyle='--', label='Median')
            
            # Add max score line
            ax.axvline(float(criterion.max_score), color='gray', linestyle='--', label='Max Score')
            
            # Customize subplot
            ax.set_title(f"{criterion.name} Distribution")
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            
        plt.tight_layout()
        if title:
            plt.suptitle(title)
        else:
            plt.suptitle(f"Performance Analytics - {rubric.title}")
            
        return self._save_figure(f"performance_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), stats
        
    def generate_comparison_chart(
        self,
        rubric: Rubric,
        group_scores: Dict[str, List[Dict[str, Decimal]]],
        title: Optional[str] = None
    ) -> str:
        """
        Generate a box plot comparing scores across different groups.
        
        Args:
            rubric: The rubric being evaluated
            group_scores: Dictionary mapping group names to lists of score dictionaries
            title: Optional title for the chart
            
        Returns:
            Path to the generated chart
        """
        # Prepare data
        data = []
        for group, scores in group_scores.items():
            df = pd.DataFrame(scores)
            df['group'] = group
            data.append(df)
            
        combined_df = pd.concat(data)
        
        # Create box plots
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=combined_df.melt(id_vars=['group']), x='variable', y='value', hue='group')
        
        # Customize plot
        plt.xlabel('Criterion')
        plt.ylabel('Score')
        plt.title(title or f"Score Comparison - {rubric.title}")
        plt.xticks(rotation=45)
        plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        return self._save_figure(f"comparison_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
    def generate_summary_report(
        self,
        rubric: Rubric,
        scores: Dict[str, Decimal],
        historical_scores: Optional[List[Dict[str, Decimal]]] = None,
        group_scores: Optional[Dict[str, List[Dict[str, Decimal]]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report with all visualizations and analytics.
        
        Args:
            rubric: The rubric being evaluated
            scores: Current scores
            historical_scores: Optional list of historical scores
            group_scores: Optional dictionary of group scores
            
        Returns:
            Dictionary containing all visualizations and analytics
        """
        report = {
            'rubric_title': rubric.title,
            'total_score': float(rubric.calculate_score(scores)),
            'max_possible_score': float(rubric.total_points),
            'score_percentage': float(rubric.calculate_score(scores) / rubric.total_points * 100),
            'visualizations': {},
            'analytics': {}
        }
        
        # Generate score chart
        report['visualizations']['score_chart'] = self.generate_score_chart(rubric, scores)
        
        # Generate progress tracking if historical data is available
        if historical_scores:
            dates = [datetime.now() for _ in range(len(historical_scores))]  # Replace with actual dates if available
            report['visualizations']['progress_tracking'] = self.generate_progress_tracking(
                rubric, historical_scores, dates
            )
            
        # Generate performance analytics
        all_scores = historical_scores or [scores]
        report['visualizations']['performance_analytics'], report['analytics']['statistics'] = (
            self.generate_performance_analytics(rubric, all_scores)
        )
        
        # Generate comparison chart if group data is available
        if group_scores:
            report['visualizations']['comparison_chart'] = self.generate_comparison_chart(
                rubric, group_scores
            )
            
        # Add criterion-specific feedback
        report['feedback'] = rubric.generate_feedback(scores)
        
        return report 