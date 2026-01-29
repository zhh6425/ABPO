#!/usr/bin/env python3
"""
CSV数据相关性分析与可视化脚本
计算两个CSV文件（如TensorBoard训练曲线）的多种相关性指标并生成可视化报告

作者: Correlation Analyzer
版本: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import json


class CorrelationAnalyzer:
    """CSV数据相关性分析器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化分析器
        
        参数:
        config: 配置字典
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        self._setup_matplotlib_style()
        
        # 配色方案
        self.colors = {
            'curve1': '#5B9BD5',
            'curve2': '#C55A11',
            'scatter': '#636363',
            'residual': '#70AD47'
        }
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'figure_size': (12, 12),
            'dpi': 300,
            'font_size': 20,
            'title_font_size': 22,
            'interpolate': True,  # Whether to interpolate and align sequences
            'smooth_window': 0,   # Smoothing window size (0 means no smoothing)
            'dtw_window': None    # DTW window constraint (None means no constraint)
        }
    
    def _setup_matplotlib_style(self):
        """设置matplotlib样式"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': self.config['font_size'],
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'axes.linewidth': 1.2,
            'figure.dpi': self.config['dpi'],
            'savefig.dpi': self.config['dpi'],
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white'
        })
    
    def load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess CSV file
        
        Parameters:
        csv_path: CSV file path
        
        Returns:
        Cleaned DataFrame
        """
        df = pd.read_csv(csv_path)
        
        # Remove Wall time column
        if 'Wall time' in df.columns:
            df = df.drop('Wall time', axis=1)
        
        # Check required columns
        if 'Step' not in df.columns or 'Value' not in df.columns:
            raise ValueError(f"CSV file must contain 'Step' and 'Value' columns")
        
        # Data type conversion
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['Step', 'Value'])
        
        # Sort by Step
        df = df.sort_values('Step').reset_index(drop=True)
        
        return df
    
    def align_sequences(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align two sequences (interpolate or truncate)
        
        Parameters:
        df1, df2: Two DataFrames
        
        Returns:
        (steps, values1, values2) - Aligned arrays
        """
        if self.config['interpolate']:
            # Find common Step range
            min_step = max(df1['Step'].min(), df2['Step'].min())
            max_step = min(df1['Step'].max(), df2['Step'].max())
            
            # Create common Step grid
            num_points = min(len(df1), len(df2))
            common_steps = np.linspace(min_step, max_step, num_points)
            
            # Interpolate
            values1 = np.interp(common_steps, df1['Step'].values, df1['Value'].values)
            values2 = np.interp(common_steps, df2['Step'].values, df2['Value'].values)
            
            return common_steps, values1, values2
        else:
            # Simple truncation to shorter length
            min_len = min(len(df1), len(df2))
            return (df1['Step'].values[:min_len], 
                   df1['Value'].values[:min_len], 
                   df2['Value'].values[:min_len])
    
    def smooth_curve(self, values: np.ndarray, window: int) -> np.ndarray:
        """
        Smooth curve
        
        Parameters:
        values: Raw values
        window: Window size
        
        Returns:
        Smoothed values
        """
        if window <= 1:
            return values
        
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(values, size=window, mode='nearest')
    
    def calculate_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Calculate Dynamic Time Warping distance
        
        Parameters:
        seq1, seq2: Two sequences
        
        Returns:
        DTW distance
        """
        try:
            from scipy.spatial.distance import euclidean
            from fastdtw import fastdtw
            distance, _ = fastdtw(seq1, seq2, dist=euclidean)
            return distance
        except ImportError:
            # Use simple implementation if fastdtw is unavailable
            return self._simple_dtw(seq1, seq2)
    
    def _simple_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Simple DTW implementation"""
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        return dtw_matrix[n, m]
    
    def calculate_trend_consistency(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Calculate trend consistency (proportion of same direction changes)
        
        Parameters:
        seq1, seq2: Two sequences
        
        Returns:
        Consistency score [0, 1]
        """
        if len(seq1) < 2 or len(seq2) < 2:
            return 0.0
        
        # Calculate differences (direction of change)
        diff1 = np.diff(seq1)
        diff2 = np.diff(seq2)
        
        # Calculate sign consistency
        same_direction = np.sign(diff1) == np.sign(diff2)
        consistency = np.mean(same_direction)
        
        return float(consistency)
    
    def calculate_cumulative_pearson(self, values1: np.ndarray, values2: np.ndarray, min_points: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cumulative Pearson correlation at each step
        
        Parameters:
        values1, values2: Two sequences
        min_points: Minimum number of points required to calculate correlation
        
        Returns:
        (indices, correlations) - Arrays of valid indices and their correlations
        """
        n = len(values1)
        correlations = []
        indices = []
        
        for i in range(min_points, n + 1):
            # Calculate Pearson correlation from start to current position
            corr, _ = stats.pearsonr(values1[:i], values2[:i])
            correlations.append(corr)
            indices.append(i - 1)  # 0-indexed
        
        return np.array(indices), np.array(correlations)
    
    def calculate_cumulative_metrics(self, values1: np.ndarray, values2: np.ndarray, min_points: int = 10) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate cumulative metrics at each step
        
        Parameters:
        values1, values2: Two sequences
        min_points: Minimum number of points required
        
        Returns:
        Dictionary with metric name -> (indices, values) pairs
        """
        n = len(values1)
        value_range = max(values1.max(), values2.max()) - min(values1.min(), values2.min())
        
        results = {
            'pearson': ([], []),
            'pearson_squared': ([], []),
            'nrmse': ([], []),
            'nmae': ([], [])
        }
        
        indices = []
        
        for i in range(min_points, n + 1):
            sub_val1 = values1[:i]
            sub_val2 = values2[:i]
            
            # Pearson correlation
            corr, _ = stats.pearsonr(sub_val1, sub_val2)
            results['pearson'][1].append(corr)
            results['pearson_squared'][1].append(corr ** 2)
            
            # NRMSE
            rmse = np.sqrt(np.mean((sub_val1 - sub_val2) ** 2))
            nrmse = rmse / value_range if value_range > 0 else 0
            results['nrmse'][1].append(nrmse)
            
            # NMAE
            mae = np.mean(np.abs(sub_val1 - sub_val2))
            nmae = mae / value_range if value_range > 0 else 0
            results['nmae'][1].append(nmae)
            
            indices.append(i - 1)
        
        # Convert lists to arrays
        indices_array = np.array(indices)
        for key in results:
            results[key] = (indices_array, np.array(results[key][1]))
        
        return results
    
    def calculate_all_metrics(self, values1: np.ndarray, values2: np.ndarray) -> Dict:
        """
        Calculate key correlation metrics
        
        Parameters:
        values1, values2: Two aligned sequences
        
        Returns:
        Dictionary containing all metrics
        """
        metrics = {}
        
        # 1. Pearson Correlation Coefficient
        pearson_corr, pearson_pval = stats.pearsonr(values1, values2)
        metrics['pearson_correlation'] = {
            'value': float(pearson_corr),
            'p_value': float(pearson_pval),
            'interpretation': self._interpret_correlation(pearson_corr),
            'description': 'Linear Correlation'
        }
        
        # 2. Root Mean Square Error
        rmse = np.sqrt(np.mean((values1 - values2) ** 2))
        value_range = max(values1.max(), values2.max()) - min(values1.min(), values2.min())
        nrmse = rmse / value_range if value_range > 0 else 0
        metrics['rmse'] = {
            'value': float(rmse),
            'normalized': float(nrmse),
            'interpretation': f"{'Low' if nrmse < 0.1 else 'Medium' if nrmse < 0.3 else 'High'} Error",
            'description': 'Root Mean Square Error'
        }
        
        # 3. Mean Absolute Error
        mae = np.mean(np.abs(values1 - values2))
        nmae = mae / value_range if value_range > 0 else 0
        metrics['mae'] = {
            'value': float(mae),
            'normalized': float(nmae),
            'interpretation': f"{'Low' if nmae < 0.1 else 'Medium' if nmae < 0.3 else 'High'} Error",
            'description': 'Mean Absolute Error'
        }
        
        return metrics
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def create_visualization(self, 
                           steps: np.ndarray,
                           values1: np.ndarray,
                           values2: np.ndarray,
                           metrics: Dict,
                           label1: str = "Curve 1",
                           label2: str = "Curve 2") -> plt.Figure:
        """
        Create correlation visualization charts
        
        Parameters:
        steps: Step array
        values1, values2: Two sequence values
        metrics: Dictionary of calculated metrics
        label1, label2: Curve labels
        
        Returns:
        matplotlib Figure object
        """
        fig = plt.figure(figsize=self.config['figure_size'])
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, 
                             height_ratios=[1, 1, 0.3],
                             top=0.90, bottom=0.06, left=0.08, right=0.95)
        
        # 1. Main plot: Comparison of two curves
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(steps, values1, color=self.colors['curve1'], 
                linewidth=2.5, label=label1, alpha=0.8)
        ax1.plot(steps, values2, color=self.colors['curve2'], 
                linewidth=2.5, label=label2, alpha=0.8)
        ax1.set_xlabel('Training Steps', fontsize=self.config['font_size'])
        ax1.set_ylabel('Value', fontsize=self.config['font_size'])
        ax1.set_title('Curve Comparison', 
                     fontsize=self.config['title_font_size'], fontweight='bold')
        ax1.legend(loc='best', fontsize=self.config['font_size'])
        ax1.grid(True, alpha=0.3)
        
        # Calculate cumulative metrics
        cum_metrics = self.calculate_cumulative_metrics(values1, values2)
        cum_steps = steps[cum_metrics['pearson'][0]]
        
        # 2. Cumulative Pearson Correlation
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(cum_steps, cum_metrics['pearson'][1], color='#2E75B6', linewidth=2.5, alpha=0.9)
        ax2.axhline(y=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Target: 0.9')
        ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Good: 0.8')
        ax2.set_xlabel('Training Steps', fontsize=self.config['font_size'])
        ax2.set_ylabel('Cumulative Pearson r', fontsize=self.config['font_size'])
        ax2.set_title('Cumulative Pearson Correlation', 
                     fontsize=self.config['font_size'], fontweight='bold')
        ax2.legend(fontsize=12, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([max(-1, cum_metrics['pearson'][1].min() - 0.05), 1.02])
        
        # 3. Cumulative Pearson Squared
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(cum_steps, cum_metrics['pearson_squared'][1], color='#C55A11', linewidth=2.5, alpha=0.9)
        ax3.axhline(y=0.81, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Target: 0.81 (0.9²)')
        ax3.axhline(y=0.64, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Good: 0.64 (0.8²)')
        ax3.set_xlabel('Training Steps', fontsize=self.config['font_size'])
        ax3.set_ylabel('Cumulative Pearson r²', fontsize=self.config['font_size'])
        ax3.set_title('Cumulative Pearson Squared (r²)', 
                     fontsize=self.config['font_size'], fontweight='bold')
        ax3.legend(fontsize=12, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([max(0, cum_metrics['pearson_squared'][1].min() - 0.05), 1.02])
        
        # 4. Metrics summary table (horizontal layout, compact)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Prepare table data - horizontal layout
        table_data = [
            ['Metric', 'Pearson r', 'Pearson r²', 'NRMSE', 'NMAE'],
            ['Value', 
             f"{metrics['pearson_correlation']['value']:.4f}",
             f"{metrics['pearson_correlation']['value']**2:.4f}",
             f"{metrics['rmse']['normalized']:.4f}",
             f"{metrics['mae']['normalized']:.4f}"]
        ]
        
        # Create table with smaller bbox
        table = ax4.table(cellText=table_data, cellLoc='center',
                         loc='center', bbox=[0.15, 0.1, 0.7, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1, 2.0)
        
        # Set header column style (first column)
        for i in range(2):
            table[(i, 0)].set_facecolor('#4472C4')
            table[(i, 0)].set_text_props(weight='bold', color='white')
        
        # Set value cells colors
        for j in range(1, 5):
            table[(1, j)].set_facecolor('#E7E6E6')
        
        # Set column widths for better appearance
        table.auto_set_column_width(list(range(5)))
        
        # plt.suptitle('Correlation Analysis Report', 
        #             fontsize=self.config['title_font_size'] + 2, 
        #             fontweight='bold', y=0.97)
        
        return fig
    
    def analyze(self, 
               csv1_path: str,
               csv2_path: str,
               output_prefix: str = "correlation_analysis",
               label1: str = None,
               label2: str = None,
               output_dir: str = ".") -> Dict:
        """
        Complete correlation analysis workflow
        
        Parameters:
        csv1_path: First CSV file path
        csv2_path: Second CSV file path
        output_prefix: Output file prefix
        label1, label2: Curve labels
        output_dir: Output directory
        
        Returns:
        Analysis results dictionary
        """
        print("🚀 Starting correlation analysis...")
        print(f"📊 CSV 1: {csv1_path}")
        print(f"📊 CSV 2: {csv2_path}")
        
        # 1. Load data
        print("\n📂 Loading data...")
        df1 = self.load_and_preprocess(csv1_path)
        df2 = self.load_and_preprocess(csv2_path)
        print(f"   CSV 1: {len(df1)} data points")
        print(f"   CSV 2: {len(df2)} data points")
        
        # 2. Align sequences
        print("\n🔄 Aligning sequences...")
        steps, values1, values2 = self.align_sequences(df1, df2)
        print(f"   After alignment: {len(steps)} data points")
        
        # 3. Smooth (optional)
        if self.config['smooth_window'] > 1:
            print(f"\n🎨 Applying smoothing (window={self.config['smooth_window']})...")
            values1 = self.smooth_curve(values1, self.config['smooth_window'])
            values2 = self.smooth_curve(values2, self.config['smooth_window'])
        
        # 4. Calculate metrics
        print("\n📐 Calculating correlation metrics...")
        metrics = self.calculate_all_metrics(values1, values2)
        
        # 5. Generate labels
        if label1 is None:
            label1 = Path(csv1_path).stem
        if label2 is None:
            label2 = Path(csv2_path).stem
        
        # 6. Create visualization
        print("\n🎨 Generating visualization...")
        fig = self.create_visualization(steps, values1, values2, metrics, label1, label2)
        
        # 7. Save results
        print("\n💾 Saving results...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save charts
        for fmt in ['png', 'pdf']:
            output_path = Path(output_dir) / f"{output_prefix}_{timestamp}.{fmt}"
            fig.savefig(output_path, format=fmt, dpi=self.config['dpi'],
                       bbox_inches='tight', facecolor='white')
            print(f"   ✅ {output_path}")
        
        plt.close(fig)
        
        # Save JSON report
        report = {
            'timestamp': timestamp,
            'files': {
                'csv1': csv1_path,
                'csv2': csv2_path,
                'label1': label1,
                'label2': label2
            },
            'data_info': {
                'original_length_1': len(df1),
                'original_length_2': len(df2),
                'aligned_length': len(steps),
                'step_range': [float(steps.min()), float(steps.max())]
            },
            'metrics': metrics,
            'config': self.config
        }
        
        json_path = Path(output_dir) / f"{output_prefix}_{timestamp}_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {json_path}")
        
        # 8. Print summary
        self._print_summary(metrics)
        
        print(f"\n🎉 Analysis complete!")
        return report
    
    def _print_summary(self, metrics: Dict):
        """Print analysis summary"""
        print("\n" + "="*70)
        print("📈 Correlation Analysis Summary")
        print("="*70)
        
        print(f"\n🔍 Linear Correlation:")
        print(f"   Pearson Correlation: {metrics['pearson_correlation']['value']:.4f} "
              f"({metrics['pearson_correlation']['interpretation']})")
        print(f"   p-value: {metrics['pearson_correlation']['p_value']:.6f}")
        
        print(f"\n🔍 Error Metrics:")
        print(f"   Normalized RMSE: {metrics['rmse']['normalized']:.4f} "
              f"({metrics['rmse']['interpretation']})")
        print(f"   Normalized MAE: {metrics['mae']['normalized']:.4f} "
              f"({metrics['mae']['interpretation']})")
        
        print("\n" + "="*70)


def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CSV Data Correlation Analysis and Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Basic analysis
  python correlation_analyzer.py file1.csv file2.csv
  
  # Specify output directory and labels
  python correlation_analyzer.py file1.csv file2.csv \\
      --output-dir ./results \\
      --label1 "Algorithm A" \\
      --label2 "Algorithm B"
  
  # Apply smoothing
  python correlation_analyzer.py file1.csv file2.csv --smooth 5
        """
    )
    
    parser.add_argument('csv1', help='First CSV file path')
    parser.add_argument('csv2', help='Second CSV file path')
    parser.add_argument('--output-prefix', default='correlation_analysis',
                       help='Output file prefix')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory')
    parser.add_argument('--label1', help='Label for first curve')
    parser.add_argument('--label2', help='Label for second curve')
    parser.add_argument('--smooth', type=int, default=0,
                       help='Smoothing window size (0 means no smoothing)')
    parser.add_argument('--no-interpolate', action='store_true',
                       help='Do not interpolate and align sequences')
    
    args = parser.parse_args()
    
    # Validate files
    if not Path(args.csv1).exists():
        print(f"❌ Error: File not found {args.csv1}")
        return 1
    if not Path(args.csv2).exists():
        print(f"❌ Error: File not found {args.csv2}")
        return 1
    
    # Create configuration
    config = {
        'smooth_window': args.smooth,
        'interpolate': not args.no_interpolate
    }
    
    # Create analyzer and run
    analyzer = CorrelationAnalyzer(config)
    
    try:
        analyzer.analyze(
            csv1_path=args.csv1,
            csv2_path=args.csv2,
            output_prefix=args.output_prefix,
            label1=args.label1,
            label2=args.label2,
            output_dir=args.output_dir
        )
        return 0
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("📚 CSV Data Correlation Analysis Tool")
        print("\nUsage:")
        print("  python correlation_analyzer.py <CSV1> <CSV2> [options]")
        print("\nView full help:")
        print("  python correlation_analyzer.py --help")
        print("\nQuick example:")
        print("  python correlation_analyzer.py experiment1.csv experiment2.csv")
        sys.exit(0)
    
    sys.exit(main())