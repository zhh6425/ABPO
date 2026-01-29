#!/usr/bin/env python3
"""
TensorBoard CSV批量可视化脚本
自动读取、处理、合并多个CSV文件并生成可视化图表

作者: TensorBoard Visualizer
版本: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
import json
import glob
import os
from typing import List, Dict, Tuple, Optional

class TensorBoardVisualizer:
    """TensorBoard CSV数据可视化处理器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化可视化器
        
        参数:
        config: 配置字典，包含样式和参数设置
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # 设置matplotlib样式
        self._setup_matplotlib_style()
        
        # 低饱和度配色方案
        self.colors = [
            '#8CAD73', '#C55A11', '#5B9BD5', '#636363',
            '#843C0C', '#B7B7B7', '#FFC000', '#8497B0', '#264478',
            '#D6AC00', '#70AD47', '#A5A5A5', 
        ]
    
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            'figure_size': (8, 6),
            'dpi': 300,
            'line_width': 3.5,
            'marker_size': 5,       # 标记大小
            'font_size': 18,
            'title_font_size': 20,  # 标题字体大小
            'grid_alpha': 0.3,
            'output_formats': ['png', 'pdf'],
            'save_stats': False,
            'show_avg_lines': False,
            'annotate_key_points': False,
            'smooth_lines': True,
            'smooth_window': 5
        }
    
    def _setup_matplotlib_style(self):
        """设置matplotlib样式"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': self.config['font_size'],
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True,
            'figure.dpi': self.config['dpi'],
            'savefig.dpi': self.config['dpi'],
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white'
        })
    
    def load_csv_files(self, file_patterns: List[str]) -> Dict[str, pd.DataFrame]:
        """
        批量加载CSV文件
        
        参数:
        file_patterns: 文件路径模式列表，支持通配符
        
        返回:
        字典：{文件名: DataFrame}
        """
        datasets = {}
        
        for pattern in file_patterns:
            files = glob.glob(pattern)
            if not files:
                print(f"⚠️  警告: 未找到匹配模式 '{pattern}' 的文件")
                continue
            
            for file_path in files:
                try:
                    df = self._preprocess_csv(file_path)
                    file_key = Path(file_path).stem  # 使用文件名(不含扩展名)作为键
                    datasets[file_key] = df
                    print(f"✅ 成功加载: {file_path} ({len(df)} 数据点)")
                except Exception as e:
                    print(f"❌ 加载失败: {file_path} - {str(e)}")
        
        return datasets
    
    def _preprocess_csv(self, file_path: str) -> pd.DataFrame:
        """
        预处理单个CSV文件
        
        参数:
        file_path: CSV文件路径
        
        返回:
        清理后的DataFrame
        """
        # 读取CSV
        df = pd.read_csv(file_path)
        
        # 移除Wall time列
        if 'Wall time' in df.columns:
            df = df.drop('Wall time', axis=1)
        
        # 检查必要列
        if 'Step' not in df.columns:
            raise ValueError(f"缺少 'Step' 列")
        if 'Value' not in df.columns:
            raise ValueError(f"缺少 'Value' 列")
        
        # 数据类型转换
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # 移除无效数据
        original_len = len(df)
        df = df.dropna(subset=['Step', 'Value'])
        if len(df) < original_len:
            print(f"  ⚠️  移除了 {original_len - len(df)} 行无效数据")
        
        # 按Step排序
        df = df.sort_values('Step').reset_index(drop=True)
        
        return df
    
    def create_visualization(self, 
                           datasets: Dict[str, pd.DataFrame],
                           title: str = "Training Performance Comparison",
                           y_label: str = "Value",
                           x_label: str = "Training Steps",
                           labels: Dict[str, str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        创建可视化图表
        
        参数:
        datasets: 数据集字典
        title: 图表标题
        y_label: Y轴标签
        x_label: X轴标签  
        labels: 自定义标签映射 {文件名: 显示标签}
        
        返回:
        (figure, axes) 元组
        """
        if not datasets:
            raise ValueError("没有可用的数据集")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        fig.patch.set_facecolor('white')
        
        # 统计信息
        stats = {}
        
        # 绘制每个数据集
        for i, (dataset_name, df) in enumerate(datasets.items()):
            color = self.colors[i % len(self.colors)]
            
            # 确定显示标签
            if labels and dataset_name in labels:
                display_label = labels[dataset_name]
            else:
                display_label = dataset_name.replace('_', ' ') #.upper()
            
            # 数据平滑(可选)
            values = df['Value'].values
            if self.config['smooth_lines'] and len(values) > self.config['smooth_window']:
                from scipy.ndimage import uniform_filter1d
                values = uniform_filter1d(values, size=self.config['smooth_window'])
            
            # 绘制主折线
            ax.plot(df['Step'], values,
                   color=color,
                   linewidth=self.config['line_width'],
                   marker='o',
                   markersize=self.config['marker_size'],
                   markerfacecolor=color,
                   markeredgecolor='white',
                   markeredgewidth=2,  # 从1增加到2
                   alpha=0.9,
                   label=display_label,
                   zorder=3)
            
            # 添加平均值线(可选)
            if self.config['show_avg_lines']:
                avg_value = df['Value'].mean()
                ax.axhline(y=avg_value, color=color, linestyle='--', 
                          alpha=0.5, linewidth=3)  # 从1.5增加到3
            
            # 收集统计信息
            stats[display_label] = {
                'count': len(df),
                'mean': df['Value'].mean(),
                'std': df['Value'].std(),
                'min': df['Value'].min(),
                'max': df['Value'].max(),
                'final': df['Value'].iloc[-1] if len(df) > 0 else np.nan
            }
        
        # 设置图表属性
        ax.set_title(title, fontsize=self.config['title_font_size'], 
                    fontweight='bold', pad=20)
        ax.set_xlabel(x_label, fontsize=self.config['font_size'], fontweight='medium')
        ax.set_ylabel(y_label, fontsize=self.config['font_size'], fontweight='medium')
        
        # 网格设置
        ax.grid(True, alpha=self.config['grid_alpha'], linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 图例设置
        legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                          fontsize=self.config['font_size'])
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#dee2e6')
        
        # 坐标轴美化
        ax.spines['left'].set_color('#000000')
        ax.spines['bottom'].set_color('#000000') 
        ax.tick_params(colors='#000000', which='both')
        
        # 自动调整坐标轴范围
        self._auto_adjust_axes(ax, datasets)
        
        plt.tight_layout()
        
        return fig, ax, stats
    
    def _auto_adjust_axes(self, ax: plt.Axes, datasets: Dict[str, pd.DataFrame]):
        """自动调整坐标轴范围"""
        all_steps = []
        all_values = []
        
        for df in datasets.values():
            all_steps.extend(df['Step'].tolist())
            all_values.extend(df['Value'].tolist())
        
        if all_steps and all_values:
            # X轴范围和刻度设置
            step_min, step_max = min(all_steps), max(all_steps)
            step_margin = (step_max - step_min) * 0.02
            ax.set_xlim(step_min - step_margin, step_max + step_margin)
            
            # 设置X轴刻度为整数，间隔为配置的step_interval
            step_interval = self.config.get('step_interval', 20)
            # 计算合适的起始点和结束点
            tick_start = int(step_min // step_interval) * step_interval
            tick_end = int((step_max // step_interval)) * step_interval
            x_ticks = list(range(max(0, tick_start), tick_end, step_interval))
            x_ticks.pop(0)
            
            
            # 确保包含数据范围内的关键点
            if step_min not in x_ticks and step_min <= max(x_ticks):
                x_ticks.insert(0, int(step_min))
            if step_max not in x_ticks and step_max >= min(x_ticks):
                x_ticks.append(int(step_max))
            
            ax.set_xticks(sorted(x_ticks))
            
            # Y轴范围
            val_min, val_max = min(all_values), max(all_values)
            val_margin = (val_max - val_min) * 0.1
            ax.set_ylim(val_min - val_margin, val_max + val_margin)
    
    def save_visualization(self, 
                          fig: plt.Figure,
                          output_prefix: str,
                          stats: Dict = None,
                          output_dir: str = ".") -> List[str]:
        """
        保存可视化结果
        
        参数:
        fig: matplotlib图形对象
        output_prefix: 输出文件前缀
        stats: 统计信息字典
        output_dir: 输出目录
        
        返回:
        保存的文件路径列表
        """
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = []
        
        # 保存图表文件
        for fmt in self.config['output_formats']:
            filename = f"{output_prefix}_{timestamp}.{fmt}"
            filepath = Path(output_dir) / filename
            
            try:
                fig.savefig(filepath,
                           format=fmt,
                           dpi=self.config['dpi'] if fmt == 'png' else None,
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none')
                saved_files.append(str(filepath))
                print(f"✅ 保存图表: {filepath}")
            except Exception as e:
                print(f"❌ 保存失败 {filepath}: {e}")
        
        # 保存统计信息
        if self.config['save_stats'] and stats:
            stats_filename = f"{output_prefix}_{timestamp}_stats.json"
            stats_filepath = Path(output_dir) / stats_filename
            
            try:
                # 转换numpy类型为Python原生类型
                clean_stats = {}
                for key, value in stats.items():
                    clean_stats[key] = {k: float(v) if not np.isnan(v) else None 
                                       for k, v in value.items()}
                
                with open(stats_filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': timestamp,
                        'config': self.config,
                        'statistics': clean_stats
                    }, f, indent=2, ensure_ascii=False)
                
                saved_files.append(str(stats_filepath))
                print(f"✅ 保存统计: {stats_filepath}")
            except Exception as e:
                print(f"❌ 保存统计失败: {e}")
        
        return saved_files
    
    def process_batch(self,
                     file_patterns: List[str],
                     output_prefix: str = "tensorboard_comparison",
                     title: str = "Training Performance Comparison",
                     y_label: str = "Value", 
                     x_label: str = "Training Steps",
                     labels: Dict[str, str] = None,
                     output_dir: str = ".") -> List[str]:
        """
        批量处理完整流程
        
        参数:
        file_patterns: 文件模式列表
        output_prefix: 输出文件前缀
        title: 图表标题
        y_label: Y轴标签
        x_label: X轴标签
        labels: 标签映射
        output_dir: 输出目录
        
        返回:
        保存的文件路径列表
        """
        print("🚀 开始批量处理TensorBoard CSV文件...")
        print(f"📂 输出目录: {output_dir}")
        print(f"🔍 搜索模式: {file_patterns}")
        
        # 1. 加载数据
        datasets = self.load_csv_files(file_patterns)
        if not datasets:
            raise ValueError("未找到任何有效的CSV文件")
        
        print(f"\n📊 成功加载 {len(datasets)} 个数据集")
        
        # 2. 创建可视化
        print("🎨 创建可视化图表...")
        fig, ax, stats = self.create_visualization(
            datasets=datasets,
            title=title,
            y_label=y_label,
            x_label=x_label,
            labels=labels
        )
        
        # 3. 保存结果
        print("💾 保存可视化结果...")
        saved_files = self.save_visualization(
            fig=fig,
            output_prefix=output_prefix,
            stats=stats,
            output_dir=output_dir
        )
        
        # 4. 关闭图形避免内存泄漏
        plt.close(fig)
        
        # 5. 打印摘要
        self._print_summary(stats, saved_files)
        
        return saved_files
    
    def _print_summary(self, stats: Dict, saved_files: List[str]):
        """打印处理摘要"""
        print(f"\n📈 数据统计摘要:")
        print("=" * 60)
        
        for name, stat in stats.items():
            print(f"{name}:")
            print(f"  📊 数据点: {stat['count']}")
            print(f"  📊 平均值: {stat['mean']:.6f}")
            print(f"  📊 标准差: {stat['std']:.6f}")
            print(f"  📊 范围: {stat['min']:.6f} - {stat['max']:.6f}")
            print(f"  📊 最终值: {stat['final']:.6f}")
            print()
        
        print(f"🎉 处理完成! 共生成 {len(saved_files)} 个文件:")
        for file_path in saved_files:
            print(f"   📁 {file_path}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='TensorBoard CSV批量可视化工具')
    
    parser.add_argument('files', nargs='+', help='CSV文件路径(支持通配符)')
    parser.add_argument('--output-prefix', default='tensorboard_comparison', 
                       help='输出文件前缀')
    parser.add_argument('--title', default='Training Performance Comparison',
                       help='图表标题')
    parser.add_argument('--y-label', default='Value', help='Y轴标签')
    parser.add_argument('--x-label', default='Training Steps', help='X轴标签') 
    parser.add_argument('--output-dir', default='.', help='输出目录')
    parser.add_argument('--config', help='配置文件路径(JSON格式)')
    parser.add_argument('--labels', help='标签映射(JSON格式)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 解析标签映射
    labels = None
    if args.labels:
        try:
            labels = json.loads(args.labels)
        except json.JSONDecodeError:
            print("⚠️  警告: 标签映射JSON格式错误，将使用默认标签")
    
    # 创建可视化器
    visualizer = TensorBoardVisualizer(config)
    
    # 执行批量处理
    try:
        saved_files = visualizer.process_batch(
            file_patterns=args.files,
            output_prefix=args.output_prefix,
            title=args.title,
            y_label=args.y_label,
            x_label=args.x_label,
            labels=labels,
            output_dir=args.output_dir
        )
        print(f"\n✅ 全部完成! 输出文件: {len(saved_files)} 个")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        print("📚 TensorBoard CSV批量可视化工具")
        print("\n🔧 基本用法:")
        print("python script.py file1.csv file2.csv --title 'My Comparison'")
        print("\n🔧 高级用法:")
        print("python script.py *.csv --output-dir ./results --y-label 'Reward Score'")
        print("\n🔧 使用标签映射:")
        print('python script.py *.csv --labels \'{"file1": "Algorithm A", "file2": "Algorithm B"}\'')
        
        # 运行示例(如果有CSV文件)
        csv_files = glob.glob("*.csv")
        if csv_files:
            print(f"\n🎯 检测到 {len(csv_files)} 个CSV文件，运行示例...")
            
            visualizer = TensorBoardVisualizer()
            saved_files = visualizer.process_batch(
                file_patterns=["*.csv"],
                title="TensorBoard Training Comparison",
                y_label="Reward Score"
            )
        else:
            print("\n💡 在当前目录放置CSV文件后重新运行以查看示例")
    else:
        exit(main())