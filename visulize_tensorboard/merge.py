#!/usr/bin/env python3
"""
PDF横向并排合并脚本
将两个PDF文件（如TensorBoard可视化图表）横向并排合并为一个新PDF

作者: PDF Merger
版本: 1.0
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
import io


class PDFSideBySideMerger:
    """PDF横向并排合并处理器"""
    
    def __init__(self, spacing: float = 20, page_size: str = 'auto'):
        """
        初始化合并器
        
        参数:
        spacing: 两个PDF之间的间距（点）
        page_size: 输出页面尺寸 ('auto', 'letter', 'A4', 或自定义(width, height))
        """
        self.spacing = spacing
        self.page_size = page_size
    
    def merge_side_by_side(self, 
                          pdf1_path: str, 
                          pdf2_path: str,
                          output_path: str = None,
                          alignment: str = 'center') -> str:
        """
        横向并排合并两个PDF文件
        
        参数:
        pdf1_path: 第一个PDF文件路径（左侧）
        pdf2_path: 第二个PDF文件路径（右侧）
        output_path: 输出文件路径（可选，默认自动生成）
        alignment: 垂直对齐方式 ('top', 'center', 'bottom')
        
        返回:
        输出文件路径
        """
        print("🚀 开始横向合并PDF文件...")
        print(f"📄 左侧PDF: {pdf1_path}")
        print(f"📄 右侧PDF: {pdf2_path}")
        
        # 读取PDF文件
        reader1 = PdfReader(pdf1_path)
        reader2 = PdfReader(pdf2_path)
        
        # 确定要处理的页数（取两者较大值）
        num_pages = max(len(reader1.pages), len(reader2.pages))
        print(f"📊 检测到最大页数: {num_pages}")
        
        # 创建输出writer
        writer = PdfWriter()
        
        # 处理每一页
        for page_num in range(num_pages):
            print(f"🎨 处理第 {page_num + 1}/{num_pages} 页...")
            
            # 获取页面（如果该PDF页数不足则使用最后一页）
            page1 = reader1.pages[min(page_num, len(reader1.pages) - 1)]
            page2 = reader2.pages[min(page_num, len(reader2.pages) - 1)]
            
            # 创建合并页面
            merged_page = self._create_merged_page(page1, page2, alignment)
            writer.add_page(merged_page)
        
        # 生成输出路径
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"merged_side_by_side_{timestamp}.pdf"
        
        # 保存合并后的PDF
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"✅ 合并完成: {output_path}")
        print(f"📏 输出页数: {len(writer.pages)}")
        
        return str(output_path)
    
    def _create_merged_page(self, page1, page2, alignment: str = 'center'):
        """
        创建横向合并的单页
        
        参数:
        page1: 左侧页面
        page2: 右侧页面
        alignment: 垂直对齐方式
        
        返回:
        合并后的页面对象
        """
        # 获取页面尺寸
        box1 = page1.mediabox
        box2 = page2.mediabox
        
        width1 = float(box1.width)
        height1 = float(box1.height)
        width2 = float(box2.width)
        height2 = float(box2.height)
        
        # 计算新页面尺寸
        new_width = width1 + width2 + self.spacing
        new_height = max(height1, height2)
        
        # 创建空白页面
        from pypdf.generic import RectangleObject
        from pypdf import PageObject
        
        merged_page = PageObject.create_blank_page(width=new_width, height=new_height)
        
        # 计算垂直位置（根据对齐方式）
        if alignment == 'top':
            y1 = new_height - height1
            y2 = new_height - height2
        elif alignment == 'bottom':
            y1 = 0
            y2 = 0
        else:  # center
            y1 = (new_height - height1) / 2
            y2 = (new_height - height2) / 2
        
        # 添加第一个PDF（左侧）
        merged_page.merge_transformed_page(
            page1,
            Transformation().translate(tx=0, ty=y1)
        )
        
        # 添加第二个PDF（右侧）
        merged_page.merge_transformed_page(
            page2,
            Transformation().translate(tx=width1 + self.spacing, ty=y2)
        )
        
        return merged_page
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """
        获取PDF文件信息
        
        参数:
        pdf_path: PDF文件路径
        
        返回:
        包含PDF信息的字典
        """
        reader = PdfReader(pdf_path)
        page = reader.pages[0]
        box = page.mediabox
        
        return {
            'path': pdf_path,
            'pages': len(reader.pages),
            'width': float(box.width),
            'height': float(box.height),
            'size_inches': (float(box.width) / 72, float(box.height) / 72)
        }
    
    def print_pdf_info(self, pdf_path: str):
        """打印PDF文件信息"""
        info = self.get_pdf_info(pdf_path)
        print(f"\n📋 PDF信息: {Path(pdf_path).name}")
        print(f"   页数: {info['pages']}")
        print(f"   尺寸: {info['width']:.1f} x {info['height']:.1f} 点")
        print(f"   英寸: {info['size_inches'][0]:.2f} x {info['size_inches'][1]:.2f} 英寸")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description='横向并排合并两个PDF文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本合并
  python merge_pdfs_side_by_side.py left.pdf right.pdf
  
  # 指定输出路径
  python merge_pdfs_side_by_side.py left.pdf right.pdf -o merged.pdf
  
  # 调整间距和对齐方式
  python merge_pdfs_side_by_side.py left.pdf right.pdf --spacing 30 --align top
  
  # 查看PDF信息
  python merge_pdfs_side_by_side.py left.pdf right.pdf --info
        """
    )
    
    parser.add_argument('pdf1', help='第一个PDF文件（左侧）')
    parser.add_argument('pdf2', help='第二个PDF文件（右侧）')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--spacing', type=float, default=20,
                       help='两个PDF之间的间距（点），默认20')
    parser.add_argument('--align', choices=['top', 'center', 'bottom'],
                       default='center', help='垂直对齐方式，默认center')
    parser.add_argument('--info', action='store_true',
                       help='仅显示PDF信息，不进行合并')
    
    args = parser.parse_args()
    
    # 验证输入文件
    pdf1_path = Path(args.pdf1)
    pdf2_path = Path(args.pdf2)
    
    if not pdf1_path.exists():
        print(f"❌ 错误: 找不到文件 {pdf1_path}")
        return 1
    
    if not pdf2_path.exists():
        print(f"❌ 错误: 找不到文件 {pdf2_path}")
        return 1
    
    # 创建合并器
    merger = PDFSideBySideMerger(spacing=args.spacing)
    
    # 如果只是查看信息
    if args.info:
        print("📊 PDF文件信息:")
        print("=" * 60)
        merger.print_pdf_info(str(pdf1_path))
        merger.print_pdf_info(str(pdf2_path))
        
        # 计算合并后的尺寸
        info1 = merger.get_pdf_info(str(pdf1_path))
        info2 = merger.get_pdf_info(str(pdf2_path))
        merged_width = info1['width'] + info2['width'] + args.spacing
        merged_height = max(info1['height'], info2['height'])
        
        print(f"\n📐 合并后预期尺寸:")
        print(f"   {merged_width:.1f} x {merged_height:.1f} 点")
        print(f"   {merged_width/72:.2f} x {merged_height/72:.2f} 英寸")
        return 0
    
    # 执行合并
    try:
        output_path = merger.merge_side_by_side(
            str(pdf1_path),
            str(pdf2_path),
            args.output,
            args.align
        )
        print(f"\n🎉 成功! 输出文件: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    # 如果没有参数，显示帮助信息
    if len(sys.argv) == 1:
        print("📚 PDF横向并排合并工具")
        print("\n使用方法:")
        print("  python merge_pdfs_side_by_side.py <左侧PDF> <右侧PDF> [选项]")
        print("\n查看完整帮助:")
        print("  python merge_pdfs_side_by_side.py --help")
        print("\n快速示例:")
        print("  python merge_pdfs_side_by_side.py chart1.pdf chart2.pdf")
        print("  python merge_pdfs_side_by_side.py chart1.pdf chart2.pdf -o comparison.pdf")
        sys.exit(0)
    
    sys.exit(main())