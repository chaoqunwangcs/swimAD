import argparse
import os
import shutil
from pathlib import Path

def delete_log_lines(input_file, start_line, end_line):
    """删除指定行号范围的内容，包含起始和结束行"""
    # 创建备份文件
    backup_path = Path(input_file).with_suffix('.log.bak')
    shutil.copy2(input_file, backup_path)
    print(f"已创建备份文件：{backup_path}")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 转换用户输入的行号为Python索引(从0开始)
    start_idx = start_line - 1
    end_idx = end_line - 1

    # 验证行号有效性
    if start_idx < 0 or end_idx >= len(lines):
        raise ValueError(f"行号超出范围 (文件共{len(lines)}行)")

    # 生成新的内容列表
    new_lines = [
        line for idx, line in enumerate(lines)
        if not (start_idx <= idx <= end_idx)
    ]

    # 写入修改后的内容
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"已删除第{start_line}-{end_line}行，共删除{len(lines)-len(new_lines)}行")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='删除日志文件指定行范围')
    parser.add_argument('file', help='.log文件路径')
    parser.add_argument('-a', type=int, required=True, help='起始行号')
    parser.add_argument('-b', type=int, required=True, help='结束行号')
    
    args = parser.parse_args()

    try:
        if args.a > args.b:
            raise ValueError("起始行号不能大于结束行号")
        delete_log_lines(args.file, args.a, args.b)
    except Exception as e:
        print(f"错误：{str(e)}")
        exit(1)