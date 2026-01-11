"""
结果导出模块：将进化结果导出为易读的格式
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.interfaces import Program

def export_results(task_id: str, output_dir: str = "results") -> None:
    """
    导出进化结果到易读的文件
    
    参数:
        task_id: 任务 ID
        output_dir: 输出目录（默认 "results"）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 从数据库加载程序
    from database_agent.agent import InMemoryDatabaseAgent
    db = InMemoryDatabaseAgent()
    
    # 获取所有相关程序
    all_programs = []
    # 由于 InMemoryDatabaseAgent 是异步的，我们需要手动加载
    if os.path.exists("program_database.json"):
        with open("program_database.json", 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    if 'programs' in data:
                        all_programs = data['programs']
                    else:
                        # 旧格式：dict of dicts
                        all_programs = list(data.values())
            except Exception as e:
                print(f"⚠️ 无法加载数据库: {e}")
                return
    
    # 过滤任务相关的程序
    task_programs = [p for p in all_programs if task_id in p.get('id', '')]
    
    if not task_programs:
        print(f"⚠️ 未找到任务 '{task_id}' 的程序")
        return
    
    # 按适应度排序（考虑 kissing_number_5d）
    def get_sort_key(prog: Dict) -> tuple:
        fitness = prog.get('fitness_scores', {})
        return (
            fitness.get('correctness', 0.0),
            fitness.get('sota_score_5d', 0.0),
            fitness.get('kissing_number_5d', 0.0),
            fitness.get('kissing_number_5d_valid', 0.0),
            -fitness.get('runtime_ms', float('inf')),
            -prog.get('generation', 0)
        )
    
    task_programs.sort(key=get_sort_key, reverse=True)
    
    # 生成结果文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{task_id}_results_{timestamp}.txt")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"OpenAlpha_Evolve 进化结果报告\n")
        f.write(f"任务 ID: {task_id}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 统计信息
        valid_programs = [p for p in task_programs if p.get('fitness_scores', {}).get('correctness', 0) > 0]
        best_5d_programs = [
            p for p in task_programs 
            if p.get('fitness_scores', {}).get('kissing_number_5d', 0) > 0
            and p.get('fitness_scores', {}).get('kissing_number_5d_valid', 0) == 1.0
        ]
        
        f.write(f"统计信息:\n")
        f.write(f"  总程序数: {len(task_programs)}\n")
        f.write(f"  有效程序数: {len(valid_programs)}\n")
        f.write(f"  包含 5D 结果的有效程序数: {len(best_5d_programs)}\n")
        f.write("\n")
        
        # 最佳程序（Top 5）
        f.write("=" * 80 + "\n")
        f.write("最佳程序（Top 5）:\n")
        f.write("=" * 80 + "\n\n")
        
        for i, prog in enumerate(task_programs[:5], 1):
            fitness = prog.get('fitness_scores', {})
            f.write(f"排名 {i}: {prog.get('id', 'N/A')}\n")
            f.write(f"  代数: {prog.get('generation', 'N/A')}\n")
            f.write(f"  适应度分数:\n")
            f.write(f"    - 正确率 (correctness): {fitness.get('correctness', 0):.4f}\n")
            f.write(f"    - 运行时间 (runtime_ms): {fitness.get('runtime_ms', 'N/A')}\n")
            
            # 5D kissing number 信息
            kissing_5d = fitness.get('kissing_number_5d', None)
            if kissing_5d is not None:
                sota_score = fitness.get('sota_score_5d', 0.0)
                is_valid = fitness.get('kissing_number_5d_valid', 0.0) == 1.0
                f.write(f"    - 5D Kissing Number: {int(kissing_5d)}\n")
                f.write(f"    - SOTA 评分: {sota_score:.4f}\n")
                f.write(f"    - 排列有效性: {'✅ 有效' if is_valid else '❌ 无效'}\n")
                
                # 与 SOTA 比较
                if is_valid:
                    diff = int(kissing_5d) - 40  # 当前 SOTA 下界
                    if diff >= 0:
                        f.write(f"    - 相对 SOTA (40): +{diff} ({'达到或超过 SOTA' if diff >= 4 else '接近 SOTA' if diff >= 0 else '低于 SOTA'})\n")
                    else:
                        f.write(f"    - 相对 SOTA (40): {diff} (低于 SOTA)\n")
            
            f.write(f"  状态: {prog.get('status', 'N/A')}\n")
            if prog.get('errors'):
                f.write(f"  错误: {prog.get('errors', [])[:3]}\n")  # 只显示前3个错误
            f.write("\n")
        
        # 5D Kissing Number 排行榜
        if best_5d_programs:
            f.write("=" * 80 + "\n")
            f.write("5D Kissing Number 排行榜 (Top 10):\n")
            f.write("=" * 80 + "\n\n")
            
            # 按 kissing_number_5d 排序
            best_5d_programs.sort(
                key=lambda p: (
                    p.get('fitness_scores', {}).get('kissing_number_5d', 0),
                    p.get('fitness_scores', {}).get('sota_score_5d', 0),
                    p.get('fitness_scores', {}).get('correctness', 0)
                ),
                reverse=True
            )
            
            for i, prog in enumerate(best_5d_programs[:10], 1):
                fitness = prog.get('fitness_scores', {})
                kissing_5d = int(fitness.get('kissing_number_5d', 0))
                sota_score = fitness.get('sota_score_5d', 0.0)
                correctness = fitness.get('correctness', 0.0)
                
                f.write(f"排名 {i}: {kissing_5d} 个球\n")
                f.write(f"  程序 ID: {prog.get('id', 'N/A')}\n")
                f.write(f"  代数: {prog.get('generation', 'N/A')}\n")
                f.write(f"  SOTA 评分: {sota_score:.4f}\n")
                f.write(f"  正确率: {correctness:.4f}\n")
                f.write("\n")
        
        # 最佳程序代码
        if task_programs:
            best_prog = task_programs[0]
            f.write("=" * 80 + "\n")
            f.write("最佳程序完整代码:\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"程序 ID: {best_prog.get('id', 'N/A')}\n")
            f.write(f"适应度: {best_prog.get('fitness_scores', {})}\n")
            f.write("\n代码:\n")
            f.write("-" * 80 + "\n")
            f.write(best_prog.get('code', 'N/A'))
            f.write("\n" + "-" * 80 + "\n")
    
    # 同时生成 JSON 格式（便于程序处理）
    json_file = os.path.join(output_dir, f"{task_id}_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'task_id': task_id,
            'timestamp': timestamp,
            'total_programs': len(task_programs),
            'valid_programs': len(valid_programs),
            'best_5d_programs': len(best_5d_programs),
            'top_5_programs': [
                {
                    'id': p.get('id'),
                    'generation': p.get('generation'),
                    'fitness_scores': p.get('fitness_scores', {}),
                    'code': p.get('code', ''),
                    'status': p.get('status')
                }
                for p in task_programs[:5]
            ],
            'best_5d_programs': [
                {
                    'id': p.get('id'),
                    'generation': p.get('generation'),
                    'kissing_number_5d': int(p.get('fitness_scores', {}).get('kissing_number_5d', 0)),
                    'sota_score_5d': p.get('fitness_scores', {}).get('sota_score_5d', 0.0),
                    'fitness_scores': p.get('fitness_scores', {}),
                    'code': p.get('code', '')
                }
                for p in best_5d_programs[:10]
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已导出到:")
    print(f"   文本格式: {result_file}")
    print(f"   JSON 格式: {json_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python results_exporter.py <task_id>")
        print("示例: python results_exporter.py kissing_number_optimized_5d")
        sys.exit(1)
    
    task_id = sys.argv[1]
    export_results(task_id)

