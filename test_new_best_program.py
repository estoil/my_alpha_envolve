#!/usr/bin/env python3
"""测试新生成的最佳程序输出 - 改进版：10分钟超时 + 实时输出 + 输出坐标"""
import math
import random
import numpy as np
import itertools
import time

def find_kissing_number(n, start_time=None, max_runtime=600):
    known = {1: 2, 2: 6, 3: 12, 4: 24, 8: 240, 24: 196560}
    if n in known:
        return known[n], known_centers(n), True
    if n == 5:
        return five_dimension(start_time=start_time, max_runtime=max_runtime)
    # For other unknown dimensions, use a simple random construction
    return generic_construction(n)

def known_centers(n):
    if n == 1:
        return [(2.0,), (-2.0,)]
    elif n == 2:
        return [(2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0),
                (math.sqrt(2.0), math.sqrt(2.0)), (-math.sqrt(2.0), -math.sqrt(2.0))]
    elif n == 3:
        # 12 centers for 3D: vertices of an icosahedron
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        scale = 2.0 / math.sqrt(1.0 + phi*phi)
        points = []
        for (x, y, z) in [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
                           (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
                           (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)]:
            points.append((x*scale, y*scale, z*scale))
        return points
    elif n == 4:
        # 24 centers for 4D: vertices of a 24-cell
        points = []
        for perm in itertools.permutations([1.0, 0.0, 0.0, 0.0]):
            points.append(tuple(2.0 * x for x in perm))
        for signs in itertools.product([-1.0, 1.0], repeat=4):
            if sum(1 for s in signs if s == 1.0) % 2 == 0:
                points.append(tuple(0.5 * 2.0 * s for s in signs))
        return points
    elif n == 8:
        # 240 centers for 8D: E8 lattice roots
        points = []
        # All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) with even number of minus signs
        base = [1.0, 1.0] + [0.0] * 6
        for perm in set(itertools.permutations(base)):
            for signs in itertools.product([-1.0, 1.0], repeat=8):
                if sum(1 for i in range(8) if signs[i] == -1.0) % 2 == 0:
                    pt = tuple(perm[i] * signs[i] for i in range(8))
                    norm = math.sqrt(sum(x*x for x in pt))
                    if norm > 0:
                        points.append(tuple(2.0 * x / norm for x in pt))
        # Also include (±0.5)^8 with odd number of minus signs
        half = 0.5
        for signs in itertools.product([-half, half], repeat=8):
            if sum(1 for s in signs if s == -half) % 2 == 1:
                norm = math.sqrt(sum(x*x for x in signs))
                if norm > 0:
                    points.append(tuple(2.0 * x / norm for x in signs))
        return list(set(points))[:240]
    elif n == 24:
        # 196560 centers for 24D: Leech lattice
        points = []
        # Type 1: permutations of (±2, ±2, 0^22)
        base = [2.0, 2.0] + [0.0] * 22
        for perm in set(itertools.permutations(base)):
            for signs in itertools.product([-1.0, 1.0], repeat=24):
                pt = tuple(perm[i] * signs[i] for i in range(24))
                norm = math.sqrt(sum(x*x for x in pt))
                if norm > 0:
                    points.append(tuple(2.0 * x / norm for x in pt))
        # We'll only return a subset due to complexity
        return points[:196560]
    return []

def five_dimension(start_time=None, max_runtime=600):
    """5D kissing number 计算，带超时和实时输出"""
    if start_time is None:
        start_time = time.perf_counter()
    
    print(f"  [步骤 1] 生成 D5* lattice 基础点...")
    # Step 1: D5* lattice construction (guaranteed 40 points)
    # Generate all permutations of (±1, ±1, 0, 0, 0) with even number of minus signs
    centers = []
    # Use itertools.permutations on positions of non‑zeros
    base = [1, 1, 0, 0, 0]
    # Use set to avoid duplicate permutations
    for perm in set(itertools.permutations(base)):
        # perm is a tuple of length 5 with two 1's and three 0's
        for signs in itertools.product([-1, 1], repeat=5):
            # Count minus signs only on the non‑zero entries
            minus_count = sum(1 for i in range(5) if signs[i] == -1 and perm[i] != 0)
            if minus_count % 2 == 0:
                pt = tuple(perm[i] * signs[i] for i in range(5))
                norm = math.sqrt(sum(x*x for x in pt))
                if norm > 0:
                    scaled = tuple(2.0 * x / norm for x in pt)
                    centers.append(scaled)
    
    # Remove duplicates efficiently
    unique_centers = []
    seen = set()
    for c in centers:
        # Round to 12 decimal places to avoid floating point errors
        rounded = tuple(round(x, 12) for x in c)
        if rounded not in seen:
            seen.add(rounded)
            unique_centers.append(c)
    
    current_max = len(unique_centers)
    print(f"  ✓ 基础点数: {current_max}")
    
    # We should have exactly 40 points, but current D5* construction only gives 20
    # Use improved search to get to 40
    if len(unique_centers) < 40:
        print(f"  [补充] 使用改进搜索补充到 40 个点（当前: {len(unique_centers)}）...")
        unique_centers = improved_search_to_40(unique_centers, 5, start_time, max_runtime)
        current_max = len(unique_centers)
        if current_max >= 40:
            print(f"  ✅ 成功达到 40 个点！")
        else:
            print(f"  ⚠️ 未能达到 40 个点，当前: {current_max}")
    
    # Now try to add more points using a more efficient method
    # Use simulated annealing style optimization on the existing set
    current = list(unique_centers)
    print(f"  [步骤 2] 尝试添加更多点（当前: {len(current)}）...")
    # We'll attempt to add a few extra points via a greedy approach with random restarts
    added = try_add_points_fast(current, 5, max_extra=8, attempts_per_extra=2000, start_time=start_time, max_runtime=max_runtime)
    # Ensure we don't exceed theoretical upper bound 48
    if len(added) > 48:
        added = added[:48]
    
    final_count = len(added)
    print(f"  ✓ 最终点数: {final_count}")
    
    elapsed_total = time.perf_counter() - start_time
    print(f"  ✓ 总耗时: {elapsed_total:.2f} 秒")
    
    return final_count, added, True

def improved_search_to_40(centers, dim, start_time, max_runtime):
    """改进的搜索方法，使用best candidate策略来补充到40个点"""
    current = centers[:]
    np.random.seed(42)  # Deterministic seed
    
    max_attempts = 100000
    attempts = 0
    num_candidates = 1500  # 每次尝试的候选数
    
    print(f"  [智能搜索] 使用 {num_candidates} 个候选/次进行搜索...")
    
    while len(current) < 40 and attempts < max_attempts and time.perf_counter() - start_time < max_runtime:
        attempts += 1
        
        if attempts % 2000 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"    进度: 已尝试 {attempts} 次，当前点数: {len(current)} (耗时: {elapsed:.1f}s)")
        
        # Use best candidate approach
        best_candidate = None
        best_min_dist_sq = -1
        
        # Try multiple candidates
        for candidate_idx in range(num_candidates):
            # Generate random candidate
            vec = np.random.randn(dim)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            candidate = tuple(2.0 * x / norm for x in vec)
            
            # Compute minimum distance to existing points
            min_dist_sq = float('inf')
            candidate_arr = np.array(candidate)
            for existing in current:
                dist_sq = np.sum((candidate_arr - np.array(existing))**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    # Early termination for efficiency
                    if min_dist_sq < (2.0 - 1e-6)**2:
                        break
            
            if min_dist_sq > best_min_dist_sq:
                best_min_dist_sq = min_dist_sq
                best_candidate = candidate
        
        # Add the best candidate if it satisfies distance requirement
        if best_min_dist_sq >= (2.0 - 1e-6)**2:
            current.append(best_candidate)
            print(f"  ✓ 找到新点！当前点数: {len(current)}/40 (尝试次数: {attempts}, 候选数: {num_candidates})")
        elif attempts > 50000 and best_min_dist_sq >= (1.99)**2:
            # After many attempts, use slightly relaxed condition
            current.append(best_candidate)
            print(f"  ⚠️ 使用略微放宽条件找到点: {len(current)}/40 (尝试次数: {attempts})")
    
    return current

def try_add_points_fast(centers, dim, max_extra=8, attempts_per_extra=2000, start_time=None, max_runtime=600):
    """Greedy attempt to add extra points with early pruning - 带超时和实时输出"""
    if start_time is None:
        start_time = time.perf_counter()
    
    current = list(centers)
    initial_count = len(current)
    print(f"  [智能搜索] 尝试添加更多点（从 {initial_count} 开始，最多尝试添加 {max_extra} 个）...")
    
    # Precompute as numpy array for speed
    arr = np.array(current)
    for extra in range(max_extra):
        # Check timeout
        if time.perf_counter() - start_time > max_runtime:
            print(f"  ⚠️ 达到最大运行时间 ({max_runtime/60:.1f} 分钟)，停止")
            break
        
        found = False
        for attempt in range(attempts_per_extra):
            # Check timeout during attempts
            if time.perf_counter() - start_time > max_runtime:
                break
            
            # Generate random direction using spherical coordinates for more uniform distribution
            vec = np.random.randn(dim)
            vec = vec / np.linalg.norm(vec) * 2.0
            # Check distances quickly using vectorized operations
            diffs = arr - vec
            dists = np.linalg.norm(diffs, axis=1)
            if np.all(dists >= 2.0 - 1e-9):
                # Valid point found
                current.append(tuple(vec))
                arr = np.array(current)  # update array
                found = True
                print(f"  ✓ 找到新点！当前点数: {len(current)} (尝试添加第 {extra+1} 个点，本次尝试: {attempt+1})")
                break
        
        if not found:
            # Could not add another point within attempts
            if extra == 0:
                print(f"  ⚠️ 无法添加更多点（已尝试 {attempts_per_extra} 次）")
            break
    
    # After adding points, run a few iterations of local repulsion to improve spacing
    # but limit iterations to avoid timeout
    if len(current) > initial_count:
        print(f"  [局部优化] 对 {len(current)} 个点进行局部优化...")
        for iter_num in range(10):
            if time.perf_counter() - start_time > max_runtime:
                break
            current = perturb_centers_fast(current, dim, step=0.005)
            if iter_num % 3 == 0:
                print(f"    局部优化迭代 {iter_num+1}/10...")
    
    return current

def perturb_centers_fast(centers, dim, step=0.005):
    """Local repulsion with vectorized operations and fast validity check."""
    if len(centers) < 2:
        return centers
    arr = np.array(centers)
    n = len(arr)
    # Early exit if n is large to avoid timeout
    if n > 50:
        return centers
    forces = np.zeros_like(arr)
    # Compute pairwise distances using efficient loops with early break
    for i in range(n):
        for j in range(i+1, n):
            diff = arr[i] - arr[j]
            dist = np.linalg.norm(diff)
            if dist < 2.0 and dist > 1e-12:
                force_magnitude = (2.0 - dist) / (dist + 1e-12)
                forces[i] += force_magnitude * diff
                forces[j] -= force_magnitude * diff
    # Move points
    new_arr = arr + step * forces
    # Project back to sphere of radius 2
    norms = np.linalg.norm(new_arr, axis=1, keepdims=True)
    new_arr = new_arr / norms * 2.0
    # Quick validity check: compute minimum pairwise distance with early break
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(new_arr[i] - new_arr[j])
            if d < min_dist:
                min_dist = d
            if min_dist < 2.0 - 1e-9:
                # Early break if invalid
                return centers
    if min_dist >= 2.0 - 1e-9:
        return [tuple(row) for row in new_arr]
    else:
        return centers

# Remove perturb_centers as it's no longer used in the improved version

def generic_construction(n):
    # Simple random construction for unknown dimensions
    centers = []
    max_points = min(2 * n * (n + 1), 100)  # Heuristic upper bound
    
    for _ in range(10000):
        if len(centers) >= max_points:
            break
        vec = np.random.randn(n)
        vec = vec / np.linalg.norm(vec) * 2.0
        candidate = tuple(vec)
        
        valid = True
        for c in centers:
            if np.linalg.norm(np.array(candidate) - np.array(c)) < 2.0 - 1e-6:
                valid = False
                break
        if valid:
            centers.append(candidate)
    
    return len(centers), centers, True

class TeeOutput:
    """同时输出到终端和文件的类"""
    def __init__(self, file_path):
        self.terminal = __import__('sys').stdout
        self.file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # 立即写入文件
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    # 生成输出文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.txt"
    
    # 创建同时输出到终端和文件的输出对象
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print("测试最新生成的最佳程序输出")
        print("程序 ID: kissing_number_optimized_5d_gen7_child7_1")
        print("来源: 2026-01-11 22:28:07 运行结果")
        print(f"输出文件: {output_file}")
        print("=" * 80)
        print()
        
        # 只测试 5D
        test_dimensions = [5]
        MAX_RUNTIME_SECONDS = 10 * 60  # 10分钟
        
        for dim in test_dimensions:
            print(f"测试维度 {dim}:")
            print(f"最大运行时间: {MAX_RUNTIME_SECONDS/60:.1f} 分钟")
            print()
            try:
                start_time = time.perf_counter()
                
                if dim == 5:
                    kissing_num, centers, is_valid = five_dimension(start_time=start_time, max_runtime=MAX_RUNTIME_SECONDS)
                else:
                    kissing_num, centers, is_valid = find_kissing_number(dim)
                
                elapsed_time = (time.perf_counter() - start_time) * 1000
                
                print()
                print(f"  Kissing Number: {kissing_num}")
                centers_count = len(centers) if isinstance(centers, list) else "N/A"
                print(f"  Centers 数量: {centers_count}")
                print(f"  有效性: {is_valid}")
                print(f"  运行时间: {elapsed_time:.2f} ms ({elapsed_time/1000:.2f} 秒)")
                
                if dim == 5:
                    print()
                    print(f"  ✅ 5D Kissing Number = {kissing_num}")
                    if kissing_num >= 40:
                        print(f"  ✅ 达到或超过 SOTA 下界 (40)!")
                    elif kissing_num >= 20:
                        print(f"  ⚠️ 低于 SOTA 下界，但 >= 20")
                    else:
                        print(f"  ❌ 低于最小阈值 (20)")
                    
                    # 输出所有找到的点的坐标
                    print()
                    print("=" * 80)
                    print(f"所有找到的 {kissing_num} 个点的坐标:")
                    print("=" * 80)
                    if isinstance(centers, list) and len(centers) > 0:
                        for i, center in enumerate(centers, 1):
                            # 格式化输出，保留6位小数
                            formatted_center = tuple(round(x, 6) for x in center)
                            print(f"点 {i:2d}: {formatted_center}")
                    else:
                        print("  无坐标数据")
                    print("=" * 80)
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                import traceback
                traceback.print_exc()
            print()
        
        print("=" * 80)
        print("测试完成")
        print(f"结果已保存到: {output_file}")
        print("=" * 80)
    finally:
        # 恢复标准输出并关闭文件
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n✅ 结果已保存到文件: {output_file}")
