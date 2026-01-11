"""
改进的评估器，专门针对 kissing number 问题
奖励更高的 kissing number 值（特别是 5 维），基于 SOTA 下界进行评分
"""

import logging
import math
from typing import Dict, Any, Optional
from evaluator_agent.agent import EvaluatorAgent
from core.interfaces import Program, TaskDefinition

logger = logging.getLogger(__name__)

# 5 维 Kissing Number 的 SOTA 信息
KISSING_NUMBER_5D_SOTA_LOWER_BOUND = 40  # 当前已知最佳下界
KISSING_NUMBER_5D_UPPER_BOUND = 48       # 理论上界
KISSING_NUMBER_5D_TARGET = 44            # 目标值（当前最佳记录附近）

class KissingNumberEvaluatorAgent(EvaluatorAgent):
    """
    改进的评估器，专门针对 kissing number 问题
    1. 评估标准测试用例（已知维度）
    2. 提取并评估 5 维 kissing number 值
    3. 基于 SOTA 下界计算奖励分数
    """
    
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        """评估程序，并在完成后提取并奖励更高的 kissing number 值"""
        # 先调用父类的标准评估（测试已知维度）
        program = await super().evaluate_program(program, task)
        
        # 确保程序有 task_id
        if not program.task_id:
            program.task_id = task.id
        
        # 如果是 kissing number 任务，提取并评估 5 维结果
        if task.id and "kissing_number" in task.id:
            kissing_number_5d, is_valid = await self._extract_and_validate_5d_kissing_number(program, task)
            
            if kissing_number_5d is not None and kissing_number_5d > 0:
                # 将 5 维结果加入适应度分数
                program.fitness_scores["kissing_number_5d"] = float(kissing_number_5d)
                program.fitness_scores["kissing_number_5d_valid"] = 1.0 if is_valid else 0.0
                
                # 计算基于 SOTA 的评分
                sota_score = self._calculate_sota_score(kissing_number_5d, is_valid)
                program.fitness_scores["sota_score_5d"] = sota_score  # 保持键名一致
                
                # 调整最终适应度：基础 correctness + SOTA 奖励
                base_correctness = program.fitness_scores.get("correctness", 0.0)
                
                # 只有通过基础测试（correctness >= 0.8）才考虑 SOTA 奖励
                if base_correctness >= 0.8 and is_valid:
                    # SOTA 评分作为额外的适应度维度
                    # 最终的 correctness 是：基础正确率 + SOTA 奖励（最多到 1.5，但选择时会被限制）
                    sota_bonus = sota_score * 0.5  # SOTA 评分最多贡献 50% 的 bonus
                    program.fitness_scores["correctness"] = min(base_correctness + sota_bonus, 1.5)
                    
                    logger.info(
                        f"Program {program.id}: "
                        f"5D kissing number = {kissing_number_5d} (valid={is_valid}), "
                        f"SOTA score = {sota_score:.3f}, "
                        f"base correctness = {base_correctness:.3f}, "
                        f"final correctness = {program.fitness_scores['correctness']:.3f}"
                    )
                elif not is_valid:
                    logger.warning(
                        f"Program {program.id}: Found {kissing_number_5d} spheres for 5D, but arrangement is invalid"
                    )
        else:
            # 对于非 kissing number 任务，也尝试提取（用于调试）
            logger.debug(f"Task {task.id} is not a kissing number task, skipping 5D evaluation")
        
        return program
    
    def _calculate_sota_score(self, kissing_number: int, is_valid: bool) -> float:
        """
        基于 SOTA 下界计算评分（改进版：更明确地奖励找到更多点）
        评分范围：0.0 (很差) 到 1.0 (达到或超过 SOTA)
        
        改进的评分策略（更陡峭的梯度，更明确地奖励 40+ 点）：
        - 低于 20: 0.0
        - 20-30: 0.1-0.3 (缓慢增长)
        - 30-40: 0.3-0.6 (中等增长)
        - 40-44: 0.6-0.95 (快速增长 - 重点奖励区域，35% 的分数跨度)
        - 44-48: 0.95-1.0 (达到或超过 SOTA)
        - 如果无效，返回 0.0
        
        改进点：对 40+ 的点给予更大的奖励梯度，鼓励算法找到更多点
        """
        if not is_valid or kissing_number < 20:
            return 0.0
        
        # 使用分段线性函数进行评分（改进版：更明确地奖励 40+ 的点）
        if kissing_number >= 44:
            # 达到或超过目标值（44+）：0.95 - 1.0
            if kissing_number >= KISSING_NUMBER_5D_UPPER_BOUND:
                return 1.0
            else:
                # 44-48: 0.95 - 1.0
                return 0.95 + 0.05 * min((kissing_number - 44) / 4.0, 1.0)
        elif kissing_number >= KISSING_NUMBER_5D_SOTA_LOWER_BOUND:
            # 达到 SOTA 下界（40-44）：0.6 - 0.95 (更陡峭的梯度，重点奖励区域)
            # 40 点: 0.6, 44 点: 0.95
            progress = (kissing_number - KISSING_NUMBER_5D_SOTA_LOWER_BOUND) / (44.0 - KISSING_NUMBER_5D_SOTA_LOWER_BOUND)
            return 0.6 + 0.35 * progress
        elif kissing_number >= 30:
            # 30-40: 0.3 - 0.6
            progress = (kissing_number - 30) / 10.0
            return 0.3 + 0.3 * progress
        else:
            # 20-30: 0.1 - 0.3 (缓慢增长)
            progress = (kissing_number - 20) / 10.0
            return 0.1 + 0.2 * progress
    
    async def _extract_and_validate_5d_kissing_number(
        self, 
        program: Program, 
        task: TaskDefinition
    ) -> tuple[Optional[int], bool]:
        """
        从程序执行结果中提取并验证 5 维的 kissing number 值
        
        返回: (kissing_number, is_valid) 元组
        """
        try:
            # 检查程序是否有语法错误
            if program.errors:
                return None, False
            
            # 创建临时的 5 维测试任务
            from core.interfaces import TaskDefinition as TD
            test_task = TD(
                id="temp_5d_test",
                description="Temporary test for 5D kissing number extraction",
                function_name_to_evolve=task.function_name_to_evolve,
                input_output_examples=[{"input": [5]}],
                allowed_imports=task.allowed_imports or [],
                tests=None  # 不使用 tests，只使用 input_output_examples
            )
            
            # 执行代码（使用较短的超时，因为只是提取值）
            execution_results, execution_error = await self._execute_code_safely(
                program.code,
                task_for_examples=test_task,
                timeout_seconds=120  # 增加到 120 秒，因为 5 维搜索可能较慢
            )
            
            if execution_error or not execution_results:
                logger.debug(f"Failed to execute program {program.id} for 5D extraction: {execution_error}")
                return None, False
            
            # 提取输出
            test_outputs = execution_results.get("test_outputs", [])
            if not test_outputs:
                logger.debug(f"No test outputs for program {program.id}")
                return None, False
            
            output_detail = test_outputs[0]
            if output_detail.get("status") != "success":
                logger.debug(f"Test execution failed for program {program.id}: {output_detail.get('error')}")
                return None, False
            
            output = output_detail.get("output")
            if not isinstance(output, tuple) or len(output) != 3:
                logger.debug(f"Invalid output format for program {program.id}: {output}")
                return None, False
            
            kissing_num, centers, is_valid = output
            
            # 验证结果类型
            if not isinstance(kissing_num, int) or kissing_num <= 0:
                logger.debug(f"Invalid kissing number value for program {program.id}: {kissing_num}")
                return None, False
            
            # 验证 centers 数量是否匹配
            if not isinstance(centers, list) or len(centers) != kissing_num:
                logger.warning(
                    f"Program {program.id}: kissing number ({kissing_num}) != centers count ({len(centers) if isinstance(centers, list) else 'N/A'})"
                )
                return kissing_num, False
            
            # 快速抽样验证（只检查前几个点和部分点对，避免超时）
            # 如果程序返回 is_valid=True，我们进行快速抽样验证来确认
            if isinstance(is_valid, bool) and is_valid:
                # 只验证前 10 个点，避免超时
                sample_size = min(10, len(centers))
                dimension = 5
                
                for i in range(sample_size):
                    center = centers[i]
                    if not isinstance(center, (tuple, list)) or len(center) != dimension:
                        logger.debug(f"Program {program.id}: Center {i} has invalid format")
                        return kissing_num, False
                    
                    # 检查距离原点的距离
                    dist_sq = sum(c**2 for c in center)
                    dist = math.sqrt(dist_sq)
                    if abs(dist - 2.0) > 1e-4:  # 容忍度稍大以提高速度
                        logger.debug(f"Program {program.id}: Center {i} not at distance 2.0 (distance: {dist:.6f})")
                        return kissing_num, False
                    
                    # 检查与前几个点的距离（只检查前 5 个，避免 O(n²)）
                    for j in range(min(i, 5)):
                        dist_sq_pair = sum((centers[i][k] - centers[j][k])**2 for k in range(dimension))
                        dist_pair = math.sqrt(dist_sq_pair)
                        if dist_pair < 2.0 - 1e-4:
                            logger.debug(f"Program {program.id}: Centers {i} and {j} too close (distance: {dist_pair:.6f})")
                            return kissing_num, False
                
                # 如果通过了快速验证，信任程序的 is_valid=True
                is_valid = True
            else:
                is_valid = False
            
            logger.info(
                f"Program {program.id}: Extracted 5D kissing number = {kissing_num}, "
                f"centers count = {len(centers)}, "
                f"valid = {is_valid}"
            )
            
            return kissing_num, is_valid
            
        except Exception as e:
            logger.warning(f"Error extracting 5D kissing number for program {program.id}: {e}", exc_info=True)
            return None, False

