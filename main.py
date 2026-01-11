"""
Main entry point for the AlphaEvolve Pro application.
Orchestrates the different agents and manages the evolutionary loop.
"""
import asyncio
import logging
import sys
import os
import yaml
import argparse
import warnings

# 抑制 LiteLLM 相关的 Pydantic 序列化警告（不影响功能，只是日志噪音）
# 这些警告来自 LiteLLM 内部使用 Pydantic 模型时的序列化问题
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*Pydantic.*serializer.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)
                                               
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
from config import settings

                   
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)

def load_task_from_yaml(yaml_path: str) -> tuple[list, str, str, str, list, str]:
    """Load task configuration and test cases from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # Get task configuration
            task_id = data.get('task_id')
            task_description = data.get('task_description')
            function_name = data.get('function_name')
            allowed_imports = data.get('allowed_imports', [])
            expert_knowledge = data.get('expert_knowledge', '')  # 加载专家知识
            
            # Convert test cases from YAML format to input_output_examples format
            input_output_examples = []
            for test_group in data.get('tests', []):
                for test_case in test_group.get('test_cases', []):
                    if 'output' in test_case:
                        input_output_examples.append({
                            'input': test_case['input'],
                            'output': test_case['output']
                        })
                    elif 'validation_func' in test_case:
                        input_output_examples.append({
                            'input': test_case['input'],
                            'validation_func': test_case['validation_func']
                        })
            
            return input_output_examples, task_id, task_description, function_name, allowed_imports, expert_knowledge
    except Exception as e:
        logger.error(f"Error loading task from YAML: {e}")
        return [], "", "", "", [], ""

async def main():
    parser = argparse.ArgumentParser(description="Run OpenAlpha_Evolve with a specified YAML configuration file.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    yaml_path = args.yaml_path

    logger.info("Starting OpenAlpha_Evolve autonomous algorithmic evolution")
    logger.info(f"Configuration: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")

    # Load task configuration and test cases from YAML file
    test_cases, task_id, task_description, function_name, allowed_imports, expert_knowledge = load_task_from_yaml(yaml_path)
    
    if not task_id or not task_description or not function_name:
        logger.error("Missing required task configuration in YAML file. Exiting.")
        return
    
    # 重新加载 YAML 以获取完整的 tests 结构（保留 level 信息）
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        tests_structure = yaml_data.get('tests', [])
        # 如果 expert_knowledge 为空，尝试从重新加载的数据中获取
        if not expert_knowledge:
            expert_knowledge = yaml_data.get('expert_knowledge', '')
    except Exception as e:
        logger.warning(f"Failed to load tests structure from YAML: {e}. Using input_output_examples only.")
        tests_structure = []
    
    task = TaskDefinition(
        id=task_id,
        description=task_description,
        function_name_to_evolve=function_name,
        input_output_examples=test_cases if test_cases else None,  # 保持向后兼容
        tests=tests_structure if tests_structure else None,  # 使用新的 tests 结构（优先）
        allowed_imports=allowed_imports,
        expert_knowledge=expert_knowledge if expert_knowledge else None  # 添加专家知识
    )
    
    logger.info(f"Loaded task '{task_id}' with {len(tests_structure) if tests_structure else 0} test groups and {len(test_cases)} total test cases")

    task_manager = TaskManagerAgent(
        task_definition=task
    )

    best_programs = await task_manager.execute()

    if best_programs:
        logger.info(f"Evolutionary process completed. Best program(s) found: {len(best_programs)}")
        for i, program in enumerate(best_programs):
            logger.info(f"Final Best Program {i+1} ID: {program.id}")
            logger.info(f"Final Best Program {i+1} Fitness: {program.fitness_scores}")
            
            # 如果是 kissing number 任务，显示详细信息
            kissing_5d = program.fitness_scores.get("kissing_number_5d")
            if kissing_5d is not None and kissing_5d > 0:
                sota_score = program.fitness_scores.get("sota_score_5d", 0.0)
                is_valid = program.fitness_scores.get("kissing_number_5d_valid", 0.0) == 1.0
                logger.info(f"Final Best Program {i+1} - 5D Kissing Number: {int(kissing_5d)}")
                logger.info(f"Final Best Program {i+1} - SOTA Score: {sota_score:.4f} (Valid: {is_valid})")
            
            logger.info(f"Final Best Program {i+1} Code:\n{program.code}")
    else:
        logger.info("Evolutionary process completed, but no suitable programs were found.")

    logger.info("OpenAlpha_Evolve run finished.")
    logger.info("=" * 80)
    logger.info("📊 结果查看:")
    logger.info(f"  1. 详细日志: {settings.LOG_FILE}")
    logger.info(f"  2. 数据库: {settings.DATABASE_PATH}")
    logger.info(f"  3. 结果文件: results/{task_id}_results_*.txt 和 *.json")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
