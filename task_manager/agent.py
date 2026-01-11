import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, BaseAgent,
    PromptDesignerInterface, CodeGeneratorInterface, EvaluatorAgentInterface,
    DatabaseAgentInterface, SelectionControllerInterface
)
from config import settings

from prompt_designer.agent import PromptDesignerAgent
from code_generator.agent import CodeGeneratorAgent
from evaluator_agent.agent import EvaluatorAgent
from evaluator_agent.kissing_number_evaluator import KissingNumberEvaluatorAgent
from database_agent.agent import InMemoryDatabaseAgent
from selection_controller.agent import SelectionControllerAgent

logger = logging.getLogger(__name__)

class TaskManagerAgent(TaskManagerInterface):
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        self.prompt_designer: PromptDesignerInterface = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeGeneratorAgent()
        
        # 如果是 kissing number 任务，使用改进的评估器
        if task_definition.id and "kissing_number" in task_definition.id:
            self.evaluator: EvaluatorAgentInterface = KissingNumberEvaluatorAgent(task_definition=self.task_definition)
            logger.info(f"Using KissingNumberEvaluatorAgent for task: {task_definition.id}")
        else:
            self.evaluator: EvaluatorAgentInterface = EvaluatorAgent(task_definition=self.task_definition)
            logger.info(f"Using standard EvaluatorAgent for task: {task_definition.id}")
        
        # Simplified database initialization:
        self.database: DatabaseAgentInterface = InMemoryDatabaseAgent()
        logger.info(f"Using {settings.DATABASE_TYPE} database (InMemoryDatabaseAgent).")
            
        self.selection_controller: SelectionControllerInterface = SelectionControllerAgent()

        self.population_size = settings.POPULATION_SIZE
        self.num_generations = settings.GENERATIONS
        self.num_parents_to_select = self.population_size // 2
        self.num_islands = settings.NUM_ISLANDS
        self.programs_per_island = self.population_size // self.num_islands

    async def initialize_population(self) -> List[Program]:
        logger.info(f"Initializing population for task: {self.task_definition.id}")
        initial_population = []
        
        initial_model = settings.LLM_SECONDARY_MODEL # Use secondary for broad initial generation
        logger.info(f"Using model '{initial_model}' for initial population generation.")

        # Generate initial programs
        tasks = []
        for i in range(self.population_size):
            initial_prompt = self.prompt_designer.design_initial_prompt()
            tasks.append(self.code_generator.generate_code(initial_prompt, model_name=initial_model, temperature=0.8))

        generated_codes = await asyncio.gather(*tasks)
        for i, generated_code in enumerate(generated_codes):
            program_id = f"{self.task_definition.id}_gen0_prog{i}"
            logger.debug(f"Generated initial program {i+1}/{self.population_size} with id {program_id}")
            program = Program(
                id=program_id,
                code=generated_code,
                generation=0,
                status="unevaluated",
                task_id=self.task_definition.id
            )
            initial_population.append(program)
            await self.database.save_program(program)

        # Initialize islands with the initial population
        self.selection_controller.initialize_islands(initial_programs=initial_population)
        
        logger.info(f"Initialized population with {len(initial_population)} programs across {self.num_islands} islands.")
        return initial_population

    async def evaluate_population(self, population: List[Program]) -> List[Program]:
        logger.info(f"Evaluating population of {len(population)} programs.")
        evaluated_programs = []
        evaluation_tasks = [self.evaluator.evaluate_program(prog, self.task_definition) for prog in population if prog.status != "evaluated"]
        
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            original_program = population[i]
            if isinstance(result, Exception):
                logger.error(f"Error evaluating program {original_program.id}: {result}", exc_info=result)
                original_program.status = "failed_evaluation"
                original_program.errors.append(str(result))
                evaluated_programs.append(original_program)
            else:
                evaluated_programs.append(result)
            await self.database.save_program(evaluated_programs[-1])
            
        logger.info(f"Finished evaluating population. {len(evaluated_programs)} programs processed.")
        return evaluated_programs

    async def manage_evolutionary_cycle(self):
        logger.info(f"Starting evolutionary cycle for task: {self.task_definition.description[:50]}...")
        current_population = await self.initialize_population()
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            # Select parents from islands
            parents = self.selection_controller.select_parents(current_population, self.num_parents_to_select)
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected. Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents.")

            # Generate offspring
            offspring_population = []
            num_offspring_per_parent = (self.population_size + len(parents) - 1) // len(parents)
            
            generation_tasks = []
            for i, parent in enumerate(parents):
                for j in range(num_offspring_per_parent):
                    child_id = f"{self.task_definition.id}_gen{gen}_child{i}_{j}"
                    generation_tasks.append(self.generate_offspring(parent, gen, child_id))
            
            generated_offspring_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            for result in generated_offspring_results:
                if isinstance(result, Exception):
                    logger.error(f"Error generating offspring: {result}", exc_info=result)
                elif result:
                    offspring_population.append(result)
                    await self.database.save_program(result)

            logger.info(f"Generation {gen}: Generated {len(offspring_population)} offspring.")
            if not offspring_population:
                logger.warning(f"Generation {gen}: No offspring generated. May indicate issues with LLM or prompting.")
                if not parents:
                    break

            # Evaluate offspring
            offspring_population = await self.evaluate_population(offspring_population)

            # Select survivors using island model
            current_population = self.selection_controller.select_survivors(current_population, offspring_population, self.population_size)
            logger.info(f"Generation {gen}: New population size: {len(current_population)}.")

            # Log best program in this generation (using enhanced fitness key for kissing_number tasks)
            if current_population:
                best_program_this_gen = sorted(
                    current_population,
                    key=self.selection_controller._get_fitness_key,
                    reverse=True
                )[0]
                
                fitness_str = f"Fitness={best_program_this_gen.fitness_scores}"
                # 如果是 kissing number 任务，显示额外的信息
                if best_program_this_gen.fitness_scores.get("kissing_number_5d", 0) > 0:
                    kissing_5d = int(best_program_this_gen.fitness_scores.get("kissing_number_5d", 0))
                    sota_score = best_program_this_gen.fitness_scores.get("sota_score_5d", 0.0)
                    fitness_str += f", 5D Kissing Number={kissing_5d}, SOTA Score={sota_score:.3f}"
                
                logger.info(f"Generation {gen}: Best program: ID={best_program_this_gen.id}, {fitness_str}")
            else:
                logger.warning(f"Generation {gen}: No programs in current population after survival selection.")
                break

        logger.info("Evolutionary cycle completed.")
        
        # 获取最佳程序（使用增强的排序键）
        # 优先使用当前种群，否则从数据库加载
        if current_population and len(current_population) > 0:
            task_programs = current_population
            logger.info(f"Using current population ({len(task_programs)} programs) for best program selection")
        else:
            # 如果当前种群为空，从数据库加载
            all_programs = await self.database.get_all_programs()
            task_programs = [
                p for p in all_programs 
                if (self.task_definition.id in p.id or 
                    self.task_definition.id in getattr(p, 'task_id', '') or
                    ('_gen' in p.id and self.task_definition.id in p.id.split('_gen')[0]))
            ]
            logger.info(f"Loaded {len(task_programs)} programs from database for task {self.task_definition.id}")
        
        if task_programs:
            # 使用增强的适应度键排序
            sorted_programs = sorted(
                task_programs,
                key=self.selection_controller._get_fitness_key,
                reverse=True
            )
            final_best = sorted_programs[:1]  # 只返回最佳的一个
            
            if final_best:
                best_prog = final_best[0]
                logger.info(f"Overall Best Program: {best_prog.id}")
                logger.info(f"Fitness: {best_prog.fitness_scores}")
                
                # 如果是 kissing number 任务，显示详细信息
                kissing_5d = best_prog.fitness_scores.get("kissing_number_5d")
                if kissing_5d is not None and kissing_5d > 0:
                    sota_score = best_prog.fitness_scores.get("sota_score_5d", 0.0)
                    is_valid = best_prog.fitness_scores.get("kissing_number_5d_valid", 0.0) == 1.0
                    logger.info(f"5D Kissing Number: {int(kissing_5d)} (SOTA Score: {sota_score:.4f}, Valid: {is_valid})")
                    
                    # 与 SOTA 比较
                    diff = int(kissing_5d) - 40
                    if diff >= 0:
                        logger.info(f"相对 SOTA 下界 (40): +{diff} ({'✅ 达到或超过' if diff >= 4 else '✅ 接近'} SOTA)")
                    else:
                        logger.info(f"相对 SOTA 下界 (40): {diff} (低于 SOTA)")
                
                logger.info(f"Code:\n{best_prog.code}")
                
                # 自动导出结果
                try:
                    from results_exporter import export_results
                    export_results(self.task_definition.id)
                    logger.info("Results exported to 'results/' directory")
                except Exception as e:
                    logger.warning(f"Failed to export results: {e}")
        else:
            logger.info("No programs found for this task.")
            final_best = []
        
        return final_best

    async def generate_offspring(self, parent: Program, generation_num: int, child_id: str) -> Optional[Program]:
        logger.debug(f"Generating offspring from parent {parent.id} for generation {generation_num}")
        
        prompt_type = "mutation"
        # Default to primary for bug fixes or high-fitness mutations
        chosen_model = settings.LLM_PRIMARY_MODEL 

        if parent.errors and parent.fitness_scores.get("correctness", 1.0) < settings.BUG_FIX_CORRECTNESS_THRESHOLD:
            primary_error = parent.errors[0]
            execution_details = None
            if len(parent.errors) > 1 and isinstance(parent.errors[1], str) and ("stdout" in parent.errors[1].lower() or "stderr" in parent.errors[1].lower()):
                execution_details = parent.errors[1]
            
            mutation_prompt = self.prompt_designer.design_bug_fix_prompt(
                program=parent,
                error_message=primary_error,
                execution_output=execution_details
            )
            logger.info(f"Attempting bug fix for parent {parent.id} using diff. Model: {chosen_model}. Error: {primary_error}")
            prompt_type = "bug_fix"
            # chosen_model remains LLM_PRIMARY_MODEL for bug fixing
        else:
            # Decide model for mutation based on parent fitness
            if parent.fitness_scores.get("correctness", 0.0) >= settings.HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM:
                # chosen_model is already LLM_PRIMARY_MODEL
                logger.info(f"Parent {parent.id} has high fitness, using primary model {chosen_model} for mutation.")
            else:
                chosen_model = settings.LLM_SECONDARY_MODEL # Use secondary for lower-fitness mutations
                logger.info(f"Parent {parent.id} has lower fitness, using secondary model {chosen_model} for mutation.")

            feedback = {
                "errors": parent.errors,
                "correctness_score": parent.fitness_scores.get("correctness"),
                "runtime_ms": parent.fitness_scores.get("runtime_ms")
            }
            
            # 对于 kissing number 任务，添加 5D kissing number 相关信息
            if self.task_definition.id and "kissing_number" in self.task_definition.id:
                kissing_5d = parent.fitness_scores.get("kissing_number_5d")
                sota_score = parent.fitness_scores.get("sota_score_5d")
                kissing_5d_valid = parent.fitness_scores.get("kissing_number_5d_valid")
                
                if kissing_5d is not None:
                    feedback["kissing_number_5d"] = kissing_5d
                if sota_score is not None:
                    feedback["sota_score_5d"] = sota_score
                if kissing_5d_valid is not None:
                    feedback["kissing_number_5d_valid"] = kissing_5d_valid
            
            feedback = {k: v for k, v in feedback.items() if v is not None}

            mutation_prompt = self.prompt_designer.design_mutation_prompt(program=parent, evaluation_feedback=feedback)
            logger.info(f"Attempting mutation for parent {parent.id} using diff. Model: {chosen_model}")
        
        generated_code = await self.code_generator.execute(
            prompt=mutation_prompt,
            model_name=chosen_model,
            temperature=0.75,
            output_format="diff",
            parent_code_for_diff=parent.code
        )

        if not generated_code.strip():
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) resulted in empty code/diff. Skipping.")
            return None
        
        if generated_code == parent.code:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) using diff resulted in no change to the code. Skipping.")
            return None
        
        if "<<<<<<< SEARCH" in generated_code and "=======" in generated_code and ">>>>>>> REPLACE" in generated_code:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) seems to have returned raw diff. LLM or diff application may have failed. Skipping. Content:\n{generated_code[:500]}")
            return None
        
        if "# Error:" in generated_code[:100]:
            logger.warning(f"Failed to generate valid code for offspring of {parent.id} ({prompt_type}). LLM Output indicates error: {generated_code[:200]}")
            return None

        offspring = Program(
            id=child_id,
            code=generated_code,
            generation=generation_num,
            parent_id=parent.id,
            island_id=parent.island_id,  # Inherit island ID from parent
            status="unevaluated",
            task_id=self.task_definition.id
        )
        logger.info(f"Successfully generated offspring {offspring.id} from parent {parent.id} ({prompt_type}).")
        return offspring

    async def execute(self) -> Any:
        return await self.manage_evolutionary_cycle()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    task_manager = TaskManagerAgent(task_definition=sample_task)

    sample_task = TaskDefinition(
        id="sum_list_task_001",
        description="Write a Python function called `solve(numbers)` that takes a list of integers `numbers` and returns their sum. The function should handle empty lists correctly by returning 0.",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0},
            {"input": [-1, 0, 1], "output": 0},
            {"input": [10, 20, 30, 40, 50], "output": 150}
        ],
        evaluation_criteria={"target_metric": "correctness", "goal": "maximize"},
        initial_code_prompt = "Please provide a Python function `solve(numbers)` that sums a list of integers. Handle empty lists by returning 0."
    )
    
    task_manager.num_generations = 3
    task_manager.population_size = 5
    task_manager.num_parents_to_select = 2

    async def run_task():
        try:
            best_programs = await task_manager.manage_evolutionary_cycle()
            if best_programs:
                print(f"\n*** Evolution Complete! Best program found: ***")
                print(f"ID: {best_programs[0].id}")
                print(f"Generation: {best_programs[0].generation}")
                print(f"Fitness: {best_programs[0].fitness_scores}")
                print(f"Code:\n{best_programs[0].code}")
            else:
                print("\n*** Evolution Complete! No suitable program was found. ***")
        except Exception as e:
            logger.error("An error occurred during the task management cycle.", exc_info=True)

    asyncio.run(run_task()) 