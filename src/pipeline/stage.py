from typing import Dict, List, Optional, Any, Callable
import yaml
from pathlib import Path
import time
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from core.config.configurations import PipelineConfig




class PipelineStage:
    """Single stage in pipeline"""
    def __init__(self,
                 name: str,
                 function: Callable,
                 dependencies: Optional[List[str]] = None):
        self.name = name
        self.function = function
        self.dependencies = dependencies or []
        
        # Stage state
        self.is_complete = False
        self.output = None
        self.error = None
        
    def run(self, inputs: Dict[str, Any] = None) -> Any:
        """Run stage function"""
        try:
            self.output = self.function(**(inputs or {}))
            self.is_complete = True
            return self.output
        except Exception as e:
            self.error = e
            raise

class Pipeline:
    """Manage execution of pipeline stages"""
    def __init__(self,
                 config: PipelineConfig):
        self.config = config
        
        # Initialize components
        self.stages: Dict[str, PipelineStage] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.logger = self._setup_logger()
        
        # Pipeline state
        self.stage_outputs = {}
        self.failed_stages = set()
        self.retries = defaultdict(int)
        
    def add_stage(self,
                  name: str,
                  function: Callable,
                  dependencies: Optional[List[str]] = None):
        """Add stage to pipeline"""
        stage = PipelineStage(name, function, dependencies)
        self.stages[name] = stage
        
    def run(self) -> Dict[str, Any]:
        """Run complete pipeline"""
        self.logger.info(f"Starting pipeline: {self.config.pipeline_name}")
        start_time = time.time()
        
        try:
            # Validate pipeline
            self._validate_pipeline()
            
            # Run stages in dependency order
            stage_order = self._get_execution_order()
            
            for stage_name in stage_order:
                self._run_stage(stage_name)
                
            duration = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {duration:.2f}s")
            
            return self.stage_outputs
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            self.executor.shutdown()
            
    def _validate_pipeline(self):
        """Validate pipeline configuration"""
        # Check for cycles
        if self._has_cycles():
            raise ValueError("Pipeline contains cyclic dependencies")
            
        # Check dependency existence
        for stage in self.stages.values():
            for dep in stage.dependencies:
                if dep not in self.stages:
                    raise ValueError(f"Unknown dependency: {dep}")
                    
    def _has_cycles(self) -> bool:
        """Check for cyclic dependencies"""
        visited = set()
        path = set()
        
        def visit(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
                
            path.add(node)
            visited.add(node)
            
            for dep in self.stages[node].dependencies:
                if visit(dep):
                    return True
                    
            path.remove(node)
            return False
            
        return any(visit(node) for node in self.stages)
    
    def _get_execution_order(self) -> List[str]:
        """Determine stage execution order"""
        order = []
        visited = set()
        
        def visit(node: str):
            if node in visited:
                return
                
            visited.add(node)
            
            # Visit dependencies first
            for dep in self.stages[node].dependencies:
                visit(dep)
                
            order.append(node)
            
        for stage_name in self.stages:
            visit(stage_name)
            
        return order
    
    def _run_stage(self, stage_name: str):
        """Run single pipeline stage"""
        stage = self.stages[stage_name]
        
        # Check dependencies
        if not self._check_dependencies(stage_name):
            self.failed_stages.add(stage_name)
            raise ValueError(f"Dependencies failed for stage: {stage_name}")
            
        self.logger.info(f"Running stage: {stage_name}")
        
        try:
            # Prepare inputs from dependencies
            inputs = self._get_stage_inputs(stage_name)
            
            # Execute stage
            output = stage.run(inputs)
            
            # Store output
            self.stage_outputs[stage_name] = output
            
            self.logger.info(f"Stage completed: {stage_name}")
            
        except Exception as e:
            self.logger.error(f"Stage failed: {stage_name}")
            self.logger.error(str(e))
            
            # Handle retries
            if self.retries[stage_name] < self.config.retry_attempts:
                self.retries[stage_name] += 1
                self.logger.info(f"Retrying stage: {stage_name} (attempt {self.retries[stage_name]})")
                time.sleep(self.config.retry_delay)
                self._run_stage(stage_name)
            else:
                self.failed_stages.add(stage_name)
                raise
                
    def _check_dependencies(self, stage_name: str) -> bool:
        """Check if dependencies completed successfully"""
        dependencies = self.stages[stage_name].dependencies
        return all(
            dep not in self.failed_stages and
            dep in self.stage_outputs
            for dep in dependencies
        )
        
    def _get_stage_inputs(self, stage_name: str) -> Dict[str, Any]:
        """Get inputs for stage from dependencies"""
        return {
            dep: self.stage_outputs[dep]
            for dep in self.stages[stage_name].dependencies
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger"""
        logger = logging.getLogger(self.config.pipeline_name)
        logger.setLevel(self.config.log_level)
        
        # Add file handler
        log_path = Path("logs") / f"{self.config.pipeline_name}.log"
        log_path.parent.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        logger.addHandler(handler)
        return logger
