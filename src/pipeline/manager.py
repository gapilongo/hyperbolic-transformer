from typing import Dict, List, Optional, Any, Callable
import yaml
from collections import defaultdict
from core.config.configurations import PipelineConfig
from stage import Pipeline
from monitor import PipelineMonitor

class PipelineManager:
    """Manage multiple pipelines"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Track pipelines
        self.pipelines: Dict[str, Pipeline] = {}
        self.pipeline_status = defaultdict(lambda: "not_started")
        
        # Pipeline monitoring
        self.monitor = PipelineMonitor()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configurations"""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
            
    def add_pipeline(self,
                    name: str,
                    stages: Dict[str, Dict[str, Any]]):
        """Add new pipeline"""
        pipeline_config = PipelineConfig(
            pipeline_name=name,
            **self.config.get('default_config', {})
        )
        
        pipeline = Pipeline(pipeline_config)
        
        # Add stages
        for stage_name, stage_config in stages.items():
            pipeline.add_stage(
                name=stage_name,
                function=stage_config['function'],
                dependencies=stage_config.get('dependencies')
            )
            
        self.pipelines[name] = pipeline
        
    def run_pipeline(self, name: str) -> Dict[str, Any]:
        """Run specific pipeline"""
        if name not in self.pipelines:
            raise ValueError(f"Unknown pipeline: {name}")
            
        self.pipeline_status[name] = "running"
        
        try:
            # Run pipeline
            results = self.pipelines[name].run()
            self.pipeline_status[name] = "completed"
            
            # Update monitoring
            self.monitor.record_completion(name, True)
            
            return results
            
        except Exception as e:
            self.pipeline_status[name] = "failed"
            self.monitor.record_completion(name, False, error=str(e))
            raise
            
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Run all pipelines"""
        results = {}
        
        for name in self.pipelines:
            try:
                results[name] = self.run_pipeline(name)
            except Exception as e:
                results[name] = {"error": str(e)}
                
        return results
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all pipelines"""
        return dict(self.pipeline_status)

