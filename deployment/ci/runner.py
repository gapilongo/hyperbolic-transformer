from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import unittest
import yaml
import shutil

import subprocess
from src.core.config.configurations import BenchmarkConfig
from benchmarks.benchmark import ModelBenchmark
from tests.test_validation import run_all_tests

class ContinuousIntegrationRunner:
    """Manage CI/CD pipeline execution"""
    def __init__(self, 
                 repo_path: str,
                 config_path: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize components
        self.test_runner = unittest.TextTestRunner(verbosity=2)
        self.benchmark_runner = ModelBenchmark(BenchmarkConfig(**self.config.get('benchmark', {})))
        
        # Setup logging
        self.logger = logging.getLogger("ci")
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load CI configuration"""
        with open(config_path) as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.repo_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "ci.log")
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        
    def run_pipeline(self) -> bool:
        """Run complete CI pipeline"""
        try:
            # Run tests
            self.logger.info("Running tests...")
            test_success = self.run_tests()
            
            if not test_success:
                self.logger.error("Tests failed!")
                return False
                
            # Run benchmarks
            self.logger.info("Running benchmarks...")
            benchmark_success = self.run_benchmarks()
            
            if not benchmark_success:
                self.logger.error("Benchmarks failed!")
                return False
                
            # Generate reports
            self.logger.info("Generating reports...")
            self.generate_reports()
            
            self.logger.info("CI pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"CI pipeline failed: {str(e)}")
            return False
            
    def run_tests(self) -> bool:
        """Run test suite"""
        result = run_all_tests()
        return result.wasSuccessful()
        
    def run_benchmarks(self) -> bool:
        """Run benchmark suite"""
        try:
            self.benchmark_runner.run_benchmarks()
            return True
        except Exception as e:
            self.logger.error(f"Benchmark error: {str(e)}")
            return False
            
    def generate_reports(self):
        """Generate CI reports"""
        report_dir = self.repo_path / "reports"
        report_dir.mkdir(exist_ok=True)
        
        # Generate test coverage report
        self._generate_coverage_report(report_dir)
        
        # Generate benchmark report
        self._generate_benchmark_report(report_dir)
        
        # Generate combined report
        self._generate_combined_report(report_dir)
        
    def _generate_coverage_report(self, report_dir: Path):
        """Generate test coverage report"""
        subprocess.run([
            "coverage", "run", "-m", "pytest", "tests/",
            f"--cov-report=html:{report_dir}/coverage"
        ])
        
    def _generate_benchmark_report(self, report_dir: Path):
        """Generate benchmark report"""
        # Copy benchmark results
        benchmark_results = Path("benchmark_results")
        if benchmark_results.exists():
            shutil.copytree(
                benchmark_results,
                report_dir / "benchmarks",
                dirs_exist_ok=True
            )
            
    def _generate_combined_report(self, report_dir: Path):
        """Generate combined HTML report"""
        template = """
        <html>
        <head>
            <title>CI Pipeline Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; }
                .metrics { display: flex; gap: 20px; }
                .metric { padding: 10px; border: 1px solid #ccc; }
            </style>
        </head>
        <body>
            <h1>CI Pipeline Report</h1>
            <div class="section">
                <h2>Test Results</h2>
                {test_results}
            </div>
            <div class="section">
                <h2>Benchmark Results</h2>
                {benchmark_results}
            </div>
        </body>
        </html>
        """
        
        # Generate report content
        test_results = self._format_test_results()
        benchmark_results = self._format_benchmark_results()
        
        report_content = template.format(
            test_results=test_results,
            benchmark_results=benchmark_results
        )
        
        # Write report
        with open(report_dir / "report.html", 'w') as f:
            f.write(report_content)
            
    def _format_test_results(self) -> str:
        """Format test results for report"""
        coverage_data = self._get_coverage_data()
        
        return f"""
        <div class="metrics">
            <div class="metric">
                <h3>Coverage</h3>
                <p>{coverage_data['total_coverage']:.1f}%</p>
            </div>
            <div class="metric">
                <h3>Tests Passed</h3>
                <p>{coverage_data['tests_passed']}/{coverage_data['total_tests']}</p>
            </div>
        </div>
        """
        
    def _format_benchmark_results(self) -> str:
        """Format benchmark results for report"""
        benchmark_data = self._get_benchmark_data()
        
        metrics_html = ""
        for metric, value in benchmark_data.items():
            metrics_html += f"""
            <div class="metric">
                <h3>{metric}</h3>
                <p>{value}</p>
            </div>
            """
            
        return f"""
        <div class="metrics">
            {metrics_html}
        </div>
        """
        
    def _get_coverage_data(self) -> Dict[str, Any]:
        """Get test coverage data"""
        # Implementation of coverage data extraction
        return {
            'total_coverage': 0.0,
            'tests_passed': 0,
            'total_tests': 0
        }
        
    def _get_benchmark_data(self) -> Dict[str, Any]:
        """Get benchmark data"""
        # Implementation of benchmark data extraction
        return {
            'Latency': '0ms',
            'Throughput': '0 tokens/s',
            'Memory': '0GB'
        }
        
