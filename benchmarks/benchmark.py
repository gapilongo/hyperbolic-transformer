import time
import json
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import plotly.graph_objects as go
from collections import defaultdict
from src.model.transformer import HyperbolicTransformer
from src.core.config.configurations import BenchmarkConfig

class ModelBenchmark:
    """Comprehensive model benchmarking"""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = defaultdict(dict)
        self.baseline = self._load_baseline() if config.compare_baseline else None
        
        # Setup logging
        self.logger = logging.getLogger("benchmark")
        self.logger.setLevel(logging.INFO)
        
    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline benchmark results"""
        if not self.config.baseline_path:
            return None
            
        try:
            with open(self.config.baseline_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load baseline: {e}")
            return None
            
    def run_benchmarks(self):
        """Run all benchmark tests"""
        self.logger.info("Starting benchmark suite")
        
        for model_variant in self.config.model_variants:
            self.logger.info(f"\nBenchmarking model: {model_variant}")
            
            # Initialize model
            model = self._load_model(model_variant)
            model.to(self.config.device)
            model.eval()
            
            # Run benchmarks
            self.results[model_variant] = {
                'latency': self._benchmark_latency(model),
                'memory': self._benchmark_memory(model),
                'throughput': self._benchmark_throughput(model),
                'scaling': self._benchmark_scaling(model)
            }
            
        # Save results
        self._save_results()
            
    def _load_model(self, variant: str) -> torch.nn.Module:
        """Load model variant"""
        # Implementation depends on model architecture
        pass
    
    def _benchmark_latency(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Benchmark inference latency"""
        latency_results = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:
                times = []
                
                # Warmup
                for _ in range(self.config.warmup_iterations):
                    inputs = self._create_inputs(batch_size, seq_length)
                    with torch.no_grad():
                        _ = model(**inputs)
                        
                # Timing runs
                for _ in range(self.config.num_iterations):
                    inputs = self._create_inputs(batch_size, seq_length)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(**inputs)
                    torch.cuda.synchronize()
                    times.append(time.time() - start_time)
                    
                latency_results[f"b{batch_size}_s{seq_length}"] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'p50': np.percentile(times, 50),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99)
                }
                
        return latency_results
    
    def _benchmark_memory(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Benchmark memory usage"""
        memory_results = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
                # Run inference
                inputs = self._create_inputs(batch_size, seq_length)
                with torch.no_grad():
                    _ = model(**inputs)
                torch.cuda.synchronize()
                
                memory_results[f"b{batch_size}_s{seq_length}"] = {
                    'allocated': torch.cuda.max_memory_allocated(),
                    'reserved': torch.cuda.max_memory_reserved(),
                    'active': torch.cuda.memory_allocated(),
                    'total': torch.cuda.get_device_properties(0).total_memory
                }
                
        return memory_results
    
    def _benchmark_throughput(self, model: torch.nn.Module) -> Dict[str, float]:
        """Benchmark maximum throughput"""
        throughput_results = {}
        
        for batch_size in self.config.batch_sizes:
            total_tokens = 0
            start_time = time.time()
            
            while time.time() - start_time < 10:  # 10-second test
                inputs = self._create_inputs(batch_size, self.config.sequence_lengths[0])
                with torch.no_grad():
                    _ = model(**inputs)
                total_tokens += batch_size * self.config.sequence_lengths[0]
                
            throughput = total_tokens / (time.time() - start_time)
            throughput_results[f"b{batch_size}"] = throughput
            
        return throughput_results
    
    def _benchmark_scaling(self, model: torch.nn.Module) -> Dict[str, float]:
        """Benchmark scaling efficiency"""
        scaling_results = {}
        
        # Get baseline performance
        base_batch_size = self.config.batch_sizes[0]
        base_time = self._measure_batch_time(model, base_batch_size)
        
        for batch_size in self.config.batch_sizes[1:]:
            batch_time = self._measure_batch_time(model, batch_size)
            
            # Calculate scaling efficiency
            ideal_time = base_time * (batch_size / base_batch_size)
            efficiency = ideal_time / batch_time
            
            scaling_results[f"b{batch_size}"] = efficiency
            
        return scaling_results
    
    def _measure_batch_time(self, model: torch.nn.Module, batch_size: int) -> float:
        """Measure average time for one batch"""
        times = []
        
        for _ in range(self.config.num_iterations):
            inputs = self._create_inputs(batch_size, self.config.sequence_lengths[0])
            
            start_time = time.time()
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
            
        return np.mean(times)
    
    def _create_inputs(self, batch_size: int, seq_length: int) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for benchmarking"""
        return {
            'input_ids': torch.randint(
                0, 1000,
                (batch_size, seq_length),
                device=self.config.device
            ),
            'attention_mask': torch.ones(
                batch_size,
                seq_length,
                device=self.config.device
            )
        }
        
    def _save_results(self):
        """Save benchmark results"""
        output_path = Path(self.config.metrics_output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_path / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json", 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Generate comparison if baseline exists
        if self.baseline:
            comparison = self._compare_with_baseline()
            with open(output_path / "baseline_comparison.json", 'w') as f:
                json.dump(comparison, f, indent=2)
                
        # Generate visualizations
        self._create_visualizations()
        
    def _compare_with_baseline(self) -> Dict[str, Any]:
        """Compare results with baseline"""
        comparison = {}
        
        for model_variant in self.results:
            if model_variant not in self.baseline:
                continue
                
            variant_comparison = {}
            
            # Compare latency
            for config, metrics in self.results[model_variant]['latency'].items():
                if config in self.baseline[model_variant]['latency']:
                    baseline_metrics = self.baseline[model_variant]['latency'][config]
                    
                    variant_comparison[f"latency_{config}"] = {
                        'change_mean': (metrics['mean'] - baseline_metrics['mean']) / baseline_metrics['mean'],
                        'change_p95': (metrics['p95'] - baseline_metrics['p95']) / baseline_metrics['p95']
                    }
                    
            # Compare throughput
            for config, throughput in self.results[model_variant]['throughput'].items():
                if config in self.baseline[model_variant]['throughput']:
                    baseline_throughput = self.baseline[model_variant]['throughput'][config]
                    
                    variant_comparison[f"throughput_{config}"] = {
                        'change': (throughput - baseline_throughput) / baseline_throughput
                    }
                    
            comparison[model_variant] = variant_comparison
            
        return comparison
    
    def _create_visualizations(self):
        """Create benchmark visualizations"""
        output_path = Path(self.config.metrics_output_path)
        
        # Latency by configuration
        self._plot_latency_comparison(output_path / "latency_comparison.html")
        
        # Memory usage
        self._plot_memory_usage(output_path / "memory_usage.html")
        
        # Throughput scaling
        self._plot_throughput_scaling(output_path / "throughput_scaling.html")
        
    def _plot_latency_comparison(self, path: Path):
        """Plot latency comparison"""
        fig = go.Figure()
        
        for model_variant in self.results:
            latencies = []
            configs = []
            
            for config, metrics in self.results[model_variant]['latency'].items():
                latencies.append(metrics['mean'] * 1000)  # Convert to ms
                configs.append(config)
                
            fig.add_trace(
                go.Bar(
                    name=model_variant,
                    x=configs,
                    y=latencies,
                    text=[f"{x:.1f}ms" for x in latencies],
                    textposition='auto',
                )
            )
            
        fig.update_layout(
            title='Inference Latency by Configuration',
            xaxis_title='Batch Size / Sequence Length',
            yaxis_title='Latency (ms)',
            barmode='group'
        )
        
        fig.write_html(str(path))
        
    def _plot_memory_usage(self, path: Path):
        """Plot memory usage"""
        fig = go.Figure()
        
        for model_variant in self.results:
            allocated = []
            configs = []
            
            for config, metrics in self.results[model_variant]['memory'].items():
                allocated.append(metrics['allocated'] / 1e9)  # Convert to GB
                configs.append(config)
                
            fig.add_trace(
                go.Bar(
                    name=model_variant,
                    x=configs,
                    y=allocated,
                    text=[f"{x:.1f}GB" for x in allocated],
                    textposition='auto',
                )
            )
            
        fig.update_layout(
            title='GPU Memory Usage by Configuration',
            xaxis_title='Batch Size / Sequence Length',
            yaxis_title='Memory (GB)',
            barmode='group'
        )
        
        fig.write_html(str(path))
        
    def _plot_throughput_scaling(self, path: Path):
        """Plot throughput scaling"""
        fig = go.Figure()
        
        for model_variant in self.results:
            batch_sizes = []
            throughput = []
            
            for config, value in self.results[model_variant]['throughput'].items():
                batch_sizes.append(int(config[1:]))  # Remove 'b' prefix
                throughput.append(value)
                
            fig.add_trace(
                go.Scatter(
                    name=model_variant,
                    x=batch_sizes,
                    y=throughput,
                    mode='lines+markers',
                    text=[f"{x:.0f} tokens/s" for x in throughput],
                    textposition='top center'
                )
            )
            
        fig.update_layout(
            title='Throughput Scaling with Batch Size',
            xaxis_title='Batch Size',
            yaxis_title='Throughput (tokens/s)',
            xaxis_type='log',
            yaxis_type='log'
        )
        
        fig.write_html(str(path))

# GitHub Actions Workflow Configuration
GITHUB_WORKFLOW = """
name: Model Testing and Benchmarking

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9]
        
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
        
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=src/ --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
      with:
        files: ./coverage.xml
        
  benchmark:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run benchmarks
      run: python benchmarks/run_benchmarks.py
      
    - name: Upload benchmark results
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: benchmark_results/
"""

def setup_ci():
    """Setup continuous integration configuration"""
    # Create workflow directory
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    # Write workflow file
    with open(workflow_dir / "test_and_benchmark.yml", 'w') as f:
        f.write(GITHUB_WORKFLOW)

class BenchmarkRunner:
    """Run model benchmarks with various configurations"""
    def __init__(self,
                 model: HyperbolicTransformer,
                 config: BenchmarkConfig):
        self.model = model
        self.config = config
        
        # Initialize trackers
        self.benchmark_history = defaultdict(list)
        self.resource_usage = defaultdict(list)
        
    def run_benchmark_suite(self):
        """Run complete benchmark suite"""
        # Setup model for benchmarking
        self.model.eval()
        
        # Run different benchmark sets
        self.run_latency_benchmarks()
        self.run_throughput_benchmarks()
        self.run_memory_benchmarks()
        self.run_scaling_benchmarks()
        
        # Generate report
        self.generate_benchmark_report()
        
    def run_latency_benchmarks(self):
        """Run latency-focused benchmarks"""
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:
                latencies = self._measure_latency(batch_size, seq_length)
                
                self.benchmark_history['latency'].append({
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'p95': np.percentile(latencies, 95)
                })
                
    def run_throughput_benchmarks(self):
        """Run throughput-focused benchmarks"""
        for batch_size in self.config.batch_sizes:
            throughput = self._measure_throughput(batch_size)
            
            self.benchmark_history['throughput'].append({
                'batch_size': batch_size,
                'tokens_per_second': throughput
            })
            
    def run_memory_benchmarks(self):
        """Run memory-focused benchmarks"""
        for batch_size in self.config.batch_sizes:
            memory_stats = self._measure_memory_usage(batch_size)
            
            self.benchmark_history['memory'].append({
                'batch_size': batch_size,
                **memory_stats
            })
            
    def run_scaling_benchmarks(self):
        """Run scaling efficiency benchmarks"""
        base_perf = None
        
        for batch_size in self.config.batch_sizes:
            perf = self._measure_scaling(batch_size)
            
            if base_perf is None:
                base_perf = perf
                efficiency = 1.0
            else:
                efficiency = (perf / base_perf) * (self.config.batch_sizes[0] / batch_size)
                
            self.benchmark_history['scaling'].append({
                'batch_size': batch_size,
                'performance': perf,
                'scaling_efficiency': efficiency
            })
            
    def _measure_latency(self,
                        batch_size: int,
                        seq_length: int) -> List[float]:
        """Measure inference latency"""
        latencies = []
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            inputs = self._create_inputs(batch_size, seq_length)
            with torch.no_grad():
                _ = self.model(**inputs)
                
        # Measurement
        for _ in range(self.config.num_iterations):
            inputs = self._create_inputs(batch_size, seq_length)
            
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(**inputs)
            torch.cuda.synchronize()
            
            latencies.append(time.time() - start_time)
            
        return latencies
    
    def _measure_throughput(self, batch_size: int) -> float:
        """Measure processing throughput"""
        total_tokens = 0
        start_time = time.time()
        
        while time.time() - start_time < 10:  # 10-second measurement
            inputs = self._create_inputs(batch_size, self.config.sequence_lengths[0])
            
            with torch.no_grad():
                _ = self.model(**inputs)
                
            total_tokens += batch_size * self.config.sequence_lengths[0]
            
        return total_tokens / (time.time() - start_time)
    
    def _measure_memory_usage(self, batch_size: int) -> Dict[str, int]:
        """Measure GPU memory usage"""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        inputs = self._create_inputs(batch_size, self.config.sequence_lengths[0])
        
        with torch.no_grad():
            _ = self.model(**inputs)
        torch.cuda.synchronize()
        
        return {
            'allocated': torch.cuda.max_memory_allocated(),
            'reserved': torch.cuda.max_memory_reserved()
        }
        
    def _measure_scaling(self, batch_size: int) -> float:
        """Measure scaling performance"""
        total_tokens = 0
        total_time = 0
        
        for _ in range(self.config.num_iterations):
            inputs = self._create_inputs(batch_size, self.config.sequence_lengths[0])
            
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(**inputs)
            torch.cuda.synchronize()
            
            total_time += time.time() - start_time
            total_tokens += batch_size * self.config.sequence_lengths[0]
            
        return total_tokens / total_time
    
    def _create_inputs(self,
                      batch_size: int,
                      seq_length: int) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for benchmarking"""
        return {
            'input_ids': torch.randint(
                0, 1000,
                (batch_size, seq_length),
                device=self.model.device
            ),
            'attention_mask': torch.ones(
                batch_size,
                seq_length,
                device=self.model.device
            )
        }
        
    def generate_benchmark_report(self):
        """Generate detailed benchmark report"""
        report_dir = Path("benchmark_reports")
        report_dir.mkdir(exist_ok=True)
        
        # Save raw results
        with open(report_dir / "raw_results.json", 'w') as f:
            json.dump(self.benchmark_history, f, indent=2)
            
        # Generate visualizations
        self._create_benchmark_plots(report_dir)
        
        # Create HTML report
        self._create_html_report(report_dir)
        
    def _create_benchmark_plots(self, report_dir: Path):
        """Create benchmark visualization plots"""
        # Latency plot
        self._plot_latencies(report_dir / "latency.html")
        
        # Throughput plot
        self._plot_throughput(report_dir / "throughput.html")
        
        # Memory usage plot
        self._plot_memory_usage(report_dir / "memory.html")
        
        # Scaling efficiency plot
        self._plot_scaling(report_dir / "scaling.html")
        
    def _plot_latencies(self, path: Path):
        """Plot latency results"""
        fig = go.Figure()
        
        latencies = self.benchmark_history['latency']
        configs = [f"b{l['batch_size']}_s{l['seq_length']}" for l in latencies]
        means = [l['mean'] * 1000 for l in latencies]  # Convert to ms
        p95s = [l['p95'] * 1000 for l in latencies]
        
        fig.add_trace(
            go.Bar(
                name='Mean',
                x=configs,
                y=means,
                text=[f"{x:.1f}ms" for x in means],
                textposition='auto'
            )
        )
        
        fig.add_trace(
            go.Bar(
                name='P95',
                x=configs,
                y=p95s,
                text=[f"{x:.1f}ms" for x in p95s],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title='Inference Latency',
            xaxis_title='Configuration',
            yaxis_title='Latency (ms)',
            barmode='group'
        )
        
        fig.write_html(str(path))