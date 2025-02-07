from trainer import DistributedTrainer
import torch
from collections import defaultdict
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
import psutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class DistributedMonitor:
    """Monitor distributed training performance"""
    def __init__(self,
                 dist_trainer: DistributedTrainer,
                 log_dir: str = "distributed_logs"):
        self.dist_trainer = dist_trainer
        self.log_dir = log_dir
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(log_dir)
        
        # Performance tracking
        self.sync_times = defaultdict(list)
        self.communication_times = defaultdict(list)
        self.computation_times = defaultdict(list)
        
        # Node statistics
        self.node_stats = defaultdict(lambda: defaultdict(list))
        
        # Initialize profiler
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=1
            ),
            on_trace_ready=self._trace_handler,
            with_stack=True
        )
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.profiler.start()
        
    def step(self):
        """Record monitoring step"""
        self.profiler.step()
        
        # Collect node statistics
        self._collect_node_stats()
        
        # Record sync times
        self._record_sync_times()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.profiler.stop()
        
        # Generate final report
        self._generate_report()
        
    def _trace_handler(self, prof) -> None:
        """Handle profiler trace"""
        # Record computation times
        for event in prof.key_averages():
            if event.key not in ['DataLoader', 'Optimizer', 'Synchronize']:
                self.computation_times[event.key].append(event.cuda_time_total)
                
    def _collect_node_stats(self):
        """Collect statistics from all nodes"""
        rank = self.dist_trainer.dist_config.rank
        
        # Collect GPU stats
        if torch.cuda.is_available():
            gpu_stats = {
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved(),
                'utilization': torch.cuda.utilization()
            }
            self.node_stats[rank]['gpu'].append(gpu_stats)
            
        # Collect CPU stats
        cpu_stats = {
            'utilization': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent
        }
        self.node_stats[rank]['cpu'].append(cpu_stats)
        
        # Collect network stats
        net_stats = {
            'bytes_sent': psutil.net_io_counters().bytes_sent,
            'bytes_recv': psutil.net_io_counters().bytes_recv
        }
        self.node_stats[rank]['network'].append(net_stats)
        
    def _record_sync_times(self):
        """Record synchronization times"""
        rank = self.dist_trainer.dist_config.rank
        
        # Record gradient sync time
        sync_start = time.time()
        self.dist_trainer.sync_gradients()
        sync_time = time.time() - sync_start
        
        self.sync_times[rank].append(sync_time)
        
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        report = {
            'performance': self._analyze_performance(),
            'bottlenecks': self._identify_bottlenecks(),
            'recommendations': self._generate_recommendations()
        }
        
        # Log report
        self.logger.info("Distributed Training Report:")
        self._log_report(report)
        
        return report
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze distributed training performance"""
        analysis = {}
        
        # Analyze computation/communication ratio
        comp_time = np.mean([np.sum(times) for times in self.computation_times.values()])
        comm_time = np.mean([np.sum(times) for times in self.communication_times.values()])
        
        analysis['comp_comm_ratio'] = comp_time / (comm_time + 1e-8)
        
        # Analyze load balancing
        gpu_utils = []
        for rank in self.node_stats:
            gpu_stats = self.node_stats[rank]['gpu']
            if gpu_stats:
                avg_util = np.mean([stats['utilization'] for stats in gpu_stats])
                gpu_utils.append(avg_util)
                
        analysis['load_balance'] = np.std(gpu_utils) if gpu_utils else 0
        
        # Analyze network efficiency
        network_efficiency = {}
        for rank in self.node_stats:
            net_stats = self.node_stats[rank]['network']
            if net_stats:
                bytes_sent = [stats['bytes_sent'] for stats in net_stats]
                bytes_recv = [stats['bytes_recv'] for stats in net_stats]
                
                network_efficiency[rank] = {
                    'send_throughput': np.diff(bytes_sent).mean(),
                    'recv_throughput': np.diff(bytes_recv).mean()
                }
                
        analysis['network_efficiency'] = network_efficiency
        
        return analysis
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify training bottlenecks"""
        bottlenecks = []
        
        # Check GPU utilization bottlenecks
        for rank in self.node_stats:
            gpu_stats = self.node_stats[rank]['gpu']
            if gpu_stats:
                avg_util = np.mean([stats['utilization'] for stats in gpu_stats])
                if avg_util < 70:  # Less than 70% utilization
                    bottlenecks.append({
                        'type': 'gpu_underutilization',
                        'rank': rank,
                        'severity': 'high' if avg_util < 50 else 'medium',
                        'details': f"GPU utilization at {avg_util:.1f}%"
                    })
                    
        # Check communication bottlenecks
        sync_times = np.array([np.mean(times) for times in self.sync_times.values()])
        if np.std(sync_times) > 0.1 * np.mean(sync_times):
            bottlenecks.append({
                'type': 'communication_imbalance',
                'severity': 'high',
                'details': f"High variance in sync times: {np.std(sync_times):.3f}s"
            })
            
        # Check memory bottlenecks
        for rank in self.node_stats:
            gpu_stats = self.node_stats[rank]['gpu']
            if gpu_stats:
                avg_mem = np.mean([stats['memory_allocated'] / stats['memory_cached'] 
                                 for stats in gpu_stats])
                if avg_mem > 0.95:  # More than 95% memory usage
                    bottlenecks.append({
                        'type': 'memory_pressure',
                        'rank': rank,
                        'severity': 'high',
                        'details': f"High memory usage: {avg_mem*100:.1f}%"
                    })
                    
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        bottlenecks = self._identify_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'gpu_underutilization':
                recommendations.append(
                    f"Increase batch size or gradient accumulation steps for rank {bottleneck['rank']} "
                    "to improve GPU utilization"
                )
            elif bottleneck['type'] == 'communication_imbalance':
                recommendations.extend([
                    "Consider using gradient compression or quantization",
                    "Check network connectivity between nodes",
                    "Try adjusting the number of gradient buckets for overlapping"
                ])
            elif bottleneck['type'] == 'memory_pressure':
                recommendations.extend([
                    f"Reduce batch size for rank {bottleneck['rank']}",
                    "Enable gradient checkpointing",
                    "Consider using mixed precision training"
                ])
                
        return recommendations
    
    def _log_report(self, report: Dict[str, Any]):
        """Log monitoring report"""
        # Log performance metrics
        self.logger.info("\nPerformance Analysis:")
        for metric, value in report['performance'].items():
            if isinstance(value, dict):
                self.logger.info(f"\n{metric}:")
                for k, v in value.items():
                    self.logger.info(f"  {k}: {v}")
            else:
                self.logger.info(f"{metric}: {value}")
                
        # Log bottlenecks
        self.logger.info("\nIdentified Bottlenecks:")
        for bottleneck in report['bottlenecks']:
            self.logger.info(
                f"[{bottleneck['severity'].upper()}] {bottleneck['type']}: "
                f"{bottleneck['details']}"
            )
            
        # Log recommendations
        self.logger.info("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            self.logger.info(f"{i}. {rec}")
            
    def plot_performance_metrics(self):
        """Generate performance visualization plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot GPU utilization over time
        plt.figure(figsize=(12, 6))
        for rank in self.node_stats:
            gpu_stats = self.node_stats[rank]['gpu']
            if gpu_stats:
                utils = [stats['utilization'] for stats in gpu_stats]
                plt.plot(utils, label=f'Rank {rank}')
                
        plt.title('GPU Utilization Over Time')
        plt.xlabel('Step')
        plt.ylabel('Utilization (%)')
        plt.legend()
        plt.savefig(f"{self.log_dir}/gpu_utilization.png")
        
        # Plot sync times distribution
        plt.figure(figsize=(12, 6))
        sync_times = []
        ranks = []
        for rank, times in self.sync_times.items():
            sync_times.extend(times)
            ranks.extend([rank] * len(times))
            
        sns.boxplot(x=ranks, y=sync_times)
        plt.title('Gradient Sync Time Distribution by Rank')
        plt.xlabel('Rank')
        plt.ylabel('Sync Time (s)')
        plt.savefig(f"{self.log_dir}/sync_times.png")
        
        # Plot memory usage over time
        plt.figure(figsize=(12, 6))
        for rank in self.node_stats:
            gpu_stats = self.node_stats[rank]['gpu']
            if gpu_stats:
                mem_used = [stats['memory_allocated'] / (1024**3) for stats in gpu_stats]  # Convert to GB
                plt.plot(mem_used, label=f'Rank {rank}')
                
        plt.title('GPU Memory Usage Over Time')
        plt.xlabel('Step')
        plt.ylabel('Memory Used (GB)')
        plt.legend()
        plt.savefig(f"{self.log_dir}/memory_usage.png")
        
        plt.close('all')