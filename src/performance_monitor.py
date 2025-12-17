import time
import threading
import psutil
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

@dataclass
class PerformanceMetrics:
    """Data class to hold performance metrics."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_usage_start: Optional[float] = None
    cpu_usage_end: Optional[float] = None
    memory_usage_start: Optional[float] = None
    memory_usage_end: Optional[float] = None
    threading_enabled: bool = False
    worker_count: int = 1
    items_processed: int = 0
    success: bool = True
    error_message: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self):
        """Finalize the metrics calculation."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
        
    def get_throughput(self) -> Optional[float]:
        """Calculate items processed per second."""
        if self.duration and self.duration > 0 and self.items_processed > 0:
            return self.items_processed / self.duration
        return None
    
    def get_cpu_change(self) -> Optional[float]:
        """Calculate CPU usage change."""
        if self.cpu_usage_start is not None and self.cpu_usage_end is not None:
            return self.cpu_usage_end - self.cpu_usage_start
        return None
    
    def get_memory_change(self) -> Optional[float]:
        """Calculate memory usage change in MB."""
        if self.memory_usage_start is not None and self.memory_usage_end is not None:
            return (self.memory_usage_end - self.memory_usage_start) / 1024 / 1024
        return None

class PerformanceMonitor:
    """Enhanced performance monitoring for threaded operations."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.lock = threading.Lock()
        
        # System info
        self.cpu_count = psutil.cpu_count()
        self.logical_cpu_count = psutil.cpu_count(logical=True)
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                "cpu_physical_cores": self.cpu_count,
                "cpu_logical_cores": self.logical_cpu_count,
                "cpu_usage_percent": cpu_percent,
                "memory_total_gb": memory.total / 1024**3,
                "memory_available_gb": memory.available / 1024**3,
                "memory_usage_percent": memory.percent,
                "threading_support": threading.active_count() > 1
            }
        except Exception as e:
            return {"error": str(e)}
    
    @contextmanager
    def monitor_operation(self, operation_name: str, threading_enabled: bool = False,
                         worker_count: int = 1, items_to_process: int = 0):
        """Context manager for monitoring operation performance."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            threading_enabled=threading_enabled,
            worker_count=worker_count,
            items_processed=items_to_process
        )
        
        # Capture initial system state
        try:
            metrics.cpu_usage_start = psutil.cpu_percent(interval=0.1)
            metrics.memory_usage_start = psutil.Process(os.getpid()).memory_info().rss
        except:
            pass
        
        self.current_metrics = metrics
        
        try:
            print(f"🔍 [Monitor] Starting {operation_name}")
            if threading_enabled:
                print(f"🧵 [Monitor] Threading enabled with {worker_count} workers")
            
            yield metrics
            
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            print(f"❌ [Monitor] Error in {operation_name}: {e}")
            raise
            
        finally:
            # Capture final system state
            metrics.end_time = time.time()
            try:
                metrics.cpu_usage_end = psutil.cpu_percent(interval=0.1)
                metrics.memory_usage_end = psutil.Process(os.getpid()).memory_info().rss
            except:
                pass
            
            metrics.finalize()
            
            # Add to history
            with self.lock:
                self.metrics_history.append(metrics)
            
            # Print performance summary
            self._print_performance_summary(metrics)
    
    def _print_performance_summary(self, metrics: PerformanceMetrics):
        """Print a formatted performance summary."""
        print(f"\n{'='*60}")
        print(f"🎯 PERFORMANCE SUMMARY: {metrics.operation_name}")
        print(f"{'='*60}")
        
        # Basic timing
        if metrics.duration:
            print(f"⏱️  Duration: {metrics.duration:.2f} seconds")
            
            if metrics.items_processed > 0:
                throughput = metrics.get_throughput()
                print(f"🚀 Throughput: {throughput:.2f} items/second")
                print(f"📊 Items processed: {metrics.items_processed}")
        
        # Threading info
        if metrics.threading_enabled:
            print(f"🧵 Threading: ✅ ENABLED ({metrics.worker_count} workers)")
            if metrics.duration and metrics.worker_count > 1:
                estimated_sequential_time = metrics.duration * metrics.worker_count * 0.7  # Account for overhead
                speedup = estimated_sequential_time / metrics.duration
                print(f"⚡ Estimated speedup: {speedup:.1f}x faster than sequential")
        else:
            print(f"🧵 Threading: ❌ DISABLED")
        
        # System resource usage
        cpu_change = metrics.get_cpu_change()
        if cpu_change is not None:
            print(f"💻 CPU impact: {cpu_change:+.1f}% change")
        
        memory_change = metrics.get_memory_change()
        if memory_change is not None:
            print(f"🧠 Memory impact: {memory_change:+.1f} MB change")
        
        # Success status
        if metrics.success:
            print(f"✅ Status: SUCCESS")
        else:
            print(f"❌ Status: FAILED ({metrics.error_message})")
        
        # Extra metadata
        if metrics.extra_metadata:
            print(f"📋 Additional info:")
            for key, value in metrics.extra_metadata.items():
                print(f"   {key}: {value}")
        
        print(f"{'='*60}\n")
    
    def get_performance_history(self, operation_name: Optional[str] = None) -> List[PerformanceMetrics]:
        """Get performance history, optionally filtered by operation name."""
        with self.lock:
            if operation_name:
                return [m for m in self.metrics_history if m.operation_name == operation_name]
            return self.metrics_history.copy()
    
    def get_threading_comparison(self, operation_name: str) -> Dict[str, Any]:
        """Compare threading vs non-threading performance for an operation."""
        metrics = self.get_performance_history(operation_name)
        
        threaded_metrics = [m for m in metrics if m.threading_enabled and m.success]
        sequential_metrics = [m for m in metrics if not m.threading_enabled and m.success]
        
        comparison = {
            "operation": operation_name,
            "threaded_runs": len(threaded_metrics),
            "sequential_runs": len(sequential_metrics)
        }
        
        if threaded_metrics:
            avg_threaded_time = sum(m.duration for m in threaded_metrics) / len(threaded_metrics)
            avg_threaded_throughput = sum(m.get_throughput() or 0 for m in threaded_metrics) / len(threaded_metrics)
            comparison.update({
                "avg_threaded_duration": avg_threaded_time,
                "avg_threaded_throughput": avg_threaded_throughput
            })
        
        if sequential_metrics:
            avg_sequential_time = sum(m.duration for m in sequential_metrics) / len(sequential_metrics)
            avg_sequential_throughput = sum(m.get_throughput() or 0 for m in sequential_metrics) / len(sequential_metrics)
            comparison.update({
                "avg_sequential_duration": avg_sequential_time,
                "avg_sequential_throughput": avg_sequential_throughput
            })
        
        # Calculate improvement
        if threaded_metrics and sequential_metrics:
            time_improvement = (comparison["avg_sequential_duration"] - comparison["avg_threaded_duration"]) / comparison["avg_sequential_duration"] * 100
            throughput_improvement = (comparison["avg_threaded_throughput"] - comparison["avg_sequential_throughput"]) / comparison["avg_sequential_throughput"] * 100
            
            comparison.update({
                "time_improvement_percent": time_improvement,
                "throughput_improvement_percent": throughput_improvement,
                "speedup_factor": comparison["avg_sequential_duration"] / comparison["avg_threaded_duration"]
            })
        
        return comparison
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "total_operations": len(self.metrics_history),
            "operations_by_type": {},
            "threading_analysis": {},
            "recommendations": []
        }
        
        # Group operations by type
        operations = {}
        for metric in self.metrics_history:
            if metric.operation_name not in operations:
                operations[metric.operation_name] = []
            operations[metric.operation_name].append(metric)
        
        report["operations_by_type"] = {
            name: {
                "total_runs": len(metrics),
                "successful_runs": len([m for m in metrics if m.success]),
                "avg_duration": sum(m.duration or 0 for m in metrics if m.success) / max(1, len([m for m in metrics if m.success])),
                "threading_enabled": any(m.threading_enabled for m in metrics)
            }
            for name, metrics in operations.items()
        }
        
        # Threading analysis
        for op_name in operations.keys():
            comparison = self.get_threading_comparison(op_name)
            if "speedup_factor" in comparison:
                report["threading_analysis"][op_name] = comparison
        
        # Generate recommendations
        recommendations = []
        
        system_info = report["system_info"]
        if isinstance(system_info, dict) and "cpu_logical_cores" in system_info:
            if system_info["cpu_logical_cores"] >= 4:
                recommendations.append("✅ System has sufficient CPU cores for threading benefits")
            else:
                recommendations.append("⚠️ System has limited CPU cores - threading benefits may be minimal")
        
        for op_name, analysis in report["threading_analysis"].items():
            if analysis.get("speedup_factor", 1) > 1.5:
                recommendations.append(f"🚀 {op_name}: Threading shows {analysis['speedup_factor']:.1f}x speedup - excellent!")
            elif analysis.get("speedup_factor", 1) > 1.2:
                recommendations.append(f"✅ {op_name}: Threading shows {analysis['speedup_factor']:.1f}x speedup - good improvement")
            else:
                recommendations.append(f"⚠️ {op_name}: Threading shows minimal improvement - consider optimizing")
        
        report["recommendations"] = recommendations
        
        return report
    
    def print_performance_report(self):
        """Print a formatted performance report."""
        report = self.generate_performance_report()
        
        print(f"\n{'='*80}")
        print(f"🎯 COMPREHENSIVE PERFORMANCE REPORT")
        print(f"{'='*80}")
        print(f"Generated: {report['generated_at']}")
        print(f"Total Operations Monitored: {report['total_operations']}")
        
        # System info
        print(f"\n🖥️  SYSTEM INFORMATION:")
        sys_info = report['system_info']
        if isinstance(sys_info, dict):
            print(f"   CPU Cores: {sys_info.get('cpu_physical_cores', 'N/A')} physical, {sys_info.get('cpu_logical_cores', 'N/A')} logical")
            print(f"   Memory: {sys_info.get('memory_total_gb', 0):.1f} GB total, {sys_info.get('memory_usage_percent', 0):.1f}% used")
            print(f"   Current CPU: {sys_info.get('cpu_usage_percent', 0):.1f}%")
        
        # Operations summary
        print(f"\n📊 OPERATIONS SUMMARY:")
        for op_name, stats in report['operations_by_type'].items():
            threading_status = "✅ ENABLED" if stats['threading_enabled'] else "❌ DISABLED"
            print(f"   {op_name}:")
            print(f"      Runs: {stats['successful_runs']}/{stats['total_runs']} successful")
            print(f"      Avg Duration: {stats['avg_duration']:.2f}s")
            print(f"      Threading: {threading_status}")
        
        # Threading analysis
        if report['threading_analysis']:
            print(f"\n🧵 THREADING PERFORMANCE ANALYSIS:")
            for op_name, analysis in report['threading_analysis'].items():
                if 'speedup_factor' in analysis:
                    print(f"   {op_name}:")
                    print(f"      Speedup Factor: {analysis['speedup_factor']:.2f}x")
                    print(f"      Time Improvement: {analysis['time_improvement_percent']:+.1f}%")
                    if 'throughput_improvement_percent' in analysis:
                        print(f"      Throughput Improvement: {analysis['throughput_improvement_percent']:+.1f}%")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   {rec}")
        
        print(f"{'='*80}\n")
    
    def clear_history(self):
        """Clear performance history."""
        with self.lock:
            self.metrics_history.clear()
        print("🧹 Performance history cleared")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience functions
@contextmanager
def monitor_threaded_operation(operation_name: str, worker_count: int = 4, items_to_process: int = 0):
    """Convenience function for monitoring threaded operations."""
    with performance_monitor.monitor_operation(
        operation_name=operation_name,
        threading_enabled=True,
        worker_count=worker_count,
        items_to_process=items_to_process
    ) as metrics:
        yield metrics

@contextmanager
def monitor_sequential_operation(operation_name: str, items_to_process: int = 0):
    """Convenience function for monitoring sequential operations."""
    with performance_monitor.monitor_operation(
        operation_name=operation_name,
        threading_enabled=False,
        worker_count=1,
        items_to_process=items_to_process
    ) as metrics:
        yield metrics