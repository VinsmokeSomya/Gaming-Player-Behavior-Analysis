#!/usr/bin/env python3
"""
Performance test runner for the player retention analytics system.
Runs comprehensive performance benchmarks and generates a summary report.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_test_suite(test_file: str, test_class: str = None, markers: str = None) -> dict:
    """Run a test suite and capture results."""
    cmd = ["python", "-m", "pytest", "-v", "-s"]
    
    if test_class:
        cmd.append(f"{test_file}::{test_class}")
    else:
        cmd.append(test_file)
    
    if markers:
        cmd.extend(["-m", markers])
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        end_time = time.time()
        
        return {
            'command': ' '.join(cmd),
            'duration': end_time - start_time,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'command': ' '.join(cmd),
            'duration': 300,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Test timed out after 5 minutes',
            'success': False
        }


def main():
    """Run all performance test suites."""
    print("Player Retention Analytics - Performance Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Define test suites to run
    test_suites = [
        {
            'name': 'End-to-End Integration Tests',
            'file': 'tests/test_end_to_end_integration.py',
            'class': 'TestEndToEndPipeline',
            'description': 'Complete pipeline from events to visualizations'
        },
        {
            'name': 'Database Query Performance Tests',
            'file': 'tests/test_performance_benchmarks.py',
            'class': 'TestRetentionQueryPerformance',
            'description': 'Retention queries with large datasets'
        },
        {
            'name': 'ML Pipeline Performance Tests',
            'file': 'tests/test_performance_benchmarks.py',
            'class': 'TestMLPipelinePerformance',
            'description': 'ML pipeline with realistic data volumes'
        },
        {
            'name': 'Model Training Performance Tests',
            'file': 'tests/test_model_training_performance.py',
            'class': 'TestModelTrainingPerformance',
            'description': 'Model training with various dataset sizes'
        },
        {
            'name': 'Visualization Rendering Performance Tests',
            'file': 'tests/test_visualization_rendering_performance.py',
            'class': 'TestVisualizationRenderingPerformance',
            'description': 'Chart rendering with various data sizes'
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for suite in test_suites:
        print(f"\n🧪 {suite['name']}")
        print(f"   {suite['description']}")
        
        result = run_test_suite(
            suite['file'], 
            suite['class'],
            markers="performance and not slow"  # Run performance tests but skip slow ones
        )
        
        result['suite_name'] = suite['name']
        results.append(result)
        
        if result['success']:
            print(f"   ✅ PASSED in {result['duration']:.2f}s")
        else:
            print(f"   ❌ FAILED in {result['duration']:.2f}s")
            if result['stderr']:
                print(f"   Error: {result['stderr'][:200]}...")
    
    total_duration = time.time() - total_start_time
    
    # Generate summary report
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY REPORT")
    print("="*80)
    print(f"Total execution time: {total_duration:.2f} seconds")
    print(f"Test suites run: {len(results)}")
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(results)*100):.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 80)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {result['suite_name']:<50} {result['duration']:>8.2f}s")
        
        if not result['success'] and result['stderr']:
            print(f"     Error: {result['stderr'][:100]}...")
    
    # Performance benchmarks summary (if available)
    print("\nPerformance Benchmarks:")
    print("-" * 80)
    
    benchmark_notes = [
        "• Retention queries should complete within 30 seconds for standard reports",
        "• Model training should achieve ≥80% accuracy on test data", 
        "• Visualization rendering should complete within 5 seconds",
        "• Feature engineering should process >1000 players/second",
        "• Model predictions should process >500 predictions/second"
    ]
    
    for note in benchmark_notes:
        print(note)
    
    # Save detailed results to file
    report_file = Path("performance_test_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"Performance Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"Suite: {result['suite_name']}\n")
            f.write(f"Duration: {result['duration']:.2f}s\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Command: {result['command']}\n")
            
            if result['stdout']:
                f.write("STDOUT:\n")
                f.write(result['stdout'])
                f.write("\n")
            
            if result['stderr']:
                f.write("STDERR:\n")
                f.write(result['stderr'])
                f.write("\n")
            
            f.write("-" * 80 + "\n\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if failed > 0:
        print(f"\n⚠️  {failed} test suite(s) failed. Check the detailed report for more information.")
        sys.exit(1)
    else:
        print(f"\n🎉 All {passed} test suites passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()