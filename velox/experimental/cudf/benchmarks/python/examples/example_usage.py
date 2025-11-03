#!/usr/bin/env python3
"""
Example usage of the Velox CUDF TPC-H Python bindings.

This script demonstrates how to use the Python bindings to run TPC-H benchmarks
programmatically from Python.
"""

import sys
import argparse
from cudf_tpch_benchmark import CudfTpchBenchmark, BenchmarkError


def main():
    parser = argparse.ArgumentParser(
        description='Run Velox CUDF TPC-H benchmarks from Python'
    )
    parser.add_argument(
        '--data-path',
        required=True,
        help='Path to TPC-H data directory'
    )
    parser.add_argument(
        '--data-format',
        default='parquet',
        help='Data format (default: parquet)'
    )
    parser.add_argument(
        '--query',
        type=int,
        choices=range(1, 23),
        help='Run a specific query (1-22). If not specified, runs all queries.'
    )
    parser.add_argument(
        '--num-drivers',
        type=int,
        default=4,
        help='Number of driver threads (default: 4)'
    )
    parser.add_argument(
        '--gpu-batch-size',
        type=int,
        default=100000,
        help='GPU batch size in rows (default: 100000)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Velox CUDF TPC-H Benchmark")
    print("=" * 80)
    print(f"Data Path: {args.data_path}")
    print(f"Data Format: {args.data_format}")
    print(f"Number of Drivers: {args.num_drivers}")
    print(f"GPU Batch Size: {args.gpu_batch_size}")
    print("=" * 80)
    
    try:
        # Create benchmark instance using context manager
        with CudfTpchBenchmark(
            data_path=args.data_path,
            data_format=args.data_format,
            num_drivers=args.num_drivers,
            num_splits_per_file=10,
            cudf_gpu_batch_size_rows=args.gpu_batch_size,
            cudf_chunk_read_limit=1024 * 1024 * 1024 * 1,
            cudf_pass_read_limit=0,
            velox_cudf_table_scan=True
        ) as benchmark:
            
            if args.query:
                # Run single query
                print(f"\nRunning Query {args.query}...")
                result = benchmark.run_query(args.query)
                print(f"✓ Query {args.query} completed")
                print(f"  Execution Time: {result.execution_time_ms:.2f} ms")
                print(f"  Raw Input Bytes: {result.raw_input_bytes:,} bytes")
                print(f"  Throughput: {result.throughput_mbps:.2f} MB/s")
                print(f"  Splits: {result.num_finished_splits}/{result.num_total_splits}")
            else:
                # Run all queries
                print("\nRunning all TPC-H queries...")
                results = benchmark.run_all_queries()
                
                total_time = 0
                total_bytes = 0
                successful = 0
                failed = 0
                
                print("\nResults:")
                print("-" * 80)
                for query_id in range(1, 23):
                    result = results[query_id]
                    if isinstance(result, BenchmarkError):
                        print(f"✗ Query {query_id:2d}: FAILED - {result}")
                        failed += 1
                    else:
                        print(f"✓ Query {query_id:2d}: {result.execution_time_ms:8.2f} ms  "
                              f"{result.throughput_mbps:8.2f} MB/s  "
                              f"{result.num_finished_splits}/{result.num_total_splits} splits")
                        total_time += result.execution_time_ms
                        total_bytes += result.raw_input_bytes
                        successful += 1
                
                print("-" * 80)
                print(f"\nSummary:")
                print(f"  Successful: {successful}/22")
                print(f"  Failed: {failed}/22")
                print(f"  Total Time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
                print(f"  Total Data: {total_bytes:,} bytes ({total_bytes/(1024**3):.2f} GB)")
                if total_time > 0:
                    avg_throughput = (total_bytes / (1024 * 1024)) / (total_time / 1000.0)
                    print(f"  Average Throughput: {avg_throughput:.2f} MB/s")
        
        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80)
        return 0
        
    except BenchmarkError as e:
        print(f"\n✗ Benchmark error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
