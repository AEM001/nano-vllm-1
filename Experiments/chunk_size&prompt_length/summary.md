Final experiment summary
By chunk size
128
Avg TTFT: 1.431s
Avg Throughput: 1.03 seq/s
256
Avg TTFT: 1.410s
Avg Throughput: 1.03 seq/s
512
Avg TTFT: 1.378s
Avg Throughput: 1.04 seq/s
1024
Avg TTFT: 1.589s
Avg Throughput: 1.01 seq/s
By prompt length
256
Avg TTFT: 1.171s
Avg Throughput: 1.21 seq/s
512
Avg TTFT: 1.324s
Avg Throughput: 1.07 seq/s
1024
Avg TTFT: 1.860s
Avg Throughput: 0.80 seq/s
Conclusion
The experiment is now correct and runnable end-to-end.
The results now reflect chunk-size behavior, not prompt-cache artifacts.
Longer prompts clearly increase TTFT and reduce throughput.
In this run, 512 gave the best average balance among the tested chunk sizes.
Status
Code fixed
Experiment fixed
Run completed successfully
Results saved to experiment_results.json