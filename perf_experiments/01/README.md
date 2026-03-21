# Experiment A: Prefill Throughput vs Batch Size

This experiment tests how different `max_num_batched_tokens` values affect prefill throughput in nano-vllm.

## Usage

### Basic Usage (with default prompts.txt)
```bash
python a.py
```

### Custom prompts file
```bash
python a.py --prompts my_prompts.txt
```

### Custom batch sizes
```bash
python a.py --batch-sizes "1024,2048,4096,8192"
```

### Full customization
```bash
python a.py \
  --prompts short_prompts.txt \
  --batch-sizes "512,1024,2048,4096" \
  --max-seqs 10 \
  --max-tokens 2048 \
  --model /path/to/your/model
```

## Parameters

- `--prompts`: Path to prompts file (default: prompts.txt)
- `--model`: Path to model directory (default: /home/albert/learn/models/Qwen3-0.6B/)
- `--batch-sizes`: Comma-separated list of batch sizes to test (default: 512,1024,2048,4096,8192,16384,32768,65536)
- `--max-seqs`: Maximum number of sequences (default: 25)
- `--max-tokens`: Maximum tokens to generate (default: 4096)

## Prompts File Format

Create a text file with one prompt per line:

```
This is prompt 1.
This is prompt 2.
This is prompt 3.
```

## Output

The script will:
1. Test each batch size configuration
2. Measure prefill throughput (tokens/second)
3. Display a results summary table
4. Identify the optimal batch size

## Example Output

```
================================================================================
EXPERIMENT A: RESULTS SUMMARY
================================================================================
Batch Size   Time (s)   Prefill (t/s)   Generated   Seqs
--------------------------------------------------------------------------------
512         45.23      89.12          4096        20
1024        23.45      171.89         4096        20
2048        12.67      317.45         4096        20
4096        7.89       509.23         4096        20

================================================================================
OPTIMAL BATCH SIZE: 4096
Prefill throughput: 509.23 tokens/s
================================================================================
```
