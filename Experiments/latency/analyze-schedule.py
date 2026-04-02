import re

def analyze_timing_data(log_file="Experiments/latency/time_log.txt"):
    """Analyze timing data from log file with start/end timestamps."""
    
    # Read all lines
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Parse timestamps
    schedule_times = []
    step_times = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "start_schedule:" in line:
            start_time = float(line.split(":")[1])
            if i + 1 < len(lines) and "end_schedule:" in lines[i+1]:
                end_time = float(lines[i+1].split(":")[1])
                schedule_time = end_time - start_time
                schedule_times.append(schedule_time)
                
                # Look for next start_schedule to calculate step time
                if i + 2 < len(lines) and "start_schedule:" in lines[i+2]:
                    next_start = float(lines[i+2].split(":")[1])
                    step_time = next_start - start_time
                    step_times.append(step_time)
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    if not schedule_times:
        print("No schedule timing data found")
        return
    
    # Calculate schedule statistics
    schedule_count = len(schedule_times)
    schedule_sum = sum(schedule_times)
    schedule_avg = schedule_sum / schedule_count
    schedule_min = min(schedule_times)
    schedule_max = max(schedule_times)
    
    print("=== Schedule Timing Statistics ===")
    print(f"Schedule calls: {schedule_count}")
    print(f"Sum: {schedule_sum:.6f} seconds")
    print(f"Average: {schedule_avg:.6f} seconds ({schedule_avg*1000:.3f} ms)")
    print(f"Min: {schedule_min:.6f} seconds ({schedule_min*1000:.3f} ms)")
    print(f"Max: {schedule_max:.6f} seconds ({schedule_max*1000:.3f} ms)")
    
    # Calculate step statistics (time between schedule starts)
    if step_times:
        step_count = len(step_times)
        step_sum = sum(step_times)
        step_avg = step_sum / step_count
        step_min = min(step_times)
        step_max = max(step_times)
        
        print(f"\n=== Step Timing Statistics ===")
        print(f"Steps: {step_count}")
        print(f"Sum: {step_sum:.6f} seconds")
        print(f"Average: {step_avg:.6f} seconds ({step_avg*1000:.3f} ms)")
        print(f"Min: {step_min:.6f} seconds ({step_min*1000:.3f} ms)")
        print(f"Max: {step_max:.6f} seconds ({step_max*1000:.3f} ms)")
        
        # Calculate overhead
        total_schedule_time = schedule_sum
        total_step_time = step_sum
        overhead_percentage = (total_schedule_time / total_step_time) * 100
        
        print(f"\n=== Schedule Overhead ===")
        print(f"Total schedule time: {total_schedule_time:.6f}s")
        print(f"Total step time: {total_step_time:.6f}s")
        print(f"Schedule overhead: {overhead_percentage:.2f}% of total step time")
        print(f"Average overhead per step: {schedule_avg*1000:.3f} ms")

if __name__ == "__main__":
    analyze_timing_data()