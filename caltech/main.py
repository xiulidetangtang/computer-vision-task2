
import os
from experiments.finetune import run_finetune_experiments
from experiments.from_scratch_experiments import run_from_scratch_experiments
from utils.visualization import plot_training_curves, compare_pretrained_vs_scratch

def main():

    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    
    print("开始微调实验...")
    pretrained_results = run_finetune_experiments()
    
    print("开始从零训练实验...")
    scratch_results = run_from_scratch_experiments()
    
    print("生成可视化结果...")
    plot_training_curves(pretrained_results)
    compare_pretrained_vs_scratch(pretrained_results, scratch_results)
    
    print("实验完成！")

if __name__ == "__main__":
    main()