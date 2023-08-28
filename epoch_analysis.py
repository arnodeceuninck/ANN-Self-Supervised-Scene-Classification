from training import load_log_info, TrainLogger, TrainConfig

def optimized_run_analysis(folder):
    optimized_log = load_log_info(f"{folder}/log.pickle")
    optimized_log.output_folder = folder
    optimized_log.analysis()

    lr = optimized_log.train_config.lr
    batch_size = optimized_log.train_config.batch_size

    print(f"learning rate {lr} and batch_size {batch_size}")

# optimized_run_analysis("outputs/optimized runs/supervised-optimized_20230823_014852")

# optimized_run_analysis("outputs/optimized runs/rot-only-optimized_20230823_041728_lr2933")
# optimized_run_analysis("outputs/optimized runs/rot-only-optimized-2_20230823_125417")
# optimized_run_analysis("outputs/optimized runs/clf-rot-pretext-optimized_20230823_213359")
#
# optimized_run_analysis("outputs/optimized runs/pert-only-optimized_20230823_034435")
# optimized_run_analysis("outputs/optimized runs/clf-pert-pretext-optimized-final_20230824_191530")
optimized_run_analysis("outputs/optimized runs/pretext_only_perturbation_lr10-5_bs15_ep10_20230824_225325")
# optimized_run_analysis("outputs/optimized runs/clf-pert-pretext-optimized-final-lr-5-ep9_20230825_045609")
