import numpy as np
import os

datasets = ["Boston", "Concrete", "Energy_Efficiency", "Kin8nm", "KriSp_Precip", "Naval", "Power", "Wine", "Yacht"]
# results = "results_20220922"
results = "results_bf_20220923"
repetitions = 5

def get_rmses(dataset, model):
    pbp_rmses = []
    sspbp_rmses = []
    for rep in range(repetitions):
        fn = os.path.join(results, f"{dataset}_{model}_{rep}_of_{repetitions}.npy")
        result = np.load(fn, allow_pickle=True).item()
        pbp_rmses.append(result["PBP"]["test_rmse"])
        sspbp_rmses.append(result["SSPBP"]["test_rmse"])
    return ((np.mean(pbp_rmses), np.std(pbp_rmses) / np.sqrt(repetitions)),
            (np.mean(sspbp_rmses), np.std(sspbp_rmses) / np.sqrt(repetitions)))

def fmt(number):
    return "%0.3f" % number
            
def print_latex(modelA, modelB):
    print(f"Comparison of {modelA}, {modelB}:\n")
    for dataset in datasets:
        tokens = [None] * 5
        tokens[0] = dataset.replace("_", "\_")
        pbp_A, sspbp_A = get_rmses(dataset, modelA)
        pbp_B, sspbp_B = get_rmses(dataset, modelB)
        if pbp_A[0] < sspbp_A[0]:
            tokens[1] = "\\textbf{%s$\\pm$%s}" % (fmt(pbp_A[0]), fmt(pbp_A[1]))
            tokens[2] = "%s$\\pm$%s" % (fmt(sspbp_A[0]), fmt(sspbp_A[1]))
        else:
            tokens[1] = "%s$\\pm$%s" % (fmt(pbp_A[0]), fmt(pbp_A[1]))
            tokens[2] = "\\textbf{%s$\\pm$%s}" % (fmt(sspbp_A[0]), fmt(sspbp_A[1]))
        if pbp_B[0] < sspbp_B[0]:
            tokens[3] = "\\textbf{%s$\\pm$%s}" % (fmt(pbp_B[0]), fmt(pbp_B[1]))
            tokens[4] = "%s$\\pm$%s" % (fmt(sspbp_B[0]), fmt(sspbp_B[1]))
        else:
            tokens[3] = "%s$\\pm$%s" % (fmt(pbp_B[0]), fmt(pbp_B[1]))
            tokens[4] = "\\textbf{%s$\\pm$%s}" % (fmt(sspbp_B[0]), fmt(sspbp_B[1]))

        print("%s \\\\" % "\t&\t".join(tokens))
    print()
print_latex([50], [10, 10])
print_latex([10], [100])
