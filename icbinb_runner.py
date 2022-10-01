from PBP.PBP_net import PBP_net as PBP
from SSPBP.PBP_net import PBP_net as SSPBP
import numpy as np
from data import get_dataset
import os
import scipy.stats as ss
from tqdm import tqdm
import theano

datasets = ["Boston", "Concrete", "Energy_Efficiency", "Kin8nm", "KriSp_Precip", "Naval", "Power", "Wine", "Yacht"]

models = [[50], [10, 10], [10], [100]]

N_EPOCHS = 40

normalize = True

repetitions = 5

biasfree=True

seed = 0x79b7bff4

test_percent = 0.2

output = "results_bf_20220923"

np.random.seed(seed)
os.makedirs(output, exist_ok = True)

def rmse(m, y):
    return np.sqrt(np.mean((m - y) ** 2))

def log_lik(p, m, v, v_noise, y):
    # Z ~ p N(m, v) + (1-p) \delta_0
    # \epsilon ~ N(0, v_noise)
    # Y = Z + \epsilon
    # Y ~ p N(m, v + v_noise) + (1-p) N(y | 0, v_noise)
    slab = p * ss.norm.pdf(y, loc=m, scale=np.sqrt(v + v_noise))
    spike = (1 - p) * ss.norm.pdf(y, loc=0.0, scale=np.sqrt(v_noise))
    log_l = np.mean(np.log(slab + spike))
    return log_l


for model in models:
    for dataset in datasets:
        X_all, y_all = get_dataset(dataset)
        N_all = len(y_all)
        all_idxs = np.arange(N_all)
        for rep in tqdm(range(repetitions), desc=f"{dataset}, {model}"):
            result_fn = os.path.join(output, f"{dataset}_{model}_{rep}_of_{repetitions}.npy")
            if os.path.exists(result_fn):
            	continue
            test_idxs = np.random.choice(all_idxs, 
                                    int(np.round(N_all * test_percent)), 
                                    replace=False)
            X_test = X_all[test_idxs, :]
            y_test = y_all[test_idxs]
                                    
            train_idxs = np.setdiff1d(all_idxs, test_idxs)
            X_train = X_all[train_idxs, :]
            y_train = y_all[train_idxs]
            
            result = {
                "dataset": dataset,
                "model": model,
                "rep number": rep,
                "repetitions": repetitions,
                "test_idxs": test_idxs,
                "n_epochs": N_EPOCHS,
                "normalize": normalize,
                "X_test": X_test,
                "y_test": y_test,
                "biasfree": biasfree,
                "PBP": {},
                "SSPBP": {}
            }
            
            pbp = PBP(X_train, y_train, model, n_epochs = N_EPOCHS,
                    normalize = normalize, biasfree=biasfree)
            
            m, v, v_noise = pbp.predict(X_test)
            result["PBP"]["m"] = m
            result["PBP"]["v"] = v
            result["PBP"]["v_noise"] = v_noise
            result["PBP"]["test_rmse"] = rmse(m, y_test)
            result["PBP"]["test_ll"] = log_lik(1, m, v, v_noise, y_test)
            
                                    
            sspbp = SSPBP(X_train, y_train, model, n_epochs = N_EPOCHS,
                    normalize = normalize, biasfree=biasfree)
            
            m, v, p, v_noise = sspbp.predict(X_test)
            result["SSPBP"]["m"] = m
            result["SSPBP"]["v"] = v
            result["SSPBP"]["p"] = p
            result["SSPBP"]["v_noise"] = v_noise
            result["SSPBP"]["test_rmse"] = rmse(m, y_test)
            result["SSPBP"]["test_ll"] = log_lik(p, m, v, v_noise, y_test)
            

            np.save(result_fn, result)
            
