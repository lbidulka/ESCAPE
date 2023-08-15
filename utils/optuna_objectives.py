import numpy as np


class optuna_objective:
    def __init__(self, config, cnet, R_cnet, test):
        self.config = config
        self.cnet = cnet
        self.R_cnet = R_cnet
        self.test = test
    
    def __call__(self, trial):
        self.config.test_adapt_lr = trial.suggest_float('test_adapt_lr', 1e-4, 1e-3, log=True) # 5e-4
        self.config.adapt_steps = trial.suggest_int('adapt_steps', 1, 3)    # 3

        # Eval Settings
        self.config.test_adapt = True
        self.config.test_eval_limit = 512 # 50_000    For debugging cnet testing (3DPW has 35515 test samples)
        if self.config.testset == 'PW3D':
            self.config.EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
            self.config.EVAL_JOINTS.sort()
            PW3D_testlen = 35_515
            self.config.test_eval_subset = np.random.choice(PW3D_testlen, 
                                                            min(self.config.test_eval_limit, PW3D_testlen), 
                                                            replace=False)
        else:
            raise NotImplementedError

        # Run eval
        test_summary = self.test(self.cnet, self.R_cnet, self.config, print_summary=False)

        # Collect gains from using CNet+TTT over just CNet
        TTT_gain1 = []
        TTT_gain2 = []
        for backbone in test_summary.keys():
            rel_cn_pampjpe = test_summary[backbone]['w/CN']['PA-MPJPE'] - test_summary[backbone]['vanilla']['PA-MPJPE']
            rel_TTT_pampjpe = test_summary[backbone]['+TTT']['PA-MPJPE'] - test_summary[backbone]['vanilla']['PA-MPJPE']

            rel_cn_mpjpe = test_summary[backbone]['w/CN']['MPJPE'] - test_summary[backbone]['vanilla']['MPJPE']
            rel_TTT_mpjpe = test_summary[backbone]['+TTT']['MPJPE'] - test_summary[backbone]['vanilla']['MPJPE']

            TTT_gain1.append(rel_TTT_pampjpe - rel_cn_pampjpe)
            TTT_gain2.append(rel_TTT_mpjpe - rel_cn_mpjpe)
            
        return np.median(TTT_gain1), np.median(TTT_gain2)