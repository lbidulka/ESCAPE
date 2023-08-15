import numpy as np

from cnet.full_body import adapt_net

class optuna_objective:
    def __init__(self, task, config, cnet, R_cnet, test):
        self.config = config
        self.task = task
        self.cnet = cnet
        self.R_cnet = R_cnet
        self.test = test
    
    def __call__(self, trial):
        if self.task == 'CNet':
            cnet_lr = trial.suggest_float('cnet_lr', 1e-5, 1e-3, log=True)
            cnet_eps = trial.suggest_int('cnet_eps', 3, 10)

            self.cnet = adapt_net(self.config, target_kpts=self.config.cnet_targets,
                        in_kpts=[kpt for kpt in range(17) if kpt not in [9,10]],
                        lr=cnet_lr, eps=cnet_eps)
            self.cnet.train()

            self.config.test_adapt = False
            self.config.test_eval_limit = 50_000

        elif self.task == 'TTT':
            self.config.test_adapt_lr = trial.suggest_float('test_adapt_lr', 1e-4, 1e-3, log=True) # 5e-4
            self.config.adapt_steps = trial.suggest_int('adapt_steps', 1, 3)    # 3

            self.config.test_adapt = True 
            self.config.test_eval_limit = 512
        else:
            raise NotImplementedError
            
        if self.config.testset == 'PW3D':
            self.config.EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
            self.config.EVAL_JOINTS.sort()
            PW3D_testlen = 35_515
            self.config.test_eval_subset = np.random.choice(PW3D_testlen, min(self.config.test_eval_limit, PW3D_testlen), replace=False)
        else:
            raise NotImplementedError

        # Run eval
        test_summary = self.test(self.cnet, self.R_cnet, self.config, print_summary=False)
        gain_1 = []
        gain_2 = []
        for backbone in test_summary.keys():
            if self.task == 'CNet':
                 gain_1.append((test_summary[backbone]['w/CN']['PA-MPJPE'] - test_summary[backbone]['vanilla']['PA-MPJPE']))
                 gain_2.append((test_summary[backbone]['w/CN']['MPJPE'] - test_summary[backbone]['vanilla']['MPJPE']))
            elif self.task == 'TTT':
                 gain_1.append((test_summary[backbone]['+TTT']['PA-MPJPE'] - test_summary[backbone]['vanilla']['PA-MPJPE']) - 
                               (test_summary[backbone]['w/CN']['PA-MPJPE'] - test_summary[backbone]['vanilla']['PA-MPJPE']))
                 gain_2.append((test_summary[backbone]['+TTT']['MPJPE'] - test_summary[backbone]['vanilla']['MPJPE']) -
                                 (test_summary[backbone]['w/CN']['MPJPE'] - test_summary[backbone]['vanilla']['MPJPE']))
        print(f"improvement (PA-MPJPE): avg: ({round(np.mean(gain_1),4)}), {gain_1}")
        print(f"improvement (MPJPE): avg: ({round(np.mean(gain_2),4)}), {gain_2}")
        
        return np.mean(gain_1), np.mean(gain_2)