import torch
import numpy as np

from core.cnet_eval import cnet_TTT_loss



def get_inference_time(config, cnet, R_cnet):
    cnet.eval()
    R_cnet.eval()

    # dummy backbone keypoint sample
    sample = torch.randn(1, 17, 3, dtype=torch.float).to(config.device)
    
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50_000
    timings = np.zeros((repetitions,1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = cnet(sample)
    print("Measuring...")
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = cnet(sample)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    # REPORT    
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f'\nINFERENCE TIME OF CNet, {repetitions} REPS:')
    print(f'mean: {round(mean_syn, 4)} ms,  std: {round(std_syn,4)} ms')


    def set_bn_eval(module):
        ''' Batch Norm layer freezing '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    if config.test_adapt:
        repetitions = 50_000
        timings = np.zeros((repetitions,1))
        for rep in range(repetitions):
            starter.record()

            # THIS IS SUPER INNEFICIENT, AND WASTES MY TIME :(
            cnet.load_cnets(print_str=False)
            cnet.cnet.train()
            # Setup Optimizer
            cnet.cnet.apply(set_bn_eval)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnet.cnet.parameters()),
                                            lr=config.test_adapt_lr)
            cnet_in = sample.detach().clone()
            for i in range(config.adapt_steps):
                optimizer.zero_grad()
                corrected_pred, corr_idx = cnet(cnet_in, ret_corr_idxs=True,)

                # only do TTT if there are some corrected samples
                if corr_idx.sum() > 0:
                    Rcnet_pred = R_cnet(corrected_pred)
                    loss = cnet_TTT_loss(config, sample, Rcnet_pred, corrected_pred, None)
                    loss.backward()
                    optimizer.step()
            
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        # REPORT    
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(f'\nTTT INFERENCE TIME OF CNet, {repetitions} REPS:')
        print(f'mean: {round(mean_syn, 4)} ms,  std: {round(std_syn,4)} ms')
