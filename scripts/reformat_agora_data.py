if __name__ == '__main__':
    import sys
    import os
    import shutil
    import numpy as np    
    import pickle
    import os.path as osp
    from tqdm import tqdm
    import pandas as pd
    from glob import glob
    #df.loc[:,"imgPath"].head()

    '''
    H36M kpts:
    'left_hip_extra', 'left_knee', 'left_ankle',    # 3
    'right_hip_extra', 'right_knee', 'right_ankle', # 6
    'neck_extra',                    # 8
    'headtop',                        # 10
    'left_shoulder', 'left_elbow', 'left_wrist',    # 13
    'right_shoulder', 'right_elbow', 'right_wrist', # 16
    '''

    BEDLAM_CLIFF_to_H36M_17 = [
        # 8,
        12, 13, 14, # ...,  LAnkle
        9, 10, 11,
        1, # Neck
        0, #0,  # Nose
        5, 6, 7, # LShoulder, LElbow, LWrist
        2, 3, 4, # RShoulder, RElbow, RWrist
    ]

    AGORA_gt_to_H36M_17 = [
        # 0,
        1, 4, 7,    # LL
        2, 5, 8,    # RL
        12,      # Neck
        15, #15,    # Nose
        16, 18, 20, # LA
        17, 19, 21, # RA
    ]   

    # [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]


    task = 'compile_corr_preds'


    # file_dir = '/data/lbidulka/CLIFF_AGORA_preds/'
    file_dir = '/data/lbidulka/BEDLAM/data/predictions/' + 'AGORA-CLIFF/'
    pred_files =  sorted(os.listdir(file_dir))

    cnet_data_dir = '/data/lbidulka/adapt_3d/' + 'AGORA/'

    # remove folders from list
    pred_files = [f for f in pred_files if (osp.isfile(osp.join(file_dir, f)) and f.endswith('.pkl'))]

    if task == 'rename_add_res':
        for file_name in pred_files:
            # remove "1280x720" from the name
            new_file_name = file_name.replace("person", "1280x720_person")
            shutil.move(os.path.join(file_dir, file_name), os.path.join(file_dir, new_file_name))
    elif task == 'rename_rem_res':
        for file_name in pred_files:
            # remove "1280x720" from the name
            new_file_name = file_name.replace("1280x720_", "")
            shutil.move(os.path.join(file_dir, file_name), os.path.join(file_dir, new_file_name))
    elif task == 'upscale_2d_joints':
        for file_name in tqdm(pred_files):
            path = os.path.join(file_dir, file_name)
            # load the dict from pkl file and rescale 2d coords
            dict = np.load(path, allow_pickle=True)
            dict['joints'] *= (2160/720)
            with open(path, 'wb') as f:
                pickle.dump(dict, f, protocol=2)
    elif task == 'downscale_2d_joints':
        for file_name in tqdm(pred_files):
            path = os.path.join(file_dir, file_name)
            # load the dict from pkl file and rescale 2d coords
            dict = np.load(path, allow_pickle=True)
            dict['joints'] /= (2160/720)
            with open(path, 'wb') as f:
                pickle.dump(dict, f, protocol=2)

    elif task == 'check_max_joints':
        max_vals = []
        for file_name in tqdm(pred_files):
            path = os.path.join(file_dir, file_name)
            dict = np.load(path, allow_pickle=True)
            max_vals.append(np.max(dict['joints']))
        print(np.max(max_vals))


    # GT Dataframe Tasks
    elif task == 'make_small_res_df':
        gt_df_path_root = '/data/lbidulka/pose_datasets/AGORA/gt_dataframe/'
        all_df = glob(os.path.join(gt_df_path_root, '*.pkl'))
        for df_path in tqdm(all_df):
            df = pd.read_pickle(df_path) #'/data/lbidulka/pose_datasets/AGORA/gt_dataframe/validation_0_withjv.pkl')

            # rescale gt_joints_2d_720p
            df['gt_joints_2d'] =  df['gt_joints_2d'].apply(lambda x: np.array(x)*(720/2160))
            # rename imgPath
            df['imgPath'] = df['imgPath'].apply(lambda x: x.replace('.png', '_1280x720.png'))

            # save the df
            df.to_pickle(df_path.replace('gt_dataframe', 'gt_dataframe_1280x720'))
    


    elif task == 'trim_BEDLAM_to_h36m_17':
        # trim the BEDLAM CLIFF predictions in the pred_files
        for file_name in tqdm(pred_files):
            path = os.path.join(file_dir, file_name)
            dict = np.load(path, allow_pickle=True)
            dict['allSmplJoints3d'] = dict['allSmplJoints3d'][BEDLAM_CLIFF_to_H36M_17]
            dict['joints'] = dict['joints'][BEDLAM_CLIFF_to_H36M_17]

            out_dir = file_dir.replace('AGORA-CLIFF', 'AGORA-CLIFF_h36m14')
            out_path = os.path.join(out_dir, file_name)
            with open(out_path, 'wb') as f:
                pickle.dump(dict, f, protocol=2)
        # trim gts in the gt_df
        gt_df_path_root = '/data/lbidulka/pose_datasets/AGORA/gt_dataframe_1280x720/'
        all_df = glob(os.path.join(gt_df_path_root, '*.pkl'))
        for df_path in tqdm(all_df):
            df = pd.read_pickle(df_path)
            # 2D
            trimmed_gts = []
            for gt_batch in df['gt_joints_2d'].values:
                trimmed_gt_batch = []
                for gt in gt_batch:
                    trimmed_gt = gt[AGORA_gt_to_H36M_17]
                    trimmed_gt_batch.append(trimmed_gt)
                trimmed_gts.append(trimmed_gt_batch)
            df['gt_joints_2d'] = trimmed_gts
            # 3D
            trimmed_gts = []
            for gt_batch in df['gt_joints_3d'].values:
                trimmed_gt_batch = []
                for gt in gt_batch:
                    trimmed_gt = gt[AGORA_gt_to_H36M_17]
                    trimmed_gt_batch.append(trimmed_gt)
                trimmed_gts.append(trimmed_gt_batch)
            df['gt_joints_3d'] = trimmed_gts
            
            df.to_pickle(df_path.replace('gt_dataframe_1280x720', 'gt_dataframe_1280x720_h36m14'))

    elif task == 'export_to_cnet':
        # export the trimmed H36M_14 predictions from the pred_files to cnet format (single numpy array)
        all_preds = []
        for file_name in tqdm(pred_files):
            path = os.path.join(file_dir, file_name).replace('AGORA-CLIFF', 'AGORA-CLIFF_h36m14')

            dict = np.load(path, allow_pickle=True)
            cnet_pred = dict['allSmplJoints3d']
            all_preds.append(cnet_pred)
        
        # reformat preds to numpy array
        all_preds = np.array(all_preds)
        all_preds = all_preds.reshape(all_preds.shape[0], -1)
        all_preds = np.array([all_preds, all_preds+1e-6])

        out_dir = file_dir.replace('AGORA-CLIFF', 'AGORA-CLIFF_h36m14')
        out_path = os.path.join(cnet_data_dir, 'bedlam_cliff_test.npy')
        np.save(out_path, all_preds)
    
    elif task == 'compile_corr_preds':
        # copy original predictions, but update them with the new corrected preds from CNet framework

        # load the new corr preds
        corr_preds_path = '/data/lbidulka/adapt_3d/AGORA/bedlam_cliff_corrected' + '.npy'
        # corr_preds_path = '/data/lbidulka/adapt_3d/AGORA/bedlam_cliff_corrected' + '_+TTT.npy'
        corr_preds = np.load(corr_preds_path)

        # iter of each of the original preds, replace the 3d preds with new ones, and save to new location
        for i, file_name in enumerate(tqdm(pred_files)):
            path = os.path.join(file_dir, file_name).replace('AGORA-CLIFF', 'AGORA-CLIFF_h36m14')
            dict = np.load(path, allow_pickle=True)

            dict['allSmplJoints3d'] =  corr_preds[i]

            out_dir = file_dir.replace('AGORA-CLIFF', 'AGORA-CLIFF_h36m14-corr')
            out_path = os.path.join(out_dir, file_name)
            with open(out_path, 'wb') as f:
                pickle.dump(dict, f, protocol=2)