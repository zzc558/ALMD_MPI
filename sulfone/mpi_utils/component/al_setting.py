
DEFAULT_AL_SETTING = {
        "global":{ # settings of global control
            "res_dir":"results/TestRun", # directory to save all metadata and results
            "pl_process": 4,
            "md_process": 93,
            "qm_process": 50,
            "ml_process": 4,
            "elements": ['S', 'O', 'O', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H'],
            "n_states": 7, # number of states (ground + excited),
            "model_per_pl": 1,
            'dft_time': 900, # estimated dft running time (energy+force) in seconds
            'n_gpu': 4, # number of gpu per node
            'update_time': 1800, # time interval to save the progress
            },
        "manager":{
            'std_thresh': 0.5, # threshold of model predictions std
            'std_thresh_record': 0.5, # threshold for recording the std values
            'retrain_step': 50, # increment of training set
            },
        "passive":{ # settings of passive learner process
            'gpu_list': [0, 1, 2, 3], # list of gpus     
            "path": "models/pl_models", # path of pre-trained models
            'model_name': 'eg',
            "test_path": "data/testing_23604_random_order_correct.json",
            },
        'model_hyper':{
            'general':{
                'model_type' : 'mlp_eg',
                'batch_size_predict': 265,
                },
            'model':{
                'use_dropout': False,
                "dropout": 0.0,
                'atoms': 38,
                'states': 7,
                'depth' : 6,
                'nn_size' : 5000,   # number of neurons per layer
                'use_reg_weight' : {'class_name': 'L2', 'config': {'l2': 1e-4}},
                "activ": {"class_name": "sharp_softplus", "config": {"n": 10.0}},
                'invd_index' : True,
                #'activ': 'relu',
                },
            'retraining':{
                "energy_only": False,
                "masked_loss": True,
                "auto_scaling": {"x_mean": True, "x_std": True, "energy_std": True, "energy_mean": True},
                "loss_weights": {"energy": 1, "force": 1},
                'learning_rate': 1e-06,
                "initialize_weights": False,
                "val_disjoint": True,
                'normalization_mode' : 1,
                'epo': 10000,
                'val_split' : 0.25,
                'batch_size' : 32,
                "epostep": 10,
                "exp_callback": {"use": False, "factor_lr": 0.9, "epomin": 100, "learning_rate_start": 1e-06, "epostep": 20},
                }
            },
        'ml':{ # settings of machine learning process
            "path": "models/ml_models", # path of pre-trained models
            },
        'md':{ # settings of molecular dynamic process
            'input_file': 'input_gsh',
            'initial_cond_path': 'data/initial_conditions_all_from_src.npy',
            'bond_index': {
                'OS': [[0, 1], [0, 2]], 'CS': [[0, 3], [0, 13]],\
                'CC': [[3, 4], [3, 5], [4, 6], [5, 8], [6, 10], [6, 34],\
                       [8, 10], [13, 14], [13, 15], [14, 16], [15, 18],\
                       [16, 20], [18, 20], [20, 23], [23, 24], [23, 25],\
                       [24, 26], [25, 28], [26, 30], [28, 30]],\
                'CH': [[4, 7], [5, 9], [8, 11], [10, 12], [14, 17], [15, 19],\
                       [16, 21], [18, 22], [24, 27], [25, 29], [26, 31],\
                       [28, 32], [30, 33], [34, 35], [34, 36], [34, 37]]},
            'bond_limit':{
                'OS': 2.5, 'CS': 4.0, 'CC': 2.3, 'CH': 6.0
                }
            }
        }
