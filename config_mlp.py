
def config(args):
    if args.dataset == 'tls23':
        # ===================================
        args.expriment_type = 'metasize'  # metasize, unannotated
        args.warmup_dir_path = './results/models/warmup/TLS23/MLP/'
        args.deepre_dir_path = './results/models/deepre/tls23/'
        args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_50r_33ep.pth'
        args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        # change in different experiments
        if args.expriment_type == 'metasize':
            args.corruption_type = 'sym'  # asym sym
            args.corruption_prob = 0.2  # 0, 0.2 , 0.4, 0.6, 0.8

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'  # note annote
            args.meta_size_prob = 200  # 500, 200

            args.lr_vent = 0.1

            # args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asym_80r_10ep.pth'
            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_sym_20r_43ep.pth'
            # args.model_kmeans_results_path = args.deepre_dir_path + 'model_kmeans_feaemb_sym0.8.pkl'
            # args.meta_ex_results_path = args.deepre_dir_path + 'meta_data_feaemb_sym0.8.npz'
            args.train_kmeans_info_results_path = args.deepre_dir_path + 'model_kmeans_labels_map_'+args.cluster_fea_type+'_'+ args.corruption_type + str(args.corruption_prob) + '.npz'

        elif args.expriment_type == 'unannotated':
            args.corruption_type = 'ask'  # ask
            args.corruption_prob = 0.2   # 0.2 0.5

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'   #note annote
            args.meta_size_prob = 200   #500, 200

            args.lr_vent = 0.1

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_ask_2r_76ep.pth'
            # args.model_results_path = args.warmup_dir_path + 'model_checkpoint_sym_80r_53ep.pth'
            # args.model_kmeans_results_path = args.deepre_dir_path + 'model_kmeans_feaemb_sym0.8.pkl'
            # args.meta_ex_results_path = args.deepre_dir_path + 'meta_data_feaemb_sym0.8.npz'
            args.train_kmeans_info_results_path = args.deepre_dir_path + 'model_kmeans_labels_map_feaemb_asym0.npz'
        # ====================================

        args.data_path = './data/tls23/'
        args.results_dir_path = './results/'
        args.ifload = True
        args.fea_flag = True

        args.feature_size = 42
        args.num_classes = 23
        args.metadata_name = 'meta_data_balance_200.npz'
        # args.metadata_name = '_'.join(['meta_data', args.cluster_fea_type, args.meta_type, str(args.meta_size_prob)]) + '.npz'
        # args.label_kmeans_name = '_'.join(['label_kmeans', args.cluster_fea_type]) + '.npz'
        # args.model_kmeans_results_path = args.deepre_dir_path + 'model_kmeans_' + args.cluster_fea_type + args.corruption_type + str(int(args.corruption_prob)*10) + '.pkl'
        args.showiter = 23

        args.epochs = 120
        args.epochs_wp = 80
        args.epochs_model_wp = 100
        args.epochs_g_delay = 70

        args.lr = 1e-2
        args.batch_size = 1024
        args.weight_decay = 5e-4
        args.momentum = 0.9

        args.cls = 'mlp'

        args.categories = ['benign', 'Banker_None_TLS_CC', 'BuerLoader_None_TLS_CC', 'Caphaw.AH_None_TLS_CC',
                           'Caphaw.A_None_TLS_CC', 'CobaltStrike_None_TLS_CC', 'DridexCC_None_TLS_CC',
                           'Drixed_None_TLS_CC', 'Dynamer!ac_None_TLS_CC', 'Panda.BZA!tr_None_TLS_CC',
                           'PandaZeuSCC_None_TLS_CC', 'Qakbot_None_TLS_CC', 'Shifu.A_None_TLS_CC', 'Tiggre_None_TLS_CC',
                           'Tor_None_TLS_CC', 'Totbrick_None_TLS_CC', 'TrickBotCC_None_TLS_CC', 'Upatre_None_TLS_CC',
                           'Vawtrak.C_None_TLS_CC', 'arachni_arachni_TLS_scan', 'burpsuite_burpsuite_TLS_scan',
                           'golistmero_golistmero_TLS_scan', 'nessus_nessus_TLS_scan']
        return args

    elif args.dataset == 'ids17c':
        # ===================================
        # ===================================
        args.expriment_type = 'metasize'  #  metasize, unannotated
        args.warmup_dir_path = './results/models/warmup/IDS17c/MLP/'
        args.deepre_dir_path = './results/models/deepre/ids17c/'
        args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_50r_33ep.pth'
        args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        # change in different experiments
        if args.expriment_type == 'metasize':
            args.corruption_type = 'sym'  # asym sym
            args.corruption_prob = 0.8  # 0, 0.2, 0.4, 0.6, 0.8

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'  # note annote
            args.meta_size_prob = 200  # 500, 200

            args.lr_vent = 0.1

            # args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asym_80r_0ep.pth'
            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_sym_80r_6ep.pth'
            # args.model_kmeans_results_path = args.deepre_dir_path + 'model_kmeans_feaemb_sym0.8.pkl'
            # args.meta_ex_results_path = args.deepre_dir_path + 'meta_data_feaemb_sym0.8.npz'
            args.train_kmeans_info_results_path = args.deepre_dir_path + 'model_meta_labels_map_'+args.cluster_fea_type+'_'+ args.corruption_type + str(args.corruption_prob) + '.npz'
            # args.train_kmeans_info_results_path = args.deepre_dir_path + 'model_kmeans_labels_map_feaemb_'+args.corruption_type+str(args.corruption_prob)+'.npz'

        if args.expriment_type == 'unannotated':
            args.corruption_type = 'ask'  # ask
            args.corruption_prob = 0.1  # 0.1  0.5

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'  # note annote
            args.meta_size_prob = 200  # 500, 200

            args.lr_vent = 0.1

            # args.model_results_path = args.warmup_dir_path + 'model_checkpoint_ask_random_5r_18ep.pth'
            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_ask_close_1r_54ep.pth'
            # args.model_kmeans_results_path = args.deepre_dir_path + 'model_kmeans_feaemb_sym0.8.pkl'
            # args.meta_ex_results_path = args.deepre_dir_path + 'meta_data_feaemb_sym0.8.npz'
            args.train_kmeans_info_results_path = args.deepre_dir_path + 'model_meta_labels_map_feaemb_asym0.npz'
            # args.train_kmeans_info_results_path = args.deepre_dir_path + 'model_kmeans_labels_map_feaemb_'+args.corruption_type+str(args.corruption_prob)+'.npz'
        # ====================================

        args.data_path = './data/ids17c/'
        args.results_dir_path = './results/'
        args.ifload = True
        args.fea_flag = True

        args.feature_size = 50
        args.num_classes = 6
        args.metadata_name = 'meta_data_balance_200.npz'
        # args.metadata_name = '_'.join(['meta_data', args.cluster_fea_type, args.meta_type, str(args.meta_size_prob)]) + '.npz'
        # args.label_kmeans_name = '_'.join(['label_kmeans', args.cluster_fea_type]) + '.npz'
        # args.model_kmeans_results_path = args.deepre_dir_path + 'model_kmeans_' + args.cluster_fea_type + args.corruption_type + str(int(args.corruption_prob)*10) + '.pkl'
        args.showiter = 54

        args.epochs = 110
        args.epochs_wp = 80
        args.epochs_model_wp = 100
        args.epochs_g_delay = 70

        args.lr = 1e-3
        args.batch_size = 1024
        args.weight_decay = 5e-4
        args.momentum = 0.9

        args.cls = 'mlp'

        args.categories = ['benign', 'Patator', 'DoS', 'Infiltration', 'DDoS', 'Botnet']
        return args

def config_warmup(args):
    if args.dataset == 'tls23':
        # ===================================
        args.expriment_type = 'unannotated'    # main, metasize, unannotated
        args.warmup_dir_path = './results/models/warmup/TLS23/MLP/'
        args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_50r_33ep.pth'
        args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        # change in different experiments
        if args.expriment_type == 'metasize':
            args.corruption_type = 'sym'  # sym, asym
            args.corruption_prob = 0   #0, 0.2, 0.4, 0.6, 0.8

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'   #note annote
            args.meta_size_prob = 200   #500, 200
            args.dc_expend_ratio = 0.2

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_70r_21ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        elif args.expriment_type == 'unannotated':
            args.corruption_type = 'ask'  # ask
            args.corruption_prob = 0.2   # 0.2 0.5

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'   #note annote
            args.meta_size_prob = 200   #500, 200
            args.dc_expend_ratio = 0.2

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_70r_21ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'
        # ====================================

        args.data_path = './data/tls23/'
        args.results_dir_path = './results/'
        args.ifload = False
        args.fea_flag = True

        args.feature_size = 42
        args.num_classes = 23
        args.metadata_name = 'meta_data_balance_200.npz'
        # args.metadata_name = '_'.join(['meta_data', args.cluster_fea_type, args.meta_type, str(args.meta_size_prob)]) + '.npz'
        args.label_kmeans_name = '_'.join(['label_kmeans', args.cluster_fea_type]) + '.npz'
        args.showiter = 23

        args.epochs = 79
        args.epochs_wp = 80
        args.epochs_model_wp = 100
        args.epochs_g_delay = 70

        args.lr = 1e-2
        args.batch_size = 1024
        args.weight_decay = 5e-4
        args.momentum = 0.9

        args.cls = 'mlp'

        args.categories = ['benign', 'Banker_None_TLS_CC', 'BuerLoader_None_TLS_CC', 'Caphaw.AH_None_TLS_CC', 'Caphaw.A_None_TLS_CC', 'CobaltStrike_None_TLS_CC', 'DridexCC_None_TLS_CC', 'Drixed_None_TLS_CC', 'Dynamer!ac_None_TLS_CC', 'Panda.BZA!tr_None_TLS_CC', 'PandaZeuSCC_None_TLS_CC', 'Qakbot_None_TLS_CC', 'Shifu.A_None_TLS_CC', 'Tiggre_None_TLS_CC', 'Tor_None_TLS_CC', 'Totbrick_None_TLS_CC', 'TrickBotCC_None_TLS_CC', 'Upatre_None_TLS_CC', 'Vawtrak.C_None_TLS_CC', 'arachni_arachni_TLS_scan', 'burpsuite_burpsuite_TLS_scan', 'golistmero_golistmero_TLS_scan', 'nessus_nessus_TLS_scan']
        return args

    elif args.dataset == 'ids17c':
        # ===================================
        args.expriment_type = 'unannotated'  # main, metasize, unannotated
        args.warmup_dir_path = './results/models/warmup/IDS17c/MLP/'
        args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_50r_33ep.pth'
        args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        # change in different experiments
        if args.expriment_type == 'metasize':
            args.corruption_type = 'sym'  # sym, asym,
            args.corruption_prob = 0.2  # 0, 0.2, 0.4, 0.6, 0.8

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'  # note annote
            args.meta_size_prob = 200  # 500, 200
            args.dc_expend_ratio = 0.2

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_70r_21ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'

        elif args.expriment_type == 'unannotated':
            args.corruption_type = 'ask'  # ask
            args.corruption_prob = 0.1   # 0.1 0.5

            args.cluster_fea_type = 'feaemb'  # onlyfea,  onlyemb, feaemb
            args.meta_type = 'annote'   #note annote
            args.meta_size_prob = 200   #500, 200
            args.dc_expend_ratio = 0.2

            args.lr_vent = 0.01

            args.model_results_path = args.warmup_dir_path + 'model_checkpoint_asy_70r_21ep.pth'
            args.model_aaec_results_path = args.warmup_dir_path + 'model_aaec_checkpoint_random_001_75.pth'
        # ====================================

        args.data_path = './data/ids17c/'
        args.results_dir_path = './results/'
        args.ifload = False
        args.fea_flag = True

        args.feature_size = 50
        args.num_classes = 6
        args.metadata_name = 'meta_data_balance_200.npz'
        # args.metadata_name = '_'.join(['meta_data', args.cluster_fea_type, args.meta_type, str(args.meta_size_prob)]) + '.npz'
        args.label_kmeans_name = '_'.join(['label_kmeans', args.cluster_fea_type]) + '.npz'
        args.showiter = 54


        args.epochs = 79
        args.epochs_wp = 80
        args.epochs_model_wp = 100
        args.epochs_g_delay = 70

        args.lr = 1e-3
        args.batch_size = 1024
        args.weight_decay = 5e-4
        args.momentum = 0.9

        args.cls = 'mlp'

        args.categories = ['benign', 'Patator', 'DoS', 'Infiltration', 'DDoS', 'Botnet']
        return args