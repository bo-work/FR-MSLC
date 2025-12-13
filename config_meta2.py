
def config_meta(args):
    if args['dataset'] == 'tls23':
        # ===================================
        args = dict(algorithm='MCRe',
                    dataset='tls23',
                    datadir="./data/tls23/",
                    savedir='./results/models/deepre/tls23/',
                    train_data_file='clean_train.npz',
                    test_data_file='clean_test.npz',
                    meta_data_file = 'meta_data_balance_50.npz',
                    categories_dict={'benign': 0, 'Banker_None_TLS_CC': 1, 'BuerLoader_None_TLS_CC': 2,
                                   'Caphaw.AH_None_TLS_CC': 3, 'Caphaw.A_None_TLS_CC': 4,
                                   'CobaltStrike_None_TLS_CC': 5, 'DridexCC_None_TLS_CC': 6,
                                   'Drixed_None_TLS_CC': 7, 'Dynamer!ac_None_TLS_CC': 8,
                                   'Panda.BZA!tr_None_TLS_CC': 9, 'PandaZeuSCC_None_TLS_CC': 10,
                                   'Qakbot_None_TLS_CC': 11, 'Shifu.A_None_TLS_CC': 12,
                                   'Tiggre_None_TLS_CC': 13, 'Tor_None_TLS_CC': 14,
                                   'Totbrick_None_TLS_CC': 15, 'TrickBotCC_None_TLS_CC': 16,
                                   'Upatre_None_TLS_CC': 17, 'Vawtrak.C_None_TLS_CC': 18,
                                   'arachni_arachni_TLS_scan': 19, 'burpsuite_burpsuite_TLS_scan': 20,
                                   'golistmero_golistmero_TLS_scan': 21, 'nessus_nessus_TLS_scan': 22},

                    reload=False,
                    reload_file='',
                    reload_acc=0,

                    num_classes=23,
                    input_dim=42,

                    noise_pattern=args['noise_pattern'],  ##asy or sy
                    noise_ratio=args['noise_ratio'],

                    # batch_size=256,
                    batch_size=512,  # <moco_queue 8192
                    num_workers=1,
                    epochs=50,
                    # epochs=2,
                    adjust_lr=1,
                    learning_rate=1e-2,

                    embedding_size=32,
                    moco_queue=8192,
                    moco_m=0.999,
                    temperature=0.1,
                    alpha=0.5,
                    pseudo_th=0.8,
                    proto_m=0.999,
                    lr=0.05,
                    cos=False,
                    schedule=[40, 80],
                    w_proto=1,
                    w_inst=1,
                    print_freq=300,

                    low_dim=16,
                    train_size=0,
                    val_size=0,
                    seed=42,

                    cluster_fea_type='feaemb',  # onlyfea,  onlyemb, feaemb
                    meta_shot_type='annote',  #'note' 'annote'
                    label_init_type='kmeans',  #'kmeans' 'meta'
                    meta_soft_label_k = 3,
                    optimal_k = 52,
                    )
        return args

    elif args['dataset'] == 'ids17c':
        # ===================================
        args = dict(algorithm='MCRe',
                    dataset='ids17c',
                    datadir="./data/ids17c/",
                    savedir='./results/models/deepre/ids17c/',
                    train_data_file='clean_train.npz',
                    test_data_file='clean_test.npz',
                    meta_data_file='meta_data_balance_50.npz',
                    categories_dict = {'benign': 0,
                                       'Patator': 1,
                                       'DoS': 2,
                                       'Infiltration': 3,
                                       'DDoS': 4,
                                       'Botnet': 5},
                    reload=False,
                    reload_file='',
                    reload_acc=0,

                    num_classes=6,
                    input_dim=50,

                    noise_pattern=args['noise_pattern'],  ##asy or sy
                    noise_ratio=args['noise_ratio'],

                    # batch_size=256,
                    batch_size=512,
                    num_workers=1,
                    epochs=50,
                    # epochs=2,
                    adjust_lr=1,
                    learning_rate=1e-2,

                    embedding_size=32,
                    moco_queue=8192,
                    moco_m=0.999,
                    temperature=0.1,
                    alpha=0.5,
                    pseudo_th=0.8,
                    proto_m=0.999,
                    lr=0.05,
                    cos=False,
                    schedule=[40, 80],
                    w_proto=1,
                    w_inst=1,
                    print_freq=300,

                    low_dim=16,
                    train_size=0,
                    val_size=0,
                    seed=42,

                    cluster_fea_type='feaemb',  # onlyfea,  onlyemb, feaemb
                    meta_shot_type='annote',  # 'note' 'annote'    note or not
                    label_init_type='kmeans', #'kmeans' 'meta'    kmeans first or meta first
                    meta_soft_label_k=3,
                    optimal_k=26,
                    )
        return args

