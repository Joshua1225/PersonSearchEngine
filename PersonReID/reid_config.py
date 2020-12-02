import torchvision.transforms as T

class ReID_config:
    # ReID Config that do not need change
    # test
    dataset = 'market1501'
    arch = 'vmgn_hgnn'
    global_branch = True
    seed = 1
    use_avai_gpus = False
    use_cpu = False
    save_json=''
    workers = 0
    split_id=0
    cuhk03_labeled=False
    cuhk03_classic_split=False
    re_rank = False
    vis_ranked_res = False

config = ReID_config()
