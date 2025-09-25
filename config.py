



PATHS = {
    'dataset_dir':         'dataset/got10k/versions/1/train',
    'dataset_dir_teacher': 'dataset/got10k/teacher/train',

    'finetuned_models':    'finetuned_models',
    'distilled_models':    'distilled_models'
}


#/home/ai_center/ai_users/yardenbakish/datasets/abhimanyukarshni/got10k/versions/1





def get_config(args):
    args.paths = PATHS
