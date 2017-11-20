python mnist_deep.py --subsample_rate 10 --random_seed 100 --train_dir 'origin'
python mnist_influence.py --subsample_rate 10 --random_seed 100 --ckpt_path 'origin/model.ckpt-19999'
python mnist_deep.py --subsample_rate 10 --random_seed 100 --remove_idx_file './origin/n_best_idx.txt' --train_dir 'worse_model'
python mnist_deep.py --subsample_rate 10 --random_seed 100 --remove_idx_file './origin/n_worst_idx.txt' --train_dir 'better_model'
