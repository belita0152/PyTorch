import os
import glob
import numpy as np

src_path = os.path.join(os.getcwd(), '..', 'data', 'sleep_edfx')
base_path = sorted(glob.glob(os.path.join(src_path, 'sc*e0.npz')))
npz_list = [np.load(fp) for fp in base_path]  # x : (841, 3, 3000), y : (841, )


def group_cross_validation(paths, test_size=0.25, holdout_subject_size: int = 25):
    size = len(paths)
    train_paths, val_paths = paths[:int(size * (1 - test_size))], paths[int(size * (1 - test_size)):]
    train_paths, eval_paths = train_paths[:len(train_paths) - holdout_subject_size], \
                             train_paths[len(train_paths) - holdout_subject_size:]

    print('[K-Group Cross Validation]')
    print('   >> Train Subject Size : {}'.format(len(train_paths)))
    print('   >> Validation Subject Size : {}'.format(len(val_paths)))
    print('   >> Evaluation Subject Size : {}\n'.format(len(eval_paths)))

    return {'train_paths': train_paths, 'val_paths': val_paths, 'eval_paths': eval_paths}

total_path = group_cross_validation(base_path)


"""
[K-Group Cross Validation]
   >> Train Subject Size : 51
   >> Validation Subject Size : 26
   >> Evaluation Subject Size : 25
"""


if __name__ == '__main__':
    x1 = npz_list[0].f.x[:, 0, :]
    x2 = npz_list[0].f.x[:, 1, :]
    x3 = npz_list[0].f.x[:, 2, :]
    y = npz_list[0].f.y

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 8))
    plt.plot(npz_list[0].f.x[:, 0, :].T, color='red')
    plt.plot(npz_list[0].f.y.T, color='blue')
    plt.show()


