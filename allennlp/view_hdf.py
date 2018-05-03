import sys
import h5py


def view(in_file):
    infh = h5py.File(in_file, 'r')
    keys = list(infh.keys())
    for key in keys:
        print(key)
        print(infh[key].value)
        print(infh[key].value.shape)
        print()
    infh.close()


if __name__ == '__main__':
    view(sys.argv[1])
