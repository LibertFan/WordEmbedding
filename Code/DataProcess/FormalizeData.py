import os
import sys


class IMDB(object):
    def __init__(self):
        self._raw_data_root = os.path.abspath('../../Data')
        self._data_folder = [os.path.join(self._raw_data_root, 'raw/aclImdb/train/neg'),
                             os.path.join(self._raw_data_root, 'raw/aclImdb/train/pos'),
                             os.path.join(self._raw_data_root, 'raw/aclImdb/train/unsup')]
        lines = self._load_data()
        self._save_path = os.path.join(self._raw_data_root, 'raw/imdb.txt')
        self._formalize(lines)

    def _load_data(self):
        lines = []
        for folder in self._data_folder:
            for file_name in os.listdir(folder):
                print(file_name)
                file_path = os.path.join(folder, file_name)
                with open(file_path, 'r') as f:
                    line = f.readline().strip()
                    lines.append(line)
                f.close()
        print('Size of Data: {}'.format(len(lines)))
        return lines

    def _formalize(self, lines):
        save_path = self._save_path
        with open(save_path, 'w') as f:
            for line in lines:
                f.write(line+'\n')
        f.close()


def main():
    IMDB()


if __name__=='__main__':
    main()