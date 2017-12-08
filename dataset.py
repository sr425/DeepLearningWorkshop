import numpy as np
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import threading
import sys
from glob import glob

def loadDatasetFile(path):
    with open(path, 'r') as f:
        content = [x.replace('\n', '') for x in f.readlines()]
    return content

def loadFiles(pattern):
    files = glob(pattern)
    return sorted(files)

def loadDataset(input_string):
    if input_string.endswith('.txt'):
        return loadDatasetFile(input_string)
    else:
        return loadFiles(input_string)
    

class BufferedDataset:
    def __data_manager_async(self):
        while True:
            for datum in self.paths:
                self.__filename_buffer.put(datum)
            permutation = np.random.permutation(len(self.paths))
            self.paths = self.paths[permutation]

    def __buffer_loader(self):
        while True:
            datum = self.__filename_buffer.get()
            if isinstance(datum, str):
                loaded_data = (self.loaders[0](datum), datum)
            else:
                loaded_data = [(loader(path), path) for path, loader in zip(datum, self.loaders)]
            self.__data_buffer.put(loaded_data)

    def __init__(self, data_paths, loaders, buffer_size = 10, nr_workers=1):
        self.paths = np.array(data_paths)
        self.sample_count = len(data_paths)
        self.buffer_size = buffer_size
        self.__filename_buffer = Queue(nr_workers * buffer_size)
        self.__data_buffer = Queue(buffer_size)

        self.loaders = loaders

        self.data_manager_thread = threading.Thread(target = self.__data_manager_async, args = ())
        self.data_manager_thread.daemon = True
        self.data_manager_thread.start()
        print("Started data file manager...")

        self.data_loaders = []
        for i in range(nr_workers):
            loader = threading.Thread(target = self.__buffer_loader, args = ())
            loader.daemon = True
            loader.start()
            self.data_loaders.append(loader)
            print("Started data loader %d ..." %i)


    def clear(self):
        self.__filename_buffer = None
        self.__data_buffer = None
        self = None

    def next_batch(self, batch_size, origin=False):
        data = [self.__data_buffer.get() for i in range(0, batch_size)]

        batch = {}
        paths = {}
        for i in range(len(self.loaders)):
            batch[i] = []
            paths[i] = []
        for entry in data:
            for i, (element_data, element_path) in enumerate(entry):
                batch[i].append(element_data)
                paths[i].append(element_path)
        if origin:
            return tuple([np.array(batch[i]) for i in range(len(self.loaders))]), tuple([np.array(paths[i]) for i in range(len(self.loaders))])
        else:
            return tuple([np.array(batch[i]) for i in range(len(self.loaders))])
        
    def get_length(self):
        return self.sample_count