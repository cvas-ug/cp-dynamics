# pylint: skip-file
import torch
from torch.utils.data.sampler import Sampler

class BatchRandomSampler(Sampler):
    def __init__(self,indices,batch_size):
        self.indices=indices
        self.batch_size=batch_size
    def __iter__(self):
        batch_size=self.batch_size
        old_indice=self.indices
        new_indice=list(range(len(old_indice)))
        iter_number=int(len(new_indice)/batch_size)
        random_matrix=torch.randperm(iter_number)
        for i in range(iter_number):
            t=random_matrix[i]
            random_indice=old_indice[t*batch_size:(t+1)*batch_size]
            new_indice[i*batch_size:(i+1)*batch_size]=random_indice
        return iter(new_indice)
    def __len__(self):
        return len(self.indices)        

