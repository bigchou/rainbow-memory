import numpy as np, pdb

class Metric():
    def __init__(self, k, **kwargs):
        self.k        = k
        self.requires = ['nearest_features', 'target_labels']
        self.name     = 'e_recall@{}'.format(k)

    def __call__(self, target_labels, k_closest_classes, **kwargs):
        recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:self.k]])/len(target_labels)
        #target_labels.shape=(5000,1)
        #k_closest_classes.shape=(5000,4)
        half = len(target_labels) // 2#2500
        first = np.sum([1 for target, recalled_predictions in zip(target_labels[:half], k_closest_classes[:half]) if target in recalled_predictions[:self.k]])/len(target_labels[:half])
        second = np.sum([1 for target, recalled_predictions in zip(target_labels[half:], k_closest_classes[half:]) if target in recalled_predictions[:self.k]])/len(target_labels[half:])
        print("all: %.2f first(0~4): %.2f | second(5~9): %.2f"%(recall_at_k,first,second))
        #print("count frequency of unique labels: ",np.unique(k_closest_classes, return_counts=True))
        #pdb.set_trace()
        #if first > 0.: pdb.set_trace()
        return recall_at_k
