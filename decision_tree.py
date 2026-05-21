from data_loader import load_normalized_data
from tqdm import tqdm

class Decision_Tree:
    
    def __init__(self,label, left=None, right=None, target_repartition = {}):
        self.label = label
        self.left = left
        self.right = right
        
        # Dictionnaire: key= Target; value = (True qte, False qte)
        self.target_repartition = target_repartition

    def total_repartition_nb(self):
        return sum(true_qte+false_qte for (true_qte,false_qte) in self.target_repartition.values())
    def gini_impurity(self,target):
        if self.left== None and self.right ==None:
            return Leaves(self).gini_impurity(target)
        else:
            left = self.left
            right = self.right
            
            left_weight = left.target_repartion[target]/left.total_target_nb() \
                if self.left != None else 0
            right_weight = left.target_repartion[target]/right.total_target_nb() \
                if self.right != None else 0
            return left.target_gini_impurity(target) * left_weight + right.target_gini_impurity(target) * right_weight
class Leaves(Decision_Tree):
    
    def gini_impurity(self,target):
        total_nb_values = self.total_target_nb()
        return 1 - (self.target_repartition[target]/total_nb_values)**2 - ((total_nb_values-self.target_repartition[target])/total_nb_values)**2