from data_loader import load_normalized_data
from tqdm import tqdm

class Decision_Tree:
    
    def __init__(self,feature,label, left=None, right=None, target_repartition = {}):
        self.feature = feature
        self.label = label
        self.left = left
        self.right = right
        
        # Dictionnaire: key= Target; value = [True qte, False qte]
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

    def fit(self,X,y):
        transpose_X = [list(row) for row in zip(*list)]
        num_feature_tree_dict = {}
        token_feature_tree_dict = {}
        for feature_index,feature_values in enumerate(transpose_X):
            feature_type = type(feature_values[0])
            feature_values = list(set(feature_values))
            if feature_type == int or feature_type == float:
                sorted_feature = sorted(feature_values)

                for index in range(len(sorted_feature)):
                    if index<len(feature_values)-1 :
                        num_feature_tree_dict[(feature_index, sorted_feature[index] + sorted_feature[index+1] /2)] = Decision_Tree(feature_index, sorted_feature[index] + sorted_feature[index+1] /2)
            else:
                for value in feature_values:
                    token_feature_tree_dict[(feature_index,value)] = Decision_Tree(feature_index,value)
        for sample, target in zip(X,y):
            for feature_index,feature_value in enumerate(sample):
                feature_type = type(feature_value[0])
                
                
                if feature_type == int or feature_type == float:
                    for feature_tree in num_feature_tree_dict:
                        if feature_value < feature_tree.label:
                            feature_tree.target_repartition.setdefault(target, [0,0])
                            feature_tree.target_repartition[0]+=1
                        else :
                            feature_tree.target_repartition.setdefault(target, [0,0])
                            feature_tree.target_repartition[1]+=1
                else:
                    for feature_tree in token_feature_tree_dict:
                        if self.feature.value == feature_tree:
                            feature_tree.target_repartition.setdefault(target, [0,0])
                            feature_tree.target_repartition[0]+=1
                        else:
                            feature_tree.target_repartition.setdefault(target, [0,0])
                            feature_tree.target_repartition[1]+=1
                    
                            
            
            
                
                    
            
                
                    
class Leaves(Decision_Tree):
    
    def gini_impurity(self,target):
        total_nb_values = self.total_target_nb()
        return 1 - (self.target_repartition[target]/total_nb_values)**2 - ((total_nb_values-self.target_repartition[target])/total_nb_values)**2