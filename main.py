import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

def load_data(path, target_col="target"):
	if ".csv" in path:
		df = pandas.read_csv(path)
	elif ".xlsx" in path:
		df = pandas.read_excel(path)
	print(df[target_col].value_counts())
	X = df.drop(columns=[target_col])
	Y = df[target_col]
	return X, Y 

def nomalize(X):
	standard_scaler_object = StandardScaler()
	X_normalized = standard_scaler_object.fit_transform(X)
	return standard_scaler_object, X_normalized

def get_classifier(Name):
    classifier_dict = dict()
    return classifier_dict[Name]
    

def evaluate(X_normalized, Y, n_neighbors=5, n_splits=10, scoring="f1_weighted", classifier_name=None , **classifier_params):
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	classifier = get_classifier(classifier_name)
	knn_object = classifier(**classifier_params)

	scores = cross_val_score(knn_object, X_normalized, Y, cv=cross_validation_object, scoring=scoring)
	print(f"N neighbors : {n_neighbors}")
	print(f"{scoring} moyen {scores.mean()} +/- {scores.std()}")
	print("-"*20)
	return scores

def get_interesting_params(classifier_name):
    
    param_grid_knn = {
		"n_splits": range(1,20),
		"neighors_range": (1,100,2)
	}
    param_grid_rf = {
    	"n_estimators": range(0,100),
    	"max_depth": [None]+range(0,50),
    	"min_samples_split": range(1,20),
    	"min_samples_leaf": [1,5],
    	"max_features": ["sqrt", "log2", None],
    	"bootstrap": [True, False]
	}
    param_grid_svm = {
        "kernel": ["poly","linear","rbf"],
        "C": [x/10 for x in range(1,100)],
        "gamma": ["scale", "auto"]+[x/1000 for x in range(1,1000)],
        "degree": range(1,10),
        "coef0": [x/1000 for x in range(1000)]}
    if "knn" in classifier_name:
        return param_grid_knn
    elif "svm" in classifier_name:
        return param_grid_svm
    elif "rf" in classifier_name:
        return param_grid_rf
        
    


def find_optimal_params(X_normalized, Y, scoring="f1_weighted",classifier_name = "rf", **classifier_params):
    
	classifier = get_classifier(classifier_name)(classifier_params)
	cross_validation_object = classifier(classifier_params)
	param_grid = get_interesting_params(classifier_name)

	grid_search = GridSearchCV(classifier, param_grid, cv=cross_validation_object, scoring=scoring)
	grid_search.fit(X_normalized, Y)

	optimal_params = grid_search.best_params_
	print("-"*100)
	print(f"The optimal N params values are {optimal_params}")
	print("-"*100)
	return optimal_params


if __name__ == "__main__":
	X, Y = load_data(csv_path="bienetre.csv")
	standard_scaler_object, X_normalized = nomalize(X)

	optimal_params = find_optimal_params(X_normalized, Y, scoring="f1_weighted")