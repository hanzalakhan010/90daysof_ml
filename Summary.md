Day1


Day2 


Day3 


Day4 


Day5


Day6


Day7 (Phase 2)
    performance scores of different model:[linear_regression,Decision_tree,randomForest,GradientBoostingRegressor]
    Mini Project 1 performace metrices of classifiation model using DecisionTreeClassifier
    Mini Project 2 perfoncae metrices of linear regression model stated above 

Day8 (Phase 2)
    Classification models :[RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier]
___________________________________________________________________________________________________ 
    *Bagging* (Bootstrap Aggregating): Reduces variance by combining predictions from multiple models trained independently.
    Key Idea :  Train multiple models on random subsets of the data (with replacement) and combine their predictions.

    Examples include BaggingClassifier , RandomForestRegressor
___________________________________________________________________________________________________    
    *Boosting* Reduces bias by combining models sequentially, where each model corrects errors of its predecessor.
    Key Idea : Build models sequentially, where each subsequent model focuses on correcting the errors of the previous ones.

    Examples include GradientBostingClassifier,AdaBoostClassifier
___________________________________________________________________________________________________    
    Importants
        1)accuracy, F1 score, and recall to have the same numerical value under certain conditions, especially when the dataset is balanced and the model makes predictions with high precision and recall.
        2)n_estimators:
            Increasing this improves accuracy but increases computational time.
            Balance accuracy and efficiency based on the dataset size and problem complexity.
        3)random_state:
            Ensures consistent results across runs, critical for debugging and reproducibility.
            Changing this affects randomness in the model and, consequently, predictions.
