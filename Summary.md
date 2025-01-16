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


___________________________________________________________________________________________________    
___________________________________________________________________________________________________    
        

Day 9 (Phase 2)
    1)learned about  [Accuracy,F1 Score,ROC AUC Score,Recall Score] to evaluate classification model performance
    2)Plotting of ROC(Receiver Operating Characteristic) Curve that illustrates the diognostic ability of binary classification model.
        #It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various thresholds.

___________________________________________________________________________________________________    
    Importants
        1)Always set random_state: This ensures reproducibility, consistency, and fairness when evaluating model performance.

___________________________________________________________________________________________________    
    
Day 10 Phase(1)
    1) feature importance in classisctaion models
___________________________________________________________________________________________________    

Day 11 Phase(1)

___________________________________________________________________________________________________    

Day 12 Phase(1)
    --- Bagging (Bootstrap Aggregating)
        How it works: Trains multiple models on bootstrapped samples of the dataset and aggregates their predictions (e.g., Random Forest).
        Performance: Showed excellent results in reducing overfitting and handling noisy data.
    --- Boosting
        How it works: Models are trained sequentially, where each model learns from the errors of the previous one. (e.g., AdaBoost, Gradient Boosting, XGBoost).
        Performance: Achieved higher accuracy on the dataset, especially after hyperparameter tuning, but required careful management to prevent overfitting.
    --- Stacking
        How it works: Combines predictions from multiple diverse base models using a meta-model.
        Performance: Delivered the most balanced results, leveraging the strengths of different algorithms for better generalization.
___________________________________________________________________________________________________    

Day 13 (Phase 1)
    ✔️ How KNN works for predicting categories and continuous values.
    ✔️ Performance Analysis: Evaluating metrics like accuracy, R2 score, and mean squared error for better insights.
    ✔️ GridSearchCV: Fine-tuning hyperparameters like n_neighbors and distance metrics to optimize performance.
    ✔️ Pipelines: Streamlining preprocessing and model training into a seamless workflow.
___________________________________________________________________________________________________    

Day 14 (Phase 1):

    --- Preprocessing: Scaled the features for better model performance.
    --- Optimal Clusters: Used the Elbow Method to determine the ideal number of clusters.
    --- Model Training: Leveraged KMeans from sklearn.cluster.
    --- Visualization: Created a scatter plot to represent the clusters.
    --- Evaluation: Assessed clustering quality with the Silhouette Score.

___________________________________________________________________________________________________    

Day 15 (Phase 1)
    --- Dendrograms: Visualizing the hierarchical clustering process to determine the optimal number of clusters.
    --- Hierarchical Clustering: Understanding how agglomerative clustering works, using it to group similar data points based on their features.
    --- Davies-Bouldin Score: A metric to evaluate the clustering model by measuring intra-cluster similarity and inter-cluster differences (lower score = better clustering).
    --- Silhouette Score: Measuring how similar each point is to its own cluster compared to other clusters (higher score = better clustering).
Day 16 (Phase 1)

    