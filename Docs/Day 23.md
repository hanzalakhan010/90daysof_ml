* Diagnose and fix **model architecture errors**.  
  * **Layer Order**:  
    * Flatten 3D outputs of Conv layers **before** Dense layers.  
  * **Activation Functions**:  
    * Use sigmoid for **binary classification** (not softmax).  
  * **Regularization**:  
    * Added **Dropout** (e.g., 0.5 after Dense layers) and **L2 regularization** to prevent overfitting.  
  * **Model Depth**:  
    * Increased layers (e.g., 3 Conv2D \+ 2 Dense) for better feature extraction.  
* Preprocess data for **robust training**.  
  * **Learning Rate**: Started with a small value (e.g., 0.0001) for stable training.  
  * **Early Stopping**: Halted training when validation loss plateaued (patience=5).  
  * **Data Augmentation**: Applied rotations/flips to artificially expand the dataset.  
  *   
* Use regularization to **generalize to new data**.  
* Deploy techniques like early stopping to **save time/resources**

