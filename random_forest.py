import extract_features
from sklearn.ensemble import RandomForestClassifier

##########################################################################
#              Performs RF parameter tuning and classification           #
#             on TFIDF, LSA, TFIDF+LSA, and LSA+TFIDF features           #
##########################################################################

# Trains forest with matrix data samples
# Uses train parameter and threshold to determine true class labels
# N determines number of trees/estimators
def train_forest(train, matrix, threshold, N):
    
    # Identify class labels
    class_labels = []
    for key in train: 
        if key < threshold:
            class_labels.append(1)
        else:
            class_labels.append(0)
     
    forest = RandomForestClassifier(n_estimators = N)
    forest = forest.fit(matrix, class_labels)
    
    return forest

# Uses TFIDF features to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses trained forest
# tfidf_mat is artifact of the trained TFIDF matrix to transform incoming submission text
def classify_tfidf(forest, tfidf_mat, submission_text):
    
    # Transform submission_text to TFIDF matrix
    response = tfidf_mat.transform([submission_text])
    
    # Predict class of response
    classification = forest.predict(response)
    
    return classification

# Uses LSA featuers to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses trained forest
# count_vect and svd are artifacts of the trained LSA matrix to transform incoming submission text
def classify_lsa(forest, count_vect, svd, submission_text):

    # Transform submission_text to LSA matrix
    count = count_vect.transform([submission_text])
    response = svd.transform(count)
    
    # Predict class of response
    classification = forest.predict(response)

    return classification

# Uses TFIDF+LSA features to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses trained forest
# tfidf_mat and svd are artifacts of the trained TFIDF+LSA matrix to transform incoming submission text
def classify_tfidf_lsa(forest, tfidf_mat, svd, submission_text):

    # Transform submission_text to TFIDF+LSA matrix
    tf = tfidf_mat.transform([submission_text])
    response = svd.transform(tf)
    
    # Predict class of response 
    classification = forest.predict(response)

    return classification

# Uses LSA+TFIDF features to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses trained forest
# count_vect, tfidf_mat, and svd are artifacts of the trained LSA+TFIDF matrix to transform incoming submission text
def classify_lsa_tfidf(forest, count_vect, tfidf_mat, svd, submission_text):
    
    # Transform submission_text to LSA+TFIDF matrix
    count = count_vect.transform([submission_text])
    reduced = svd.transform(count)
    response = tfidf_mat.transform(reduced)
    
    # Predict class of response
    classification = forest.predict(response)
    
    return classification

# Tunes n_estimators parameter for RF for a particular set of features and displays table of results
# Threshold represents the number of training samples of opioid abusers (changes per fold)
# Partitioned_data is the data in 3 randomly assigned folds
# min_N and max_N create the range in which n_estimators will be searched
# features can be any of the following: 'TFIDF', 'LSA', 'TFIDF+LSA', 'LSA+TFIDF'
def param_tune(threshold, partitioned_data, min_N, max_N, features):
    
    # Display table header
    print('Random Forest with ' + features + ' Parameter Tuning Results for n_estimators in [' + str(min_N) + ', ' + str(max_N) + ']:')
    print('________________________________________________________________')
    print('N\tAccuracy\tPrecision\tRecall\t\tF1 measure\tSensitivity\tSpecificity')

    # Track max F1-measure and corresponding K
    max_f1 = -1
    optimal_N = -1
        
    #############################################################
    #                   Search for optimal k                    #
    #############################################################

    for N in range(min_N, max_N + 1):
        
        # Initialize all running sums for metrics to 0
        acc_sum = 0.0
        prec_sum = 0.0
        rec_sum = 0.0
        sens_sum = 0.0
        spec_sum = 0.0
        
        #############################################################
        #                    3-fold cross validation                #
        #############################################################

        # For each fold, determine training dataset and testing dataset
        for k in range(0, 3):
            test = partitioned_data[k]
            train = {}
            for j in range(0, 3):
                if k != j:
                    for key in partitioned_data[j]:
                        train[key] = partitioned_data[j][key]

            ######################## Train ########################### 

            # Calculate frequency matrix on training data based on features desired
            if features == 'TFIDF':
                [tfidf_mat, tfs_mat] = extract_features.build_tfidf_matrix(train)
                
                forest = train_forest(train, tfs_mat, threshold, N)
                
            elif features == 'LSA':
                [reduced_mat, count_vect, svd] = extract_features.build_lsa_matrix(train)
                
                forest = train_forest(train, reduced_mat, threshold, N)
                
            elif features == 'TFIDF+LSA':
                [reduced_mat, tfidf_mat, svd] = extract_features.build_tfidf_lsa_matrix(train)
                
                forest = train_forest(train, reduced_mat, threshold, N)
                
            elif features == 'LSA+TFIDF':
                [tfs_mat, count_vect, tfidf_mat, svd] = extract_features.build_lsa_tfidf_matrix(train)
                
                forest = train_forest(train, tfs_mat, threshold, N)

            ######################### Test ###########################

            # Initialize true positives, true negatives, etc. to 0
            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0

            # For each test document's key in dictionary 
            for key_test in test:

                # Access test document
                doc = test[key_test]
                
                # Classify text
                if features == 'TFIDF':
                    classification = classify_tfidf(forest, tfidf_mat, doc)
                elif features == 'LSA':
                    classification = classify_lsa(forest, count_vect, svd, doc)
                elif features == 'TFIDF+LSA':
                    classification = classify_tfidf_lsa(forest, tfidf_mat, svd, doc)
                elif features == 'LSA+TFIDF':
                    classification = classify_lsa_tfidf(forest, count_vect, tfidf_mat, svd, doc)
                
                if key_test >= threshold:         # ground truth negative case (a nonuser)

                    # Determine fp or tn accordingly 
                    if classification == 0:       # classified as nonuser
                        tn += 1
                    else:                         # classified as user
                        fp += 1

                elif key_test < threshold:        # ground truth positive case (an abuser)

                    # Determine tp or fn accordingly 
                    if classification == 0:        # classified as nonuser
                        fn += 1
                    else:                          # classified as user
                        tp += 1

            # Aggregate accuracy, precision, recall, specificity and sensitivity
            acc_sum += (tp+tn+1)/(tp+tn+fp+fn+1)
            prec_sum += (tp+1)/(tp+fp+1)
            rec_sum += (tp+1)/(tp+tn+1)
            sens_sum += (tp+1)/(tp+fn+1)
            spec_sum += (tn+1)/(tn+fp+1)

        # Calculate final metrics 
        accuracy = acc_sum/3.0
        precision = prec_sum/3.0
        recall = rec_sum/3.0
        f1 = (2*precision*recall)/(precision+recall)
        sensitivity = sens_sum/3.0
        specificity = spec_sum/3.0

        # Display final metrics
        print(str(N) + '\t' + str(accuracy) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(sensitivity) + '\t' + str(specificity))
    
        # Check if new f1 is greater than old max
        if f1 > max_f1:
            max_f1 = f1
            optimal_N = N
    
    print('________________________________________________________________')
    
    # Print optimal N after parameter tuning
    print('')
    print('Result: n_estimators value of ' + str(optimal_N) + ' is optimal due to max F1-measure of ' + str(max_f1))
    print ('')
    
    return optimal_N