import math
import extract_features

##########################################################################
#             Performs KNN parameter tuning and classification           #
#             on TFIDF, LSA, TFIDF+LSA, and LSA+TFIDF features           #
##########################################################################

# Uses TFIDF featuers to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses K nearest neighbors from train
# tfs_mat and tfidf_mat are artifacts of the trained TFIDF matrix to transform incoming submission text
def classify_tfidf(train, tfs_mat, tfidf_mat, submission_text, threshold, K):
    
    # Track indices of top k data points with maximum similarity 
    max_sim = {}
    max_sim[-1] = -100
    
    # Transform submission_text to TFIDF matrix
    response = tfidf_mat.transform([submission_text])

    # For each train document's key in dictionary
    i = 0
    for key_train in train:

        # Initialize document similarity components to 0
        dist_numerator = 0.0
        dist_denominator_train = 0.0
        dist_denominator_test = 0.0

        # For each term that appears in the test document
        for col in response.nonzero()[1]:

            # Access TFDIF matrix component of that term for train and test
            f_train = tfs_mat[i][col]
            f_test = response[0, col]

            # Calculate cosine similarity components
            dist_numerator += f_train*f_test
            dist_denominator_train += pow(f_train, 2)
            dist_denominator_test += pow(f_test, 2)

        # Calculate final cosine similarity for every term combined (between two docs)
        dist_denominator = math.sqrt(dist_denominator_train) + math.sqrt(dist_denominator_test)

        # Check for 0 denominator
        if dist_denominator != 0:
            dist = dist_numerator / dist_denominator
        else: 
            dist = 0

        # Keep track of K nearest neighbors
        vals = max_sim.values()
        for val in vals:

            # If new similarity is larger than any value in max_sim, 
            # replace the smallest value with new larger similarity
            if dist > val:
                if len(max_sim) >= K:
                    key = min(max_sim, key=max_sim.get)
                    del max_sim[key]
                max_sim[key_train] = dist
                break

        i += 1

    pos_votes = 0       # votes that the test is an opioid abuser
    neg_votes = 0       # votes that the test is a nonuser

    # Take input from all K nearest neighbors
    for key in max_sim:

        # If neighbor says abuser 
        if key < threshold:
            pos_votes += 1

        # If neighbor says nonuser 
        else:
            neg_votes += 1
            
    if pos_votes > neg_votes:
        classification = 1
        
    else:
        classification = 0

    return classification

# Uses LSA featuers to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses K nearest neighbors from train
# reduced_mat count_vect and svd are artifacts of the trained LSA matrix to transform incoming submission text
def classify_lsa(train, reduced_mat, count_vect, svd, submission_text, threshold, K):
    
    # Track indices of top k data points with maximum similarity 
    max_sim = {}
    max_sim[-1] = -100
    
    # Transform submission_text to LSA matrix
    count = count_vect.transform([submission_text])
    response = svd.transform(count)
    
    # For each train document's key in dictionary
    i = 0
    for key_train in train:

        # Initialize document similarity components to 0
        dist_numerator = 0.0
        dist_denominator_train = 0.0
        dist_denominator_test = 0.0

        # For each term that appears in the test document
        for col in response.nonzero()[1]:

            # Access TFDIF matrix component of that term for train and test
            f_train = reduced_mat[i, col]
            f_test = response[0, col]

            # Calculate cosine similarity components
            dist_numerator += f_train*f_test
            dist_denominator_train += pow(f_train, 2)
            dist_denominator_test += pow(f_test, 2)

        # Calculate final cosine similarity for every term combined (between two docs)
        dist_denominator = math.sqrt(dist_denominator_train) + math.sqrt(dist_denominator_test)

        # Check for 0 denominator
        if dist_denominator != 0:
            dist = dist_numerator / dist_denominator
        else: 
            dist = 0

        # Keep track of K nearest neighbors
        vals = max_sim.values()
        for val in vals:

            # If new similarity is larger than any value in max_sim, 
            # replace the smallest value with new larger similarity
            if dist > val:
                if len(max_sim) >= K:
                    key = min(max_sim, key=max_sim.get)
                    del max_sim[key]
                max_sim[key_train] = dist
                break

        i += 1

    pos_votes = 0       # votes that the test is an opioid abuser
    neg_votes = 0       # votes that the test is a nonuser

    # Take input from all K nearest neighbors
    for key in max_sim:

        # If neighbor says abuser 
        if key < threshold:
            pos_votes += 1

        # If neighbor says nonuser 
        else:
            neg_votes += 1
            
    if pos_votes > neg_votes:
        classification = 1
        
    else:
        classification = 0

    return classification

# Uses TFIDF+LSA featuers to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses K nearest neighbors from train
# reduced_mat tfidf_mat, and svd are artifacts of the trained TFIDF+LSA matrix to transform incoming submission text
def classify_tfidf_lsa(train, reduced_mat, tfidf_mat, svd, submission_text, threshold, K):
    
    # Track indices of top k data points with maximum similarity 
    max_sim = {}
    max_sim[-1] = -100
    
    # Transform submission_text to TFIDF+LSA matrix
    tf = tfidf_mat.transform([submission_text])
    response = svd.transform(tf)
    
    # For each train document's key in dictionary
    i = 0
    for key_train in train:

        # Initialize document similarity components to 0
        dist_numerator = 0.0
        dist_denominator_train = 0.0
        dist_denominator_test = 0.0

        # For each term that appears in the test document
        for col in response.nonzero()[1]:

            # Access TFDIF+LSA matrix component of that term for train and test
            f_train = reduced_mat[i, col]
            f_test = response[0, col]

            # Calculate cosine similarity components
            dist_numerator += f_train*f_test
            dist_denominator_train += pow(f_train, 2)
            dist_denominator_test += pow(f_test, 2)

        # Calculate final cosine similarity for every term combined (between two docs)
        dist_denominator = math.sqrt(dist_denominator_train) + math.sqrt(dist_denominator_test)

        # Check for 0 denominator
        if dist_denominator != 0:
            dist = dist_numerator / dist_denominator
        else: 
            dist = 0

        # Keep track of K nearest neighbors
        vals = max_sim.values()
        for val in vals:

            # If new similarity is larger than any value in max_sim, 
            # replace the smallest value with new larger similarity
            if dist > val:
                if len(max_sim) >= K:
                    key = min(max_sim, key=max_sim.get)
                    del max_sim[key]
                max_sim[key_train] = dist
                break

        i += 1

    pos_votes = 0       # votes that the test is an opioid abuser
    neg_votes = 0       # votes that the test is a nonuser

    # Take input from all K nearest neighbors
    for key in max_sim:

        # If neighbor says abuser 
        if key < threshold:
            pos_votes += 1

        # If neighbor says nonuser 
        else:
            neg_votes += 1
            
    if pos_votes > neg_votes:
        classification = 1
        
    else:
        classification = 0

    return classification

# Uses LSA+TFIDF featuers to classify submission_text into opioid abuser (1) or nonuser (0) 
# Uses K nearest neighbors from train
# tfs_mat count_vect, tfidf_mat and svd are artifacts of the trained LSA+TFIDF matrix to transform incoming submission text
def classify_lsa_tfidf(train, tfs_mat, count_vect, tfidf_mat, svd, submission_text, threshold, K):
    
    # Track indices of top k data points with maximum similarity 
    max_sim = {}
    max_sim[-1] = -100
    
    # Transform submission_text to LSA+TFIDF matrix
    count = count_vect.transform([submission_text])
    reduced = svd.transform(count)
    response = tfidf_mat.transform(reduced)
    
    # For each train document's key in dictionary   
    i = 0
    for key_train in train:

        # Initialize document similarity components to 0
        dist_numerator = 0.0
        dist_denominator_train = 0.0
        dist_denominator_test = 0.0

        # For each term that appears in the test document
        for col in response.nonzero()[1]:

            # Access LSA+TFDIF matrix component of that term for train and test
            f_train = tfs_mat[i, col]
            f_test = response[0, col]

            # Calculate cosine similarity components
            dist_numerator += f_train*f_test
            dist_denominator_train += pow(f_train, 2)
            dist_denominator_test += pow(f_test, 2)

        # Calculate final cosine similarity for every term combined (between two docs)
        dist_denominator = math.sqrt(dist_denominator_train) + math.sqrt(dist_denominator_test)

        # Check for 0 denominator
        if dist_denominator != 0:
            dist = dist_numerator / dist_denominator
        else: 
            dist = 0

        # Keep track of K nearest neighbors
        vals = max_sim.values()
        for val in vals:

            # If new similarity is larger than any value in max_sim, 
            # replace the smallest value with new larger similarity
            if dist > val:
                if len(max_sim) >= K:
                    key = min(max_sim, key=max_sim.get)
                    del max_sim[key]
                max_sim[key_train] = dist
                break

        i += 1

    pos_votes = 0       # votes that the test is an opioid abuser
    neg_votes = 0       # votes that the test is a nonuser

    # Take input from all K nearest neighbors
    for key in max_sim:

        # If neighbor says abuser 
        if key < threshold:
            pos_votes += 1

        # If neighbor says nonuser 
        else:
            neg_votes += 1
            
    if pos_votes > neg_votes:
        classification = 1
        
    else:
        classification = 0

    return classification

# Tunes k parameter for KNN for a particular set of features and displays table of results
# Threshold represents the number of training samples of opioid abusers (changes per fold)
# Partitioned_data is the data in 3 randomly assigned folds
# min_K and max_K create the range in which k will be searched
# features can be any of the following: 'TFIDF', 'LSA', 'TFIDF+LSA', 'LSA+TFIDF'
def param_tune(threshold, partitioned_data, min_K, max_K, features):
    
    # Display table header
    print('KNN with ' + features + ' Parameter Tuning Results for K in [' + str(min_K) + ', ' + str(max_K) + ']:')
    print('________________________________________________________________')
    print('K\tAccuracy\tPrecision\tRecall\t\tF1 measure\tSensitivity\tSpecificity')

    # Track max F1-measure and corresponding K
    max_f1 = -1
    optimal_K = -1
        
    #############################################################
    #                   Search for optimal k                    #
    #############################################################

    for K in range(min_K, max_K + 1):
        
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
            elif features == 'LSA':
                [reduced_mat, count_vect, svd] = extract_features.build_lsa_matrix(train)
            elif features == 'TFIDF+LSA':
                [reduced_mat, tfidf_mat, svd] = extract_features.build_tfidf_lsa_matrix(train)
            elif features == 'LSA+TFIDF':
                [tfs_mat, count_vect, tfidf_mat, svd] = extract_features.build_lsa_tfidf_matrix(train)

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
                    classification = classify_tfidf(train, tfs_mat, tfidf_mat, doc, threshold, K)
                elif features == 'LSA':
                    classification = classify_lsa(train, reduced_mat, count_vect, svd, doc, threshold, K)
                elif features == 'TFIDF+LSA':
                    classification = classify_tfidf_lsa(train, reduced_mat, tfidf_mat, svd, doc, threshold, K)
                elif features == 'LSA+TFIDF':
                    classification = classify_lsa_tfidf(train, tfs_mat, count_vect, tfidf_mat, svd, doc, threshold, K)
                
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
        print(str(K) + '\t' + str(accuracy) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(sensitivity) + '\t' + str(specificity))
    
        # Check if new f1 is greater than old max
        if f1 > max_f1:
            max_f1 = f1
            optimal_K = K
    
    print('________________________________________________________________')
    
    # Print optimal K after parameter tuning
    print('')
    print('Result: K value of ' + str(optimal_K) + ' is optimal due to max F1-measure of ' + str(max_f1))
    print ('')
    
    return optimal_K