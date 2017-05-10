import nltk
import string
import random 
import math

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

# Adapted from http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

token_dict = {}
stemmer = PorterStemmer() 

# Tokenize text using nltk and stem them
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# Stem tokens based on PorterStemmer
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def preprocess():
    
    # Access opioid abuser data
    with open('glassine_subs.txt', 'r') as f:
        lines = f.readlines()

    # Calculate index threshold based on how many positive examples are present
    # This index is the determining factor of where opioid abusers are and where nonusers are
    threshold = len(lines)

    # Acess nonuser data 
    with open('nonuser_subs.txt', 'r') as f:
        lines.extend(f.readlines())

    # Preprocess the data
    i = 0
    for text in lines:
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        token_dict[i] = no_punctuation
        i += 1
        
    return threshold

def build_tfidf_lsa_matrix(train):
    
    # Create TFIDF matrix
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(train.values())

    # Perform dimensionality reduction
    svd = TruncatedSVD(n_components=100)
    reduced_mat = svd.fit_transform(tfs) 
    
    return [reduced_mat, tfidf, svd]

def classify(train, reduced_mat, tfidf, svd, submission_text, threshold, K):
    
    # Track indices of top k data points with maximum similarity 
    max_sim = {}
    max_sim[-1] = -100
    
    # Transform submission_text to TFIDF+LSA matrix
    tf = tfidf.transform([submission_text])
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
    

# Tunes k parameter for TFIDF + LSA and displays table of results
def param_tune(min_K, max_K):
    
    # Display table header
    print('TFIDF + LSA Parameter Tuning Results for K in [' + str(min_K) + ', ' + str(max_K) + ']:')
    print('________________________________________________________________')
    print('K\tAccuracy\tPrecision\tRecall\t\tF1 measure\tSensitivity\tSpecificity')
    
    # Track max F1-measure and corresponding K
    max_f1 = -1
    optimal_K = -1
    
    #############################################################
    #                    Data preprocessing                     #
    #############################################################
    
    threshold = preprocess()
    
    # Prep 3 folds for cross validation
    # Initialize empty folds
    partitioned_data = [{}, {}, {}]

    # Randomly assign each data point to one of the 3 folds
    for i in range(0, len(token_dict)):
        doc = token_dict[i]
        partitioned_data[random.randint(0,2)][i] = doc
    
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

            #################### LSA (train) ####################### 

            [reduced_mat, tfidf, svd] = build_tfidf_lsa_matrix(train)

            ######################### Test ###########################
           
            # Initialize true positives, true negatives, etc. to 0
            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0

            # For each test document's key in dictionary 
            for key_test in test:

                # Access test document and transform to LSA matrix
                doc = test[key_test]
                
                #Classify text
                classification = classify(train, reduced_mat, tfidf, svd, doc, threshold, K)

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