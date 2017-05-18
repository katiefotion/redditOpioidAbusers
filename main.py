import string
import random 
import knn
import random_forest
import svm
import plotGraph as plotGraph
import redditCrawler as redditCrawler
import extract_features

##########################################################################
#                      Simple UI to allow user to:                       #
#   parameter tune, crawl reddit for opioid abusers, view network graph  #
##########################################################################

token_dict = {}

# Read in and preprocess submission (training) data 
def preprocess():
    
    # Access opioid abuser data
    with open('opioid_subs.txt', 'r') as f:
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

if __name__ == "__main__":
    
    optimal_ks = [13, 111, 79, 45]    
    optimal_ns = [7, 33, 52, 41]
    optimal_cs = [1000.0, 10.0, 100.0, 1000.0]
    
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
    
    print('Welcome to the Opioid Network Grapher')
    print('- - - - - - - - - - - - - - - - -- - -')
    print('Please select one of the following options from below by typing its corresponding number')
    print('')
    print('1. Perform parameter tuning on a classifier')
    print('2. Crawl through new opioid abusers')
    print('3. View network graph')
    print('4. Quit')
    print('')
    
    # Access user's input of functionality
    menu_choice = raw_input('->')
    
    while not menu_choice.startswith('4'):
    
        # Perform parameter tuning on a classifier
        if menu_choice.startswith('1'):
            
            print('')
            print('Which classifier would you like to perform parameter tuning on?')
            print('')
            print('1. K-Nearest-Neighbors')
            print('2. Random Forest')
            print('3. SVM')
            print('4. Return to main menu')
            print('')
            
            classifier_choice = raw_input('->')
            
            if classifier_choice.startswith('4'):
                break;
                
            else: 
        
                print('')
                print('What set of features would you like to use?')
                print('')
                print('1. TFIDF')
                print('2. LSA')
                print('3. TFIDF + LSA')
                print('4. LSA + TFIDF')
                print('5. Return to main menu')
                print('')

                features_choice = raw_input('->')
                
                if features_choice.startswith('5'):
                    break;
                    
                else:
            
                    # KNN
                    if classifier_choice.startswith('1'):
                        
                        print('')
                        print('What range of k (for KNN) would you like to experiment with?')
                        print('Please enter the minimum k followed by a space followed by the maximum k') 
                        print('(i.e. 1 50 will tune the classifier when k is in [1, 50])')
                        print('')

                        k_choice = raw_input('->')
                        print('')

                        k = k_choice.split(' ')
                        
                        # KNN with TFIDF
                        if features_choice.startswith('1'):
                            optimal_ks[0] = knn.param_tune(threshold, partitioned_data, int(float(k[0])), int(float(k[1])), 'TFIDF')
                        
                        # KNN with LSA 
                        elif features_choice.startswith('2'):
                            optimal_ks[1] = knn.param_tune(threshold, partitioned_data, int(float(k[0])), int(float(k[1])), 'LSA')

                        # KNN with TFIDF+LSA
                        elif features_choice.startswith('3'):
                            optimal_ks[2] = knn.param_tune(threshold, partitioned_data, int(float(k[0])), int(float(k[1])), 'TFIDF+LSA')

                        # KNN with LSA+TFIDF
                        elif features_choice.startswith('4'):
                            optimal_ks[3] = knn.param_tune(threshold, partitioned_data, int(float(k[0])), int(float(k[1])), 'LSA+TFIDF')

                    # Random Forest
                    elif classifier_choice.startswith('2'):
                        
                        print('')
                        print('What range of n_estimators (for RF) would you like to experiment with?')
                        print('Please enter the minimum n_estimators followed by a space followed by the maximum n_estimators') 
                        print('(i.e. 50 100 will tune the classifier when n_estimators is in [50, 100])')
                        print('')

                        n_choice = raw_input('->')
                        print('')

                        n = n_choice.split(' ')
                        
                        # RF with TFIDF
                        if features_choice.startswith('1'):
                            optimal_ns[0] = random_forest.param_tune(threshold, partitioned_data, int(float(n[0])), int(float(n[1])), 'TFIDF')

                        # RF with LSA
                        elif features_choice.startswith('2'):
                            optimal_ns[1] = random_forest.param_tune(threshold, partitioned_data, int(float(n[0])), int(float(n[1])), 'LSA')

                        # RF with TFIDF+LSA
                        elif features_choice.startswith('3'):
                            optimal_ns[2] = random_forest.param_tune(threshold, partitioned_data, int(float(n[0])), int(float(n[1])), 'TFIDF+LSA')
                     
                        # RF with LSA+TFIDF
                        elif features_choice.startswith('4'):
                            optimal_ns[3] = random_forest.param_tune(threshold, partitioned_data, int(float(n[0])), int(float(n[1])), 'LSA+TFIDF')
                            
                    # Support Vector Machine
                    elif classifier_choice.startswith('3'):
                        
                        print('')
                        print('What range of C (for SVM) would you like to experiment with?')
                        print('Please enter the minimum C followed by a space followed by the maximum C') 
                        print('(i.e. 0.00001 100 will tune the classifier for C values of 0.00001, 0.0001, 0.001, ..., 100)')
                        print('')

                        c_choice = raw_input('->')
                        print('')

                        c = c_choice.split(' ')
                        
                        # SVM with TFIDF
                        if features_choice.startswith('1'):
                            optimal_cs[0] = svm.param_tune(threshold, partitioned_data, float(c[0]), float(c[1]), 'TFIDF')

                        # SVM with LSA
                        elif features_choice.startswith('2'):
                            optimal_cs[1] = svm.param_tune(threshold, partitioned_data, float(c[0]), float(c[1]), 'LSA')

                        # SVM with TFIDF+LSA
                        elif features_choice.startswith('3'):
                            optimal_cs[2] = svm.param_tune(threshold, partitioned_data, float(c[0]), float(c[1]), 'TFIDF+LSA')
                     
                        # SVM with LSA+TFIDF
                        elif features_choice.startswith('4'):
                            optimal_cs[3] = svm.param_tune(threshold, partitioned_data, float(c[0]), float(c[1]), 'LSA+TFIDF')
                            
        # Crawl Reddit for opioid abusers               
        elif menu_choice.startswith('2'):
            
            print('')
            print('Random forest using LSA+TFIDF features was determined to be the best classifier.')
            print('The classification while crawling Reddit will be done with RF with n_estimators = ' + str(optimal_ns[3]))
            print('')
            
            # Build LSA+TFIDF matrix (due to superior performance)
            [tfs_mat, count_vect, tfidf, svd] = extract_features.build_lsa_tfidf_matrix(token_dict)
            
            # Train forest
            forest = random_forest.train_forest(token_dict, tfs_mat, threshold, optimal_ns[3])
            
            # Crawl reddit, classify opioid abusers and add to network graph
            redditCrawler.getData(forest, count_vect, tfidf, svd)
        
        # Graph network
        elif menu_choice.startswith('3'):
            
            # Call network graphing method 
            plotGraph.draw()
            
        # Exit
        elif menu_choice.startswith('4'):
            sys.exit(0)
        
        else:
            print('Not a recognized command. Try again.')
                
            
        print('- - - - - - - - - - - - - - - - -- - -')
        print('Please select one of the following options from below by typing its corresponding number')
        print('')
        print('1. Perform parameter tuning on a classifier')
        print('2. Crawl through new opioid abusers')
        print('3. View network graph')
        print('4. Quit')
        print('')

        menu_choice = raw_input('->')