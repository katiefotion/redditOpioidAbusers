# Python 2.7 module to run main functionalities and communicate with user

import tfidf as tfidf
import lsa as lsa
import tfidf_lsa as tfidf_lsa
import lsa_tfidf as lsa_tfidf


if __name__ == "__main__":
    
    optimal_ks = [5, 12, 15, 40]
    
    print('Welcome to the Opioid Network Grapher')
    print('- - - - - - - - - - - - - - - - -- - -')
    print('Please select one of the following options from below by typing its corresponding number')
    print('')
    print('1. Perform parameter tuning on a classifier')
    print('2. Crawl through new opioid abusers')
    print('3. View network graph')
    print('4. Quit')
    print('')
    
    menu_choice = raw_input('->')
    
    while not menu_choice.startswith('4'):
    
        if menu_choice.startswith('1'):
        
            print('')
            print('Which k-Nearest-Neighbors classifier would you like to perform parameter tuning of k on?')
            print('')
            print('1. TFIDF')
            print('2. LSA')
            print('3. TFIDF + LSA')
            print('4. LSA + TFIDF')
            print('5. Return to main menu')
            print('')

            classifier_choice = raw_input('->')
            
            print('')
            print('What range of k (for KNN) would you like to experiment with?')
            print('Please enter the minimum k followed by a space followed by the maximum k') 
            print('(i.e. 1 50 will tune the classifier when k is in [1, 50])')
            print('')
                
            k_choice = raw_input('->')
            print('')
            
            k = k_choice.split(' ')
            
            if classifier_choice.startswith('1'):
                optimal_ks[0] = tfidf.param_tune(int(float(k[0])), int(float(k[1])))
                
            elif classifier_choice.startswith('2'):
                optimal_ks[1] = lsa.param_tune(int(float(k[0])), int(float(k[1])))
                
            elif classifier_choice.startswith('3'):
                optimal_ks[2] = tfidf_lsa.param_tune(int(float(k[0])), int(float(k[1])))
                
            elif classifier_choice.startswith('4'):
                optimal_ks[3] = lsa_tfidf.param_tune(int(float(k[0])), int(float(k[1])))
                
            else:
                break
                
        elif menu_choice.startswith('2'):
            
            print('____ was determined to be the best classifier based on highest F1-measure.')
            print('The classification while crawling Reddit will be done with ____ with k = ' + optimal_ks[value])
            
            # Call Neharika's crawl method here
            # within that method, call classify_new_submission(submission_text)
            
            # MAKE SURE TO TRACK HOW MANY SUBMISSIONS WERE CRAWLED AND CLASSIFIED AS OPIOID ABUSER
            # VERSUS THOSE THAT WERE NOT (will go in paper)
            
        #elif menu_choice.startswith('3'):
            
            # Call network graphing method here
            
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