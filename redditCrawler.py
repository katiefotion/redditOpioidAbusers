
import plotGraph as plot
import praw as p
import redditCrawlerConstants as rc
import random_forest
#from docutils.nodes import comment

#Crawler function: retrieves subreddits and parses through the author names and the comments for each submssion within the subreddit

def getData(forest, count_vect, tfidf, svd):
    
    reddit = p.Reddit(user_agent=rc.my_user_agent,
    client_id=rc.my_client_id,
    client_secret=rc.my_client_secret,
    username=rc.username,
    password=rc.password)
        
        
    usersDict={}
    classVar=1
    commentClassVar=1

    abuserCount = 0
    nonuserCount = 0

    print('')
    print("Subreddit Information")
    for subR in rc.subredditList:
        subreddit = reddit.subreddit(subR)
        
        print(subreddit.display_name) 
     
        for submission in subreddit.hot(limit=5):
            
            subText=submission.selftext
            activeCommentUsersList=[]
            
            #Katie's classifer function call to classify the submission text
            classVar = random_forest.classify_lsa_tfidf(forest, count_vect, tfidf, svd, subText)
            
            if(classVar==1):
                
                #Author of a submission is identified as an abuser
                #The author name is retrieved for adding it as a node in network graphing
                #every author from each submission, and across submissions is counted only once
                
                redditor1=submission.author
                name=redditor1.name
                
                commentList=[]
                usersDict[name]=commentList
                
                abuserCount += 1
                
                submission.comments.replace_more(limit=0)
                for comment in submission.comments:
                    
                    #Katie's classifier function call to classify the comment text
                    commentClassVar = random_forest.classify_lsa_tfidf(forest, count_vect, tfidf, svd, comment.body)
                    
                    if(commentClassVar==1):
                        
                        #the author of a comment of a submission is identified as an abuser
                        #list of all classified comment authors are extracted to be added as nodes for network graphing
                        #there is an edge between every classified submission author and all its corresponding classified comment authors
                        
                        usersDict.get(name).append(comment.author)
                        
                        abuserCount += 1
                    
                    else:
                        nonuserCount += 1
                
                
                
            else:
                nonuserCount += 1
        
    
    plot.addConnections(usersDict)
      
    print('')
    print('A total of ' + str(nonuserCount+abuserCount) + ' submissions and comments were crawled and ' + str(abuserCount) + ' were classified as opioid abusers')
    print('')
        
            