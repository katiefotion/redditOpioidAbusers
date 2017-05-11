'''
Created on 10-May-2017

@author: Neharika Mazumdar
'''
import plotGraph as plot
import praw as p
import redditCrawlerConstants as rc
import lsa_tfidf

#Crawler function: retrievs subreddit and parses through the author names and the comments for each submssion within the subreddit

def getData(train, tfs_mat, count_vect, tfidf, svd, threshold, K):
    reddit = p.Reddit(user_agent=rc.my_user_agent,
    client_id=rc.my_client_id,
    client_secret=rc.my_client_secret,
    username=rc.username,
    password=rc.password)
        
        
    activeUsersList=[]
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
            
            #Katie's classsifer function call to classify the submission text
            classVar = lsa_tfidf.classify(train, tfs_mat, count_vect, tfidf, svd, subText, threshold, K)
            
            if(classVar==1):
                
                redditor1=submission.author
                name=redditor1.name
                activeUsersList.append(name)
                
                abuserCount += 1
                
                submission.comments.replace_more(limit=0)
                for comment in submission.comments:
                    
                    #Katie's classifier function call to classify the comment text
                    commentClassVar = lsa_tfidf.classify(train, tfs_mat, count_vect, tfidf, svd, comment.body, threshold, K)
                    
                    if(commentClassVar==1):
                        usersDict[name]=comment.author
                        abuserCount += 1
                    
                    else:
                        nonuserCount += 1
        
            else:
                nonuserCount += 1
        
        plot.addConnections(activeUsersList,usersDict,subR)
        
    print('')
    print('A total of ' + str(nonuserCount+abuserCount) + ' submissions and comments were crawled and ' + str(abuserCount) + ' were classified as opioid abusers')
    print('')
        
            
              