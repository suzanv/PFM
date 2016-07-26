# coding=utf-8
# python3 create_summary_for_unseen_data_TNO.py example_query_result_full_threads_improved.json example_query_result_full_threads.summary.json


# + 1. Read json file (query+result list), and extract threads
# + 2. For each thread in result list, extract post feats
# + 3. Standardize post feats
# + 4. Apply linear model
# + 5. Apply threshold
# + 6. Write to json file with for each thread, for each postid the value 1 or 0 for in/out summary, and the predicted value of the linear model


import os
import sys
import re
import string
import operator
import functools
import math
import numpy
from scipy.linalg import norm
import time
import json
import fileinput

json_filename = sys.argv[1]
outfilename = sys.argv[2]


def tokenize(t):
    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãåA-Z0-9- \']', "", text)
    wrds = text.split()
    return wrds

caps = "([A-Z])"
prefixes = "(Dhr|Mevr|Dr|Drs|Mr|Ir|Ing)[.]"
suffixes = "(BV|MA|MSc|BSc|BA)"
starters = "(Dhr|Mevr|Dr|Drs|Mr|Ir|Ing)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|nl)"

def split_into_sentences(text):
    # adapted from http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = re.sub("([\.\?!]+\)?)","\\1<stop>",text)
    if "<stop>" not in text:
        text += "<stop>"
    text = text.replace("<prd>",".")
    text = re.sub('  +',' ',text)
    sents = text.split("<stop>")
    sents = sents[:-1]
    sents = [s.strip() for s in sents]
    return sents

def count_punctuation(t):
    punctuation = string.punctuation
    punctuation_count = len(list(filter(functools.partial(operator.contains, punctuation), t)))
    textlength = len(t)
    relpc = 0
    if textlength>0:
        relpc = float(punctuation_count)/float(textlength)
    return relpc

def nrofsyllables(w):
    count = 0
    vowels = 'aeiouy'
    w = w.lower().strip(".:;?!")
    if w[0] in vowels:
        count +=1
    for index in range(1,len(w)):
        if w[index] in vowels and w[index-1] not in vowels:
            count +=1
    if w.endswith('e'):
        count -= 1
    if w.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count


def fast_cosine_sim(a, b):
    #print (a)
    if len(b) < len(a):
        a, b = b, a
    up = 0
    a_value_array = []
    b_value_array = []
    for key in a:
        a_value = a[key]
        b_value = b[key]
        a_value_array.append(a_value)
        b_value_array.append(b_value)
        up += a_value * b_value
    if up == 0:
        return 0
    return up / norm(a_value_array) / norm(b_value_array)




columns = dict() # key is feature name, value is dict with key (threadid,postid) and value the feature value

def addvaluestocolumnsoverallthreads(dictionary,feature):
    global columns
    columndict = dict()
    if feature in columns: # if this is not the first thread, we already have a columndict for this feature
        columndict = columns[feature] # key is (threadid,postid) and value the feature value
    for (threadid,postid) in dictionary:
        value = dictionary[(threadid,postid)]
        columndict[(threadid,postid)] = value
    #print feature, columndict
    columns[feature] = columndict

def standardize_values(columndict,feature):
    values = list()
    for (threadid,postid) in columndict:
        values.append(columndict[(threadid,postid)])
    mean = numpy.mean(values)
    stdev = numpy.std(values)
    normdict = dict() # key is (threadid,postid) and value the normalized feature value
    for (threadid,postid) in columndict:
        value = columndict[(threadid,postid)]
        if stdev == 0.0:
            stdev = 0.000000000001
            print ("stdev is 0! ", feature, value, mean, stdev)
        #if value != 0:
        normvalue = (float(value)-float(mean))/float(stdev)
        normdict[(threadid,postid)] = normvalue
#        if feature == "noofupvotes":
#           print threadid,postid, feature, float(value), mean, stdev, normvalue, len(columndict)

    return normdict

months_conversion = {'januari': '01', 'februari': '02', 'maart': '03', 'april': '04', 'mei': '05', 'juni': '06', 'juli': '07', 'augustus': '08', 'september': '09', 'oktober': '10', 'november': '11', 'december': '12', 'May': '05'}
postsperthread = dict() # dictionary with threadid as key and posts dictionary ((author,timestamp)->postid) as value

def findQuote (content,thread_id) :
    pattern = re.compile("\*\*\[(.*) schreef op (.*) @ ([0-9:]+)\]")
    # > quote: > > **[kattie2 schreef op 28 januari 2015 @ 16:32]
    match = pattern.search(content)
    referred_post = ""
    if match :
        #print (match)
        user = match.group(1)
        date = match.group(2)
        time = match.group(3)
        datepattern = re.compile("^[^ ]+ [^ ]+ [^ ]+$")
        if datepattern.match(date) :
            [day,month,year] = date.split()
            monthnumber = months_conversion[month]
            converteddate = day+"-"+monthnumber+"-"+year+" "+time
            if thread_id in postsperthread:
                postsforthread = postsperthread[thread_id]

                if (user,converteddate) in postsforthread:
                    referred_post = postsforthread[(user,converteddate)]
                    #sys.stderr.write("Found referred post: "+user+" "+converteddate+" :: "+postid+"\n")
                else :
                    #sys.stderr.write("Quoted post is missing from thread: "+user+" "+converteddate+" ")

                    user = "anoniem"
                    if (user,converteddate) in postsforthread :
                        referred_post = postsforthread[(user,converteddate)]
                        #sys.stderr.write("but found anoniempje at that timestamp and used that")
                    else :
                        for (u,d) in postsforthread:
                            if converteddate == d:# and re.match(".*[aA]noniem.*",u):
                                #sys.stderr.write("but found "+u+" at that timestamp and used that")
                                referred_post = postsforthread[(u,d)]
                                break
                    #sys.stderr.write ("\n")
    return referred_post

'''
MAIN: READ JSON AND EXTRACT FEATURES
'''

openingpost_for_thread = dict() # key is threadid, value is id of opening post
postids_dict = dict() # key is (threadid,postid), value is postid. Needed for pasting the columns at the end
threadids = dict() # key is (threadid,postid), value is threadid. Needed for pasting the columns at the end
threadids_list = list() # needed for feature standardization: length of list is total no of posts
threadids_dict = dict()
postids_per_threadid = dict()
upvotecounts = dict()  # key is (threadid,postid), value is # of upvotes
responsecounts = dict()  # key is (threadid,postid), value is # of replies
cosinesimilaritiesthread = dict()  # key is (threadid,postid), value is cossim with term vector for complete thread
cosinesimilaritiestitle = dict()  # key is (threadid,postid), value is cossim with term vector for title
uniquewordcounts = dict()  # key is (threadid,postid), value is unique word count in post
wordcounts = dict() # key is (threadid,postid), value is word count in post
typetokenratios = dict()  # key is (threadid,postid), value is type-token ratio in post
abspositions = dict() # key is (threadid,postid), value is absolute position in thread
relpositions = dict()  # key is (threadid,postid), value is relative position in thread
relauthorcountsinthreadforpost = dict()  # key is (threadid,postid), value is relative number of posts by author in this thread
relpunctcounts = dict() # key is (threadid,postid), value is relative punctuation count in post
avgwordlengths = dict() # key is (threadid,postid), value is average word length (nr of characters)
avgnrsyllablesinwords = dict() # key is (threadid,postid), value is average word length (nr of syllables)
avgsentlengths = dict() # key is (threadid,postid), value is average word length (nr of words)
readabilities = dict() # key is (threadid,postid), value is readability
bodies = dict()  # key is (threadid,postid), value is content of post

op_source_strings = dict() # key is threadid, value is the value of the 'source' field of the opening post
post_source_strings = dict() # key is (threadid,postid), value is the value of the 'source' field of the comment

#print time.clock(), "\t", "go through files"

json_string = ""
with open(json_filename,'r') as json_file:
    for line in json_file:
        json_string += line.rstrip()



parsed_json = json.loads(json_string)
threads = parsed_json['threads']


for thread in threads:
    threadid = str(thread)
    #print (threadid)

    postids = list()
    termvectors = dict()  # key is postid, value is dict with term -> termcount for post
    termvectorforthread = dict()  # key is term, value is termcount for full thread
    termvectorfortitle = dict()  # key is term, value is termcount for title
    authorcountsinthread = dict()  # key is authorid, value is number of posts by author in this thread

    subthreads_in_thread = threads[threadid]
    #print (len(subthreads_in_thread),"subthreads in thread")
    for subthread in subthreads_in_thread:
        # A subthread in the thread contains a message (opening post) and an array of comments (posts)
        openingpost = subthread['message']
        op_source = openingpost['_source']
        text_of_openingpost = op_source['text']
        author_of_openingpost = op_source['author']
        timestamp_of_openingpost = op_source['time']
        postid_of_openingpost = op_source['msg_id']
        openingpost_for_thread[threadid] = postid_of_openingpost

        # save all author-time combinations (including of openingpost) for postid lookup
        postsforthread = dict()
        if threadid in postsperthread:
            postsforthread = postsperthread[threadid]
        postsforthread[(author_of_openingpost,timestamp_of_openingpost)] = postid_of_openingpost
        postsperthread[threadid] = postsforthread

        #print (openingpost_for_thread[threadid])
        # In the TNO json, the msg_id of the opening post is equal to the threadid
        category = "" # no category information in json
        title = ""
        if 'thread_title' in subthread: # thread_title currently defined on post level
            title = subthread['title']


        #print (text_of_openingpost)
        posts = subthread['comments']
        noofposts = len(posts)
        #print (threadid,"no of comments in this subthread:",noofposts)

        for post in posts:
            post_source = post['_source']
            # first go through the thread to find all authors,
            postid = post_source['msg_id']
            timestamp = post_source['time']
            author = post_source['author']
            if 'thread_title' in post_source:
                title = post_source['thread_title'] # thread_title currently defined on post level


            if author in authorcountsinthread:
                authorcountsinthread[author] += 1
            else:
                authorcountsinthread[author] =1

            # and save all author-time combinations for postid lookup
            postsforthread = dict()
            if threadid in postsperthread:
                postsforthread = postsperthread[threadid]
            postsforthread[(author,timestamp)] = postid
            postsperthread[threadid] = postsforthread



        postcount = 0
        for post in posts:
            # then go through the thread again to calculate all feature values
            postcount += 1
            post_source = post['_source']
            postid = post_source['msg_id']
            timestamp = post_source['time']
            author = post_source['author']
            post_source_string = re.sub("'","\"",str(post_source))
            post_source_strings[(threadid,postid)] = post_source_string

            postidsforthread = list()
            if threadid in postids_per_threadid:
                postidsforthread = postids_per_threadid[threadid]
            postidsforthread.append(postid)
            postids_per_threadid[threadid] = postidsforthread


            body = post_source['text']
            postids.append(postid)

            postids_dict[(threadid,postid)] = postid
            threadids[(threadid,postid)] = threadid
            threadids_list.append(threadid) # needed for feature standardization: length of list is total no of posts
            threadids_dict[threadid] = 1
            parentid = ""
            if 'parent' in post_source:
                parentid = post_source['parent'] # no parent field in current version of json
            else:
                parentid = findQuote(body,threadid)
                if parentid != openingpost_for_thread[threadid]:
                    # do not save responses for openingpost because openingpost will not be in feature file
                    # (and disturbs the column for standardization)
                    if (threadid,parentid) in responsecounts:
                        responsecounts[(threadid,parentid)] += 1
                    else:
                        responsecounts[(threadid,parentid)] = 1

            upvotes = 0
            if 'upvotes' in post_source:
                upvotes = int(post_source['upvotes']) # no upvotes in current version of json (does not exist for viva)
            upvotecounts[(threadid,postid)] = upvotes

            relauthorcountsinthreadforpost[(threadid,postid)] = float(authorcountsinthread[author])/float(noofposts)

            if "smileys" in body:
                body = re.sub(r'\((http://forum.viva.nl/global/(www/)?smileys/.*.gif)\)','',body)

            if "http" in body:
                body = re.sub(r'http://[^ ]+','',body)

            bodies[(threadid,postid)] = body

            words = tokenize(body)
            wc = len(words)

            sentences = split_into_sentences(body)
            sentlengths = list()

            for s in sentences:
                sentwords = tokenize(s)
                nrofwordsinsent = len(sentwords)
                #print (s, nrofwordsinsent)
                sentlengths.append(nrofwordsinsent)
            if len(sentences) > 0:
                avgsentlength = numpy.mean(sentlengths)
                avgsentlengths[(threadid,postid)] = avgsentlength
            else:
                avgsentlengths[(threadid,postid)] = 0
            relpunctcount = count_punctuation(body)
            relpunctcounts[(threadid,postid)] = relpunctcount
            #print (body, punctcount)
            wordcounts[(threadid,postid)] = wc
            uniquewords = dict()
            wordlengths = list()
            nrofsyllablesinwords = list()
            for word in words:
                #print (word, nrofsyllables(word))
                nrofsyllablesinwords.append(nrofsyllables(word))
                wordlengths.append(len(word))
                uniquewords[word] = 1
                if word in termvectorforthread:  # dictionary over all posts
                   termvectorforthread[word] += 1
                else:
                   termvectorforthread[word] = 1

                worddict = dict()
                if postid in termvectors:
                    worddict = termvectors[postid]
                if word in worddict:
                    worddict[word] += 1
                else:
                    worddict[word] = 1
                termvectors[postid] = worddict

            uniquewordcount = len(uniquewords.keys())
            uniquewordcounts[(threadid,postid)] = uniquewordcount
            readabilities[(threadid,postid)] = 0

            if wc > 0:
                avgwordlength = numpy.mean(wordlengths)
                #avgnrsyllablesinword = numpy.mean(nrofsyllablesinwords)
                avgwordlengths[(threadid,postid)] = avgwordlength
                #avgnrsyllablesinwords[(threadid,postid)] = avgnrsyllablesinword
                #readabilities[(threadid,postid)] = readability(avgnrsyllablesinword,avgsentlength)
            else:
                avgwordlengths[(threadid,postid)] = 0

            #print (threadid, postid, wc, avgsentlengths[(threadid,postid)])

            typetokenratio = 0
            if wordcounts[(threadid,postid)] > 0:
                typetokenratio = float(uniquewordcount) / float(wordcounts[(threadid,postid)])
            typetokenratios[(threadid,postid)] = typetokenratio

            relposition = float(postcount)/float(noofposts)
            #relposition = float(postid)/float(noofposts)
            relpositions[(threadid,postid)] = relposition
            abspositions[(threadid,postid)] = postcount

            #abspositions[(threadid,postid)] = postid
        #print wordcounts
    # add zeroes for titleterms that are not in the thread vector
    titlewords = tokenize(title)
    print (title)
    for tw in titlewords:
        if tw in termvectorfortitle:
            termvectorfortitle[tw] += 1
        else:
            termvectorfortitle[tw] = 1

    print (threadid,"total no of comments for thread:",len(postids))
    for titleword in termvectorfortitle:
        if titleword not in termvectorforthread:
            termvectorforthread[titleword] = 0

    # add zeroes for terms that are not in the title vector:
    for word in termvectorforthread:
        if word not in termvectorfortitle:
            termvectorfortitle[word] = 0

    # add zeroes for terms that are not in the post vector:
    for postid in termvectors:
        worddictforpost = termvectors[postid]
        for word in termvectorforthread:
            if word not in worddictforpost:
                worddictforpost[word] = 0
        termvectors[postid] = worddictforpost

#    for term in termvectorforthread:
#        print(postid,term,termvectors[postid][term])
    cossimthread = fast_cosine_sim(termvectors[postid], termvectorforthread)
    cossimtitle = fast_cosine_sim(termvectors[postid], termvectorfortitle)
    cosinesimilaritiesthread[(threadid,postid)] = cossimthread
    cosinesimilaritiestitle[(threadid,postid)] = cossimtitle

    #print postid, cossimthread

    for postid in postids:
        #print postid, abspositions[(threadid,postid)]
        if not (threadid,postid) in cosinesimilaritiesthread:
            cosinesimilaritiesthread[(threadid,postid)] = 0.0
        if not (threadid,postid) in cosinesimilaritiestitle:
            cosinesimilaritiestitle[(threadid,postid)] = 0.0
        if not (threadid,postid) in responsecounts:
            # don't store the counts for the openingpost
            #print "postid not in responsecounts", postid, "opening post:", openingpost_for_thread[threadid]
            responsecounts[(threadid,postid)] = 0

    addvaluestocolumnsoverallthreads(threadids, "threadid")
    addvaluestocolumnsoverallthreads(postids_dict, "postid")

    addvaluestocolumnsoverallthreads(abspositions, "abspos")
    addvaluestocolumnsoverallthreads(relpositions, "relpos")
    addvaluestocolumnsoverallthreads(responsecounts, "noresponses")
    addvaluestocolumnsoverallthreads(upvotecounts, "noofupvotes")
    addvaluestocolumnsoverallthreads(cosinesimilaritiesthread, "cosinesimwthread")
    addvaluestocolumnsoverallthreads(cosinesimilaritiestitle, "cosinesimwtitle")
    addvaluestocolumnsoverallthreads(wordcounts, "wordcount")
    addvaluestocolumnsoverallthreads(uniquewordcounts, "uniquewordcount")
    addvaluestocolumnsoverallthreads(typetokenratios, "ttr")
    addvaluestocolumnsoverallthreads(relpunctcounts, "relpunctcount")
    addvaluestocolumnsoverallthreads(avgwordlengths, "avgwordlength")
    addvaluestocolumnsoverallthreads(avgsentlengths, "avgsentlength")
    addvaluestocolumnsoverallthreads(relauthorcountsinthreadforpost,"relauthorcountsinthread")


columns_std = dict()

featnames = ("threadid","postid","abspos","relpos","noresponses","noofupvotes","cosinesimwthread","cosinesimwtitle","wordcount","uniquewordcount","ttr","relpunctcount","avgwordlength","avgsentlength","relauthorcountsinthread")


for featurename in featnames:
    columndict = columns[featurename]

    columndict_with_std_values = columndict
    if featurename != "postid" and featurename != "threadid":
        columndict_with_std_values = standardize_values(columndict,featurename)
    columns_std[featurename] = columndict_with_std_values


feat_weights = dict()
feat_weights["abspos"] = -0.69456
feat_weights["relpos"] = -0.17991
feat_weights["noresponses"] = -0.11507
feat_weights["noofupvotes"] = 0 # not in viva data
feat_weights["cosinesimwthread"] = 0.32817
feat_weights["cosinesimwtitle"] = 0.13588
feat_weights["wordcount"] = -1.44997
feat_weights["uniquewordcount"] = 1.90478
feat_weights["ttr"] = -0.38033
feat_weights["relpunctcount"] = -0.12664
feat_weights["avgwordlength"] = 0.18753
feat_weights["avgsentlength"] = 0 # not significant
feat_weights["relauthorcountsinthread"] =  -0.11927

intercept = 2.45595

#(Intercept)              2.45595    0.03381  72.646  < 2e-16 ***
#abspos                  -0.69456    0.07358  -9.440  < 2e-16 ***
#relpos                  -0.17991    0.07159  -2.513 0.012025 *
#noresponses             -0.11507    0.03386  -3.399 0.000686 ***
#cosinesimwthread         0.32817    0.07379   4.447 9.02e-06 ***
#cosinesimwtitle          0.13588    0.03523   3.857 0.000117 ***
#wordcount               -1.44997    0.17763  -8.163 4.81e-16 ***
#uniquewordcount          1.90478    0.19878   9.582  < 2e-16 ***
#ttr                     -0.38033    0.06543  -5.813 6.81e-09 ***
#relpunctcount           -0.12664    0.03995  -3.170 0.001541 **
#avgwordlength            0.18753    0.04156   4.512 6.67e-06 ***
#avgsentlength           -0.01406    0.04058  -0.347 0.728958
#relauthorcountsinthread -0.11927    0.03410  -3.498 0.000476 ***

selected_posts = dict() # key is thread id, value is dict with postid -> predicted # of votes (predicted_outcome by LRM)

columnnames = list(featnames)
columnnames.append("predicted")
columnnames.append("selected_based_on_threshold")

'''
out = open(outfilename,'w')

for columnname in columnnames:
    out.write(columnname+"\t")
out.write("\n")


for (threadid,postid) in threadids:
    predicted_outcome = intercept
    #out.write(threadid+"\t"+postid+"\t")
    for featurename in featnames:
        columndict_with_std_values = columns_std[featurename]
        value = columndict_with_std_values[(threadid,postid)]
        out.write(str(value)+"\t")
        if featurename in feat_weights:
            weighted_value = feat_weights[featurename]*value
            #print value, weighted_value
            predicted_outcome += weighted_value
    out.write(str(predicted_outcome)+"\t")
    if predicted_outcome >= 3.72:
        # fixed threshold, based on tune set 5 from viva data
        out.write("1")
    else:
        out.write("0")
    out.write("\n")


out.close()
'''

predicted = dict()
include = dict()

for (threadid,postid) in threadids:
    predicted_outcome = intercept
    #out.write(threadid+"\t"+postid+"\t")
    for featurename in featnames:
        columndict_with_std_values = columns_std[featurename]
        value = columndict_with_std_values[(threadid,postid)]
        if featurename in feat_weights:
            weighted_value = feat_weights[featurename]*value
            predicted_outcome += weighted_value
    predicted[(threadid,postid)] = predicted_outcome
    if predicted_outcome >= 3.72:
        # fixed threshold, based on tune set 5 from viva data
        include[(threadid,postid)] = 1
    else:
        include[(threadid,postid)] = 0


json = open(outfilename,'w')
json.write('{')
json.write('"threads": [\n')

i=1
no_of_threads = len(threadids_dict)
for threadid in threadids_dict:
    json_for_all_posts = ''
    j=1
    no_of_posts = len(postids_per_threadid[threadid])
    for postid in postids_per_threadid[threadid]:
        predicted_value = predicted[(threadid,postid)]
        include_value = include[(threadid,postid)]
        json_for_post = '{ "_id" : "'+postid+'", '+'"summary_include": "'+str(include_value)+'", "summary_predicted": "'+str(predicted_value)+'" }'

        if j == no_of_posts:
            json_for_post+='\n'
        else :
            json_for_post+=',\n'
        json_for_all_posts += json_for_post
        j += 1

    # always include the opening post in the summary
    json.write('{"'+threadid+'": { "message": { "summary_include": "1", "summary_predicted": "1" }}, "comments": [\n'+json_for_all_posts+'] }')

    if i == no_of_threads:
        json.write('\n')
    else :
        json.write(',\n')
    i += 1

    '''

    '''

json.write(']')
json.write('}')

json.close()