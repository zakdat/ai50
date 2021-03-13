import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # create empty dict
    files = {}

    # iterate through corpus files
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # check path exists and is .txt file
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            # open contents of txt file
            with open(file_path, 'r', encoding='utf8') as f:
                # add filename and string to dict
                files[filename] = f.read()
                
    # return dict
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    list_of_words = []

    # lowercase all
    document = document.lower()
    # remove punctuation
    document = document.translate(str.maketrans('', '', string.punctuation))
    # tokenize words
    words = nltk.word_tokenize(document)

    for word in words:
        if word not in nltk.corpus.stopwords.words("english"):
            list_of_words.append(word)

    return list_of_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_dict = {}
    # iterate through documents and words in docs
    for doc in documents:
        already_used = set()

        for word in documents[doc]:
            # if the word is in the dict, but hasn't already been seen in the doc, add 1 to value
            if word in idf_dict and not already_used:
                idf_dict[word] += 1
                # add word to set of words already in doc
                already_used.add(word)          
            # if a word is not already in the dict
            elif word not in idf_dict:
                idf_dict[word] = 1
                # add word to set of words already in doc
                already_used.add(word)
    
    # iterate through dict and divide ln number of documents by number of docs word appears in
    number_of_docs = len(documents)
    for word in idf_dict:
        idf_dict[word] = math.log(number_of_docs) / idf_dict[word]

    return idf_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # create empty dict for tf and tfidf scores, filenames as keys
    tf_dict = {}
    tfidf_dict = {}

    # initialise tfidf values to 0
    for filename in files:   
        tfidf_dict[filename] = 0

    # calculate tf for each term for each doc
    # iterate through term in query
    for term in query:
        # iterate through filenames
        for filename in files:
            tf_dict[filename] = 0
            # iterate through words
            for word in files[filename]:
                # add 1 to tfdict for each time term appears in list
                if word == term:
                    tf_dict[filename] += 1
            
            # add tf-idf to dict entry
            tfidf_dict[filename] += tf_dict[filename] * idfs[term]
    
    # Files should be ranked according to the sum of tf-idf values for any 
    # word in the query that also appears in the file.
    # get list of tuples sorted by tfidf values
    tfidf_dict_sorted = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)
    filenames_list = []
    # get ordered list of filenames
    for x in range(n):
        filenames_list.append(tfidf_dict_sorted[x][0])
    
    return filenames_list

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentences_list = []

    for sentence in sentences:
        # list for sentence measures
        sentence_measures = [sentence, 0, 0]
        
        for term in query:
            if term in sentences[sentence]:
                # sum idf values
                sentence_measures[1] += idfs[term]
                # query term density is number of terms divided by length of sentence
                sentence_measures[2] += sentences[sentence].count(term) / len(sentences[sentence])
    
        sentences_list.append(sentence_measures)

    sorted_list = sorted(sentences_list, key=lambda item: (item[1], item[2]), reverse=True)
    final_list = []
    # get ordered list of sentences
    for x in range(n):
        final_list.append(sorted_list[x][0])

    return final_list


if __name__ == "__main__":
    main()
