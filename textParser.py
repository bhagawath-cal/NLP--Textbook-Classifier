from PyPDF2 import PdfReader
import os
import io
import csv
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import numpy as np

def main():
    # Filter for stopwords - list extracted from nltk.stopwords, pasted here to prevent the need to redownload that package
    filters = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
               'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
               'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
               'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
               'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
               'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
               'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
               'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 
               'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
               "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
               "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    filters.extend([*string.punctuation])
    filelist = ['2.pdf', '3.pdf', '4.pdf', '5.pdf', '6.pdf', '7.pdf', '8.pdf', '9.pdf', '10.pdf', '11.pdf', '13.pdf', '14.pdf', '15.pdf', '16.pdf', '17.pdf', '18.pdf', '19.pdf', '21.pdf', '22.pdf', '23.pdf', '24.pdf', '25.pdf', '26.pdf', '27.pdf', '28.pdf']
    filelist = ['2.pdf']
    page_converter = {
        2 : 4,
        3 : 31,
        4 : 58, 
        5 : 79,
        6 : 103,
        7 : 134, 
        8 : 160, 
        9 : 185,
        10 : 211,
        11 : 228,
        13 : 247,
        14 : 269,
        15 : 296,
        16 : 329,
        17 : 357,
        18 : 381,
        19 : 405,
        21 : 429,
        22 : 446,
        23 : 457,
        24 : 476,
        25 : 496,
        26 : 516,
        27 : 543,
        28 : 565
    }
    fields = ['Chapter', 'Page', 'Sentence']  
    rows = []
    for pdf in filelist:
        print(f"Parsing {pdf}")
        chapter = pdf.split('.')[0]
        reader = PdfReader("chapters/" + pdf)

        for i, page in enumerate(reader.pages):
        
            pageNum = i+(page_converter[int(chapter)])
            
            print(f"Parsing {pdf} page:{pageNum}")
            content = page.extract_text()

            if pageNum == 467:
                    print(content)

            # remove newlines 
            content = content.replace('\n', ' ').replace('\r', '')

            # Get rid of equations. Replace the colon that precedes the equation with a period to help sent_tokenize() 
            content = re.sub(r':.*\([0-9]+\.[0-9]+\)', '.', content)
            sentences = []
            
            if pageNum == 467:
                    print("parsed: ")
                    print(content)
            

            for k, sentence in enumerate(sent_tokenize(content)):
                
                

                # get rid of some non-ascii charactes that exist in the pdf
                sentence = sentence.encode('ascii', 'ignore').decode()

                sentence = word_tokenize(sentence)
                
                words = [word for word in sentence if word not in filters]
                
                parsed = " ".join(words)
                parsed = parsed
                sentences.append(parsed)
            
            
            # Generate random sentence combinations
            print(len(sentences))
            combo_lengths = np.random.randint(low=1, high=(len(sentences)), size=50)

            for length in combo_lengths:
                mock_page = []
                for sentence_index in np.random.randint(low=0, high=(len(sentences)), size=length):
                    mock_page.append(sentences[sentence_index])
                combined = " ".join(mock_page)
                rows.append([chapter, pageNum, combined])
            
                
    filename = "mock_page_test_2.csv"

    with open(filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar='\\') 
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)


if __name__ == "__main__":
   main()