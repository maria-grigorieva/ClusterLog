#!/usr/bin/python
import sys
import getopt
from gensim.models import Word2Vec
import pprint
from time import time
from pyonmttok import Tokenizer
#from smart_open import open
import json
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType,ArrayType
from pyspark.sql.functions import col,udf,struct,collect_list
from pyspark import SparkContext, StorageLevel
from pyspark.sql import SparkSession
import csv
import os
import re
import logging

import site



from pyspark.sql.functions import col, lit, regexp_replace, trim, lower, concat, count
import numpy as np
import pandas as pd
import nltk

import uuid



def spark_context(appname='cms', yarn=None, verbose=False, python_files=[]):
        # define spark context, it's main object which allow
        # to communicate with spark
    if  python_files:
        return SparkContext(appName=appname, pyFiles=python_files)
    else:
        return SparkContext(appName=appname)

def spark_session(appName="log-parser"):
    """
    Function to create new spark session
    """
    sc = SparkContext(appName="log-parser")
    return SparkSession.builder.config(conf=sc._conf).getOrCreate()  

@udf(returnType=StringType()) 
def clean_message(message):
    import re
    message = re.sub(r'\S+\.\S+', ' ', message)  # any URL
    message = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', message) # remove all substrings with digits
    message = re.sub(r'(\d+)', ' ', message) # remove all other digits
    message = re.sub(r'[^\w\s]', ' ', message) # removes all punctuation
    message = re.sub(r' +', r' ', message)
    message=message.lower()
    return message

def tokenize_message(message, tokenizer_type, spacer_annotate, preserve_placeholders,spacer_new):
    tokenizer = Tokenizer(tokenizer_type, spacer_annotate=spacer_annotate, preserve_placeholders= preserve_placeholders, spacer_new=spacer_new)
    return tokenizer.tokenize(message)[0]


class uniqueMex(object):
    
    
    def __init__(self,spark,month,days):
        
        self.spark=spark
        self.hdir='hdfs:///project/monitoring/archive/fts/raw/complete'
        self.month=month
        self.days=days        
            
    def fts_messages(self,verbose=False):
        """
        Parse fts HDFS records
        """
       #clean_mex_udf=udf(lambda row: clean_message(x) for x in row, StringType()) #user defined function to clean spark dataframe
        clean_mex_udf=udf(lambda x: clean_message(x), StringType())
        self.spark.udf.register('clean_mex_udf',clean_mex_udf)
        if len(self.days)==0:
            hpath=self.hdir+'/'+self.month
        else:
            hpath = [('%s/%s' % (self.hdir,self.month+iDate)) for iDate in self.days]
        # create new spark DataFrame
        schema = StructType([StructField('data', StructType([StructField('t__error_message', StringType(), nullable=True)]))])
        df=self.spark.read.json(hpath, schema)
        df=df.select(col('data.t__error_message').alias('error_message')).where('error_message <> ""')
        df.cache()
        bf_n=df.count()
        print('before cleaning %i messages'% bf_n)
        print('...cleaning messages')
        #df=df.withColumn('error_message', clean_mex_udf(struct(df['error_message']))).dropDuplicates()
        df=df.withColumn('error_message', clean_message(col('error_message'))).dropDuplicates()
        af_n=df.count()
        print('after cleaning %i different messages'% af_n)
        #df.show()
        return df,bf_n,af_n
   

    
class MyCorpus(object):
    
    """An interator that yields sentences (lists of str)."""
    
    def __init__(self,inputDf):
        self.inputDf=inputDf
        self.list_err=self.inputDf.select(collect_list("error_message")).collect()[0][0]      
    
    def __iter__(self):       
                     
        for line in self.list_err:
            tokenized=tokenize_message(line, 'space',False,True,False)
            yield tokenized
        
                     
                    

def main(argv):
    
    spark = spark_session()
    #inputfile = ''
    outputfile = '' #name of the model
    outputfile=sys.argv[1]
    nDays=int(sys.argv[2]) #number of days to train over
#     try:
#         opts, args = getopt.getopt(argv, "o:",["ofile="]) #argv=argument list to be parsed
#                                                           #options that require an argument are followed by a colon ':'                                                      
#         #opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
#     except getopt.GetoptError:
#         print
#         #'training_corpus.py -i <inputfile> -o <outputfile>'
#         'training_corpus.py -o <outputfile>'
#         sys.exit(2)
#     if opts[0] in ("-o", "--ofile"):
#         outputfile = opts[1]         
    #for opt, arg in opts:
#         if opt == '-h':
#             print
#             'test.py -i <inputfile> -o <outputfile>'
#             sys.exit()
#         #elif opt in ("-i", "--ifile"):
#             #inputfile = arg
#         elif opt in ("-o", "--ofile"):
#             outputfile = arg
                                                                              
    days_vec=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
   # days_vec=['01','02','03','04','05']
    days=[days_vec[i] for i in np.arange(0,nDays)]


    month='2020/01/'
    fts,bf_n,af_n=uniqueMex(spark,month,days).fts_messages() #bf_n and af_n number of messages
    tokenized = MyCorpus(fts)
    print('...starting training')
    try:
        start_time = time()
        model = Word2Vec(sentences=tokenized,compute_loss=True,size=300,window=7, min_count=1, workers=4, iter=30)
        tot_time=time() - start_time
        print("--- %f seconds ---" % tot_time)
        loss=model.get_latest_training_loss()
        print('latest training loss:',loss)
        with open('training_parameters.csv', mode='a',newline='') as tFile:
            file_writer = csv.writer(tFile)
            file_writer.writerow([nDays,bf_n,af_n,loss,tot_time])
        model.save(outputfile)
        print('Training has finished. Model saved in file. Thanks for coming :)')
    except Exception as e:
        print('Training model error:', e)
   
   

if __name__ == "__main__":
    main(sys.argv[1:]) # get everything after the script name
   
