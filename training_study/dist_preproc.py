import pandas as pd
import sys
#import getopt
from time import time
import json
from pyspark.sql import DataFrame, Column
from pyspark.sql.types import StructType, StructField, StringType,ArrayType,DataType
from pyspark.sql.functions import col,udf,struct,collect_list, lower, count,lit
from pyspark import SparkContext
from pyspark.sql import SparkSession
import csv
from datetime import date, timedelta
import numpy as np
from typing import Callable



# def spark_context(appname='cms', yarn=None, verbose=False, python_files=[]):
#         # define spark context, it's main object which allow
#         # to communicate with spark
#     if  python_files:
#         return SparkContext(appName=appname, pyFiles=python_files)
#     else:
#         return SparkContext(appName=appname)

# def spark_session(appName="log-parser"):
#     """
#     Function to create new spark session
#     """
#     sc = SparkContext(appName="log-parser")
#     return SparkSession.builder.config(conf=sc._conf).getOrCreate()  


class py_or_udf:
    "Introducing — `py_or_udf` — a decorator that allows a method to act as either a regular python method or a pyspark UDF"
    def __init__(self, returnType : DataType=StringType()):
        self.spark_udf_type = returnType
        
    def __call__(self, func : Callable):
        def wrapped_func(*args, **kwargs):
            if any([isinstance(arg, Column) for arg in args]) or \
                any([isinstance(vv, Column) for vv in kwargs.values()]):
                return udf(func, self.spark_udf_type)(*args, **kwargs)
               
            else:
                return func(*args, **kwargs)
            
        return wrapped_func

@py_or_udf(returnType=StringType())
def clean_message(message,clean_short=False):
   
    import re
    message = re.sub(r'[^!"()*+,-\\\/:;<=>?@[\]^_`{|}\w\s]', '', message)  # removes any foreign character
    message=message.lower()
    message = re.sub(r'(\w+:)*\/\S*', ' ', message)  # removes filepath/url
    message=re.sub(r'/\((.*?)\)/g',' ',message)#removes strings in () 
    message = re.sub(r'\S+@\S+', ' ', message) #removes strings with @
    message = re.sub(r'(\w+-)*\w+((\.|:)\w+)*(\.|:)\w+', ' ', message) #removes no-textual strings with . : - inside 
    message = re.sub(r'(\w+\d+\w*)|(\d+\w+)', ' ', message)#removes alpha-numerical string
    message = re.sub(r'\w+_\S+', ' ', message)#removes strings with underscore
    message = re.sub(r'\.\w+', ' ', message)#removes strings with .
    message = re.sub(r'-\S+-', ' ', message)#removes strings between - 
    message = re.sub(r'([^:]+:\s(?=[^:]+:[^:]))', ' ', message) #nucleus selection
    message = re.sub(r'\S+\s=\s\S+', ' ', message)#removes patterns with string equality
    message = re.sub(r'\S+=(\w+)*', ' ', message)
    message = re.sub(r'\[\w+\]', ' ', message)# removes [string]
    message = re.sub(r'(\d+)', ' ', message)#removes digits
    message = re.sub(r'[^\w\s]', ' ', message) # removes punctuation 
#         message = re.sub(r'\\', ' ', message)#removes \
    if(clean_short):
        message = re.sub(r'\s\w{1,2}\b(?<!\bno)', ' ', message) #removes strings up to 2 characters long except for no    
    else:
        message = re.sub(r'\s[a-zA-Z]{1}(\s|\Z)', ' ', message) #removes one char string
    message = re.sub(r' +', r' ', message)# removes addictional whitespace  
    message = re.sub(r'\A ', '', message)#removes whitespace at the beginning
    message = re.sub(r' \Z', '', message)#removes whitespace at the end
  
    return message

def tokenize(df_cleaned):
    """
    This function removes tokens appearing once and stopwords
    """
    from pyspark.ml.feature import Tokenizer
    from pyspark.sql.types import IntegerType
    tokenizer = Tokenizer(inputCol="error_message", outputCol="tokenized_message")
    countTokens = udf(lambda words: len(words), IntegerType())
    tokenized = tokenizer.transform(df_cleaned)
    tokenized=tokenized.select("error_message", "tokenized_message").withColumn("tokens", countTokens(col("tokenized_message")))
    return tokenized

def clean_tokenized(df_tokenized,stop_one=True):
    """
    This function removes stopwords (optional) and words occuring once
    """
    from pyspark.ml.feature import StopWordsRemover
    import pyspark.sql.functions as F
    
    #stopword list
    stopwords = ['the','with','a','an','but','of','on','to','all','has','have','been','for','in','it','its','itself',
                'this','that','those','these','is','are','were','was','be','being','having','had','does','did','doing',
                'and','if','about','again','then','so','too','by','error'] 
    
   #removes words occurring once
    if(stop_one):  
         count_df = df_tokenized.withColumn('word', F.explode(F.col('tokenized_message')))\
         .groupBy('word')\
         .count()\
         .sort('count', ascending=False)
         stoplist=count_df.where(F.col('count')==1).select('word').rdd.flatMap(lambda x: x).collect()
         stopwords=stoplist+stopwords
    
    remover = StopWordsRemover(inputCol="tokenized_message",outputCol="cleaned_tok_message", stopWords=stopwords) 
    df_tokenized=remover.transform(df_tokenized)
    
    return df_tokenized
               
    

class preprocMex(object):
    
    
    def __init__(self,spark,hpath,clean_short,stop_one):
    
        
        self.hpath=hpath
        self.spark=spark
        self.clean_short=clean_short
        self.stop_one=stop_one
            
    def process(self,verbose=False):
       
        """
        Parse fts records from HDFS
        """

        # create new spark DataFrame
        schema = StructType([StructField('data', StructType([StructField('t__error_message', StringType(), nullable=True)]))])
        df=self.spark.read.json(self.hpath, schema)
        df=df.select(col('data.t__error_message').alias('error_message')).where('error_message <> ""')
        df.cache()
        bf_n=df.count()
        #print('before cleaning %i message'% bf_n)
        #print('...cleaning message')
        df=df.withColumn('error_message', clean_message(col('error_message'),lit(self.clean_short))).dropDuplicates()
        af_n=df.count()        
        df=tokenize(df)
        df=clean_tokenized(df,stop_one=self.stop_one)           
        #print('after cleaning %i different message'% af_n)
        #df.show()
        return df,bf_n,af_n
       
   
    
    
class MyCorpus(object):
    
    """An interator that yields sentences (lists of str)."""
    
    def __init__(self,inputDf):
        self.inputDf=inputDf
        self.list_err=self.inputDf.select(collect_list("cleaned_tok_message")).collect()[0][0]      
    
    def __iter__(self):       
                     
        for line in self.list_err:            
            yield line