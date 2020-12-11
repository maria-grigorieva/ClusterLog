import pandas as pd
import sys
#import getopt
from gensim.models import Word2Vec
from time import time
import json
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType,ArrayType,DataType
from pyspark.sql.functions import col,udf,struct,collect_list
from pyspark import SparkContext
from pyspark.sql import SparkSession
import csv
import re
from pyspark.sql.functions import col, lower, count
import numpy as np
from datetime import date, timedelta
import numpy as np
import pandas as pd
from typing import Callable
from pyspark.sql import Column


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
def clean_message(message):
   
    import re
    message=message.lower()
    message = re.sub(r'\/\S*', ' ', message)  # removes filepath/url
    #message = re.sub(r'(\w+(\.|:\w+).+\w+)', ' ', message) #removes strings with . : inside
    message = re.sub(r'(\w+-)*\w+((\.|:)\w+)*(\.|:)\w+', ' ', message) #removes strings with . : - inside
    message = re.sub(r'\.\w+', ' ', message)
    #message = re.sub(r'(\w+\.)+\w+ ', ' ', message)
    message = re.sub(r'\S+\s=\s\S+', ' ', message)
    message = re.sub(r'\S+=\S+', ' ', message)
    message = re.sub(r'\w+\d+|-\w+-', ' ', message)
    
    #message = re.sub(r'(o=)(\w+\s)+\w+', 'o=', message) 
    #message = re.sub(r'(ou=)(\w+\s)+\w+', 'ou=', message) 
    #message = re.sub(r'=\w+','=', message)
    message = re.sub(r'(\d+)', ' ', message)#removes digits
    message = re.sub(r'[^\w\s]', ' ', message) # removes punctuation 
    message = re.sub(r'\s\w\s', ' ', message) #removes strings made by one letter
    message = re.sub(r'\[\w+\]', ' ', message)
    #message = re.sub(r'\s=\s', ' ', message)
    
    message = re.sub(r' +', r' ', message)# remove addictional whitespace      
    
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

def clean_tokenized(df_tokenized):
    """
    This function removes stopwords
    """
    from pyspark.ml.feature import StopWordsRemover
    stoplist = ['the','with','a','an','but','of','on','to','all','has','have','been','for','in','it','its','itself',
                'this','that','those','these','is','are','were','was','be','being','having','had','does','did','doing',
                'and','if','about','again','then','so','too','cern','cms','atlas','by']
    remover = StopWordsRemover(inputCol="tokenized_message",outputCol="cleaned_tok_message", stopWords=stoplist)
    df_tokenized=remover.transform(df_tokenized)
    return df_tokenized
               
    

class preprocMex(object):
    
    
    def __init__(self,spark,hpath):
        
        self.hpath=hpath
        self.spark=spark
            
    def process(self,verbose=False):
       
        """
        Parse fts HDFS records
        """

        # create new spark DataFrame
        schema = StructType([StructField('data', StructType([StructField('t__error_message', StringType(), nullable=True)]))])
        df=self.spark.read.json(self.hpath, schema)
        df=df.select(col('data.t__error_message').alias('error_message')).where('error_message <> ""')
        df.cache()
        #df2=df.withColumn('error_message',col('error_message'))
        bf_n=df.count()
        #print('before cleaning %i message'% bf_n)
        #print('...cleaning message')
        #df=df.withColumn('error_message', clean_mex_udf(struct(df['error_message']))).dropDuplicates()
        df=df.withColumn('error_message', clean_message(col('error_message'))).dropDuplicates()
        #df2=df2.withColumn('error_message_cleaned',clean_message(col('error_message'))).dropDuplicates()
        af_n=df.count()
        df=tokenize(df)
        df=clean_tokenized(df)       
        #print('after cleaning %i different message'% af_n)
        #df.show()
        return df,bf_n,af_n
        #return df,df2
   
    
    
class MyCorpus(object):
    
    """An interator that yields sentences (lists of str)."""
    
    def __init__(self,inputDf):
        self.inputDf=inputDf
        self.list_err=self.inputDf.select(collect_list("cleaned_tok_message")).collect()[0][0]      
    
    def __iter__(self):       
                     
        for line in self.list_err:            
            yield line