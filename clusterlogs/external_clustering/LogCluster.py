"""
Description : This file implements a wrapper around the original LogCluster code in perl
Author      : LogPAI team
License     : MIT
"""

import os
import pandas as pd
import re
import hashlib
from datetime import datetime
import subprocess


class LogParser():
    def __init__(self, messages, outdir, rex=[], support=None, rsupport=None, separator=None, lfilter=None,
                 template=None,
                 lcfunc=None, syslog=None, wsize=None, csize=None, wweight=None, weightf=None, wfreq=None, wfilter=None,
                 wsearch=None, wrplace=None, wcfunc=None, outliers=None, readdump=None,
                 writedump=None, readwords=None, writewords=None):
        """
        Arguments
        ---------
            rsupport = < relative_support >
            separator = < word_separator_regexp >
            lfilter = < line_filter_regexp >
            template = < line_conversion_template >
            lcfunc = < perl_code >
            syslog = < syslog_facility >
            wsize = < word_sketch_size >
            csize = < candidate_sketch_size >
            wweight = < word_weight_threshold >
            weightf = < word_weight_function >
            wfreq = < word_frequency_threshold >
            wfilter = < word_filter_regexp >
            wsearch = < word_search_regexp >
            wreplace = < word_replace_string >
            wcfunc = < perl_code >
            outliers = < outlier_file >
            readdump = < dump_file >
            writedump = < dump_file >
            readwords = < word_file >
            writewords = < word_file >
        """
        self.messages = messages
        self.savepath = outdir
        self.paras = [support, rsupport, separator, lfilter, template,
                      lcfunc, syslog, wsize, csize, wweight, weightf, wfreq,
                      wfilter, wsearch, wrplace, wcfunc, outliers, readdump, writedump,
                      readwords, writewords]
        self.paranames = ["support", "rsupport", "separator", "lfilter", "template", "lcfunc", "syslog",
                          "wsize", "csize", "wweight", "weightf", "wfreq", "wfilter", "wsearch", "wrplace",
                          "wcfunc", "outliers", "readdump", "writedump", "readwords", "writewords"]
        self.perl_command = "perl {} --input {}".format(os.path.join(os.path.dirname(__file__), 'logcluster.pl'),
                                                        "logcluster_input.log")
        for idx, para in enumerate(self.paras):
            if para:
                self.perl_command += " -{} {}".format(self.paranames[idx], para)
        self.perl_command += " > logcluster_output.txt"
        self.rex = rex

    def parse(self):
        start_time = datetime.now()
        self.df_log = pd.DataFrame(self.messages, columns=['Content'])
        self.df_log.insert(0, 'LineId', None)
        self.df_log['LineId'] = [i + 1 for i in range(len(self.messages))]

        with open('logcluster_input.log', 'w') as fw:
            for line in self.df_log['Content']:
                if self.rex:
                    for currentRex in self.rex:
                        line = re.sub(currentRex, ' ', line)
                        line = re.sub(' +', ' ', line)
                fw.write(line + '\n')
        try:
            print("Run LogCluster command...\n>> {}".format(self.perl_command))
            subprocess.check_call(self.perl_command, shell=True)
        except:
            print("LogCluster run failed! Please check perl installed.\n")
            raise
        result = self.constructResult()
        os.remove("logcluster_input.log")
        os.remove("logcluster_output.txt")
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        return result

    def constructResult(self):

        result = []
        with open("logcluster_output.txt", 'r') as fr:
            for line in fr:
                line = line.split('\t')
                result.append(line[0].strip())
                #result.append([line[0].strip(), line[2].strip()])
        return result
