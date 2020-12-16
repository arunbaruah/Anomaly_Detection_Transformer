import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import importlib
import re

from Parsers import Drain
from Parsers import Spell

importlib.reload(Spell)

def parse(log_source, log_file, algorithm, start_date = '', start_time = '', end_date = '', end_time = ''):  
    """
    Parses log file.

    Args:
        log_source: The source of the logs (e.g. HDFS, Openstack, Linux).
        log_file: The name of the log file.
        algorithm: Parsing algorithm: Spell or Drain.
        
    """
    
    if log_source == 'HDFS':
        input_dir = 'Dataset/' + log_source
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
        regex      = [
            r'blk_(|-)[0-9]+' , # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
        ]    
    elif log_source == 'Linux':
        print("Parse linux logs")
    elif log_source == 'Openstack':
        print("Parse Openstack logs")
        
    if algorithm == 'Spell':
        output_dir = 'Spell_result/'  # The output directory of parsing results
        tau        = 0.6  # Message type threshold (default: 0.5)
        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
        parser.parse(log_file)        
    elif algorithm == 'Drain':
        output_dir = 'Drain_result/'  # The output directory of parsing results
        st         = 0.5  # Similarity threshold
        depth      = 4  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
        parser.parse(log_file)
        
        
def backtrace(pred, log_source, algorithm):
    """
    Find log templates from sequence of log keys.

    Args:
        pred: The sequence of log keys.
        log_source: The source of the log keys.
        algorithm: Parsing algorithm: Spell or Drain.
    """
    log_template = pd.read_csv(algorithm + "_result/" + log_source + "_templates.csv") 
    y = np.squeeze(pred.tolist())
    for log in y:
        if log == -1: continue
        print(log, log_template['EventTemplate'][log-1])