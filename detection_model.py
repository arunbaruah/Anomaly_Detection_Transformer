#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import torch
import importlib
import re
import elasticsearch
from elasticsearch_dsl.search import Search

import Spell
import Transformer as tnsf

import importlib
importlib.reload(Spell)

def parse(log_source = '', machine = '', start_date = '', start_time = '', end_date = '', end_time = '',):  
    input_dir  = 'Dataset/'  # The input directory of log file
    output_dir = 'Spell_result/'  # The output directory of parsing results
    tau        = 0.6  # Message type threshold (default: 0.5)
    regex      = [r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
                  r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])', # Numbers
                ]  # Regular expression list for optional preprocessing (default: [])
    log_format = '<Month> <Day> <Time> <Machine> <Level>: <Content>'  # ES log format
    #log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format

    #Parse from Elastic Search
    if re.match(r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', log_source):
        if not machine:
            print("Please specify a machine.")
            return

        es = elasticsearch.Elasticsearch([log_source])
        res = Search(using=es, index="updated-fb*").\
                filter('range',**{"@timestamp":{'gte': start_date+'T'+start_time+':00.000Z', 'lt' : end_date+'T'+end_time+':59.999Z'}})
        response = res.execute()
        print("Filtered logs: %i" %res.count()) #There is an error on the number of matched logs
        print("Total number of logs: %i" %response.hits.total)
        
        myregex = r"\b(?=\w)" + re.escape('bigdata-vm-'+machine) + r"\b(?!\w)"
        compiled_regex = re.compile(myregex, re.IGNORECASE)
        log_file = [hit.message for hit in res.scan() if compiled_regex.search(hit.message)]
    #Parse from file
    else:
        log_file   = log_source
    
    if not log_file:
        print("No logs found for this machine during specified time.")
        return
    
    parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parser.parse(log_file)

def train(filename = ''):
    output_dir = 'Spell_result/'  # The output directory of parsing results
    if filename:
        events_template = pd.read_csv(output_dir + filename + '_templates.csv') 
        log_keys = list(map(lambda n: n, map(int, open("Spell_result/np.txt").readline().split())))
        #log_keys = "hdfs_train"
    else:
        events_template = pd.read_csv(output_dir + 'logs_templates.csv') 
        log_keys = list(map(lambda n: n, map(int, open("Spell_result/np.txt").readline().split())))

    VOCAB_SIZE, _ = events_template.shape
    criterion = tnsf.LabelSmoothing(size = VOCAB_SIZE, padding_idx=0, smoothing=0.0)
    model = tnsf.make_model(VOCAB_SIZE, VOCAB_SIZE, N=2)
    model_opt = tnsf.NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        WINDOW_SIZE = 10
        model.train()
        tnsf.run_epoch(tnsf.data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 30), model, tnsf.SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(tnsf.run_epoch(tnsf.data_gen(log_keys, VOCAB_SIZE, WINDOW_SIZE, 10, 5), model, tnsf.SimpleLossCompute(model.generator, criterion, None)))
        
    torch.save(model.state_dict(), "Model/model.pt")
    return model