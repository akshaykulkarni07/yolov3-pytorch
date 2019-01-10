from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

# Parses configuration file to get the network
def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    # store the lines in a list
    lines = file.read().split('\n')
    # remove any empty lines or comments
    lines = [x for x in lines if (len(x) > 0 and x[0] != '#')]
    lines = [x.rstrip().lstrip() for x in lines]
    
    # empty dictionary
    block = {}
    # empty list
    blocks = []
    
    for line in lines:
        # '[' marks the start of a new block
        if line[0] == '[':
            # if any block is already open and content was
            # added to it previously, it needs to be closed
            # and appended to the list of blocks, since now 
            # new block will be starting
            if len(block) != 0:
                blocks.append(block)
                # make the block empty again so it can be used for new block
                block = {}
            block['type'] = line[1 : -1].rstrip()
        else :
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks