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


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        
    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode = 'replicate')
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x
    
    
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

        
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        
    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        prediction = x
        prediction  = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence)
        return prediction
    
    
def create_modules(blocks):
    # store the information about input, pre-processing, etc.
    net_info = blocks[0]
    
    module_list = nn.ModuleList()
    
    # indexing blocks hellps with implementing route layers (skip connections)
    index = 0
    
    prev_filters = 3
    output_filters = []
    
    for x in blocks:
        module = nn.Sequential()
        
        if (x['type'] == 'net'):
            continue
            
        # if Conv Layer
        if (x['type'] == 'convolutional'):
            # get info about the layer
            activation = x['activation']
            try :
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except :
                batch_normalize = 0
                bias = True
                
            filters = int(x['filters'])
            padding = int(x['padding'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            
            if padding :
                pad = (kernel_size - 1) // 2
            else : 
                pad = 0
                
            # convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module('conv_{0}'.format(index), conv)
            
            # batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)
                
            # activation
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module('leaky_{0}'.format(index), activn)
                
        # upsampling layer
        elif (x['type'] == 'upsample'):
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
            module.add_module('upsample_{}'.format(index), upsample)
            
        # route layer
        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')
            
            # start of route 
            start = int(x['layers'][0])
            
            # end, if it exists
            try :
                end = int(x['layers'][1])
            except :
                end = 0
                
            if start > 0 :
                start = start - index
            if end > 0 :
                end = end - index
                
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)
            
            if end < 0 :
                filters = output_filters[index + start] + output_filters[index + end]
            else :
                filters = output_filters[index + start]
                
        # shortcut (skip connections)
        elif (x['type'] == 'shortcut'):
            from_ = int(x['from'])
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)
            
        elif (x['type'] == 'maxpool':
              stride = int(x['stride'])
              size = int(x['size'])
              
              if stride != 1 :
                  maxpool = nn.MaxPool2d(size, stride)
              else :
                  maxpool = MaxPoolStride1(size)
              
              module.add_module('maxpool_{}'.format(index), maxpool)
              
          # YOLO (detection layer)
          elif (x['type'] == 'yolo'):
              mask = x['mask'].split(',')
              mask = [int(x) for x in mask]
              
              anchors = x['anchors'].split(',')
              anchors = [int(a) for a in anchors]
              anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
              anchors = [anchors[i] for i in mask]
              
              detection = DetectionLayer(anchors)
              module.add_module('Detection_{}'.format(index), detection)
              
          else :
              print('Something I dont know')
              assert False
              
          module_list.append(module)
          prev_filters = filters
          output_filters.append(filters)
          index += 1
          
      return (net_info, module_list)