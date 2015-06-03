#!/usr/bin/python
import sys, os, time, glob, re
import argparse
import ROOT
import sys
import os
import pickle
from CrossValid import *
from util       import *

def include(filename):
  if os.path.exists(filename): 
    execfile(filename)

#Parse command line step
parser = argparse.ArgumentParser()
parser.add_argument('--input',  action='store', default="/afs/cern.ch/user/j/jodafons/public/valid_ringer_sample.pic")
parser.add_argument('--output', action='store', default="output.save")
parser.add_argument('--neurons', action='store', default="2")
parser.add_argument('--inits', action='store', default="10")
parser.add_argument('--outputlevel', action='store', default="2")
parser.add_argument('--sortMin', action='store', default="1")
parser.add_argument('--sortMax', action='store', default="1")
args=parser.parse_args()

print 'openning data and normalize...'

DatasetLocationInput              = args.input
objectsFromFile                   = load( DatasetLocationInput )
Data                              = normalizeSumRow( objectsFromFile[0] )
Target                            = objectsFromFile[1]
CrossValidObject                  = objectsFromFile[2]
OutputName                        = args.output
MonitoringLevel                   = int(args.outputlevel)
NumberOfInitsPerSort              = int(args.inits)
NumberOfSortsPerConfigurationMin  = int(args.sortMin)
NumberOfSortsPerConfigurationMax  = int(args.sortMax)
NumberOfNeuronsInHiddenLayerMin   = int(args.neurons)
NumberOfNeuronsInHiddenLayerMax   = int(args.neurons)
DoMultiStops                      = True
ShowEvolution                     = 4
Epochs                            = 1000

#Fastnet loop top option
include('../python/jobs/NeuralTrainingLoop_fastnet_topOption.py')



