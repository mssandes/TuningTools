#!/usr/bin/env python

import subprocess as sp, json, pprint, pickle, requests, tempfile, os
from RingerCore import ArgumentParser, loggerParser
from RingerCore import emptyArgumentsPrintHelp, Logger, printArgs

parentParser = ArgumentParser(add_help = False)
parentReqParser = parentParser.add_argument_group("required arguments", '')
parentReqParser.add_argument('-u','--url', required = True, metavar='URL',
    action='store', nargs='+',
    help = "Bigpanda urls to retrieve the job configuration.")

parser = ArgumentParser(description = 'Retrieve configuration from panda job and create a file to run the job locally',
    parents = [parentParser, loggerParser], conflict_handler = 'resolve')
parser.make_adjustments()
emptyArgumentsPrintHelp(parser)

# Retrieve parser args:
args = parser.parse_args()

mainLogger = Logger.getModuleLogger( __name__, args.output_level )
mainLogger.write = mainLogger.info
printArgs( args, mainLogger.debug )

c = tempfile.mkstemp()
with open(c[1],'rw') as f:
  # TODO Check if output was retrieved correctly:
  if sp.call(['cern-get-sso-cookie', '-u', 'https://bigpanda.cern.ch/', '-o', c[1]]):
    mainLogger.fatal("Could not get cookie")
  for url in args.url:
    r = sp.check_output(['curl', '-b', c[1], '-H', 'Accept: application/json', '-H Content-Type: application/json', url])
    d = json.loads(r)
    taskparams = d['taskparams']
    jobParams = taskparams['jobParameters']
    job = ''
    job += 'setAthena "' + taskparams['transHome'].split('-')[-1] + '";\n'
    job += taskparams['transPath'] + ' \\' + '\n'
    job += '--maxEvents=10 \\\n'
    for o in jobParams:
      if o[u'type'] == u'constant':
        job += o[u'value'] + ' \\' + '\n'
      elif o[u'type'] == u'template':
        if o['param_type'] in ('input','output'):
          job += o[u'value'].split('=')[0] + '=' + o[u'dataset'] + ' \\' + '\n'
        #else:
        #  job += o[u'value'].split('=')[0] + ' \\' + '\n'
      else:
        mainLogger.warning("Could not recognize param: %r", o)
    job = job[:-2]
    print job
    jobPath = 'job_' + url.split('/')[-2] + '.sh'
    with open(jobPath, 'w') as o:
      o.write(job)
      mainLogger.info("Created file: %s", jobPath )
