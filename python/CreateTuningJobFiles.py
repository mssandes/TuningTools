from FastNetTool.Logger import Logger
from FastNetTool.LoopingBounds import *

class CreateTuningJobFiles(Logger):
  """
    An instance of this class can be used to create all the tuning job
    needed files but the data files, which should be created with CreateData
    instead.
  """

  def __init__( self, logger = None ):
    Logger.__init__( self, logger = logger )

  @classmethod
  def _retrieveJobLoopingBoundsCol( cls, varBounds, varWindow ):
    """
      Create window bounded variables from larger range.
    """
    varIncr = varBounds.incr()
    jobWindowList = LoopingBoundsCollection()
    for jobTuple in varBounds.window( varWindow ):
      if len(jobTuple) == 1:
        jobWindowList += MatlabLoopingBounds(jobTuple[0], jobTuple[0])
      elif len(jobTuple) == 0:
        raise RuntimeError("Retrieved empty window.")
      else:
        jobWindowList += MatlabLoopingBounds(jobTuple[0], 
                                             varIncr, 
                                             jobTuple[-1])
    return jobWindowList


  def __call__(self, **kw):
    """
      Create a collection of tuning job configuration files at the output
      folder.
    """
    from FastNetTool.FileIO import save
    from FastNetTool.util   import checkForUnusedVars, mkdir_p

    # Cross validation configuration
    outputFolder = kw.pop('outputFolder',       'jobConfig'        )
    neuronBounds = kw.pop('neuronBounds', SeqLoopingBounds(5, 20)  )
    sortBounds   = kw.pop('sortBounds',   PythonLoopingBounds(50)  )
    nInits       = kw.pop('nInits',                100             )
    # Output configuration
    nNeuronsPerJob = kw.pop('nNeuronsPerJob',         1            )
    nSortsPerJob   = kw.pop('nSortsPerJob',           1            )
    nInitsPerJob   = kw.pop('nInitsPerJob',          100           )
    if 'level' in kw: self.level = kw.pop('level')
    # Make sure that bounds variables are LoopingBounds objects:
    if not isinstance( neuronBounds, SeqLoopingBounds ):
      neuronBounds = SeqLoopingBounds(neuronBounds)
    if not isinstance( sortBounds, SeqLoopingBounds ):
      sortBounds   = PythonLoopingBounds(sortBounds)
    # and delete it to avoid mistakes:
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    if nInits < 1:
      raise ValueError(("Cannot require zero or negative initialization "
          "number."))

    # Do some checking in the arguments:
    nNeurons = len(neuronBounds)
    nSorts = len(sortBounds)
    if not nSorts:
      raise RuntimeError("Sort bounds is empty.")
    if nNeuronsPerJob > nNeurons:
      self._logger.warning(("The number of neurons per job (%d) is "
        "greater then the total number of neurons (%d), changing it "
        "into the maximum possible value."), nNeuronsPerJob, nNeurons )
      nNeuronsPerJob = nNeurons
    if nSortsPerJob > nSorts:
      self._logger.warning(("The number of sorts per job (%d) is "
        "greater then the total number of sorts (%d), changing it "
        "into the maximum possible value."), nSortsPerJob, nSorts )
      nSortsPerJob = nSorts

    # Create the output folder:
    mkdir_p(outputFolder)

    # Create the windows in which each job will loop upon:
    neuronJobsWindowList = \
        CreateTuningJobFiles._retrieveJobLoopingBoundsCol( neuronBounds, 
                                                           nNeuronsPerJob )
    sortJobsWindowList = \
        CreateTuningJobFiles._retrieveJobLoopingBoundsCol( sortBounds, 
                                                           nSortsPerJob )
    initJobsWindowList = \
        CreateTuningJobFiles._retrieveJobLoopingBoundsCol( \
          PythonLoopingBounds( nInits ), \
          nInitsPerJob )

    # Loop over windows and create the job configuration
    for neuronWindowBounds in neuronJobsWindowList():
      for sortWindowBounds in sortJobsWindowList():
        for initWindowBounds in initJobsWindowList():
          self._logger.debug(('Retrieved following job configuration '
              '(bounds.vec) : '
              '[ neuronBounds=%s, sortBounds=%s, initBounds=%s]'),
              neuronWindowBounds.formattedString('hn'), 
              sortWindowBounds.formattedString('s'), 
              initWindowBounds.formattedString('i'))
          fulloutput = '{outputFolder}/job.{neuronStr}.{sortStr}.{initStr}'.format( 
                        outputFolder = outputFolder, 
                        neuronStr = neuronWindowBounds.formattedString('hn'), 
                        sortStr = sortWindowBounds.formattedString('s'),
                        initStr = initWindowBounds.formattedString('i') )
          objSave = {'version': 1,
                     'type' : 'ConfJobFile',
                     'neuronBounds' : [neuronWindowBounds.lowerBound(), 
                                       nNeuronsPerJob,
                                       neuronWindowBounds.upperBound()], 
                     'sortBounds' : [sortWindowBounds.lowerBound(),
                                     nSortsPerJob, 
                                     sortWindowBounds.upperBound()], 
                     'initBounds' : [initWindowBounds.lowerBound(),
                                     initWindowBounds.upperBound()]}
          savedFile = save( objSave, fulloutput )
          self._logger.info('Saved job option configuration at path: %s',
                            savedFile )

createTuningJobFiles = CreateTuningJobFiles()

