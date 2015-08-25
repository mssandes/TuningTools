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
    jobWindowsList = LoopingBoundsCollection(
                        MatlabLoopingBounds(jobTuple[0], 
                                            varIncr.incr(), 
                                            jobTuple[-1]) \
                        for jobTuple in varBounds.window( varWindow ))
    return jobWindowsList


  def __call__(self, **kw):
    """
      Create a collection of tuning job configuration files at the output
      folder.
    """
    from FastNetTool.FileIO import save, checkForUnusedVars, mkdir_p

    # Cross validation configuration
    outputFolder = kw.pop('outputFolder',       'jobConfig'          )
    sortBounds   = kw.pop('sortBounds',   PythonLoopingBounds(50)    )
    nInits       = kw.pop('nInits',                100               )
    neuronBounds = kw.pop('neuronBounds', SeqLoopingBounds(5, 20)    )
    # Output configuration
    nNeuronsPerJob = kw.pop('nNeuronsPerJob',         1              )
    nSortsPerJob   = kw.pop('nSortsPerJob',           1              )
    nInitsPerJob   = kw.pop('nInitsPerJob',          100             )
    if 'level' in kw: self.level = kw.pop('level')
    # and delete it to avoid mistakes:
    checkForUnusedVars( kw, self._logger.warning )
    del kw

    # Make sure that bounds variables are LoopingBounds objects:
    from FastNetTool.TuningJob import TuningJob
    neuronBoundsCol = TuningJob.fixLoopingBoundsCol( neuronBoundsCol,
                                                     SeqLoopingBounds )
    sortBoundsCol   = TuningJob.fixLoopingBoundsCol( sortBoundsCol,
                                                     PythonLoopingBounds )
    if nInits < 1:
      raise ValueError(("Cannot require zero or negative initialization "
          "number."))

    # Do some checking in the arguments:
    nSorts = len(sortBounds)
    if not nSorts:
      raise RuntimeError("Sort bounds is empty.")
    if nInitsPerJob > nSorts:
      self._logger.warning(("The number of sorts per job (%d) is "
        "greater then the total number of sorts (%d), changing it "
        "into the maximum possible value."))
      nInitsPerJob = nSorts
    if not len(neuronBounds):
      raise RuntimeError("Neuron bounds is empty.")

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
          self._logger.info(('Retrieved following job configuration "
              "(bounds.vec) : "
              "[ neuronBounds=%r, sortBounds=%r, initBounds=%-4d]'),
              neuronWindowBounds.vec(), 
              sortWindowBounds.vec(), 
              initWindowBounds.vec())
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
          savedFile = save( objSave, jobName )
          self._logger.info('Saved job option configuration with name: %s',
                            savedFile )

createTuningJobFiles = CreateTuningJobFiles()

