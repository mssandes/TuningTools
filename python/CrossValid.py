__all__ = ['CrossValidArchieve', 'CrossValid']

import numpy as np
from itertools import chain, combinations
from RingerCore.Logger import Logger
from RingerCore.util import checkForUnusedVars
from RingerCore.FileIO import save, load
from TuningTools.coreDef import retrieve_npConstants
npCurrent, _ = retrieve_npConstants()

class CrossValidArchieve( Logger ):
  """
  Context manager for Cross-Validation archives

  Version 2: Saving raw dict and rebuilding object from it when loading.
  Version 1: Renamed module to TunningTools, still saving original object.
  Version 0: Module was still called FastNetTool
  """

  _type = 'CrossValidFile'
  _version = 2
  _crossValid = None

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it

    with CrosValidArchieve("/path/to/file") as data:
      BLOCK

    CrosValidArchieve( "file/path", crossValid = CrossValid() )
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self.crossValid = kw.pop( 'crossValid', None )
    checkForUnusedVars( kw, self._logger.warning )

  @property
  def filePath( self ):
    return self._filePath

  @filePath.setter
  def filePath( self, val ):
    self._filePath = val

  @property
  def crossValid( self ):
    return self._crossValid

  @crossValid.setter
  def crossValid( self, val ):
    if not val is None and not isinstance(val, CrossValid):
      raise ValueType("Attempted to set crossValid to an object not of CrossValid type.")
    else:
      self._crossValid = val

  def getData( self ):
    if not self._crossValid:
       raise RuntimeError("Attempted to retrieve empty data from CrossValidArchieve.")
    return {'type' : self._type,
            'version' : self._version,
            'crossValid' : self._crossValid.toRawObj() }

  def save(self, compress = True):
    return save( self.getData(), self._filePath, compress = compress )

  def __enter__(self):
    from cPickle import PickleError
    try:
      crossValidInfo   = load( self._filePath )
    except PickleError:
      # It failed without renaming the module, retry renaming old module
      # structure to new one.
      import sys
      sys.modules['FastNetTool.CrossValid'] = sys.modules[__name__]
      crossValidInfo   = load( self._filePath )
    # Open crossValidFile:
    try: 
      if isinstance(crossValidInfo, dict):
        if crossValidInfo['type'] != 'CrossValidFile':
          raise RuntimeError(("Input crossValid file is not from PreProcFile " 
              "type."))
        if crossValidInfo['version'] == 2:
          crossValid = CrossValid.fromRawObj( crossValidInfo['crossValid'] )
        elif crossValidInfo['version'] == 1:
          crossValid = crossValidInfo['crossValid']
        else:
          raise RuntimeError("Unknown job configuration version.")
      elif type(crossValidInfo) == list: # Read legacy files
        crossValid = crossValidInfo[3]
      else:
        raise RuntimeError("Invalid CrossValidFile contents.")
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read cross validation file file '%s': Reason:"
          "\n\t %s") % (self._filePath, e))
    if not isinstance(crossValid, CrossValid ):
      raise ValueError(("crossValidFile \"%s\" doesnt contain a CrossValid " \
          "object!") % self._filePath)
    return crossValid
    
  def __exit__(self, exc_type, exc_value, traceback):
    # Remove bound
    self.crossValid = None 

def combinations_taken_by_multiple_groups(seq, parts, indexes=None, res=[], cur=0):
  """
    Take combinations from seq using part separations into groups.

    Taken from: http://stackoverflow.com/a/16331286/1162884
  """
  if indexes is None: # indexes to use for combinations
    indexes = range(len(seq))

  if cur >= len(parts): # base case
    yield [[seq[i] for i in g] for g in res]
    return    

  for x in combinations(indexes, r=parts[cur]):
    set_x = set(x)
    new_indexes = [i for i in indexes if i not in set_x]
    for comb in combinations_taken_by_multiple_groups(seq, 
                                                      parts, 
                                                      new_indexes, 
                                                      res = res + [x], 
                                                      cur = cur + 1):
      yield comb

class CrossValid (Logger):
  """
    CrossValid is used to sort and randomize the dataset for training step.  
  """

  # There is only need to change version if a property is added
  _version = 1

  def __init__(self, **kw ):
    Logger.__init__( self, kw  )
    from RingerCore.util import printArgs
    printArgs( kw, self._logger.debug )
    self._nSorts = kw.pop('nSorts', 50)
    self._nBoxes = kw.pop('nBoxes', 10)
    self._nTrain = kw.pop('nTrain', 6 )
    self._nValid = kw.pop('nValid', 4 )
    self._nTest  = kw.pop('nTest',  self._nBoxes - ( self._nTrain + self._nValid ) )
    self._seed   = kw.pop('seed',   None )
    checkForUnusedVars( kw, self._logger.warning )

    # Check if variables are ok:
    if (not self._nTest is None) and self._nTest < 0:
      raise ValueError("Number of test clusters is lesser than zero")
    totalSum = self._nTrain + self._nValid + (self._nTest) if self._nTest else \
               self._nTrain + self._nValid
    if totalSum != self._nBoxes:
      raise ValueError("Sum of train, validation and test boxes doesn't match.")

    np.random.seed(self._seed)

    # Test number of possible combinations (N!/((N-K)!(K)!) is greater
    # than the required sorts. If number of sorts (greater than half of the
    # possible cases) is close to the number of combinations, generate all
    # possible combinations and then gather the number of needed sorts.
    # However, as calculating factorial can be heavy, we don't do this if the
    # number of boxes is large.
    self._sort_boxes_list = []
    useRandomCreation = True
    from math import factorial
    if self._nBoxes < 201:
      totalPossibilities = ( factorial( self._nBoxes ) ) / \
          ( factorial( self._nTrain ) * \
            factorial( self._nValid ) * \
            factorial( self._nTest  ) )
      if self._nSorts > (totalPossibilities / 2):
        useRandomCreation = False
    if useRandomCreation:
      count = 0
      while True:
        random_boxes = np.random.permutation(self._nBoxes)
        random_boxes = tuple(chain(sorted(random_boxes[0:self._nTrain]),
                        sorted(random_boxes[self._nTrain:self._nTrain+self._nValid]),
                        sorted(random_boxes[self._nTrain+self._nValid:])))
        # Make sure we are not appending same sort again:
        if not random_boxes in self._sort_boxes_list:
          self._sort_boxes_list.append( random_boxes )
          count += 1
          if count == self._nSorts:
            break
    else:
      combinations = list(
          combinations_taken_by_multiple_groups(range(self._nBoxes),
                                                (self._nTrain, 
                                                 self._nVal, 
                                                 self._nTest)))
      # Pop from our list the not needed values:
      for i in range(totalPossibilities - self._nSorts):
        combinations.pop( np.random_integers(0, totalPossibilities) )
  # __init__ end

  def nSorts(self):
    """
      Retrieve number of sorts done for this instance.
    """
    return self._nSorts

  def nTrain(self):
    "Number of training boxes"
    return self._nTrain

  def nValid(self):
    "Number of validation boxes"
    return self._nVal

  def nTest(self):
    "Number of test boxes"
    return self._nTest

  def __call__(self, data, sort):
    """
      Split data into train/val/test datasets using sort index.
    """

    sort_boxes = self._sort_boxes_list[sort]

    trainData  = []
    valData    = []
    testData   = []

    for cl in data:
      # Retrieve the number of events in this class:
      evts = cl.shape[ npCurrent.odim ]
      if evts < self._nBoxes:
        raise RuntimeError("Too few events for dividing data.")
      # Calculate the remainder when we do equal splits in nBoxes:
      remainder = evts % self._nBoxes
      # Take the last events which will not be allocated to any class during
      # np.split
      evts_remainder = cl[ npCurrent.access( pidx=':', oidx=slice(evts-remainder, None) ) ]
      # And the equally divisible part of the class:
      cl = cl[ npCurrent.access( pidx=':', oidx=slice(0,evts-remainder) ) ]
      # Split it
      cl = np.split(cl, self._nBoxes, axis=npCurrent.odim )

      # Now we allocate the remaining events in each one of the nth first
      # class, where n is the remainder size
      for idx in range(remainder):
        evts_remainder[ npCurrent.access( pidx=np.newaxis, oidx=idx ) ].shape
        cl[idx] = np.append(cl[idx], evts_remainder[ npCurrent.access( pidx=':', oidx=slice(idx,idx+1) ) ], axis = npCurrent.odim )
        
      # With our data split in nBoxes for this class, concatenate them into the
      # train, validation and test datasets
      trainData.append( np.concatenate( [cl[trnBoxes] for trnBoxes in self.getTrnBoxIdxs(sort)], axis = npCurrent.odim ) )
      valData.append(   np.concatenate( [cl[valBoxes] for valBoxes in self.getValBoxIdxs(sort)], axis = npCurrent.odim ) )
      if self._nTest:
        testData.append(np.concatenate( [cl[tstBoxes] for tstBoxes in self.getTstBoxIdxs(sort)], axis = npCurrent.odim ) )

    self._logger.info('Train      #Events/class: %r', 
                      [cTrnData.shape[npCurrent.odim] for cTrnData in trainData])
    self._logger.info('Validation #Events/class: %r', 
                      [cValData.shape[npCurrent.odim] for cValData in valData])
    if self._nTest:  
      self._logger.info('Test #Events/class: %r', 
                        [cTstData.shape[npCurrent.odim] for cTstData in testData])


     #default format
    return trainData, valData, testData
  # __call__ end

  def getBoxIdxs(self, ds, sort):
    """
    Retrieve boxes for the input datasets and for a sort index
    """
    from TuningTools.FilterEvents import Dataset
    if ds is Dataset.Train:
      return self.getTrnBoxIdxs(sort)
    elif ds is Dataset.Validation:
      return self.getValBoxIdxs(sort)
    elif ds is Dataset.Test:
      return self.getTstBoxIdxs(sort)
    elif ds is Dataset.Operation:
      return True
    else:
      return False

  def getTrnBoxIdxs(self, sort):
    """
    Retrieve training box indexes for a sort index
    """
    sort_boxes = self._sort_boxes_list[sort]
    return sort_boxes[:self._nTrain]

  def getValBoxIdxs(self, sort):
    """
    Retrieve valdation box indexes for a sort index
    """
    sort_boxes = self._sort_boxes_list[sort]
    return sort_boxes[self._nTrain:self._nTrain+self._nValid]

  def getTstBoxIdxs(self, sort):
    """
    Retrieve test box indexes for a sort index
    """
    sort_boxes = self._sort_boxes_list[sort]
    return sort_boxes[self._nTrain+self._nValid:]

  def isWithin(self, ds, sort, idx, maxEvts):
    """
    Check if index is within input dataset.
    """
    from TuningTools.FilterEvents import Dataset
    if ds is Dataset.Train:
      return self.isWithinTrn(sort,idx,maxEvts)
    elif ds is Dataset.Validation:
      return self.isWithinTest(sort,idx,maxEvts)
    elif ds is Dataset.Test:
      return self.isWithinTest(sort,idx,maxEvts)
    elif ds is Dataset.Operation:
      return True
    else:
      return False

  def isWithinTrain(self, sort, idx, maxEvts):
    """
    Check if index is within training dataset.
    """
    for box in self.getTrnBoxIdxs(sort):
      startPos, endPos = self.getBoxPosition(sort, maxEvts=maxEvts)
      if idx >= startPos and idx < endPos:
        return True
    return False

  def isWithinValid(self, sort, idx, maxEvts):
    """
    Check if index is within validation dataset.
    """
    for box in self.getValBoxIdxs(sort):
      startPos, endPos = self.getBoxPosition(sort, maxEvts=maxEvts)
      if idx >= startPos and idx < endPos:
        return True
    return False

  def isWithinTest(self, sort, idx, maxEvts):
    """
    Check if index is within test dataset.
    """
    for box in self.getTstBoxIdxs(sort):
      startPos, endPos = self.getBoxPosition(sort, maxEvts=maxEvts)
      if idx >= startPos and idx < endPos:
        return True
    return False

  def whichDS(self, sort, idx, maxEvts):
    """
    Return a TuningTools.CrossValidStat object determinig which dataset the
    index is contained.
    """
    from TuningTools.FilterEvents import Dataset
    if self.isWithinTrain(sort, idx, maxEvts):
      return Dataset.Train
    elif self.isWithinValid(sort, idx, maxEvts):
      return Dataset.Validation
    else:
      if not self.isWithinTest(sort, idx, maxEvts):
        raise RuntimeError("This event is not in any dataset!")
      return Dataset.Test

  def getBoxPosition(self, sort, boxIdx, *sets, **kw):
    """
      Returns start and end position from a box index in continuous data
      representation merged after repositioning a split into equally distributed
      sets into this cross validation object number of boxes.

      startPos, endPos = crossVal.getBoxPosition( sort, boxIdx, maxEvts=maxEvts )

      If you also want to retrieve the index with respect to the divided set,
      then inform the sets as the *args argument list, in this case, it will
      return the instance where the box is in. It will also treat the
      remainders which were added to the sets!

      startPos, endPos, set, cStartPos, cEndPos = \
          crossVal.getBoxPosition( sort, boxIdx, trnData,
                                   valData[, tstData=None, 
                                             evtsPerBox = None,
                                             remainder = None,
                                             maxEvts = None])
      
    """
    evtsPerBox = kw.pop( 'evtsPerBox', None )
    remainder = kw.pop( 'remainder', None )
    maxEvts = kw.pop( 'maxEvts', None )
    takeFrom = None
    from math import floor
    # Check parameters
    if maxEvts is not None: 
      if maxEvts < 0:
        raise TypeError("Number of events must be postitive")
      if evtsPerBox is not None or remainder is not None:
        raise ValueError("Cannot set remainder or evtsPerBox when maxEvts is set.")
      evtsPerBox = floor( maxEvts / self._nBoxes)
      remainder = maxEvts % self._nBoxes
    # The sorted boxes:
    sort_boxes = self._sort_boxes_list[sort]
    # The index where this box is in the sorts:
    box_pos_in_sort = sort_boxes.index(boxIdx)
    # Retrieve evtsPerBox if it was not input:
    if evtsPerBox is None:
      if not sets:
        raise TypeError(("It is needed to inform the sets or the number of "
            "events per box"))
      # Retrieve total number of events:
      evts = cTrnData.shape[npCurrent.odim] \
           + cValData.shape[npCurrent.odim] \
           + (cTstData.shape[npCurrent.odim] if cTstData.size else 0)
      # The number of events in each splitted box:
      evtsPerBox = floor( evts / self._nBoxes )
    if sets:
      # Calculate the remainder when we do equal splits in nBoxes:
      if remainder is None:
        remainder = evts % self._nBoxes
    # The position where the box start and end
    startPos = box_pos_in_sort * evtsPerBox 
    endPos = startPos + evtsPerBox
    # Discover which data from which we will take this box:
    if remainder:
      # Retrieve the number of boxes which were increased by the remainder:
      increaseSize = sum( [box < remainder for box in sort_boxes[:box_pos_in_sort] ] )
      # The start position and end position of the current box:
      startPos += increaseSize
      endPos += increaseSize + (1 if boxIdx < remainder and not sets else 0)
    if sets:
      # Finally, check from which set should we take this box:
      takeFrom = sets[0]
      if box_pos_in_sort >= self._nTrain + self._nValid:
        if len(sets) > 2:
          takeFrom = sets[2]
          # We must remove the size from the train and validation dataset:
          startPos -= sets[0].shape[npCurrent.odim] + sets[1].shape[npCurrent.odim]
          endPos   -= sets[0].shape[npCurrent.odim] + sets[1].shape[npCurrent.odim]
        else:
          raise RuntimeError(("Test dataset was not given as an input, but it "
            "seems that the current box is at the test dataset."))
      elif box_pos_in_sort >= self._nTrain:
        if len(sets) > 1:
          takeFrom = sets[1]
          # We must remove the size from the train dataset:
          startPos -= sets[0].shape[npCurrent.odim]
          endPos   -= sets[0].shape[npCurrent.odim]
        else:
          raise RuntimeError(("Validation dataset was not given as an input, "
            "but it seems that the current box is at the validation dataset."))
    if not takeFrom is None:
      return startPos, endPos, takeFrom
    else:
      return startPos, endPos
  # getBoxPosition end


  def revert(self, trnData, valData, tstData=None, **kw):
    """
      Revert sort using the training, validation and testing datasets.

      data = cross.revert( trnData, valData[, tstData=None], sort = sortValue)
    """
    from math import floor

    try:
      sort = kw.pop('sort')
    except:
      TypeError('Needed argument "sort" not specified')

    data = []

    if not tstData:
      tstData = [npCurrent.fp_array([]) for i in range(len(trnData))]
    for cTrnData, cValData, cTstData in zip(trnData, valData, tstData):
      # Retrieve total number of events:
      evts = cTrnData.shape[npCurrent.odim] \
           + cValData.shape[npCurrent.odim] \
           + (cTstData.shape[npCurrent.odim] if cTstData.size else 0)
      # Allocate the numpy array to hold 
      cData = npCurrent.fp_zeros(
                                 shape=npCurrent.shape(npat=cTrnData.shape[npCurrent.pdim], 
                                                       nobs=evts)
                                )
      # Calculate the remainder when we do equal splits in nBoxes:
      remainder = evts % self._nBoxes
      # The number of events in each splitted box:
      evtsPerBox = floor( evts / self._nBoxes )
      # Create a holder for the remainder events, which must be in the end of the
      # data array
      remainderData = npCurrent.fp_zeros( shape=npCurrent.shape( npat=cTrnData.shape[npCurrent.pdim],nobs=remainder ) )
      for boxIdx in range(self._nBoxes):
        # Get the indexes where we will put our data in cData:
        cStartPos = boxIdx * evtsPerBox 
        cEndPos = cStartPos + evtsPerBox
        # And get the indexes and dataset where we will copy the values from:
        startPos, endPos, ds = self.getBoxPosition( sort, 
                                                    boxIdx,
                                                    cTrnData, 
                                                    cValData, 
                                                    cTstData,
                                                    evtsPerBox = evtsPerBox,
                                                    remainder = remainder )
        # Copy this box values to data:
        cData[ npCurrent.access( pidx=':', oidx=slice(cStartPos,cEndPos) ) ] = ds[ npCurrent.access( pidx=':', oidx=slice(startPos,endPos) ) ]
        # We also want to copy this box remainder if it exists to the remainder
        # data:
        if boxIdx < remainder:
          # Take the row added to the end of dataset:
          remainderData[ npCurrent.access( pidx=':', oidx=boxIdx) ] = ds[ npCurrent.access( pidx=':', oidx=endPos ) ]
      # We finished looping over the boxes, now we copy the remainder data to
      # the last positions of our original data np.array:
      if remainder:
        cData[ npCurrent.access( pidx=':', oidx=slice(evtsPerBox * self._nBoxes, None) ) ] = remainderData
      # Finally, append the numpy array holding this class information to the
      # data list:
      data.append(cData)
    return data
  # revert end

  def __str__(self):
    """
      String representation of the object.
    """
    string = ""
    for i, sort in enumerate(self._sort_boxes_list):
      string += "%-10s:{Train:%s|Valid:%s%s}" % ( "Sort%d" % i,
          sort[0:self._nTrain],
          sort[self._nTrain:self._nTrain+self._nValid],
          "|Tst:%s" % sort[(self._nTrain+self._nValid):] if self._nTest else "")
      if i != self._nSorts-1:
        string+='\n'
    return string

  def toRawObj(self):
    "Return a raw dict object from itself"
    from copy import copy # Every complicated object shall be changed to a rawCopyObj
    raw = copy(self.__dict__)
    raw['version'] = self.__class__._version
    raw.pop('_logger') # remove logger
    return raw

  def buildFromDict(self, d):
    if d.pop('version') == self.__class__._version:
      for k, val in d.iteritems():
        self.__dict__[k] = d[k]
    self._logger = Logger.getModuleLogger(self.__class__.__name__, self._level )
    return self

  @classmethod
  def fromRawObj(cls, obj):
    from copy import copy
    obj = copy(obj)
    self = cls().buildFromDict(obj)
    return self
