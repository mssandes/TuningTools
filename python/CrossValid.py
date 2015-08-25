import numpy as np
from itertools import chain, combinations
from FastNetTool.Logger import Logger

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

  def __init__(self, **kw ):
    Logger.__init__( self, kw  )
    from FastNetTool.util import printArgs
    printArgs( kw, self._logger.debug )
    self._nSorts = kw.pop('nSorts', 50)
    self._nBoxes = kw.pop('nBoxes', 10)
    self._nTrain = kw.pop('nTrain', 6 )
    self._nValid = kw.pop('nValid', 4 )
    self._nTest  = kw.pop('nTest',  self._nBoxes - ( self._nTrain + self._nValid ) )
    self._seed   = kw.pop('seed',   None )
    from FastNetTool.util import checkForUnusedVars
    checkForUnusedVars( kw, self._logger.warning )

    # Check if variables are ok:
    if self._nTest and self._nTest < 0:
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
      evts = cl.shape[0]
      # Calculate the remainder when we do equal splits in nBoxes:
      remainder = evts % self._nBoxes
      # Take the last events which will not be allocated to any class during
      # np.split
      evts_remainder = cl[evts-remainder:]
      # And the equally divisible part of the class:
      cl = cl[0:evts-remainder]
      # Split it
      cl = np.split(cl, self._nBoxes)

      # Now we allocate the remaining events in each one of the nth first
      # class, we n is the remainder size
      for idx in range(remainder):
        cl[idx] = np.append(cl[idx], evts_remainder[idx, np.newaxis], axis = 0)

      # With our data split in nBoxes for this class, concatenate them into the
      # train, validation and test datasets
      trainData.append( np.concatenate( [cl[trnBoxes] for trnBoxes in sort_boxes[:self._nTrain]] ) )
      valData.append(   np.concatenate( [cl[valBoxes] for valBoxes in sort_boxes[self._nTrain:
                                                      self._nTrain+self._nValid]] ) )
      if self._nTest:
        testData.append(np.concatenate( [cl[tstBoxes] for tstBoxes in sort_boxes[self._nTrain+self._nValid:]] ) )

    self._logger.info('Train      #Events/class: %r', 
                      [cTrnData.shape[0] for cTrnData in trainData])
    self._logger.info('Validation #Events/class: %r', 
                      [cValData.shape[0] for cValData in valData])
    if self._nTest:  
      self._logger.info('Test #Events/class: %r', 
                        [cTstData.shape[0] for cTstData in testData])
    return trainData, valData, testData
  # __call__ end

  def getBoxPosition(self, sort, boxIdx, *sets, **kw):
    """
      Returns start and end position from a box index in continuous data
      representation merged after repositioning a split into equally distributed
      sets into this cross validation object number of boxes.

      WARNING: This does not count the position with respect to the remainders!

      startPos, endPos = crossVal.getBoxPosition( sort, boxIdx, evtsPerBox )

      If you also want to retrieve the index with respect to the divided set,
      then inform the sets as the *args argument list, in this case, it will
      also return the instance where the box is in. It will also treat the 
      remainders which were added to the sets!

      startPos, endPos, set, cStartPos, cEndPos = \
          crossVal.getBoxPosition( sort, boxIdx, trnData,
                                   valData[, tstData=None, 
                                             evtsPerBox = None,
                                             remainder = None])
      
    """
    evtsPerBox = kw.pop( 'evtsPerBox', None )
    remainder = kw.pop( 'remainder', None )

    # The sorted boxes:
    sort_boxes = self._sort_boxes_list[sort]
    # Retrieve evtsPerBox if it was not input:
    if evtsPerBox is None:
      if not sets:
        raise TypeError(("It is needed to inform the sets or the number of "
            "events per box"))
      from math import floor
      # Retrieve total number of events:
      evts = cTrnData.shape[0] + cValData.shape[0] + cTstData.shape[0]
      # The number of events in each splitted box:
      evtsPerBox = floor( evts / self._nBoxes )
    # The position where the box start and end
    startPos = box_pos_in_sort * evtsPerBox 
    endPos = startPos + evtsPerBox
    # Discover which data from which we will take this box:
    if sets:
      # Calculate the remainder when we do equal splits in nBoxes:
      if remainder is None:
        remainder = evts % self._nBoxes
      # The index where this box is in the sorts:
      box_pos_in_sort = sort_boxes.index(box)
      # Retrieve the number of boxes which were increased by the remainder:
      increaseSize = sum( box_pos_in_sort[:box_pos_in_sort-1] < remainder  )
      # The start position and end position of the current box:
      startPos += increaseSize
      endPos += increaseSize
      # Finally, check from which set should we take this box:
      takeFrom = sets[0]
      if box_pos_in_sort > self._nTrain:
        if len(sets) > 1:
          takeFrom = sets[1]
          # We must remove the size from the train dataset:
          startPos -= sets[0].shape[0]
          endPos   -= sets[0].shape[0]
        else:
          raise RuntimeError(("Validation dataset was not given as an input, "
            "but it seems that the current box is at the validation dataset."))
      elif box_pos_in_sort > self._nTrain + self._nValid:
        if len(sets) > 2:
          takeFrom = sets[2]
          # We must remove the size from the train and validation dataset:
          startPos -= sets[0].shape[0] + sets[1].shape[0]
          endPos   -= sets[0].shape[0] + sets[1].shape[0]
        else:
          raise RuntimeError(("Test dataset was not given as an input, but it "
            "seems that the current box is at the test dataset."))
    else:
      takeFrom = None
    return startPos, endPos, takefrom
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

    sort_boxes = self._sort_boxes_list[sort]

    # This will hold the data information:
    data = []

    for cTrnData, cValData, cTstData in zip(trnData, valData, tstData):
      # Retrieve total number of events:
      evts = cTrnData.shape[0] + cValData.shape[0] + cTstData.shape[0]
      # Allocate the numpy array to hold 
      cData = np.zeros(shape=(evts,cTrnData.shape[1]), dtype='float32')
      # Calculate the remainder when we do equal splits in nBoxes:
      remainder = evts % self._nBoxes
      # The number of events in each splitted box:
      evtsPerBox = floor( evts / self._nBoxes )
      # Create a holder for the remainder events, which must be in the end of the
      # data array
      remainderData = []
      for box in range(self._nBoxes):
        # Get the indexes where we will put our data in cData:
        cStartPos = box * evtsPerBox 
        cEndPos = startPos + evtsPerBox
        # And get the indexes and dataset where we will copy the values from:
        startPos, endPos, ds = self.getBoxPosition( sort, 
                                                    boxIdx,
                                                    cTrnData, 
                                                    cValData, 
                                                    cTstData,
                                                    evtsPerBox = evtsPerBox,
                                                    remainder = remainder )
        # Copy this box values to data:
        cData[cStartPos:cEndPos,] = ds[startPos:endPos,]
        # We also want to copy this box remainder if it exists to the remainder
        # data:
        if box < remainder:
          # Take the row added to the end of dataset:
          remainderData.append( ds[ endPos, ] )
      # We finished looping over the boxes, now we copy the remainder data to
      # the last positions of our original data np.array:
      cData[ (evtsPerBox * self._nBoxes): ,] = remainderData
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

