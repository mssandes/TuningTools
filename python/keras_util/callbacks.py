__all__ = ['PerformanceHistory']

import keras.callbacks as callbacks
from collections import OrderedDict

from RingerCore import Logger
from TuningTools.Neural import Roc

#class DefaultProgBar(keras.callbacks.Callback, Logger):
#
#  def on_batch_end(self, batch, logs={''}):

class PerformanceHistory(callbacks.History, Logger):

  def __init__(self, model = None, trnData = None, valData = None, tstData = None, references = None, display=None):
    callbacks.History.__init__(self)
    Logger.__init__(self)
    self.model      = model
    self.trnData    = trnData
    self.valData    = valData
    self.tstData    = tstData
    self.trnRoc     = Roc()
    self.valRoc     = Roc()
    self.tstRoc     = Roc()
    self.references = references
    self.display    = display
 
  def on_train_begin(self, logs={}):
    callbacks.History.on_train_begin(self, logs)

  def on_epoch_end(self, epoch, logs={}):
    #self.trnRoc( self.model.predict(self.trnData[0]), self.trnData[1] )
    self.valRoc( self.model.predict(self.valData[0]), self.valData[1] )
    #if self.tstData:
    #  self.tstRoc( self.model.predict(self.tstData[0]), self.tstData[1] )

    # Add information to the logs so that other callbacks can have access to
    # them:
    for bench in self.references:
      #trnPoint = self.trnRoc.retrieve( bench, extraLabel='trn_' ); logs.update( trnPoint.__dict__ )
      valPoint = self.valRoc.retrieve( bench, extraLabel='val_' ); logs.update( valPoint.__dict__ )
      #if self.tstData:
      #  tstPoint = self.tstRoc.retrieve( bench, extraLabel='tst_' ); logs.update( trnPoint.__dict__ )

    if self.display and not(epoch % self.display):
      self._info( "epoch %d/%d: %s", epoch + 1, self.params['nb_epoch'], str(sorted(logs.items()))[1:-1] )

    callbacks.History.on_epoch_end(self, epoch, logs)

  def on_train_end(self, logs={}):
    epoch = self.epoch[-1]
    if epoch + 1 != self.params['nb_epoch']:
      if not(self.display) or epoch % self.display:
        self._info("EarlyStopping, performances are: %s", str(sorted(logs.items()))[1:-1])
      else:
        self._info("EarlyStopping...")
    else:
      if not(self.display) or epoch % self.display:
        self._info("Finished tuning, performances are: %s", str(sorted(logs.items()))[1:-1])

