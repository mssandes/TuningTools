
__all__ = ['EarlyStopping']

from keras.callbacks import Callback
from RingerCore import Logger
from libTuningTools import genRoc
import numpy as np

class EarlyStopping(Callback,Logger):

  def __init__(self, display=1, doMultiStop=False, patience = 25, save_the_best=True, **kw):
    
    # initialize all base objects
    super(Callback, self).__init__()
    Logger.__init__(self,**kw)
    # number of max fails
    self._patience = {'max':patience,'score':0, 'loss':0}
    self._display = display
    self._save_the_best = save_the_best
    # used to hold the best configs
    self._best_weights = None
    # used to hold the SP value
    self._current_score = 0.0


  def on_epoch_end(self, epoch, logs={}):
    
    self.data=self.validation_data[0]
    y_pred = self.model.predict(self.validation_data[0],batch_size=int(self.validation_data[0].shape[0]/4.))
    y_true = self.validation_data[1]
    sp, det, fa, thresholds = self.roc( y_pred, y_true )
    # get the max sp value
    knee = np.argmax( sp ); sp_max = sp[knee]
    
    # check ig the current sp score is maximal
    if sp_max > self._current_score:
      self._logger.debug( ('Best SP reached is: %1.4f (DET=%1.4f, FA=%1.4f)')%(sp_max*100,det[knee]*100, fa[knee]*100) )
      self._current_score=sp_max
      self._patience['score']=0
      if self._save_the_best:
      	self._best_weights = self.model.get_weights()
    else:
      self._patience['score']+=1

    if self._display and not(epoch % self._display):
      self._logger.info('Epoch %d/%d: loss = %1.4f, acc = %1.4f, val_loss = %1.4f, val_acc = %1.4f and [Patience = %d/%d]',
          epoch+1, self.params['epochs'], logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc'],
          self._patience['score'], self._patience['max'])

    # Stop the fitting
    if self._patience['score'] > self._patience['max']:
      self._logger.info('Stopping the Training by SP...')
      self.model.stop_training = True


  def on_train_end(self, logs={}):
    # Loading the best model
    if self._save_the_best:
      self._logger.info('Reload the best configuration into the current model...')
      self.model.set_weights( self._best_weights )
    self._logger.info("Finished tuning")
 

  @classmethod
  def roc(cls, pred, target, resolution=0.01):
    signal = pred[np.where(target==1)]; noise = pred[np.where(target==-1)]
    return genRoc( signal, noise, 1, -1, resolution)
      

