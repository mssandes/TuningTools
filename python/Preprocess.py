from FastNetTool.Logger import Logger
import numpy as np

class Normalize(Logger):
  def __init__(self, **kw):
    Logger.__init__(self, **kw)
    self._type  = kw.pop('Norm','totalEnergy')
    self.appName = 'preprocessNormalizeTool'
    self._logger.info('Normalization tool was created.')

  def __call__(self, data):
    if self._type == 'totalEnergy': return self.__total_energy(data)

  def __total_energy(self, data):
      norms = data.sum(axis=1)
      norms[norms==0] = 1
      return data / norms[:, np.newaxis ]


class RingerRp(Logger):
  def __init__(self, **kw):
    Logger.__init__(self, **kw)
    self._alpha = kw.pop('alpha', 1)
    self._beta  = kw.pop('beta' , 1)
    self.appName = 'PreprocessRingerRpTool'
    #Layers resolution
    PS      = 0.025 * np.arange(8)
    EM1     = 0.003125 * np.arange(64)
    EM2     = 0.025 * np.arange(8)
    EM3     = 0.05 * np.arange(8)
    HAD1    = 0.1 * np.arange(4)
    HAD2    = 0.1 * np.arange(4)
    HAD3    = 0.2 * np.arange(4)
    rings = np.concatenate((PS,EM1,EM2,EM3,HAD1,HAD2,HAD3))
    self._rVec = np.power( rings, self._beta)
    self._logger.info('RingerRp tool was created using alpha = %d and beta =\
                      %d.',self._alpha, self._beta)

  def __call__(self, data):
    norms = np.power(data, self._alpha)
    norms = data.sum(axis=1)
    norms[norms==0] = 1
    return (data*self._rVec )/ norms[:, np.newaxis ]










