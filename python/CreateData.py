__all__ = ['TuningDataArchieve', 'CreateData', 'createData']

_noProfilePlot = False
try:
  import scipy.stats 
except ImportError as _noProfileImportError:
  _noProfilePlot = True
try:
  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
except ImportError as _noProfileImportError:
  _noProfilePlot = True
from RingerCore import Logger, checkForUnusedVars, reshape, save, load, traverse, \
                       retrieve_kw, NotSet, appendToFileName, progressbar
from TuningTools.coreDef import retrieve_npConstants

npCurrent, _ = retrieve_npConstants()
import numpy as np

# FIXME This should be integrated into a class so that save could check if it
# is one instance of this base class and use its save method
class TuningDataArchieve( Logger ):
  """
  Context manager for Tuning Data archives

  Version 5: - Changes _rings for _patterns
  Version 4: - keeps the operation requested by user when creating data
  Version 3: - added eta/et bins compatibility
             - added benchmark efficiency information
             - improved fortran/C integration
             - loads only the indicated bins to memory
  Version 2: - started fotran/C order integration
  Version 1: - save compressed npz file
             - removed target information: classes are flaged as
               signal_rings/background_rings
  Version 0: - save pickle file with numpy data
  """

  _type = np.array('TuningData', dtype='|S10')
  _version = np.array(5)

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with TuningDataArchieve("/path/to/file", 
                           [eta_bin = None],
                           [et_bin = None]) as data:
      data['signal_patterns'] # access patterns from signal dataset 
      data['background_patterns'] # access patterns from background dataset
      data['benchmark_effs'] # access benchmark efficiencies

    When setting eta_bin or et_bin to None, the function will return data and
    efficiency for all bins instead of the just one selected.

    TuningDataArchieve( signal_patterns = np.array(...),
                       background_patterns = np.array(...),
                       eta_bins = np.array(...),
                       et_bins = np.array(...),
                       benchmark_effs = np.array(...), )
    """
    # Both
    Logger.__init__(self, kw)
    self._filePath                      = filePath
    # Saving
    self._signal_patterns               = kw.pop( 'signal_patterns',               npCurrent.fp_array([]) )
    self._background_patterns           = kw.pop( 'background_patterns',           npCurrent.fp_array([]) )
    self._eta_bins                      = kw.pop( 'eta_bins',                      npCurrent.fp_array([]) )
    self._et_bins                       = kw.pop( 'et_bins',                       npCurrent.fp_array([]) )
    self._signal_efficiencies           = kw.pop( 'signal_efficiencies',           None                   )
    self._background_efficiencies       = kw.pop( 'background_efficiencies',       None                   )
    self._signal_cross_efficiencies     = kw.pop( 'signal_cross_efficiencies',     None                   )
    self._background_cross_efficiencies = kw.pop( 'background_cross_efficiencies', None                   )
    self._toMatlab                      = kw.pop( 'toMatlab',                      False                  )
    # Loading
    self._eta_bin                       = kw.pop( 'eta_bin',                       None                   )
    self._et_bin                        = kw.pop( 'et_bin',                        None                   )
    self._operation                     = kw.pop( 'operation',                     None                   )
    self._label                         = kw.pop( 'label',                         NotSet                 )
    self._collectGraphs = []
    checkForUnusedVars( kw, self._logger.warning )
    # Make some checks:
    if type(self._signal_patterns) != type(self._background_patterns):
      self._logger.fatal("Signal and background types do not match.", TypeError)
    if type(self._signal_patterns) == list:
      if len(self._signal_patterns) != len(self._background_patterns) \
          or len(self._signal_patterns[0]) != len(self._background_patterns[0]):
        self._logger.fatal("Signal and background patterns lenghts do not match.",TypeError)
    if type(self._eta_bins) is list: self._eta_bins=npCurrent.fp_array(self._eta_bins)
    if type(self._et_bins) is list: self._et_bins=npCurrent.fp_array(self._eta_bins)
    if self._eta_bins.size == 1 or self._eta_bins.size == 1:
      self._logger.fatal("Eta or et bins size are 1.",ValueError)

  @property
  def filePath( self ):
    return self._filePath

  @property
  def signal_patterns( self ):
    return self._signal_patterns

  @property
  def background_patterns( self ):
    return self._background_patterns

  def getData( self ):
    from TuningTools.ReadData import RingerOperation
    kw_dict =  {
                'type': self._type,
             'version': self._version,
            'eta_bins': self._eta_bins,
             'et_bins': self._et_bins,
           'operation': RingerOperation.retrieve( self._operation ),
               }
    max_eta = self.__retrieve_max_bin(self._eta_bins)
    max_et = self.__retrieve_max_bin(self._et_bins)
    # Handle patterns:
    if max_eta is None and max_et is None:
      kw_dict['signal_patterns'] = self._signal_patterns
      kw_dict['background_patterns'] = self._background_patterns
    else:
      if max_eta is None: max_eta = 0
      if max_et is None: max_et = 0
      for et_bin in range( max_et + 1 ):
        for eta_bin in range( max_eta + 1 ):
          bin_str = self.__get_bin_str(et_bin, eta_bin) 
          sgn_key = 'signal_patterns_' + bin_str
          kw_dict[sgn_key] = self._signal_patterns[et_bin][eta_bin]
          bkg_key = 'background_patterns_' + bin_str
          kw_dict[bkg_key] = self._background_patterns[et_bin][eta_bin]
        # eta loop
      # et loop
    # Handle efficiencies
    from copy import deepcopy
    kw_dict['signal_efficiencies']           = deepcopy(self._signal_efficiencies)
    kw_dict['background_efficiencies']       = deepcopy(self._background_efficiencies)
    kw_dict['signal_cross_efficiencies']     = deepcopy(self._signal_cross_efficiencies)
    kw_dict['background_cross_efficiencies'] = deepcopy(self._background_cross_efficiencies)
    def efficiency_to_raw(d):
      for key, val in d.iteritems():
        for cData, idx, parent, _, _ in traverse(val):
          if parent is None:
            d[key] = cData.toRawObj()
          else:
            parent[idx] = cData.toRawObj()
    if self._signal_efficiencies and self._background_efficiencies:
      efficiency_to_raw(kw_dict['signal_efficiencies'])
      efficiency_to_raw(kw_dict['background_efficiencies'])
    if self._signal_cross_efficiencies and self._background_cross_efficiencies:
      efficiency_to_raw(kw_dict['signal_cross_efficiencies'])
      efficiency_to_raw(kw_dict['background_cross_efficiencies'])
    return kw_dict
  # end of getData

  def _toMatlabDump(self, data):
    import scipy.io as sio
    crossval = None
    kw_dict_aux = dict()

    # Retrieve efficiecies
    for key_eff in ['signal_','background_']:# sgn and bkg efficiencies
      key_eff+='efficiencies'
      kw_dict_aux[key_eff] = dict()
      for key_trigger in data[key_eff].keys():# Trigger levels
        kw_dict_aux[key_eff][key_trigger] = list()
        etbin = 0; etabin = 0
        for obj  in data[key_eff][key_trigger]: #Et
          kw_dict_aux[key_eff][key_trigger].append(list())
          for obj_  in obj: # Eta
            obj_dict = dict()
            obj_dict['count']  = obj_['_count'] if obj_.has_key('_count') else 0
            obj_dict['passed'] = obj_['_passed'] if obj_.has_key('_passed') else 0
            if obj_dict['count'] > 0:
              obj_dict['efficiency'] = obj_dict['passed']/float(obj_dict['count']) * 100
            else:
              obj_dict['efficiency'] = 0
            obj_dict['branch'] = obj_['_branch']
            kw_dict_aux[key_eff][key_trigger][etbin].append(obj_dict)
          etbin+=1 

    # Retrieve patterns 
    for key in data.keys():
      if 'rings' in key or \
         'patterns' in key:
        kw_dict_aux[key] = data[key]

    # Retrieve crossval
    try:
      crossVal = data['signal_cross_efficiencies']['L2CaloAccept'][0][0]['_crossVal']
    except KeyError:
      crossVal = data['signal_cross_efficiencies']['LHLoose'][0][0]['_crossVal']
    except IndexError:
      crossVal = None
    if crossVal is None:
      from TuningTools import CrossValid
      crossVal = CrossValid().toRawObj()
    kw_dict_aux['crossVal'] = {
                                'nBoxes'          : crossVal['nBoxes'],
                                'nSorts'          : crossVal['nSorts'],
                                'nTrain'          : crossVal['nTrain'],
                                'nTest'           : crossVal['nTest'],
                                'nValid'          : crossVal['nValid'],
                                'sort_boxes_list' : crossVal['sort_boxes_list'],
                              }

    self._logger.info( 'Saving data to matlab...')
    sio.savemat(self._filePath+'.mat', kw_dict_aux)
  #end of matlabDump

  def drawProfiles(self):
    from itertools import product
    for etBin, etaBin in progressbar(product(range(self.nEtBins()),range(self.nEtaBins())), self.nEtBins()*self.nEtaBins(),
                                     logger = self._logger, prefix = "Drawing profiles "):
      sdata = self._signal_patterns[etBin][etaBin]
      bdata = self._background_patterns[etBin][etaBin]
      if sdata is not None:
        self._makeGrid(sdata,'signal',etBin,etaBin)
      if bdata is not None:
        self._makeGrid(bdata,'background',etBin,etaBin)


  def _makeGrid(self,data,bckOrSgn,etBin,etaBin):
    colors=[(0.1706, 0.5578, 0.9020),
            (0.1427, 0.4666, 0.7544),
            (0.1148, 0.3754, 0.6069),
            (0.0869, 0.2841, 0.4594),
            (0.9500, 0.3000, 0.3000),
            (0.7661, 0.2419, 0.2419),
            (0.5823, 0.1839, 0.1839)]
    upperBounds= np.zeros(100)
    lowerBounds= np.zeros(100)
    dataT = np.transpose(data)
    underFlows= np.zeros(100)
    overFlows= np.zeros(100)
    nLayersRings= np.array([8,64,8,8,4,4,4])
    layersEdges = np.delete(np.cumsum( np.append(-1, nLayersRings)),0)
    opercent=np.ones(100)
    nonzeros = []
    self._oSeparator(dataT,opercent,nonzeros)

    for i in range(len(nonzeros)):
      if len(nonzeros[i]):
        upperBounds[i]= max(nonzeros[i])
        lowerBounds[i]= min(nonzeros[i])
        self._forceLowerBound(i,lowerBounds,nonzeros)
        self._takeUnderFlows(i,underFlows,lowerBounds,nonzeros)
        self._findParcialUpperBound(i,underFlows,upperBounds,nonzeros)
        self._makeCorrections(i,lowerBounds,upperBounds, layersEdges )
        self._takeOverFlows(i,overFlows,upperBounds,nonzeros)
        self._takeUnderFlows(i,underFlows,lowerBounds,nonzeros)

    for i in range(len(nonzeros)):
      if len(nonzeros[i]):
        if( i <  8*11):
          plt.subplot2grid((8,14), (i%8,i/8))
          self._plotHistogram(np.array( nonzeros[i]), np.where(layersEdges >= i )[0][0],i,lowerBounds,upperBounds,opercent,underFlows,overFlows,colors)
        else:
          plt.subplot2grid((8,14), ((i-88)% 4,(i-88)/4+11 ))
          self._plotHistogram(np.array(nonzeros[i]),np.where(layersEdges >= i )[0][0],i,lowerBounds,upperBounds,opercent,
            underFlows,overFlows,colors)
      else:
        if( i <  8*11):
          plt.subplot2grid((8,14), (i%8,i/8))
          self._representNullRing(i)
        else:
          plt.subplot2grid((8,14), ((i-88)% 4,(i-88)/4+11 ))
          self._representNullRing(i)  
        
    plt.subplot2grid((8,14), (8-4,14-3),colspan=3,rowspan=4)
    verts= [(0,1),(0,0.7),(1,0.7),(1,1)] 
    ax= plt.gca()
    ax.add_patch( patches.Rectangle((0,0.7),1,0.3,facecolor='none'))
    aux = bckOrSgn[0].upper()+bckOrSgn[1::]
    color=colors[1] if bckOrSgn == 'signal' else colors[-2]
    plt.text(0.1,0.75,"Rings Energy(MeV)\nhistograms for\n{}\nEt[{}] Eta[{}] ".format(aux,etBin,etaBin),color=color,multialignment='center',
        size='large',fontweight='bold')

    if self._label is not NotSet:
      plt.text(0.25,0.3,'{}'.format(self._label),fontsize=12)

    plt.text(0.1,0.07,'Number of clusters for this\ndataset:\n{}'.format(data.shape[0])
        ,multialignment='center',fontweight='bold',fontsize=9)

    self._makeColorsLegend(colors)

    for line in ax.spines.values() :
      line.set_visible(False)
   
    for line in ax.yaxis.get_ticklines() + ax.xaxis.get_ticklines():
       line.set_visible(False)

    for tl in ax.get_xticklabels() + ax.get_yticklabels():
      tl.set_visible(False)

    figure = plt.gcf() # get current figure
    figure.set_size_inches(16,9)
   
    plt.savefig('ring_distribution_{}_etBin{}_etaBin{}.pdf'.format(bckOrSgn,etBin,etaBin),dpi=100,bbox_inches='tight')

  def _makeColorsLegend(self,colors):
    import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.text(0.15,0.56
        ,'Layer Color Legend:',fontsize=12)
    text=['PS','EM1','EM2','EM3','HAD1','HAD2','HAD3']
    x0,x = 0.1,0.1
    y0,y = 0.5,0.5
    for i in range(4):
      plt.text(x,y,text[i],color=colors[i])
      x=x+0.2

    y=y-0.05
    x=x0+0.1
    for i in ( np.arange(3)+4):
      plt.text(x,y,text[i],color=colors[i])
      x=x+0.2 

  def _oSeparator(self,dataT,opercent,nonzeros):
    for index in range(dataT.shape[0]):
      counter = 0 
      ocounter = 0
      no0= np.array([])
      for aux in dataT[index] :
        if( aux != 0):
          no0 = np.append(no0, aux)
        else:
          ocounter = ocounter +1
        counter = counter +1
      liist = no0.tolist() 
      
      nonzeros.append (liist)
      opercent[index] =(ocounter*100.0)/counter

  def _plotHistogram(self,data,layer,ring,lowerBounds,upperBounds,opercent,underFlows,overFlows, colors,nbins=60):
    statistcs = scipy.stats.describe(data) 
    if type(statistcs) is tuple:
      class DescribeResult(object):
        def __init__(self, t):
          self.mean = statistcs[2]
          self.variance= statistcs[3]
          self.skewness= statistcs[4]
          self.kurtosis= statistcs[5]
      statistcs = DescribeResult(statistcs)
    m=statistcs.mean
    plotingData=[]
    statstring='{:0.1f}\n{:0.1f}\n{:0.1f}\n{:0.1f}'.format(statistcs.mean,statistcs.variance**0.5,
        statistcs.skewness,statistcs.kurtosis)

    binSize=( upperBounds[ring]-lowerBounds[ring])/(nbins + -2.0)
    underflowbound= lowerBounds[ring]- binSize
    overFlowbound= upperBounds[ring] + binSize
    
    for n in data:
      if n >   lowerBounds[ring] and  n < upperBounds[ring]:
        plotingData.append(n)
      elif n >upperBounds[ring]:

        plotingData.append( upperBounds[ring] + binSize/2.0)
      else:
        plotingData.append( lowerBounds[ring] - binSize/2.0)
    
    n, bins, patches = plt.hist(plotingData,nbins,[underflowbound,overFlowbound],edgecolor=colors[layer])
    mbins=[]

    for i in range(nbins):
      mbins.append( (bins[i]+bins[i+1])/2.0)

    plt.axis([underflowbound,overFlowbound,0,max(n)])
    ax  = plt.gca()  

    for tl in ax.get_xticklabels() + ax.get_yticklabels():
      tl.set_visible(False)
     
    of= overFlows[ring]*100.0
    uf=underFlows[ring]*100.0

    plt.ylabel ('#{} U:{:0.1f}% | O:{:0.1f}%'.format(ring+1,uf,of),labelpad=0,fontsize=5 )
    plt.xlabel('{:0.0f}    {:0.1f}%    {:0.0f}'.format(lowerBounds[ring] , opercent[ring],
      upperBounds[ring]),fontsize=5,labelpad=2)

    xtext= underflowbound + (-underflowbound + overFlowbound)*0.75
    ytext= max(n)/1.5

    plt.text(xtext,ytext,statstring,fontsize=4,multialignment='right')

    try:
      mdidx=np.where(n==max(n))[0][0]
      midx = np.where(bins > m)[0][0]-1

      xm=[mbins[midx],mbins[midx]]
      xmd=[mbins[mdidx],mbins[mdidx]]
      x0=[0,0]
      y=[0,max(n)/2]
      plt.plot(x0,y,'k',dashes=(1,1),linewidth=0.5) # 0
      plt.plot(xmd,y,'k',linewidth=0.5,dashes=(5,1)) # mode
      plt.plot(xm,y,'k-',linewidth=0.5) # mean
    except IndexError:
      pass

    for line in ax.yaxis.get_ticklines() + ax.xaxis.get_ticklines():
       line.set_visible(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

  def _representNullRing(self,i):
    ax  = plt.gca()  
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
      tl.set_visible(False)
    plt.ylabel('#{}'.format(i),multialignment='left',fontsize='5',labelpad=0)
    for line in ax.yaxis.get_ticklines() + ax.xaxis.get_ticklines():
       line.set_visible(False)

  def _lowerPowerofTen(self,x):
    from math import log10,floor
    num = 0
    if x > 0:
      num = x
    else:
      num = - x
    lg=log10(num)
    return floor(lg)

  def _forceLowerBound(self,i,lowerBounds,nonzeros):
    parcial=-500
    while   np.sum(np.array(nonzeros[i])<parcial)/float(len(nonzeros[i])) > (0.005):
      parcial += -500
    lowerBounds[i] = parcial

  def _findParcialUpperBound(self,i,underFlows,upperBounds,nonzeros):
    cMax= upperBounds[i]
    power10= self._lowerPowerofTen (cMax)
    cPerc = 1 - underFlows[i]
    while cPerc > .99:
      if cMax  - 10**(power10-1) < 0.8*cMax:
        power10 -= 1
      cMax  = cMax - 10**(power10-1)
      cPerc = np.sum(nonzeros[i]<=cMax)/float(len(nonzeros[i])) - underFlows[i]
    while cPerc < .99 - 0.001 and cPerc > .99:
      cMax += 10**(power10-2)
      cPerc= np.sum(nonzeros[i]<=cMax)/float(len(nonzeros[i])) - underFlows[i]
      
    power10= self._lowerPowerofTen(cMax)
    for j  in np.arange(2,11)*10**power10:
      if abs(cMax) < j:
        if cMax<0:
          upperBounds[i]= - j
        else:
          upperBounds[i]= j
        break


  def _makeCorrections(self,i,lowerBounds,upperBounds,LayerEdges):
    if lowerBounds[i] < 0 and -lowerBounds[i] > upperBounds[i] and upperBounds[i]>0:
      lowerBounds[i] = -upperBounds[i]
    if lowerBounds[i] < 0 and -lowerBounds[i] < upperBounds[i] and upperBounds[i] <= 1000 and upperBounds[i]>0:
      lowerBounds[i] = -upperBounds[i]
    
    if i < 8:
      lidx = 0
    else:
      lidx= LayerEdges[(np.where(LayerEdges < i)[0][-1] ) ] +1
    
    for j in range( lidx , i):
      if lowerBounds[i]<lowerBounds[j]:
        lowerBounds[j]=lowerBounds[i]
    
      if upperBounds[i] > upperBounds[j]:
        upperBounds[j] = upperBounds[i]
    
      if lowerBounds[j]<0 and -lowerBounds[j] > upperBounds[j] and upperBounds[i]>0:
        lowerBounds[j] = -upperBounds[j]

      if lowerBounds[j]<0 and -lowerBounds[j] < upperBounds[j] and upperBounds[j] <= 1000 and upperBounds[i]>0 :
        lowerBounds[j] = -upperBounds[j]


  def _takeUnderFlows(self,i,underFlows,lowerBounds,nonzeros):
    lower= lowerBounds[i]
    underFlows[i] = float(np.sum(np.array(nonzeros[i])<lower))/len(nonzeros[i])

  def _takeOverFlows(self,i,overFlows,upperBounds,nonzeros):
    upper= upperBounds[i]
    overFlows[i] = np.sum(np.array(nonzeros[i])>upper)/float(len(nonzeros[i]))

  def __generateMeanGraph (self, canvas, data, kind, etbounds, etabounds, color, idx = 0):
    from ROOT import TGraph, gROOT, kTRUE
    gROOT.SetBatch(kTRUE)
    xLabel = "Ring #"
    yLabel = "Energy (MeV)"
 
    if data is None or not len(data):
      self._logger.error("Data is unavaliable")
    else:
      x = np.arange( 100 ) + 1.0
      y = data.mean(axis=0 ,dtype='f8')
      n =data.shape[1]
    
      canvas.cd(idx)
      canvas.SetGrid()
      graph = TGraph(n , x , y )
      self._collectGraphs.append( graph )
      graph.SetTitle( ( kind + " et = [%d ,  %d] eta = [%.2f,  %.2f]" ) % (etbounds[0],etbounds[1],etabounds[0],etabounds[1]))
      graph.GetXaxis().SetTitle(xLabel)
      graph.GetYaxis().SetTitle(yLabel)
      graph.GetYaxis().SetTitleOffset(1.9)
      graph.SetFillColor(color)
      graph.Draw("AB")

  def plotMeanPatterns(self):
    from ROOT import TCanvas, gROOT, kTRUE
    gROOT.SetBatch(kTRUE)

    for etBin in range(self.nEtBins()):
      for etaBin in range(self.nEtaBins()):
        c1 = TCanvas("plot_patternsMean_et%d_eta%d" % (etBin, etaBin), "a",0,0,800,400)

        signal = self._signal_patterns[etBin][etaBin]
        background = self._background_patterns[etBin][etaBin]

        if (signal is not None) and (background is not None):
          c1.Divide(2,1)

        etBound = self._et_bins[etBin:etBin+2]
        etaBound = self._eta_bins[etaBin:etaBin+2]

        self.__generateMeanGraph( c1, signal,     "Signal",     etBound, etaBound, 34, 1 )
        self.__generateMeanGraph( c1, background, "Background", etBound, etaBound, 2,  2 )

        c1.SaveAs('plot_patterns_mean_et_%d_eta%d.pdf' % (etBin, etaBin))
        c1.Close()
    self._collectGraphs = []

  def save(self):
    self._logger.info( 'Saving data using following numpy flags: %r', npCurrent)
    data = self.getData()
    if self._toMatlab:  self._toMatlabDump(data)
    return save(data, self._filePath, protocol = 'savez_compressed')



  def __enter__(self):
    data = {'et_bins' : npCurrent.fp_array([]),
            'eta_bins' : npCurrent.fp_array([]),
            'operation' : None,
            'signal_patterns' : npCurrent.fp_array([]),
            'background_patterns' : npCurrent.fp_array([]),
            'signal_efficiencies' : {},
            'background_efficiencies' : {},
            'signal_efficiencies' : {},
            'background_efficiencies' : {},
            }
    npData = load( self._filePath )
    try:
      if type(npData) is np.ndarray:
        # Legacy type:
        data = reshape( npData[0] ) 
        target = reshape( npData[1] ) 
        self._signal_patterns, self._background_patterns = TuningDataArchieve.__separateClasses( data, target )
        data = {'signal_patterns' : self._signal_patterns, 
                'background_patterns' : self._background_patterns}
      elif type(npData) is np.lib.npyio.NpzFile:
        # Retrieve operation, if any
        if npData['version'] <= np.array(4):
          sgn_base_key = 'signal_rings'
          bkg_base_key = 'background_rings'
        else:
          sgn_base_key = 'signal_patterns'
          bkg_base_key = 'background_patterns'
        if npData['version'] >= np.array(4):
          data['operation'] = npData['operation']
        else:
          from TuningTools.ReadData import RingerOperation
          data['operation'] = RingerOperation.EFCalo
        # Retrieve bins information, if any
        if npData['version'] <= np.array(5) and npData['version'] >= np.array(3): # self._version:
          eta_bins = npData['eta_bins'] if 'eta_bins' in npData else \
                     npCurrent.array([])
          et_bins  = npData['et_bins'] if 'et_bins' in npData else \
                     npCurrent.array([])
          self.__check_bins(eta_bins, et_bins)
          max_eta = self.__retrieve_max_bin(eta_bins)
          max_et = self.__retrieve_max_bin(et_bins)
          if self._eta_bin == self._et_bin == None:
            data['eta_bins'] = npCurrent.fp_array(eta_bins) if max_eta else npCurrent.fp_array([])
            data['et_bins'] = npCurrent.fp_array(et_bins) if max_et else npCurrent.fp_array([])
          else:
            data['eta_bins'] = npCurrent.fp_array([eta_bins[self._eta_bin],eta_bins[self._eta_bin+1]]) if max_eta else npCurrent.fp_array([])
            data['et_bins'] = npCurrent.fp_array([et_bins[self._et_bin],et_bins[self._et_bin+1]]) if max_et else npCurrent.fp_array([])
        # Retrieve data (and efficiencies):
        from TuningTools.ReadData import BranchEffCollector, BranchCrossEffCollector
        def retrieve_raw_efficiency(d, et_bins = None, eta_bins = None, cl = BranchEffCollector):
          if d is not None:
            if type(d) is np.ndarray:
              d = d.item()
            for key, val in d.iteritems():
              if et_bins is None or eta_bins is None:
                for cData, idx, parent, _, _ in traverse(val):
                  if parent is None:
                    d[key] = cl.fromRawObj(cData)
                  else:
                    parent[idx] = cl.fromRawObj(cData)
              else:
                if type(et_bins) == type(eta_bins) == list:
                  d[key] = []
                  for cEtBin, et_bin in enumerate(et_bins):
                    d[key].append([])
                    for eta_bin in eta_bins:
                      d[key][cEtBin].append(cl.fromRawObj(val[et_bin][eta_bin]))
                else:
                  d[key] = cl.fromRawObj(val[et_bins][eta_bins])
          return d
        if npData['version'] <= np.array(5) and npData['version'] >= np.array(3): # self._version:
          if self._eta_bin is None and max_eta is not None:
            self._eta_bin = range( max_eta + 1 )
          if self._et_bin is None and max_et is not None:
            self._et_bin = range( max_et + 1)
          if self._et_bin is None and self._eta_bin is None:
            data['signal_patterns'] = npData[sgn_base_key]
            data['background_patterns'] = npData[bkg_base_key]
            try:
              data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'])
              data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'])
            except KeyError:
              pass
            try:
              data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'], BranchCrossEffCollector)
              data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'], BranchCrossEffCollector)
            except (KeyError, TypeError):
              pass
          else:
            # FIXME This is where _eta_bin is being assigned and makes impossible to load file twice!
            if self._eta_bin is None: self._eta_bin = 0
            if self._et_bin is None: self._et_bin = 0
            if type(self._eta_bin) == type(self._eta_bin) != list:
              bin_str = self.__get_bin_str(self._et_bin, self._eta_bin) 
              sgn_key = sgn_base_key + '_' + bin_str
              bkg_key = bkg_base_key + '_' + bin_str
              data['signal_patterns']                  = npData[sgn_key]
              data['background_patterns']              = npData[bkg_key]
              try:
                data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
                data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'],
                                                                                self._et_bin, self._eta_bin)
              except KeyError:
                pass
              try:
                data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'],
                                                                                self._et_bin, self._eta_bin, BranchCrossEffCollector)
                data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'],
                                                                                self._et_bin, self._eta_bin, BranchCrossEffCollector)
              except (KeyError, TypeError):
                pass
            else:
              if not type(self._eta_bin) is list:
                self._eta_bin = [self._eta_bin]
              if not type(self._et_bin) is list:
                self._et_bin = [self._et_bin]
              sgn_list = []
              bkg_list = []
              for et_bin in self._et_bin:
                sgn_local_list = []
                bkg_local_list = []
                for eta_bin in self._eta_bin:
                  bin_str = self.__get_bin_str(et_bin, eta_bin) 
                  sgn_key = sgn_base_key + '_' + bin_str
                  sgn_local_list.append(npData[sgn_key])
                  bkg_key = bkg_base_key + '_' + bin_str
                  bkg_local_list.append(npData[bkg_key])
                # Finished looping on eta
                sgn_list.append(sgn_local_list)
                bkg_list.append(bkg_local_list)
              # Finished retrieving data
              data['signal_patterns'] = sgn_list
              data['background_patterns'] = bkg_list
              indexes = self._eta_bin[:]; indexes.append(self._eta_bin[-1]+1)
              data['eta_bins'] = eta_bins[indexes]
              indexes = self._et_bin[:]; indexes.append(self._et_bin[-1]+1)
              data['et_bins'] = et_bins[indexes]
              try:
                data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
                data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
              except KeyError:
                pass
              try:
                data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'], 
                                                                                self._et_bin, self._eta_bin, 
                                                                                BranchCrossEffCollector)
                data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'], 
                                                                                self._et_bin, self._eta_bin, 
                                                                                BranchCrossEffCollector)
              except (KeyError, TypeError):
                pass
        elif npData['version'] <= np.array(2): # self._version:
          data['signal_patterns']     = npData['signal_patterns']
          data['background_patterns'] = npData['background_patterns']
        else:
          self._logger.fatal("Unknown file version!")
      elif isinstance(npData, dict) and 'type' in npData:
        self._logger.fatal("Attempted to read archive of type: %s_v%d" % (npData['type'],
                                                                          npData['version']))
      else:
        self._logger.fatal("Object on file is of unkown type.")
    except RuntimeError, e:
      self._logger.fatal(("Couldn't read TuningDataArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))
    data['eta_bins'] = npCurrent.fix_fp_array(data['eta_bins'])
    data['et_bins'] = npCurrent.fix_fp_array(data['et_bins'])
    # Now that data is defined, check if numpy information fits with the
    # information representation we need:
    if type(data['signal_patterns']) is list:
      for cData, idx, parent, _, _ in traverse(data['signal_patterns'], (list,tuple,np.ndarray), 2):
        cData = npCurrent.fix_fp_array(cData)
        parent[idx] = cData
      for cData, idx, parent, _, _ in traverse(data['background_patterns'], (list,tuple,np.ndarray), 2):
        cData = npCurrent.fix_fp_array(cData)
        parent[idx] = cData
    else:
      data['signal_patterns'] = npCurrent.fix_fp_array(data['signal_patterns'])
      data['background_patterns'] = npCurrent.fix_fp_array(data['background_patterns'])
    return data
    
  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def nEtBins(self):
    """
      Return maximum eta bin index. If variable is not dependent on bin, return none.
    """
    if self._et_bins is None:
      et_max = self.__max_bin('et_bins') 
    else:
      et_max = len(self._et_bins) - 1
    return et_max  if et_max is not None else et_max

  def nEtaBins(self):
    """
      Return maximum eta bin index. If variable is not dependent on bin, return none.
    """

    if self._eta_bins is None:
      eta_max = self.__max_bin('eta_bins')
    else:
      eta_max = len(self._eta_bins) - 1
    return eta_max if eta_max is not None else eta_max

  def __max_bin(self, var):
    """
      Return maximum dependent bin index. If variable is not dependent on bin, return none.
    """
    npData = load( self._filePath )
    try:
      if type(npData) is np.ndarray:
        return None
      elif type(npData) is np.lib.npyio.NpzFile:
        try:
          if npData['type'] != self._type:
            self._logger.fatal("Input file is not of TuningData type!")
        except KeyError:
          self._logger.warning("File type is not specified... assuming it is ok...")
        arr  = npData[var] if var in npData else npCurrent.array([])
        return self.__retrieve_max_bin(arr)
    except RuntimeError, e:
      self._logger.fatal(("Couldn't read TuningDataArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))

  def __retrieve_max_bin(self, arr):
    """
    Return  maximum dependent bin index. If variable is not dependent, return None.
    """
    max_size = arr.size - 2
    return max_size if max_size >= 0 else None

  def __check_bins(self, eta_bins, et_bins):
    """
    Check if self._eta_bin and self._et_bin are ok, through otherwise.
    """
    max_eta = self.__retrieve_max_bin(eta_bins)
    max_et = self.__retrieve_max_bin(et_bins)
    # Check if eta/et bin requested can be retrieved.
    errmsg = ""
    if self._eta_bin > max_eta:
      errmsg += "Cannot retrieve eta_bin(%r) from eta_bins (%r). %s" % (self._eta_bin, eta_bins, 
          ('Max bin is: ' + str(max_eta) + '. ') if max_eta is not None else ' Cannot use eta bins.')
    if self._et_bin > max_et:
      errmsg += "Cannot retrieve et_bin(%r) from et_bins (%r). %s" % (self._et_bin, et_bins,
          ('Max bin is: ' + str(max_et) + '. ') if max_et is not None else ' Cannot use E_T bins. ')
    if errmsg:
      self._logger.fatal(errmsg)

  def __get_bin_str(self, et_bin, eta_bin):
    return 'etBin_' + str(et_bin) + '_etaBin_' + str(eta_bin)

  @classmethod
  def __separateClasses( cls, data, target ):
    """
    Function for dealing with legacy data.
    """
    sgn = data[np.where(target==1)]
    bkg = data[np.where(target==-1)]
    return sgn, bkg2


class CreateData(Logger):

  def __init__( self, logger = None ):
    Logger.__init__( self, logger = logger )
    from TuningTools.ReadData import readData
    self._reader = readData

  def __call__(self, sgnFileList, bkgFileList, ringerOperation, **kw):
    """
      Creates a numpy file ntuple with rings and its targets
      Arguments:
        - sgnFileList: A python list or a comma separated list of the root files
            containing the TuningTool TTree for the signal dataset
        - bkgFileList: A python list or a comma separated list of the root files
            containing the TuningTool TTree for the background dataset
        - ringerOperation: Set Operation type to be used by the filter
      Optional arguments:
        - pattern_oFile ['tuningData']: Path for saving the output file
        - efficiency_oFile ['tuningData']: Path for saving the output file efficiency
        - referenceSgn [Reference.Truth]: Filter reference for signal dataset
        - referenceBkg [Reference.Truth]: Filter reference for background dataset
        - treePath [<Same as ReadData default>]: set tree name on file, this may be set to
          use different sources then the default.
        - efficiencyTreePath [None]: Sets tree path for retrieving efficiency
              benchmarks.
            When not set, uses treePath as tree.
        - nClusters [<Same as ReadData default>]: Number of clusters to export. If set to None, export
            full PhysVal information.
        - getRatesOnly [<Same as ReadData default>]: Do not create data, but retrieve the efficiency
            for benchmark on the chosen operation.
        - etBins [<Same as ReadData default>]: E_T bins  (GeV) where the data should be segmented
        - etaBins [<Same as ReadData default>]: eta bins where the data should be segmented
        - ringConfig [<Same as ReadData default>]: A list containing the number of rings available in the data
          for each eta bin.
        - crossVal [<Same as ReadData default>]: Whether to measure benchmark efficiency splitting it
          by the crossVal-validation datasets
        - extractDet [<Same as ReadData default>]: Which detector to export (use Detector enumeration).
        - standardCaloVariables [<Same as ReadData default>]: Whether to extract standard track variables.
        - useTRT [<Same as ReadData default>]: Whether to export TRT information when dumping track
          variables.
        - toMatlab [False]: Whether to also export data to matlab format.
        - efficiencyValues [NotSet]: expert property to force the efficiency values to a new reference.
          This property can be [detection = 97.0, falseAlarm = 2.0] or a matrix with size
          E_T bins X Eta bins where each position is [detection, falseAlarm].
        - plotMeans [True]: Plot mean values of the patterns
        - plotProfiles [False]: Plot pattern profiles
        - label [NotSet]: Adds label to profile plots
        - supportTriggers [True]: Whether reading data comes from support triggers
    """
    """
    # TODO Add a way to create new reference files setting operation points as
    # desired. A way to implement this is:
    #"""
    #    - tuneOperationTargets [['Loose', 'Pd' , #looseBenchmarkRef],
    #                            ['Medium', 'SP'],
    #                            ['Tight', 'Pf' , #tightBenchmarkRef]]
    #      The tune operation targets which should be used for this tuning
    #      job. The strings inputs must be part of the ReferenceBenchmark
    #      enumeration.
    #      Instead of an enumeration string (or the enumeration itself),
    #      you can set it directly to a value, e.g.: 
    #        [['Loose97', 'Pd', .97,],['Tight005','Pf',.005]]
    #      This can also be set using a string, e.g.:
    #        [['Loose97','Pd' : '.97'],['Tight005','Pf','.005']]
    #      , which may contain a percentage symbol:
    #        [['Loose97','Pd' : '97%'],['Tight005','Pf','0.5%']]
    #      When set to None, the Pd and Pf will be set to the value of the
    #      benchmark correspondent to the operation level set.
    #"""
    from TuningTools.ReadData import FilterType, Reference, Dataset, BranchCrossEffCollector
    pattern_oFile         = retrieve_kw(kw, 'pattern_oFile',         'tuningData'    )
    efficiency_oFile      = retrieve_kw(kw, 'efficiency_oFile',      NotSet          )
    referenceSgn          = retrieve_kw(kw, 'referenceSgn',          Reference.Truth )
    referenceBkg          = retrieve_kw(kw, 'referenceBkg',          Reference.Truth )
    treePath              = retrieve_kw(kw, 'treePath',              NotSet          )
    efficiencyTreePath    = retrieve_kw(kw, 'efficiencyTreePath',    NotSet          )
    l1EmClusCut           = retrieve_kw(kw, 'l1EmClusCut',           NotSet          )
    l2EtCut               = retrieve_kw(kw, 'l2EtCut',               NotSet          )
    efEtCut               = retrieve_kw(kw, 'efEtCut',               NotSet          )
    offEtCut              = retrieve_kw(kw, 'offEtCut',              NotSet          )
    nClusters             = retrieve_kw(kw, 'nClusters',             NotSet          )
    getRatesOnly          = retrieve_kw(kw, 'getRatesOnly',          NotSet          )
    etBins                = retrieve_kw(kw, 'etBins',                NotSet          )
    etaBins               = retrieve_kw(kw, 'etaBins',               NotSet          )
    ringConfig            = retrieve_kw(kw, 'ringConfig',            NotSet          )
    crossVal              = retrieve_kw(kw, 'crossVal',              NotSet          )
    extractDet            = retrieve_kw(kw, 'extractDet',            NotSet          )
    standardCaloVariables = retrieve_kw(kw, 'standardCaloVariables', NotSet          )
    useTRT                = retrieve_kw(kw, 'useTRT',                NotSet          )
    toMatlab              = retrieve_kw(kw, 'toMatlab',              True            )
    efficiencyValues      = retrieve_kw(kw, 'efficiencyValues',      NotSet          )
    plotMeans             = retrieve_kw(kw, 'plotMeans',             True            )
    plotProfiles          = retrieve_kw(kw, 'plotProfiles',          False           )
    label                 = retrieve_kw(kw, 'label',                 NotSet          )
    supportTriggers       = retrieve_kw(kw, 'supportTriggers',       NotSet          )

    if plotProfiles and _noProfilePlot:
      self._logger.error("Cannot draw profiles! Reason:\n%r", _noProfileImportError)
      plotProfiles = False

    if 'level' in kw: 
      self.level = kw.pop('level') # log output level
      self._reader.level = self.level
    checkForUnusedVars( kw, self._logger.warning )
    # Make some checks:
    if type(treePath) is not list:
      treePath = [treePath]
    if len(treePath) == 1:
      treePath.append( treePath[0] )
    if efficiencyTreePath in (NotSet, None):
      efficiencyTreePath = treePath
    if type(efficiencyTreePath) is not list:
      efficiencyTreePath = [efficiencyTreePath]
    if len(efficiencyTreePath) == 1:
      efficiencyTreePath.append( efficiencyTreePath[0] )
    if etaBins is None: etaBins = npCurrent.fp_array([])
    if etBins is None: etBins = npCurrent.fp_array([])
    if type(etaBins) is list: etaBins=npCurrent.fp_array(etaBins)
    if type(etBins) is list: etBins=npCurrent.fp_array(etBins)
    if efficiency_oFile in (NotSet, None):
      efficiency_oFile = appendToFileName(pattern_oFile,'-eff')

    nEtBins  = len(etBins)-1 if not etBins is None else 1
    nEtaBins = len(etaBins)-1 if not etaBins is None else 1
    #useBins = True if nEtBins > 1 or nEtaBins > 1 else False

    #FIXME: problems to only one bin. print eff doest work as well
    useBins=True


    # Checking the efficiency values
    if efficiencyValues is not NotSet:
      if len(efficiencyValues) == 2 and (type(efficiencyValues[0]) is int or float):
        #rewrite to amatrix form
        efficiencyValues = nEtBins * [ nEtaBins * [efficiencyValues] ]
      else:
        if len(efficiencyValues) != nEtBins:
          self._logger.error(('The number of etBins (%d) does not match with efficiencyValues (%d)')%(nEtBins, len(efficiencyValues)))
          raise ValueError('The number of etbins must match!')
        if len(efficiencyValues[0]) != nEtaBins:
          self._logger.error(('The number of etaBins (%d) does not match with efficiencyValues (%d)')%(nEtaBins, len(efficiencyValues[0])))
          raise ValueError('The number of etabins must match!')
        if len(efficiencyValues[0][0]) != 2:
          self._logger.error('The reference value must be a list with 2 like: [sgnEff, bkgEff]')
          raise ValueError('The number of references must be two!')


    # List of operation arguments to be propagated
    kwargs = { 'l1EmClusCut':           l1EmClusCut,
               'l2EtCut':               l2EtCut,
               'efEtCut':               efEtCut,
               'offEtCut':              offEtCut,
               'nClusters':             nClusters,
               'etBins':                etBins,
               'etaBins':               etaBins,
               'ringConfig':            ringConfig,
               'crossVal':              crossVal,
               'extractDet':            extractDet,
               'standardCaloVariables': standardCaloVariables,
               'useTRT':                useTRT,
               'supportTriggers':       supportTriggers,
             }

    if efficiencyTreePath[0] == treePath[0]:
      self._logger.info('Extracting signal dataset information for treePath: %s...', treePath[0])
      npSgn, sgnEff, sgnCrossEff  = self._reader(sgnFileList,
                                                 ringerOperation,
                                                 filterType = FilterType.Signal,
                                                 reference = referenceSgn,
                                                 treePath = treePath[0],
                                                 **kwargs)
      if npSgn.size: self.__printShapes(npSgn, 'Signal')
    else:
      if not getRatesOnly:
        self._logger.info("Extracting signal data for treePath: %s...", treePath[0])
        npSgn, _, _  = self._reader(sgnFileList,
                                    ringerOperation,
                                    filterType = FilterType.Signal,
                                    reference = referenceSgn,
                                    treePath = treePath[0],
                                    getRates = False,
                                    **kwargs)
        self.__printShapes(npSgn, 'Signal')
      else:
        self._logger.warning("Informed treePath was ignored and used only efficiencyTreePath.")

      self._logger.info("Extracting signal efficiencies for efficiencyTreePath: %s...", efficiencyTreePath[0])
      _, sgnEff, sgnCrossEff  = self._reader(sgnFileList,
                                             ringerOperation,
                                             filterType = FilterType.Signal,
                                             reference = referenceSgn,
                                             treePath = efficiencyTreePath[0],
                                             getRatesOnly = True,
                                             **kwargs)

    if efficiencyTreePath[1] == treePath[1]:
      self._logger.info('Extracting background dataset information for treePath: %s...', treePath[1])
      npBkg, bkgEff, bkgCrossEff  = self._reader(bkgFileList,
                                                 ringerOperation,
                                                 filterType = FilterType.Background,
                                                 reference = referenceBkg,
                                                 treePath = treePath[1],
                                                 **kwargs)
    else:
      if not getRatesOnly:
        self._logger.info("Extracting background data for treePath: %s...", treePath[1])
        npBkg, _, _  = self._reader(bkgFileList,
                                    ringerOperation,
                                    filterType = FilterType.Background,
                                    reference = referenceBkg,
                                    treePath = treePath[1],
                                    getRates = False,
                                    **kwargs)
      else:
        self._logger.warning("Informed treePath was ignored and used only efficiencyTreePath.")

      self._logger.info("Extracting background efficiencies for efficiencyTreePath: %s...", efficiencyTreePath[1])
      _, bkgEff, bkgCrossEff  = self._reader(bkgFileList,
                                             ringerOperation,
                                             filterType = FilterType.Background,
                                             reference = referenceBkg,
                                             treePath = efficiencyTreePath[1],
                                             getRatesOnly= True,
                                             **kwargs)
    if npBkg.size: self.__printShapes(npBkg, 'Background')

    # Rewrite all effciency values
    if efficiencyValues is not NotSet:
      for etBin in range(nEtBins):
        for etaBin in range(nEtaBins):
          for key in sgnEff.iterkeys():
            self._logger.warning(('Rewriting the Efficiency value of %s to %1.2f')%(key, efficiencyValues[etBin][etaBin][0]))
            sgnEff[key][etBin][etaBin].setEfficiency(efficiencyValues[etBin][etaBin][0])
          for key in bkgEff.iterkeys():
            self._logger.warning(('Rewriting the Efficiency value of %s to %1.2f')%(key, efficiencyValues[etBin][etaBin][1]))
            bkgEff[key][etBin][etaBin].setEfficiency(efficiencyValues[etBin][etaBin][1])

    if not getRatesOnly:
      tdArchieve = TuningDataArchieve(pattern_oFile,
                                      signal_patterns = npSgn,
                                      background_patterns = npBkg,
                                      eta_bins = etaBins,
                                      et_bins = etBins,
                                      signal_efficiencies = sgnEff,
                                      background_efficiencies = bkgEff,
                                      signal_cross_efficiencies = sgnCrossEff,
                                      background_cross_efficiencies = bkgCrossEff,
                                      operation = ringerOperation,
                                      toMatlab = toMatlab,
                                      label = label)      

      savedPath=tdArchieve.save()
      self._logger.info('Saved data file at path: %s', savedPath )

      if plotMeans:
        tdArchieve.plotMeanPatterns()
      if plotProfiles:
        tdArchieve.drawProfiles()

    # plot number of events per bin
    if npBkg.size and npSgn.size:
      self.__plotNSamples(npSgn, npBkg)

    for etBin in range(nEtBins):
      for etaBin in range(nEtaBins):
        # plot ringer profile per bin

        for key in sgnEff.iterkeys():
          sgnEffBranch = sgnEff[key][etBin][etaBin] if useBins else sgnEff[key]
          bkgEffBranch = bkgEff[key][etBin][etaBin] if useBins else bkgEff[key]
          self._logger.info('Efficiency for %s: Det(%%): %s | FA(%%): %s', 
                            sgnEffBranch.printName,
                            sgnEffBranch.eff_str(),
                            bkgEffBranch.eff_str() )
          if crossVal not in (None, NotSet):
            for ds in BranchCrossEffCollector.dsList:
              try:
                sgnEffBranchCross = sgnCrossEff[key][etBin][etaBin] if useBins else sgnEff[key]
                bkgEffBranchCross = bkgCrossEff[key][etBin][etaBin] if useBins else bkgEff[key]
                self._logger.info( '%s_%s: Det(%%): %s | FA(%%): %s',
                                  Dataset.tostring(ds),
                                  sgnEffBranchCross.printName,
                                  sgnEffBranchCross.eff_str(ds),
                                  bkgEffBranchCross.eff_str(ds))
              except KeyError, e:
                pass
        # for eff
      # for eta
    # for et
  # end __call__



  def __plotNSamples(self, npArraySgn, npArrayBkg ):
    """Plot number of samples per bin"""
    from ROOT import TCanvas, gROOT, kTRUE, kFALSE, TH2I, TText
    gROOT.SetBatch(kTRUE)
    c1 = TCanvas("plot_patterns_signal", "a",0,0,800,400);
    c1.Draw();
    shape = npArraySgn.shape #npArrayBkg.shape should be the same
    histo1 = TH2I("name1", "Number of Filtered Samples on Signal/Background dataset", shape[0], 0, shape[0], shape[1], 0, shape[1]);
    histo2 = TH2I("name2 ", "" , shape[0], 0, shape[0], shape[1], 0, shape[1]);
    histo1.SetStats(kFALSE);
    histo2.SetStats(kFALSE);
    histo1.Draw("TEXT");
    histo1.GetXaxis().SetLabelSize(0.06);
    histo1.GetYaxis().SetLabelSize(0.06);
    histo1.SetXTitle("E_T");
    histo1.GetXaxis().SetTitleSize(0.04);
    histo1.SetYTitle("#eta");
    histo1.GetYaxis().SetTitleSize(0.05);
    histo1.SetMarkerColor(0);
    ttest = TText();
    for etBin in range(shape[0]):
      for etaBin in range(shape[1]):
        histo1.SetBinContent(etBin+1, etaBin+1, npArraySgn[etBin][etaBin].shape[0]) \
            if npArraySgn[etBin][etaBin] is not None else histo1.SetBinContent(etBin+1, etaBin+1,0)
        histo2.SetBinContent(etBin+1, etaBin+1, npArrayBkg[etBin][etaBin].shape[0]) \
            if npArrayBkg[etBin][etaBin] is not None else histo2.SetBinContent(etBin+1, etaBin+1,0)
        ttest.SetTextColor(4);
        ttest.DrawText(0.29+etBin,0.42+etaBin,"%d" % (histo1.GetBinContent(etBin+1,etaBin+1)));
        ttest.SetTextColor(1);
        ttest.DrawText(0.495+etBin,0.42+etaBin, " / ");
        ttest.SetTextColor(2);
        ttest.DrawText(0.666+etBin,0.42+etaBin,"%d" % (histo2.GetBinContent(etBin+1,etaBin+1)));
        histo1.GetXaxis().SetBinLabel(etBin+1, "Bin %d" % (etBin))
        histo1.GetYaxis().SetBinLabel(etaBin+1, "Bin %d" % (etaBin))
    c1.SetGrid();
    c1.SaveAs("nPatterns.pdf");


  def __printShapes(self, npArray, name):
    "Print numpy shapes"
    if not npArray.dtype.type is np.object_:
      self._logger.info('Extracted %s patterns with size: %r',name, (npArray.shape))
    else:
      shape = npArray.shape
      for etBin in range(shape[0]):
        for etaBin in range(shape[1]):
          self._logger.info('Extracted %s patterns (et=%d,eta=%d) with size: %r', 
                            name, 
                            etBin,
                            etaBin,
                            (npArray[etBin][etaBin].shape if npArray[etBin][etaBin] is not None else ("None")))
        # etaBin
      # etBin

createData = CreateData()

