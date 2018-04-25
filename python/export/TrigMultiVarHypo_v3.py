


class TrigMultiVarHypo_v2( Logger ):

  # the athena version
  _version = 3
  # root branches used by the discriminator file
  _discrBranches = [
                   ['unsigned int', 'nodes'  , None],
                   ['unsigned int', 'type'   , None],
                   ['double'      , 'weights', None],
                   ['double'      , 'bias'   , None],
                   ['double'      , 'etBin'  , None],
                   ['double'      , 'etaBin' , None],
                   ['double'      , 'muBin'  , None],
                   ]

  # root branches used by the threshold file
  _thresBranches = [
                   ['double'      , 'thresholds'  , None],
                   ['double'      , 'etBin'       , None],
                   ['double'      , 'etaBin'      , None],
                   ['double'      , 'muBin'       , None],
                   ['unsigned int', 'type'        , None],
                   ]
 
  def __init__( self, **kw ):

    self._toPython = retrieve_kw( kw, 'toPython', True )



  def export_weights( self, discrList )

    for model in discrList:
      
      ## Discriminator configuration
      discrData={}
      discrData['discriminator']={}
      discrData['discriminator']['type']      = [ppType]
      discrData['discriminator']['etBin']     = etBin.tolist()
      discrData['discriminator']['etaBin']    = etaBin.tolist()
      discrData['discriminator']['muBin']     = muBin
      discrData['discriminator']['nodes']     = tolist( discrDict['nodes']   )
      discrData['discriminator']['bias']      = tolist( discrDict['bias']    )
      discrData['discriminator']['weights']   = tolist( discrDict['weights'] )
      discrData['discriminator']['removeOutputTansigTF'] = removeOutputTansigTF
      discrData['threshold'] = {}
      discrData['threshold']['etBin']     = etBin.tolist()
      discrData['threshold']['etaBin']    = etaBin.tolist()
      discrData['threshold']['muBin']     = muBin
      discrData['threshold']['type']      = [thresType]
      discrData['threshold']['thresholds'] = thresValues
      discrData['metadata'] = {}
      discrData['metadata']['pileupThreshold'] = maxPileupCorrection
      discrData['metadata']['useCaloRings'] = useCaloRings
      discrData['metadata']['useTrack'] = useTrack
      discrData['metadata']['useShowerShape'] = useShowerShape
      
   
      discrParams       = { 
                        'UseCaloRings'          :useCaloRings         ,
                        'UseShowerShape'        :useShowerShape       ,
                        'UseTrack'              :useTrack             ,
                        'UseEtaVar'             :False                ,
                        'UseLumiVar'            :False                ,
                        'RemoveOutputTansigTF'  :removeOutputTansigTF ,
                        'UseNoActivationFunctionInTheLastLayer' : removeOutputTansigTF,
                        }

      thresParams       = { 
                        'DoPileupCorrection'                    :doPileupCorrection   ,
                        'LumiCut'                               : maxPileupCorrection,
                        }

    from ROOT import TFile, TTree
    from ROOT import std
    
    ### Create the discriminator root object
    fdiscr = TFile(discrFilename+'.root', 'recreate')
    createRootParameter( 'int'   , '__version__', _version).Write()
    fdiscr.mkdir('tuning') 
    fdiscr.cd('tuning')
    tdiscr = TTree('discriminators','')

    for idx, b in enumerate(discrBranches):
      b[2] = std.vector(b[0])()
      tdiscr.Branch(b[1], 'vector<%s>'%b[0] ,b[2])

    for discr in outputDict:
      for idx, b in enumerate(discrBranches):
        attachToVector( discr['discriminator'][b[1]],b[2])
      tdiscr.Fill()

    tdiscr.Write()

    ### Create the thresholds root object
    fthres = TFile(thresFilename+'.root', 'recreate')
    createRootParameter( 'int'   , '__version__', _version).Write()
    fthres.mkdir('tuning') 
    fthres.cd('tuning')
    tthres = TTree('thresholds','')

    for idx, b in enumerate(thresBranches):
      b[2] = std.vector(b[0])()
      tthres.Branch(b[1], 'vector<%s>'%b[0] ,b[2])

    for discr in outputDict:
      for idx, b in enumerate(thresBranches):
        attachToVector( discr['threshold'][b[1]],b[2])
      tthres.Fill()
 
    tthres.Write()
    fdiscr.mkdir('metadata'); fdiscr.cd('metadata')
    for key, value in discrParams.iteritems():
      logger.info('Saving metadata %s as %s', key, value)
      createRootParameter( 'int' if type(value) is int else 'bool'   , key, value).Write()

    fthres.mkdir('metadata'); fthres.cd('metadata')
    for key, value in thresParams.iteritems():
      logger.info('Saving metadata %s as %s', key, value)
      createRootParameter( 'int' if type(value) is int else 'bool'   , key, value).Write()

    fdiscr.Close()
    fthres.Close()

    return outputDict
  # exportDiscrFilesToDict 



  def self.__attachToVector( self, l, vec ):
    vec.clear()
    for value in l: vec.push_back(value)

  def self.__createRootParameter( self, type_name, name, value):
    from ROOT import TParameter
    return TParameter(type_name)(name,value)

 
