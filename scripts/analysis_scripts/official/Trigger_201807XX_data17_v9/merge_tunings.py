



from TuningTools.export import TrigMultiVarHypo_v2
export = TrigMultiVarHypo_v2()

### export weights

configs = ['Tight','Medium','Loose','VeryLoose']

for pid in configs:
  files = [
          'data/export/jpsi/TrigL2CaloRingerElectron%sConstants.root'%pid,
          'data/export/zee/TrigL2CaloRingerElectron%sConstants.root'%pid,
        ]
  #export.merge_weights( files, 'TrigL2CaloRingerElectron%sConstants'%pid )

  files = [
          'data/correction/correction_jpsi/TrigL2CaloRingerElectron%sThresholds.root'%pid,
          'data/correction/correction_zee/TrigL2CaloRingerElectron%sThresholds.root'%pid,
        ]
  export.merge_thresholds( files, 'TrigL2CaloRingerElectron%sThresholds'%pid )












