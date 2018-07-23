



from TuningTools.export import TrigMultiVarHypo_v2
export = TrigMultiVarHypo_v2()

### export weights

configs = ['Tight','Medium','Loose','VeryLoose']

for pid in configs:
  files = [
          #'data_jpsi/export/data17_20180125_v8/TrigL2CaloRingerElectron%sConstants.root'%pid,
          'data/export/jpsi/TrigL2CaloRingerElectron%sConstants.root'%pid,
          'data/export/zee/TrigL2CaloRingerElectron%sConstants.root'%pid,
        ]
  export.merge_weights( files, 'TrigL2CaloRingerElectron%sConstants'%pid )

  files = [
          #'data_jpsi/export/data17_20180125_v8/TrigL2CaloRingerElectron%sThresholds.root'%pid,
          'data/export/jpsi/TrigL2CaloRingerElectron%sThresholds.root'%pid,
          'data/export/zee/TrigL2CaloRingerElectron%sThresholds.root'%pid,
        ]
  export.merge_thresholds( files, 'TrigL2CaloRingerElectron%sThresholds'%pid )












