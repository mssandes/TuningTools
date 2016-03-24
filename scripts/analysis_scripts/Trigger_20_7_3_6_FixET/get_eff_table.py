#!/usr/bin/env python

crossValGrid = \
            [['/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916634.pic.gz'  
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916635.pic.gz'  
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916636.pic.gz'  
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916637.pic.gz']  
            ,['/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916638.pic.gz'  
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916639.pic.gz'  
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916641.pic.gz'  
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916642.pic.gz']  
            ,['/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916643.pic.gz' 
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916644.pic.gz' 
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916645.pic.gz' 
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916647.pic.gz']
            ,['/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916648.pic.gz'
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916649.pic.gz'
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916650.pic.gz'
            ,'/afs/cern.ch/user/w/wsfreund/Ringer/xAODRingerOfflinePorting/RingerTPFrameWork/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916651.pic.gz']]


#crossValGrid = [['/Users/wsfreund/Documents/Doutorado/CERN/xAOD/RingerProject/root/TuningTools/scripts/skeletons/FixET_Norm1_20.7.3.6_7916634.pic.gz']]
#configBaseList = ['EFCalo']
configBaseList = ['Medium']

configMap = [[
                # EFCalo, Et 0, Eta 0
                #  Pd, SP, Pf
              [[   7,   7,  7   ],
               [# Et 0, Eta 1
                   16,  16,  7  ],
               [# Et 0, Eta 2
                   7,  19,  6   ],
               [# Et 0, Eta 3
                  5,   11, 14   ]],
                # EFCalo, Et 1, Eta 0
                #  Pd, SP, Pf
              [[   9,  10, 7    ],
               [# Et 1, Eta 1
                   8,  16, 7    ],
               [# Et 1, Eta 2
                   11, 11, 6    ],
               [# Et 1, Eta 3
                   8,  6,  6    ]],
                # EFCalo, Et 2, Eta 0
                #  Pd, SP, Pf
              [[   5,  14, 14   ],
               [# Et 2, Eta 1
                   11, 11, 11   ],
               [# Et 2, Eta 2
                   8,  9,  13   ],
               [# Et 2, Eta 3
                   15, 15, 20   ]],
                # EFCalo, Et 3, Eta 0
                #  Pd, SP, Pf
              [[   10, 8,  16   ],
               [# Et 3, Eta 1
                   17, 7,  7    ],
               [# Et 3, Eta 2
                   8,  17, 15   ],
               [# Et 3, Eta 3
                  16,  16,  5  ]]
            ]]

#configMap = [[
#              # Et 0, eta 0
#              [[   5, 5, 5 ]]
#            ]]

from pprint import pprint
pprint(configMap)

from TuningTools.CrossValidStat import CrossValidStatAnalysis
CrossValidStatAnalysis.printTables(configBaseList, 
                                   crossValGrid,
                                   configMap)



