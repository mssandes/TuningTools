#!/usr/bin/env python

from RingerCore.Logger import LoggingLevel
from RingerCore.FileIO import expandFolders
from TuningTools.CrossValidStat import CrossValidStatAnalysis
from TuningTools.FilterEvents import RingerOperation
from pprint import pprint

crossValGrid = expandFolders('/Users/wsfreund/Documents/Doutorado/CERN/Online/ana_efetcalo_18032016_retune/','*.pic')
pprint(crossValGrid)


configList = [
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
            ]
pprint(configList)


refBenchmarkList = [["Medium_LH_EFCalo_Pd","Medium_MaxSP","Medium_LH_EFCalo_Pf"]]
pprint(refBenchmarkList)

CrossValidStatAnalysis.exportDiscrFiles(crossValGrid,
                                        RingerOperation.L2,
                                        refBenchCol=refBenchmarkList,
                                        configCol=configList)


