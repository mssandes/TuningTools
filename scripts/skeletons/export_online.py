#!/usr/bin/env python

from RingerCore import LoggingLevel, expandFolders
from TuningTools import CrossValidStatAnalysis, RingerOperation
from pprint import pprint

crossValGrid = expandFolders('tuning/201609XX/nnstat/user.jodafons.nnstat.norm1.data16_13TeV.sgn_pb.284285.rAll.bkg_eb.298967.302956.rAll.trig.eg.e24_lhmedium_nod0_il.t01/',
                              '*.pic')
pprint(crossValGrid)

# 16 bins
configList = [
                # EFCalo, Et 0, Eta 0
                #  Pd, SP, Pf
              [[   7  ],
               [# Et 0, Eta 1
                   16 ],
               [# Et 0, Eta 2
                   7  ],
               [# Et 0, Eta 3
                   5  ]],
                # EFCalo, Et 1, Eta 0
                #  Pd, SP, Pf
              [[   9  ],
               [# Et 1, Eta 1
                   8  ],
               [# Et 1, Eta 2
                   11 ],
               [# Et 1, Eta 3
                   8  ]],
                # EFCalo, Et 2, Eta 0
                #  Pd, SP, Pf
              [[   5  ],
               [# Et 2, Eta 1
                   11 ],
               [# Et 2, Eta 2
                   8  ],
               [# Et 2, Eta 3
                   15 ]],
                # EFCalo, Et 3, Eta 0
                #  Pd, SP, Pf
              [[   10 ],
               [# Et 3, Eta 1
                   17 ],
               [# Et 3, Eta 2
                   8  ],
               [# Et 3, Eta 3
                   16 ]]
            ]
pprint(configList)



refBenchmarkList = [[ # Et 0, Eta 0
                     ["Pf"],
                      # Et 0, Eta 1
                     ["Pf"],
                      # Et 0, Eta 2
                     ["SP"],
                      # Et 0, Eta 3
                     ["Pd"]],
                      # Et 1, Eta 0
                    [["Pf"],
                      # Et 1, Eta 1
                     ["Pf"],
                      # Et 1, Eta 2
                     ["Pd"],
                      # Et 1, Eta 3
                     ["Pf"]],
                      # Et 2, Eta 0
                    [["Pf"],
                      # Et 2, Eta 1
                     ["Pf"],
                      # Et 2, Eta 2
                     ["Pd"],
                      # Et 2, Eta 3
                     ["Pd"]],
                      # Et 3, Eta 0
                    [["Pf"],
                      # Et 3, Eta 1
                     ["Pf"],
                      # Et 3, Eta 2
                     ["Pd"],
                      # Et 3, Eta 3
                     ["Pf"]],
                    ]



pprint(refBenchmarkList)

triggerChains = ['e24_lhmedium']

a = CrossValidStatAnalysis.exportDiscrFiles(crossValGrid,
                                        RingerOperation.L2,
                                        triggerChains=triggerChains,
                                        refBenchCol=refBenchmarkList,
                                        configCol=configList)


