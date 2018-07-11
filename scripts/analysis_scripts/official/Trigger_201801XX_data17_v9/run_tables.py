

from TuningTools.monitoring import MonitoringTool

basepath = 'data_jpsi/reports'
dataLocation = 'data_jpsi/files/data_entries.pic.gz'
cnames = [
            '$ref$',
            '$NN_{v9}(rings,Full)$',
            '$NN_{v9}(rings,EM)$',
          ]


summaryList =  [
            basepath+'/report_FullCalo_tight/summary.pic.gz',
            basepath+'/report_OnlyEMCalo_tight/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Tight', outname='comparison_ref_v9_tight') 
MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Tight', outname='comparison_ref_v9_tight',toPDF=False) 

summaryList =  [
            basepath+'/report_FullCalo_medium/summary.pic.gz',
            basepath+'/report_OnlyEMCalo_medium/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Medium', outname='comparison_ref_v9_medium') 
MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Medium', outname='comparison_ref_v9_medium',toPDF=False) 

summaryList =  [
            basepath+'/report_FullCalo_loose/summary.pic.gz',
            basepath+'/report_OnlyEMCalo_loose/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Loose', outname='comparison_ref_v9_loose') 
MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Loose', outname='comparison_ref_v9_loose',toPDF=False) 


summaryList =  [
            basepath+'/report_FullCalo_vloose/summary.pic.gz',
            basepath+'/report_OnlyEMCalo_vloose/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) VeryLoose', outname='comparison_ref_v9_vloose') 
MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) VeryLoose', outname='comparison_ref_v9_vloose',toPDF=False) 

















