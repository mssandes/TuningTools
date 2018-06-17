

from TuningTools.monitoring import MonitoringTool

basepath = 'data_jpsi/reports'
dataLocation = 'data_jpsi/files/data_entries.pic.gz'
cnames = [
            '$ref$',
            '$NN_{v9}(rings)$',
          ]


summaryList =  [
            basepath+'/report_tight/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Tight', outname='comparison_ref_v9_tight') 

summaryList =  [
            basepath+'/report_medium/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Medium', outname='comparison_ref_v9_medium') 

summaryList =  [
            basepath+'/report_loose/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Loose', outname='comparison_ref_v9_loose') 


summaryList =  [
            basepath+'/report_vloose/summary.pic.gz',
          ]

MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) VeryLoose', outname='comparison_ref_v9_vloose') 

















