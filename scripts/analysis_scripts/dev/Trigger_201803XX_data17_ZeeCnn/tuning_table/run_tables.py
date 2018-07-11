

from TuningTools.monitoring import MonitoringTool

basepaths = 'data/monitoring'

summaryList =  [
            'report_v8_tight/summary.pic.gz',
            'report_v10_tight/summary.pic.gz',
          ]

cnames = [
            '$ref$',
            '$NN_{v8}(rings)$',
            '$CNN_{v10}(rings)$',
          ]


dataLocation = 'data_v10/files/data_entries.pic.gz'


MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (v8 Vs v10) Tight', outname='comparison_v8_v10_tight') 
#MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='MediumID Comparison', outname='medium_comparison',toPDF=False) 


basepaths = 'data/monitoring'

summaryList =  [
            'report_v8_vloose/summary.pic.gz',
            'report_v10_vloose/summary.pic.gz',
          ]

cnames = [
            '$ref$',
            '$NN_{v8}(rings)$',
            '$CNN_{v10}(rings)$',
          ]


dataLocation = 'data_v10/files/data_entries.pic.gz'


MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (v8 Vs v10) VeryLoose', outname='comparison_v8_v10_vloose') 
#MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='MediumID Comparison', outname='medium_comparison',toPDF=False) 
















