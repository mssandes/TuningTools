

from TuningTools.monitoring import MonitoringTool

basepaths = 'data/monitoring'

summaryList =  [
            'report_tight/summary.pic.gz',
          ]

cnames = [
            '$ref$',
            '$NN_{v9}(rings)$',
          ]

dataLocation = 'data_jpsi/files/data_entries.pic.gz'


MonitoringTool.compareTTsReport(cnames,summaryList,dataLocation,title='Comparison (ref Vs v9) Tight', outname='comparison_ref_v9_tight') 


















