


class BeamerMaker:

  _begin = "
           \documentclass{beamer}
           
           % For more themes, color themes and font themes, see:
           % http://deic.uab.es/~iblanes/beamer_gallery/index_by_theme.html
           %
           \mode<presentation>
           {
             \usetheme{Madrid}       % or try default, Darmstadt, Warsaw, ...
             \usecolortheme{default} % or try albatross, beaver, crane, ...
             \usefonttheme{serif}    % or try default, structurebold, ...
             \setbeamertemplate{navigation symbols}{}
             \setbeamertemplate{caption}[numbered]
           } 
           
           \usepackage[english]{babel}
           \usepackage[utf8x]{inputenc}
           \usepackage{chemfig}
           \usepackage[version=3]{mhchem}
           
           % On Overleaf, these lines give you sharper preview images.
           % You might want to `comment them out before you export, though.
           \usepackage{pgfpages}
           \pgfpagesuselayout{resize to}[%
             physical paper width=8in, physical paper height=6in]
           
           % Here's where the presentation starts, with the info for the title slide
           \title[Molecules in \LaTeX{}]{A short presentation on molecules in \LaTeX{}}
           \author{J. Hammersley}
           \institute{www.overleaf.com}
           \date{\today}
           
           \begin{document}
           
           \begin{frame}
             \titlepage
           \end{frame}
           "

  _end = "\end{document}"



  def __init__(self, outputname):

    self._output = open(outputName+'.tex', 'w')
    self._output.write(self._begin)
    self._output.write(self._end)
    self._output.close()

    


