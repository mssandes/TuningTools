

for file in user.*; 
do
  cd ${file}
  for file in *.2; do mv $file ${file//.2} ;done
  for file in *.tgz; do tar xfvz $file; done
  rm -rf *.tgz
  for file in *; do gzip -d $file; done
  cd ..
done
