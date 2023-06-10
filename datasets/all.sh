#!/bin/bash
python dibco_2009.py $1
python dibco_2011.py $1
python dibco_2013.py $1
python dibco_2016.py $1
python dibco_2017.py $1
python dibco_2018.py $1
python hdibco_2010.py $1
python hdibco_2012.py $1
python hdibco_2014.py $1
python mcs.py $1
python phibd_2012.py $1
python msi.py $1
python msi_histodoc.py $1
python denoising_dirty_documents.py $1
python nabuco.py $1

