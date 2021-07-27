Pipeline for generating data/label from MuseScore (.mscz) files
1. Run "Batch Convert Resize" in MuseScore on all .mscz files to .musicxml
2. Run removecredits.py on the generated .musicxml files
3. Run "Batch Convert Xml2png" in MuseScore on the cleaned .musicxml files to .mscz
4. Run "Batch Convert Xml2png" in MuseScore on the new .mscz to .musicxml and .png
5. Run removenolabeldata.py, removenonpolyphonic.py, removesparsesamples.py, removetitleimgs.py to clean data as needed