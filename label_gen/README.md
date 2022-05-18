# Label_gen
Code for generating symbolic sequence labels from MuseScore files. An example is "clef-G2 + keySig-FM + timeSig-4/4 + note-A4_quarter..."

# Instructions
1. Make sure you have MuseScore installed and locate the Plugins folder for it
2. Drag the two batch_export folders provided by this repo to the Plugins folder
3. Open Plugins -> Plugin Manager in MuseScore and make sure that two new plugins (Batch Convert Resize Height and Batch Convert Orig) show up
4. Run the pipeline described below

## Pipeline
1. Run "Batch Convert Resize Height" in MuseScore on all .mscz files to .musicxml
2. Run removecredits.py on the generated .musicxml files
3. Run "Batch Convert Orig" in MuseScore on the cleaned .musicxml files to .mscz
4. Run "Batch Convert Orig" in MuseScore on the new .mscz to .musicxml and .png
5. Run removenolabeldata.py, removenonpolyphonic.py, removesparsesamples.py, removetitleimgs.py to clean data (.musicxml) as needed
6. Run genlabels.py as needed to generate labels for the .musicxml files

(Note: Command line arguments for running each Python file is commented at the top of each file)
