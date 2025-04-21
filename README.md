# EECamp 2021 Minidrawing Machine - image2Gcode

## Description
With this code, one can turn image into gcode file, so that minidrawing machine can work on it

## Usage (local)

Please install  packages in requirements.txt

Image to gcode
```bash
$ python genGcode.py
# output will be saved in /out directory
```

Upload gcode to Arduino
```bash
$ python GcodeSender.py
# make sure the PORT is the same as your environment (COM4 for example)
```


