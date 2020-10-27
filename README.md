# aomenc-by-gop

Runs aomenc in parallel per group of pictures

## Usage

```aomenc-by-gop FILE AOMENC_ARGS```

View help with:  
```aomenc-by-gop --help```

Simple 2 pass encode:  
```aomenc-by-gop file.mkv```

Other examples:  
```aomenc-by-gop input.mkv --workers 6 --kf-max-dist=120 --threads=8 --cpu-used=6 -o final.webm```  
```aomenc-by-gop input.mkv -s 100 -e 200 --threads=8 --cpu-used=6```

## Requirements
- Python 3
- [Vapoursynth](http://www.vapoursynth.com/) R45+
- [ffms2](https://github.com/FFMS/ffms2) or [lsmash](https://github.com/VFR-maniac/L-SMASH-Works)
- [mkvmerge](https://mkvtoolnix.download/)
