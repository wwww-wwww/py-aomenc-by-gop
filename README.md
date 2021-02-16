# aomenc-by-gop

Runs aomenc in parallel per group of pictures

## Usage

```aomenc-by-gop -i INPUT OUTPUT AOMENC_ARGS```

View help with:  
```aomenc-by-gop --help```

Simple 2 pass encode:  
```aomenc-by-gop -i file.mkv out.mkv```

Other examples:  
```aomenc-by-gop -i input.mkv out.webm --workers 6 --kf-max-dist=120 --threads=8 --cpu-used=6```  
```aomenc-by-gop -i input.mkv out.mkv -s 100 -e 200 --threads=8 --cpu-used=6```  
```aomenc-by-gop -i script.vpy out.mkv -s 100 -e 200 --threads=8 --cpu-used=6```

Enable resuming:  
```aomenc-by-gop -i input.mkv out.mkv --working_dir project```

Save/use onepass keyframes file  
```aomenc-by-gop -i input.mkv out.mkv --keyframes input.txt```

Enable resuming using a different onepass keyframes file  
```aomenc-by-gop -i input.mkv out.mkv --working_dir project --keyframes input.txt```

Use vpxenc:  
```aomenc-by-gop -i input out.mkv --aomenc vpxenc```

## Requirements
- Python 3
- [Vapoursynth](http://www.vapoursynth.com/) R45+
- [ffms2](https://github.com/FFMS/ffms2) or [lsmash](https://github.com/VFR-maniac/L-SMASH-Works) (Available via vsrepo on Windows [lsmas](http://vsdb.top/plugins/lsmas) [ffms2](http://vsdb.top/plugins/ffms2))
- [mkvmerge](https://mkvtoolnix.download/)
