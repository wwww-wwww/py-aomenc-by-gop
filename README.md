# aomenc-by-gop

Runs aomenc in parallel per group of pictures

## Features

- Frame accurate GOP encoding
- aomenc equivalent GOP selection
- Easy resuming
- [Usage within Python for automation](#example-python-usage)
- Minimal amount of dependencies
- Dark boost

## Requirements

- Python 3
- [Vapoursynth](http://www.vapoursynth.com/) R55+
- [lsmash](https://github.com/VFR-maniac/L-SMASH-Works) or
  [ffms2](https://github.com/FFMS/ffms2)
  (Available via vsrepo on Windows [lsmas](http://vsdb.top/plugins/lsmas)
  [ffms2](http://vsdb.top/plugins/ffms2))
- [mkvmerge](https://mkvtoolnix.download/)

## Usage

`aomenc-by-gop -i INPUT OUTPUT AOMENC_ARGS`

```
aomenc-by-gop --help
usage: aomenc-by-gop [--help] -i INPUT [--workers WORKERS] [--passes PASSES]
                     [--kf-max-dist KF_MAX_DIST] [-u USE] [-s START] [-e END]
                     [-y] [--copy-timestamps] [--timestamps TIMESTAMPS]
                     [--fps FPS] [--mux] [--keyframes KEYFRAMES]
                     [--working-dir WORKING_DIR] [--keep] [--aomenc AOMENC]
                     [--vspipe VSPIPE] [--mkvmerge MKVMERGE]
                     [--mkvextract MKVEXTRACT] [--ranges RANGES] [--webm]
                     [--darkboost] [--darkboost-file DARKBOOST_FILE]
                     [--darkboost-profile DARKBOOST_PROFILE] [--show-segments]
                     output

positional arguments:
  output

optional arguments:
  --help
  -i INPUT, --input INPUT
  --workers WORKERS
  --passes PASSES
  --kf-max-dist KF_MAX_DIST
  -u USE, --use USE     VS source filter (ex. lsmas.LWLibavSource)
  -s START, --start START
                        Input start frame
  -e END, --end END     Input end frame
  -y                    Skip warning / overwrite output.
  --copy-timestamps     Copy timestamps from input file. Support for variable
                        frame rate.
  --timestamps TIMESTAMPS
                        Timestamps file
  --fps FPS             Output framerate (ex. 24000/1001). Use "auto" to
                        determine automatically.
  --mux                 Mux with contents of input file.
  --keyframes KEYFRAMES
                        Path to keyframes file
  --working-dir WORKING_DIR
                        Path to working directory. Allows resuming and does not
                        remove files after completion.
  --keep                Do not delete temporary working directory.
  --aomenc AOMENC       Path to aomenc
  --vspipe VSPIPE       Path to vspipe
  --mkvmerge MKVMERGE   Path to mkvmerge
  --mkvextract MKVEXTRACT
                        Path to mkvmerge. Required for VFR.
  --ranges RANGES       frame_n:arguments;frame_n2:arguments
  --webm
  --darkboost           Enable dark boost.
  --darkboost-file DARKBOOST_FILE
                        Path to darkboost cache
  --darkboost-profile DARKBOOST_PROFILE
                        Available profiles: conservative, light, medium
  --show-segments       Show individual segments' progress.
```

Simple 2 pass encode:  
`aomenc-by-gop -i file.mkv out.mkv`

Other examples:  
`aomenc-by-gop -i input.mkv out.mkv --copy-timestamps --workers 6 --kf-max-dist=120 --threads=8 --cpu-used=6`  
`aomenc-by-gop -i input.mkv out.webm -s 100 -e 200 --threads=8 --cpu-used=6`

Enable resuming:  
`aomenc-by-gop -i input.mkv out.mkv --working-dir project`

Save/use onepass keyframes file  
`aomenc-by-gop -i input.mkv out.mkv --keyframes input.txt`

Enable resuming using a different onepass keyframes file  
`aomenc-by-gop -i input.mkv out.mkv --working-dir project --keyframes input.txt`

Use vpxenc:  
`aomenc-by-gop -i input out.mkv --aomenc vpxenc`

## Example Python Usage

```python
from aomenc_by_gop.app import encode, DefaultArgs

args = {
  "workers": 4,
  "fps": "auto",
  "darkboost": True,
  "input": "flt/01.mkv",
  "output": "enc/01.mkv",
  "working_dir": "gop/01",
  "keyframes": "kf/01.txt",
}

aom_args = [
  "--threads=4",
  "--good",
  "--cpu-used=2",
  "--tile-columns=1",
  "--row-mt=1",
  "--cq-level=8",
  "--end-usage=q",
  "--enable-dnl-denoising=0",
  "--denoise-noise-level=10",
]

ranges = [
  (5000, ["--denoise-noise-level=1"]),  # low noise
  (6000, ["--denoise-noise-level=10"]),
  (7000, ["--cq-level=10"], True),  # ignore dark boost
]

encode(DefaultArgs(**args), aom_args, ranges)
```
