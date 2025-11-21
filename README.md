# aomenc-by-gop

Runs aomenc in parallel per group of pictures

## Features

- Frame accurate GOP encoding
- aomenc equivalent GOP selection
- Resuming
- Dark boost
- Target metric
- Zones

## Requirements

- Python 3
- [Vapoursynth](http://www.vapoursynth.com/) R55+
- [lsmash](https://github.com/VFR-maniac/L-SMASH-Works) or
  [ffms2](https://github.com/FFMS/ffms2)
  (Available via vsrepo on Windows [lsmas](http://vsdb.top/plugins/lsmas)
  [ffms2](http://vsdb.top/plugins/ffms2))
- [mkvmerge](https://mkvtoolnix.download/)

## Example Python Usage

```python
from aomenc_by_gop.app import encode, DefaultArgs

args = {
    "workers": 4,
    "fps": "auto",
    "show_segments": True,
    "use_metric": "ssimulacra2",
    "metric_tmax": 100,
    "metric_tmin": 80,
    "metric_qmin": 5,
    "metric_qmax": 25,
    "metric_q": 15,
    "input": "flt/01.mkv",
    "output": "enc/01.mkv",
    "working_dir": "gop/01",
    "keyframes": "kf/01.txt",
}

aom_args = [
    "aomenc",
    "-",
    f"--ivf",
    "-o",
    "{out}",
    "--passes={passes}",
    "--pass={pass}",
    "--fpf={fpf}",
    "--threads=8",
    "--good",
    "--cpu-used=2",
    "--tile-columns=1",
    "--row-mt=1",
    "--end-usage=q",
    "--cq-level={q}",
]

ranges = [
    (5000, [
        "--denoise-noise-level=1" # add arg
    ], {}),
    (6000, [
        "--denoise-noise-level=10"
    ], {}),
    (7000, [
        ("--cq-level={q}", "--cq-level=10") # replace arg
    ], {"use_metric": None}),
]

encode(DefaultArgs(**args), aom_args, ranges)
```

## Usage (not working, use in-python version)

`aomenc-by-gop -i INPUT OUTPUT AOMENC_ARGS`

```
aomenc-by-gop --help
usage: aomenc-by-gop [--help] [-i INPUT] [--workers WORKERS] [--passes PASSES]
                     [--kf-max-dist KF_MAX_DIST] [-u USE] [-s START] [-e END]
                     [-y] [--priority PRIORITY] [--copy-timestamps]
                     [--timestamps TIMESTAMPS] [--fps FPS] [--mux]
                     [--keyframes KEYFRAMES] [--working-dir WORKING_DIR]
                     [--keep] [--vspipe VSPIPE] [--mkvmerge MKVMERGE]
                     [--mkvextract MKVEXTRACT] [--ranges RANGES] [--webm]
                     [--darkboost] [--darkboost-file DARKBOOST_FILE]
                     [--darkboost-profile DARKBOOST_PROFILE] [--show-segments]
                     [--extra-filter EXTRA_FILTER] [--use-metric USE_METRIC]
                     [--metric-tmax METRIC_TMAX] [--metric-tmin METRIC_TMIN]
                     [--metric-qmin METRIC_QMIN] [--metric-qmax METRIC_QMAX]
                     [--metric-denoise METRIC_DENOISE] [--graph GRAPH]
                     output

positional arguments:
  output

options:
  --help
  -i INPUT, --input INPUT
  --workers WORKERS
  --passes PASSES
  --kf-max-dist KF_MAX_DIST
  -u USE, --use USE     VS source filter (ex. core.lsmas.LWLibavSource("{file}", cachefile="{cachefile}"))
  -s START, --start START
                        Input start frame
  -e END, --end END     Input end frame
  -y                    Skip warning / overwrite output.
  --priority PRIORITY   Process priority
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
  --extra-filter EXTRA_FILTER
                        Extra vapoursynth filtering (ex. cropping). Input and
                        output is clip.
  --use-metric USE_METRIC
  --metric-tmax METRIC_TMAX
  --metric-tmin METRIC_TMIN
  --metric-qmin METRIC_QMIN
  --metric-qmax METRIC_QMAX
  --metric-denoise METRIC_DENOISE
  --graph GRAPH
```

## Extra Filter

Available variables are `vs` for vapoursynth and `clip` for the clip.
