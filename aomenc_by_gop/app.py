import argparse, json, os, platform, re, shutil, subprocess, sys, tempfile
import time, traceback
import vapoursynth as vs
from collections import namedtuple
from functools import partial
from pkg_resources import resource_filename
from rich.progress import Progress, BarColumn, ProgressColumn, Text, TimeElapsedColumn, TimeRemainingColumn
from threading import Condition, Event, Lock, Thread
from typing import Callable, List, Optional

re_keyframe = r"f *([0-9]+):([0|1])"
re_aom_frame = r"Pass *([0-9]+)/[0-9]+ *frame * [0-9]+/([0-9]+)"
re_mkvmerge_track = r"Track ID ([0-9]+?): video"

script_input = """import vapoursynth as vs
clip = vs.core.{}(r\"{}\")
{}
clip.set_output()"""

script_gop = """import vapoursynth as vs
v = vs.core.{}(r\"{}\")
r = v.width / v.height
h = min(720, (v.height / 1.5) // 2 * 2)
w = min(round(r * 720), (v.width / 1.5) // 2 * 2)
v.resize.Bicubic(width=w, height=h, format=vs.YUV420P8).set_output()"""

if hasattr(subprocess, "CREATE_NO_WINDOW"):
  CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
  CREATE_NO_WINDOW = 0

if platform.system() == "Linux":
  import resource
  file_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
  cmd_limit = os.sysconf(os.sysconf_names["SC_ARG_MAX"])
else:
  file_limit = -1
  cmd_limit = 30000

priorities = {
  "-2": 0x00000040,
  "-1": 0x00004000,
  "1": 0x00008000,
  "2": 0x00000080,
}

Segment = namedtuple("Segment", ["start", "end", "n", "args", "info"])


class DefaultArgs:

  def __init__(self, **kwargs):
    self.input = None
    self.output = None
    self.workers = 4
    self.passes = 2
    self.kf_max_dist = 240
    self.use = "lsmas.LWLibavSource"
    self.start = None
    self.end = None
    self.y = False
    self.priority = 0
    self.copy_timestamps = False
    self.timestamps = None
    self.fps = None
    self.mux = False
    self.keyframes = None
    self.working_dir = None
    self.keep = False
    self.aomenc = "aomenc"
    self.vspipe = "vspipe"
    self.mkvmerge = "mkvmerge"
    self.mkvextract = "mkvextract"
    self.webm = False
    self.darkboost = False
    self.darkboost_file = None
    self.darkboost_profile = "conservative"

    # bruteforce only
    self.use_metric = None
    self.metric_tmin = 86
    self.metric_tmax = 84
    self.metric_minq = 0
    self.metric_maxq = 20
    self.metric_denoise = 1

    self.extra_filter = None

    self.show_segments = False

    self.__dict__.update(kwargs)


class Queue:

  def __init__(self, offset_start):
    self.queue = []
    self.lock = Lock()
    self.empty = Condition(self.lock)
    self.update = None
    self.offset_start = offset_start or 0

  def acquire(self, worker) -> Optional[Segment]:
    with self.lock:
      if len(self.queue) == 0:
        self.empty.wait()

      if len(self.queue) == 0:
        return None
      else:
        worker.working.clear()
        job = self.queue.pop(0)
        if self.update:
          self.update()
        if len(self.queue) == 0:
          self.empty.notify()
        return job

  def wait_empty(self):
    with self.lock:
      while len(self.queue) > 0:
        self.empty.wait()

  def submit(self, start: int, end: int, i: int, args: List[str],
             extra_args: List[str]) -> None:
    segment_info = " ".join([f"{a}:{b}" for a, b in extra_args])
    segment_info = segment_info + " " if segment_info else ""
    segment = Segment(start=self.offset_start + start,
                      end=self.offset_start + end,
                      n=i,
                      args=args,
                      info=segment_info)
    with self.lock:
      self.queue.append(segment)
      self.queue.sort(key=lambda x: x[0] - x[1])
      if self.update:
        self.update()
      self.empty.notify()


def replace_args(args1: List[str], args2: List[str]) -> List[str]:
  args2_s = [arg.split("=")[0] for arg in args2]
  new_args = [
    arg for arg in args1 if not any(arg.startswith(arg2) for arg2 in args2_s)
  ]
  new_args += args2
  return new_args


def get_cq(args):
  for arg in args:
    if arg.startswith("--cq-level="):
      return int(arg.split("=")[1])
  return None


def replace_cq(args, cq):
  new_args = [arg for arg in args if not arg.startswith("--cq-level=")]
  new_args.append(f"--cq-level={cq}")
  return new_args


def score_worse(metric, score, target):
  if metric == "butteraugli":
    return score > target

  return score < target


def list_e(items):
  return " ".join(list(filter(None, items)))


def score_worst(args, a, b):
  if a is None: return b

  if args.use_metric == "butteraugli":
    return max(a, b)

  return min(a, b)


class Tester:

  def __init__(self, args, clip):
    self.source_filter = args.source_filter
    self.metric = args.use_metric.lower()
    self.denoise = args.metric_denoise

    self.clip = clip

    if self.denoise:
      self.clip = self.clip.dfttest.DFTTest(sigma=self.denoise)

    self.clip = self.clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
    self.clip = self.clip.fpng.Write(filename="orig.png", overwrite=1)
    self.lock = Lock()

  def get(self, path, frame1, frame2):
    try:
      distorted = self.source_filter(path, cache=0)
    except:
      return None
    if len(distorted) < 3: return None
    if len(distorted) <= frame2: return None
    with self.lock:
      if self.denoise:
        distorted = distorted.dfttest.DFTTest(sigma=1)
      distorted = distorted.resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
      distorted = distorted.fpng.Write(filename="dist.png", overwrite=1)

      self.clip.get_frame(frame1)
      try:
        distorted.get_frame(frame2)
      except:
        return None
      t = subprocess.run([self.metric, "orig.png", "dist.png"],
                         capture_output=True)

      return float(t.stdout.decode("utf-8"))


class Stats:

  def __init__(self, filename):
    self.filename = filename
    self.lock = Lock()
    self.stats = {}
    if os.path.exists(self.filename):
      with open(self.filename, "r") as f:
        self.stats = json.load(f)

  def update(self, key, value):
    with self.lock:
      self.stats[str(key)] = value
      with open(self.filename, "w+") as f:
        json.dump(self.stats, f, indent=2, sort_keys=True)


class Worker:

  def __init__(self, args, queue, passes, script, update, progress):
    self.args = args
    self.queue = queue
    self.passes = passes
    self.script = script
    self.update = update
    self.progress = progress

    self.vspipe = None
    self.pipe = None
    self.working = Event()
    self.working.set()
    self.stopped = False
    self.segment = None
    self.state_lock = Lock()
    self.state_ev = Condition(self.state_lock)
    self.starting = False

    self.current_cq = -1
    self.last_cq = None
    self.last_file = None
    self.change = 0
    self.freeze = False
    self.last_score = None

    self.task = None

    self.pass1 = False

    Thread(target=self.loop, daemon=True).start()

  def update_description(self, items, reset=False):
    if reset:
      self.progress.update(self.task, completed=0)

    self.progress.update(self.task, description=list_e(items))

  def encode(self, segment: Segment):
    if self.args.show_segments:
      self.task = self.progress.add_task(
        list_e([
          f"{segment.n:4d}",
          f"{segment.start:5d}-{segment.end:<5d}",
          segment.info,
          str(self.current_cq),
          f"{self.last_score:.2f}" if self.last_score else "",
        ]),
        total=segment.end - segment.start + 1,
      )

    vspipe_cmd = [
      self.args.vspipe, self.script, "-c", "y4m", "-", "-s",
      str(segment.start), "-e",
      str(segment.end)
    ]

    segment_tmp = tempfile.mktemp(dir=self.args._working_dir,
                                  suffix=f".{self.args.segment_ext}")

    segment_args = segment.args
    if self.args.use_metric:
      segment_args = replace_cq(segment_args, self.current_cq)

    aomenc_cmd = [
      self.args.aomenc,
      "-",
      f"--{self.args.segment_ext}",
      "-o",
      segment_tmp,
      f"--passes={self.passes}",
    ] + segment_args

    for p in range(self.passes):
      if p == 0 and self.pass1: continue

      pass_cmd = aomenc_cmd + [f"--pass={p + 1}"]
      if self.passes == 2:
        segment_fpf = os.path.join(self.args._working_dir,
                                   f"segment_{segment.n}.log")
        pass_cmd.append(f"--fpf={segment_fpf}")
        if p == 0:
          pass_cmd = [
            a for a in pass_cmd if not a.startswith("--denoise-noise-level")
          ]

      frame = 0
      s_output = []
      try:
        self.vspipe = subprocess.Popen(vspipe_cmd,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL,
                                       creationflags=CREATE_NO_WINDOW)

        self.pipe = subprocess.Popen(pass_cmd,
                                     stdin=self.vspipe.stdout,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True,
                                     creationflags=self.args.priority)

        with self.state_lock:
          self.starting = False
          self.state_ev.notify_all()

        segment_length = segment.end - segment.start

        test_frames = [
          0,
          int(segment_length * .1),
          int(segment_length * .3),
          int(segment_length * .5),
          int(segment_length * .7),
          int(segment_length * .9),
        ]
        test_frames = sorted(list(set(test_frames)))
        min_score = None

        if self.args.show_segments:
          self.update_description([
            f"{segment.n:4d}", f"{segment.start:5d}-{segment.end:<5d}",
            segment.info,
            str(p + 1),
            str(self.current_cq),
            f"{self.last_score:.2f}" if self.last_score else "",
            str(test_frames[0]) if test_frames else "",
            "f" if self.freeze else ""
          ], True)

        redo = False

        while True:
          line = self.pipe.stdout.readline().strip()

          if len(line) == 0 and self.pipe.poll() is not None:
            break

          if not line: continue

          s_output.append(line)
          match = re.match(re_aom_frame, line.strip())
          if match:
            current_pass = int(match.group(1))
            new_frame = int(match.group(2))
            if new_frame > frame:
              if self.args.show_segments:
                self.progress.update(self.task, advance=new_frame - frame)

              if current_pass == self.passes:
                self.update(new_frame - frame)

              frame = new_frame

              while not self.freeze and p == 1 and test_frames and frame - test_frames[
                  0] > 1:
                score = self.args.tester.get(segment_tmp,
                                             test_frames[0] + segment.start,
                                             test_frames[0])
                if score is None: break
                test_frames.pop(0)

                new_min_score = score_worst(self.args, min_score, score)
                if new_min_score == min_score: continue
                min_score = new_min_score

                self.update_description([
                  f"{segment.n:4d}",
                  f"{segment.start:5d}-{segment.end:<5d}",
                  segment.info,
                  str(p + 1),
                  str(self.current_cq),
                  f"{self.last_score:.2f}" if self.last_score else "",
                  f"{min_score:.2f}" if min_score else "",
                  str(test_frames[0]) if test_frames else "",
                  "f" if self.freeze else "",
                ])

                if self.current_cq > self.args.metric_minq and score_worse(
                    self.args.use_metric, min_score,
                    self.args.metric_tmin):  # score is bad
                  if self.change == 1:  # regression
                    self.current_cq = self.last_cq
                    min_score = self.last_score
                    segment_tmp = self.last_file
                    self.freeze = True
                    break

                  redo = True
                  self.last_cq = self.current_cq
                  self.last_score = min_score
                  return 2

          if self.freeze: break
      except:
        print(traceback.format_exc())
      finally:
        if self.pipe.returncode != 0:
          self.update(-frame)
          if self.stopped: return False
          if redo:
            os.remove(segment_tmp)
          elif not self.freeze:
            if s_output:
              print(" ".join(vspipe_cmd), "|", " ".join(pass_cmd))
              print("\n" + "\n".join(s_output))
            return False

        self.pipe.kill()
        self.vspipe.kill()
        self.pipe = None
        self.vspipe = None

      if p == 0:
        self.pass1 = True

    if self.args.use_metric and not self.freeze:
      while test_frames:
        score = self.args.tester.get(segment_tmp,
                                     test_frames[0] + segment.start,
                                     test_frames[0])
        min_score = score_worst(self.args, min_score, score)
        test_frames.pop(0)

      if self.current_cq > self.args.metric_minq and score_worse(
          self.args.use_metric, min_score, self.args.metric_tmin):
        os.remove(segment_tmp)
        if self.last_file:  # regression
          self.current_cq = self.last_cq
          min_score = self.last_score
          segment_tmp = self.last_file
        else:
          self.update(-(segment.end - segment.start))
          return 2

      elif self.change >= 0 and self.current_cq < self.args.metric_maxq and not score_worse(
          self.args.use_metric, min_score, self.args.metric_tmax):
        if self.last_file:
          os.remove(self.last_file)
        self.last_cq = self.current_cq
        self.last_file = segment_tmp
        self.last_score = min_score
        self.update(-(segment.end - segment.start))
        return 3

    segment_output = os.path.join(
      self.args._working_dir, f"segment_{segment.n}.{self.args.segment_ext}")
    shutil.move(segment_tmp, segment_output)

    if self.last_file and self.last_file != segment_tmp:
      os.remove(self.last_file)

    if self.args.use_metric:
      return [min_score]

    return True

  def _encode(self, segment: Segment) -> bool:
    if self.args.use_metric:
      self.current_cq = get_cq(segment.args)
      self.last_cq = None
      self.last_file = None
      self.change = 0
      self.freeze = False
      self.last_score = None

    self.pass1 = False
    tries = 3

    while True:
      if tries == 0:
        break

      with self.state_lock:
        self.starting = True

      resp = self.encode(segment)

      if self.task:
        self.progress.remove_task(self.task)
        self.task = None

      if not resp:
        tries -= 1
        self.pass1 = False
      elif resp == 2:
        self.current_cq -= 1
        self.change = -1
      elif resp == 3:
        self.last_cq = self.current_cq
        self.current_cq += 1
        self.change = 1
      else:
        if self.args.use_metric:
          self.args.stats.update(segment.n, (self.current_cq, resp[0]))
        return True

      if self.stopped: return True

    return False

  def loop(self):
    while not self.stopped:
      try:
        self.segment = self.queue.acquire(self)
        if self.segment:
          self.update()
          if not self._encode(self.segment):
            exit(1)  # catastrophic failure
      finally:
        self.segment = None
        self.update()
        self.working.set()

  def kill(self):
    with self.state_lock:
      if self.starting:
        self.state_ev.wait()

    self.stopped = True

    if self.pipe:
      self.pipe.kill()
    if self.vspipe:
      self.vspipe.kill()


class FPSColumn(ProgressColumn):

  def render(self, task: "Task") -> Text:
    speed = task.finished_speed or task.speed
    if speed is None:
      return Text("?", style="progress.data.speed")
    return Text(f"{speed:.2f}fps", style="progress.data.speed")


class DarkBoost:

  def __init__(self, clip: vs.VideoNode, cachefile: str):
    self.state = None
    self.cache = {}
    self.cachefile = cachefile
    if os.path.exists(self.cachefile):
      self.cache = json.load(open(self.cachefile, "r"))

    clip = clip.std.SplitPlanes()[0]
    clip = clip.resize.Bicubic(format=vs.GRAY8)
    clip = clip.std.Levels(min_in=16, max_in=235, min_out=0, max_out=255)
    self.clip = clip

  def count(self, frame: int, threshold: int) -> List[float]:
    key_frame = str(frame)
    if key_frame not in self.cache:
      self.cache[key_frame] = {}

    key_threshold = str(threshold)

    if key_threshold not in self.cache[key_frame]:
      prop = self.clip.std.Expr(f"x {threshold} < 255 0 ?")
      prop = prop.std.PlaneStats()

      def brightness(n, clip, state, f):
        state["val"] = f.props["PlaneStatsAverage"]
        return clip

      state = {"val": None}
      clip_eval = self.clip.std.FrameEval(partial(brightness,
                                                  clip=self.clip,
                                                  state=state),
                                          prop_src=prop)
      clip_eval.get_frame(frame)
      self.cache[key_frame][key_threshold] = state["val"]

      json.dump(self.cache, open(self.cachefile, "w+"), indent=2)

    return self.cache[key_frame][key_threshold]

  def get(self, frames: List[int], extra_args: List, profile: str) -> None:
    frames = list(set(frames))

    if profile == "conservative":
      if any(self.count(frame, 32) > 0.25 for frame in frames):
        extra_args.append(("cq", -2))
      elif any(self.count(frame, 64) > 0.25 for frame in frames):
        extra_args.append(("cq", -1))
    elif profile == "light":
      if any(self.count(frame, 32) > 0.25 for frame in frames):
        extra_args.append(("cq", -2))
      elif any(self.count(frame, 64) > 0.25 for frame in frames):
        extra_args.append(("cq", -1))
      elif all(self.count(frame, 160) < 0.33 for frame in frames):
        extra_args.append(("cq", 1))
    elif profile == "medium":
      if any(self.count(frame, 32) > 0.25 for frame in frames):
        extra_args.append(("cq", -3))
      elif any(self.count(frame, 48) > 0.25 for frame in frames):
        extra_args.append(("cq", -2))
      elif any(self.count(frame, 64) > 0.25 for frame in frames):
        extra_args.append(("cq", -1))
      elif all(self.count(frame, 224) < 0.33 for frame in frames):
        extra_args.append(("cq", 2))
      elif all(self.count(frame, 160) < 0.33 for frame in frames):
        extra_args.append(("cq", 1))


def concat(args, n_segments):
  print("\nConcatenating")
  segments = [f"segment_{n + 1}.{args.segment_ext}" for n in range(n_segments)]
  segments = [os.path.join(args._working_dir, segment) for segment in segments]

  for segment in segments:
    if not os.path.exists(segment):
      raise Exception(f"Segment {segment} is missing")

  out = _concat(args.mkvmerge, args._working_dir, segments, args.output)

  if not (args.copy_timestamps or args.timestamps or args.mux or args.fps):
    os.replace(out, args.output)
    return segments

  merge = [args.mkvmerge, "-o", args.output]

  if args.copy_timestamps:
    print("Getting track id")
    trackid = 0
    r = subprocess.run([args.mkvmerge, "--identify", args.input],
                       capture_output=True,
                       universal_newlines=True)
    if r.returncode != 0:
      print(r.stdout)
      exit(1)
    assert r.returncode == 0
    lines = r.stdout.splitlines()
    tracks = [re.match(re_mkvmerge_track, line) for line in lines]
    tracks = [track for track in tracks if track]
    assert len(tracks) > 0
    trackid = tracks[0].group(1)

    print("Extracting timestamps for track", trackid)
    path_timestamps = os.path.join(args._working_dir, "timestamps.txt")

    extract = [
      args.mkvextract,
      args.input,
      "timestamps_v2",
      f"{trackid}:{path_timestamps}",
    ]

    assert subprocess.run(extract).returncode == 0

    if args._start:
      with open(path_timestamps, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        ts = lines[1:]
        ts = [float(line) for line in ts]
        ts = ts[args._start:]

      with open(path_timestamps, "w+") as f:
        f.write(lines[0] + "\n")
        f.writelines([str(line) + "\n" for line in ts])

  if args.timestamps:
    path_timestamps = args.timestamps

  if args.timestamps or args.copy_timestamps:
    merge += ["--timestamps", f"0:{path_timestamps}"]

  if args.fps:
    merge += [
      "--default-duration", f"0:{args.fps}fps",
      "--fix-bitstream-timing-information", "0"
    ]

  merge += [out]

  if args.mux:
    merge += ["-D", args.input]

  print("Merging")
  assert subprocess.run(merge).returncode == 0

  return segments


def _concat(mkvmerge: str,
            cwd: str,
            files: List[str],
            output: str,
            flip: bool = False) -> str:
  tmp_out = os.path.join(cwd, f"{output}.tmp{int(flip)}.mkv")
  cmd = [mkvmerge, "-o", tmp_out, files[0]]

  remaining = []
  for i, file in enumerate(files[1:]):
    new_cmd = cmd + [f"+{file}"]
    if sum(len(s)
           for s in new_cmd) < cmd_limit and (file_limit == -1
                                              or i < max(1, file_limit - 10)):
      cmd = new_cmd
    else:
      remaining = files[i + 1:]
      break

  assert subprocess.run(cmd).returncode == 0

  if len(remaining) > 0:
    return _concat(mkvmerge, cwd, [tmp_out] + remaining, output, not flip)
  else:
    return tmp_out


def get_attr(obj, attr, ex=False):
  try:
    s = obj
    for ns in attr.split("."):
      s = getattr(s, ns)
  except AttributeError as e:
    if ex: raise e
    return None
  return s


def get_source_filter(core):
  source_filter = get_attr(core, "lsmas.LWLibavSource")
  if source_filter:
    return "lsmas.LWLibavSource", source_filter

  source_filter = get_attr(core, "lsmas.LSMASHVideoSource")
  if source_filter:
    return "lsmas.LSMASHVideoSource", source_filter

  source_filter = get_attr(core, "ffms2.Source")
  if source_filter:
    return "ffms2.Source", source_filter

  raise Exception("No source filter found")


def require_exec(file: str, default: Optional[str] = None) -> str:
  path = shutil.which(file)
  if not path:
    if default:
      path = require_exec(default[0](*default[1]))
    else:
      print(file, "not found. Exiting.")
      exit(1)

  return path


def parse_args(args) -> vs.VideoNode:
  args.segment_ext = "webm" if args.webm else "ivf"
  args.input = os.path.abspath(args.input)
  args.workers = int(args.workers)
  args.passes = int(args.passes)
  args.kf_max_dist = int(args.kf_max_dist)
  args.min_dist = 24

  core = vs.core

  if args.use:
    args.source_filter = get_attr(core, args.use, True)
  else:
    args.use, args.source_filter = get_source_filter(core)
    print(f"Using {args.use} as source filter")

  if os.path.exists(args.output) and not args.y:
    try:
      overwrite = input(f"{args.output} already exists. Overwrite? [y/N] ")
    except KeyboardInterrupt:
      print("Not overwriting, exiting.")
      exit(0)

    if overwrite.lower().strip() != "y":
      print("Not overwriting, exiting.")
      exit(0)

  video = args.source_filter(args.input)

  num_frames = video.num_frames

  args._start = args.start
  args.start = int(args.start or 0)
  args.end = int(args.end or num_frames - 1)
  video = video[args.start:args.end + 1]

  if args.end >= num_frames or (args.start and args.end < args.start):
    raise Exception("End frame out of bounds")

  args.num_frames = args.end - args.start + 1

  args._working_dir = args.working_dir or tempfile.mkdtemp(dir=os.getcwd())
  print("Working directory:", args._working_dir)

  if not os.path.isdir(args._working_dir):
    os.mkdir(args._working_dir)

  if args.keyframes:
    args._keyframes = args.keyframes
    print("Using keyframes file:", args.keyframes)
  else:
    args._keyframes = os.path.join(args._working_dir, "keyframes.txt")

  return video


def parse_keyframe(line: str, frame: int, start: int, num_frames: int,
                   offset: int, min_dist: int, max_dist: int,
                   add_job: Callable[[int, int], int]):
  match = re.match(re_keyframe, line.strip())
  if not match: return False, frame, start, 0

  frame = int(match.group(1))
  if frame >= num_frames:
    return False, frame, start, 0

  frame += offset
  frame_type = int(match.group(2))

  completed = 0

  while frame - start > max_dist * 2:
    completed += add_job(start, start + max_dist)
    start += max_dist

  length = frame - start
  if frame - offset > 0 and frame_type == 1:
    if length > max_dist:
      completed += add_job(start, start + int(length / 2))
      completed += add_job(start + int(length / 2), frame)
      start = frame
    elif length > min_dist:
      completed += add_job(start, frame)
      start = frame

  return True, frame, start, frame_type, completed


def encode(args, aom_args, ranges):
  args.aomenc = require_exec(args.aomenc)
  args.vspipe = require_exec(args.vspipe)
  args.mkvmerge = require_exec(args.mkvmerge)

  if sys.platform == "win32" or sys.platform == "cygwin":
    onepass_keyframes = "bin/win64/onepass_keyframes.exe"
  else:
    onepass_keyframes = "bin/linux_amd64/onepass_keyframes"

  onepass_keyframes = require_exec("onepass_keyframes",
                                   (resource_filename,
                                    ("aomenc_by_gop", onepass_keyframes)))

  clip = parse_args(args)
  print(str(clip))

  if args.copy_timestamps:
    if args.timestamps:
      print("Can't have --timestamps and --copy-timestamps")
      exit(1)
    args.mkvextract = require_exec(args.mkvextract)

  if args.fps:
    if args.timestamps:
      print("Can't have --fps and --timestamps")
      exit(1)

  if args.fps == "auto":
    args.fps = str(clip.fps)

  if args.timestamps and not os.path.isfile(args.timestamps):
    print("Timestamps file not found:", args.timestamps)
    exit(1)

  extras = [f"Workers: {args.workers}", f"Passes: {args.passes}"]

  if args.fps:
    extras.append(f"Output framerate: {args.fps}")

  if args.darkboost:
    extras.append(f"Darkboost: {args.darkboost_profile}")
    if args.darkboost_file:
      darkboost_path = args.darkboost_file
    else:
      filename = os.path.basename(args.input) + "_darkboost.json"
      darkboost_path = os.path.join(os.path.dirname(args.input), filename)
    darkboost = DarkBoost(clip, darkboost_path)

  if args.use_metric:
    extras.append(" ".join([
      args.use_metric,
      f"{args.metric_tmin}<{args.metric_tmax}",
      f"{args.metric_minq}<{get_cq(aom_args)}<{args.metric_maxq}",
    ]))
    args.tester = Tester(args, clip)
    args.stats = Stats(os.path.join(args._working_dir, "stats.json"))

  print("aomenc:", args.aomenc)
  print("vspipe:", args.vspipe)
  print("mkvmerge:", args.mkvmerge)
  print("onepass_keyframes:", onepass_keyframes)
  print(" | ".join(extras))
  print("Encoder arguments:", " ".join(aom_args))

  if "--enable-keyframe-filtering=0" not in aom_args:
    print("WARNING: --enable-keyframe-filtering=0 is not set")

  if CREATE_NO_WINDOW and args.priority:
    args.priority = priorities[str(args.priority)]
  else:
    args.priority = 0

  script_name = os.path.join(args._working_dir, "video.vpy")

  with open(script_name, "w+") as script_f:
    script_f.write(
      script_input.format(args.use, args.input, args.extra_filter or ""))

  script_name_gop = os.path.join(args._working_dir, "gop.vpy")

  with open(script_name_gop, "w+") as script_f:
    script_f.write(script_gop.format(args.use, args.input))

  if args.workers <= 0:
    print("Number of workers set to 0, only getting keyframes")

  with Progress(
      "[progress.description]{task.description}",
      BarColumn(None),
      "[progress.percentage]{task.percentage:>3.0f}%",
      "[progress.download]{task.completed}/{task.total}",
      FPSColumn(),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      expand=True,
  ) as progress:
    task_firstpass = progress.add_task("Keyframes",
                                       total=args.num_frames,
                                       queue=False)
    task_encode = progress.add_task("", total=args.num_frames, queue=True)

    queue = Queue(args.start)
    workers = []
    frame = 0

    def update(n=None):
      active_workers = [worker for worker in workers if worker.segment]
      s = f"queue: {len(queue.queue)} workers: {len(active_workers)}"
      if frame < args.num_frames - 1:
        progress.update(task_firstpass, completed=frame)

      if n:
        progress.update(task_encode, advance=n, refresh=True, description=s)
      else:
        progress.update(task_encode, refresh=True, description=s)

    update()

    queue.update = update

    segment_count = [0]
    start = 0
    offset = 0
    output_log = []

    def add_job(start_frame: int, end_frame: int):
      segment_count[0] += 1
      segment_output = os.path.join(
        args._working_dir, f"segment_{segment_count[0]}.{args.segment_ext}")
      if os.path.isfile(segment_output):
        return end_frame - start_frame
      else:
        segment_args = aom_args.copy()

        extra_args = []
        if args.darkboost:
          segment_length = end_frame - start_frame
          db_frames = [
            int(start_frame + segment_length * .25),
            int(start_frame + segment_length * .5),
            int(start_frame + segment_length * .75),
          ]
          darkboost.get(db_frames, extra_args, args.darkboost_profile)

        # apply ranges and extra args
        seg_ranges = [r for r in ranges if r[0] <= start_frame + args.start]
        if seg_ranges:
          segment_args = replace_args(segment_args, seg_ranges[-1][1])

        for extra_arg in extra_args:
          if extra_arg[0] == "cq":
            cq_arg = [arg.split("=") for arg in segment_args]
            cq_arg = [arg for arg in cq_arg if arg[0] == "--cq-level"]
            if cq_arg:
              new_cq = int(cq_arg[0][1]) + extra_arg[1]
              segment_args = replace_args(segment_args,
                                          [f"--cq-level={new_cq}"])

        if seg_ranges and len(seg_ranges[-1]) > 2 and seg_ranges[-1][2]:
          segment_args = replace_args(segment_args, seg_ranges[-1][1])

        queue.submit(start_frame, end_frame - 1, segment_count[0],
                     segment_args, extra_args)

      return 0

    if args._keyframes and os.path.isfile(args._keyframes):
      completed = 0
      with open(args._keyframes, "r") as f:
        for line in f.readlines():
          _, frame, start, _, _completed = parse_keyframe(
            line, frame, start, args.num_frames, offset, args.min_dist,
            args.kf_max_dist, add_job)
          completed += _completed

      progress.update(task_encode, completed=completed)
      update()

    for worker_id in range(args.workers):
      workers.append(
        Worker(args, queue, args.passes, script_name, update, progress))

    if frame < args.num_frames - 1:
      offset = max(0, frame - 3)
      args.start += offset
    else:
      args.start = frame

    if args.start < args.end:
      get_gop = [args.vspipe, script_name_gop, "-c", "y4m", "-"]

      if args.start > 0:
        get_gop.extend(["-s", str(args.start)])

      if args.end < args.num_frames - 1:
        get_gop.extend(["-e", str(args.end)])

      gop_lines = []
      try:
        with open(args._keyframes, "a+") as keyframes_file:
          vspipe_pipe = subprocess.Popen(get_gop,
                                         stdout=subprocess.PIPE,
                                         creationflags=CREATE_NO_WINDOW)

          pipe = subprocess.Popen(onepass_keyframes,
                                  stdin=vspipe_pipe.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True,
                                  creationflags=CREATE_NO_WINDOW)

          while True:
            line = pipe.stderr.readline()

            if len(line) == 0 and pipe.poll() is not None:
              break

            output_log.append(line)

            match, frame, start, frame_type, _ = parse_keyframe(
              line, frame, start, args.num_frames, offset, args.min_dist,
              args.kf_max_dist, add_job)

            if match and (offset == 0 or frame - offset > 3):
              if args._keyframes:
                gop_lines.append(f"f {frame}:{frame_type}\n")
                if frame_type:
                  keyframes_file.writelines(gop_lines)
                  keyframes_file.flush()
                  gop_lines.clear()
              update()

          if pipe.returncode == 0:
            if frame < args.num_frames - 1:
              gop_lines.extend([
                f"f {frame + i + 1}:0\n"
                for i in range(args.num_frames - frame - 1)
              ])
            keyframes_file.writelines(gop_lines)
            keyframes_file.flush()

      except KeyboardInterrupt:
        print("\nCancelled")
        graceful_exit = True
      except:
        print(traceback.format_exc())
      finally:
        vspipe_pipe.kill()
        pipe.kill()
        if pipe.returncode != 0:
          for worker in workers:
            worker.kill()

          for worker in workers:
            worker.working.wait()

          if "graceful_exit" not in locals():
            print()
            print(" ".join(get_gop))
            print("".join(output_log))
          exit(1)

    if args.num_frames > start:
      while args.num_frames - start > args.kf_max_dist * 2:
        add_job(start, start + args.kf_max_dist)
        start += args.kf_max_dist

      length = args.num_frames - start
      if length > args.kf_max_dist:
        add_job(start, start + int(length / 2))
        add_job(start + int(length / 2), args.num_frames)
      else:
        add_job(start, args.num_frames)

    update()

    if len(workers) <= 0:
      print("Completed getting keyframes")
      return

    progress.remove_task(task_firstpass)

    try:
      queue.wait_empty()
      for worker in workers:
        worker.working.wait()
    except KeyboardInterrupt:
      print("\nCancelled")
      for worker in workers:
        worker.kill()
      for worker in workers:
        worker.working.wait()
      exit(1)

  segments = concat(args, segment_count[0])

  if not args.working_dir and not args.keep:
    print("Cleaning up")
    # remove temporary files
    tmp_files = [
      os.path.join(args._working_dir, f"{args.output}.tmp0.mkv"),
      os.path.join(args._working_dir, f"{args.output}.tmp1.mkv"),
      os.path.join(args._working_dir, "gop.vpy"),
      os.path.join(args._working_dir, "video.vpy"),
      os.path.join(args._working_dir, "timestamps.txt"),
    ]

    for file in tmp_files:
      if os.path.exists(file):
        os.remove(file)

    for segment in segments:
      os.remove(segment)
      if args.passes == 2:
        fpf = f"{os.path.splitext(segment)[0]}.log"
        os.remove(fpf)

    if not args.keyframes:
      os.remove(args._keyframes)

    os.rmdir(args._working_dir)

  print("Completed")


def main():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--help", action="help")

  parser.add_argument("-i", "--input", required=True)
  parser.add_argument("output")
  parser.add_argument("--workers", default=4)
  parser.add_argument("--passes", default=2)
  parser.add_argument("--kf-max-dist", default=240)
  parser.add_argument("-u",
                      "--use",
                      help="VS source filter (ex. lsmas.LWLibavSource)")
  parser.add_argument("-s", "--start", default=None, help="Input start frame")
  parser.add_argument("-e", "--end", default=None, help="Input end frame")
  parser.add_argument("-y",
                      help="Skip warning / overwrite output.",
                      action="store_true")

  if CREATE_NO_WINDOW:
    parser.add_argument("--priority", default=0, help="Process priority")

  parser.add_argument("--copy-timestamps",
                      default=False,
                      action="store_true",
                      help="Copy timestamps from input file.\n" \
                      "Support for variable frame rate.")
  parser.add_argument("--timestamps", default=None, help="Timestamps file")
  parser.add_argument("--fps",
                      default=None,
                      help="Output framerate (ex. 24000/1001).\n" \
                      "Use \"auto\" to determine automatically.")
  parser.add_argument("--mux",
                      default=False,
                      action="store_true",
                      help="Mux with contents of input file.")

  parser.add_argument("--keyframes",
                      default=None,
                      help="Path to keyframes file")
  parser.add_argument("--working-dir",
                      default=None,
                      help="Path to working directory.\n" \
                      "Allows resuming and does not remove files after completion.")
  parser.add_argument("--keep",
                      default=False,
                      action="store_true",
                      help="Do not delete temporary working directory.")

  parser.add_argument("--aomenc", default="aomenc", help="Path to aomenc")
  parser.add_argument("--vspipe", default="vspipe", help="Path to vspipe")
  parser.add_argument("--mkvmerge",
                      default="mkvmerge",
                      help="Path to mkvmerge")
  parser.add_argument("--mkvextract",
                      default="mkvextract",
                      help="Path to mkvmerge. Required for VFR.")
  parser.add_argument("--ranges",
                      default=None,
                      help="frame_n:arguments;frame_n2:arguments")

  parser.add_argument("--webm", default=False, action="store_true")

  parser.add_argument("--darkboost",
                      default=False,
                      action="store_true",
                      help="Enable dark boost.")
  parser.add_argument("--darkboost-file",
                      default=None,
                      help="Path to darkboost cache")
  parser.add_argument("--darkboost-profile",
                      default="conservative",
                      help="Available profiles: conservative, light, medium")

  parser.add_argument("--show-segments",
                      default=False,
                      action="store_true",
                      help="Show individual segments' progress.")

  parser.add_argument("--extra-filter",
                      default=None,
                      help="Extra vapoursynth filtering (ex. cropping).\n" \
                      "Input and output is clip.")

  parser.add_argument("--use-metric", default=None)
  parser.add_argument("--metric-tmax", default=86)
  parser.add_argument("--metric-tmin", default=85)
  parser.add_argument("--metric-minq", default=0)
  parser.add_argument("--metric-maxq", default=20)
  parser.add_argument("--metric-denoise", default=1)

  args, aom_args = parser.parse_known_args()

  ranges = []
  if args.ranges:
    for part_s in args.ranges.split(";"):
      part = part_s.split(":")
      part_frame = int(part[0])
      part_aom_args = (":".join(part[1:])).split(" ")
      ranges.append((part_frame, part_aom_args))
      print("range:", part_frame, part_aom_args)

  encode(args, aom_args, ranges)


if __name__ == "__main__":
  main()
