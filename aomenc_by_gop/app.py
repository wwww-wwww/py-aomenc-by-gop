import argparse, json, os, platform, re, shutil, subprocess, sys, tempfile
import time, traceback
import vapoursynth as vs
from functools import partial
from pkg_resources import resource_filename
from threading import Condition, Event, Lock, Thread
from typing import List

re_keyframe = r"f *([0-9]+):([0|1])"
re_aom_frame = r"Pass *([0-9]+)/[0-9]+ *frame * [0-9]+/([0-9]+)"
re_mkvmerge_track = r"Track ID ([0-9]+?): video"

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
    self.__dict__.update(kwargs)


class Queue:
  def __init__(self, offset_start):
    self.queue = []
    self.lock = Lock()
    self.empty = Condition(self.lock)
    self.update = None
    self.offset_start = offset_start or 0

  def acquire(self, worker):
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

  def submit(self, start, end, i, args):
    segment = (self.offset_start + start, self.offset_start + end, i, args)
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


class Worker:
  def __init__(self, args, queue, aom_args, ranges, passes, script, update):
    self.queue = queue
    self.args = args
    self.aom_args = aom_args
    self.ranges = ranges
    self.passes = passes
    self.script = script
    self.vspipe = None
    self.pipe = None
    self.update = update
    self.working = Event()
    self.working.set()
    self.stopped = False
    self.segment = None
    self.state_lock = Lock()
    self.state_ev = Condition(self.state_lock)
    self.starting = False
    Thread(target=self.loop, daemon=True).start()

  def encode(self, segment):
    aom_args = self.aom_args

    # todo: move this into add_job
    ranges = [r for r in self.ranges if r[0] <= segment[0]]
    if ranges:
      aom_args = replace_args(aom_args, ranges[-1][1])

    for extra_arg in segment[3]:
      if extra_arg[0] == "cq":
        cq_arg = [arg.split("=") for arg in aom_args]
        cq_arg = [arg for arg in cq_arg if arg[0] == "--cq-level"]
        if cq_arg:
          new_cq = int(cq_arg[0][1]) + extra_arg[1]
          aom_args = replace_args(aom_args, [f"--cq-level={new_cq}"])

    if ranges and len(ranges[-1]) > 2 and ranges[-1][2]:
      aom_args = replace_args(aom_args, ranges[-1][1])

    vspipe_cmd = [
      self.args.vspipe, self.script, "-c", "y4m", "-", "-s",
      str(segment[0]), "-e",
      str(segment[1])
    ]

    segment_tmp = tempfile.mktemp(dir=self.args._working_dir,
                                  suffix=f".{self.args.segment_ext}")

    aomenc_cmd = [
      self.args.aomenc,
      "-",
      f"--{self.args.segment_ext}",
      "-o",
      segment_tmp,
      f"--passes={self.passes}",
    ] + aom_args

    for p in range(self.passes):
      pass_cmd = aomenc_cmd + [f"--pass={p + 1}"]
      if self.passes == 2:
        segment_fpf = os.path.join(self.args._working_dir,
                                   f"segment_{segment[2]}.log")
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

        while True:
          line = self.pipe.stdout.readline().strip()

          if len(line) == 0 and self.pipe.poll() is not None:
            break

          if not line: continue

          s_output.append(line)
          match = re.match(re_aom_frame, line.strip())
          if match:
            current_pass = int(match.group(1))
            if current_pass == self.passes:
              new_frame = int(match.group(2))
              if new_frame > frame:
                self.update(new_frame - frame)
                frame = new_frame

      except:
        print(traceback.format_exc())
      finally:
        if self.pipe.returncode != 0:
          if self.stopped: return False
          self.update(-frame)
          if s_output:
            print(" ".join(vspipe_cmd), "|", " ".join(pass_cmd))
            print("\n" + "\n".join(s_output))
          return False

        self.pipe.kill()
        self.vspipe.kill()
        self.pipe = None
        self.vspipe = None

    segment_output = os.path.join(
      self.args._working_dir, f"segment_{segment[2]}.{self.args.segment_ext}")
    shutil.move(segment_tmp, segment_output)
    return True

  def _encode(self, segment):
    for _ in range(3):
      with self.state_lock:
        self.starting = True
      if self.encode(segment): return True
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


class Progress:
  def __init__(self, total):
    self.total = total
    self.description = ""
    self.lock = Lock()
    self.n = 0
    self._n = 0
    self.started = time.time()
    self.last_line = ""

  def update(self, n=None, description=None):
    if description:
      self.description = description

    if n != None:
      with self.lock:
        self.n += n
        self._n += n

    self.print()

  def print(self):
    elapsed = round(time.time() - self.started)
    pct = int(self.n / self.total * 100)
    fps = self._n / (elapsed or 1)
    if fps < 1:
      fps = f"{round(fps * 60, 2)}fpm"
    else:
      fps = f"{round(fps, 2)}fps"

    line = " ".join([
      self.description,
      f"{pct}%",
      f"{self.n}/{self.total}",
      fps,
      f"{int(elapsed / 60):02d}:{elapsed % 60:02d}",
    ])

    padding = " " * (len(self.last_line) - len(line))

    self.last_line = line
    print(line + padding, end="\r")


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


def _concat(mkvmerge, cwd, files, output, flip=False):
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


def require_exec(file, default=None):
  path = shutil.which(file)
  if not path:
    if default:
      path = require_exec(default[0](*default[1]))
    else:
      print(file, "not found. Exiting.")
      exit(1)

  return path


def parse_args(args):
  args.segment_ext = "webm" if args.webm else "ivf"
  args.input = os.path.abspath(args.input)
  args.workers = int(args.workers)
  args.passes = int(args.passes)
  args.kf_max_dist = int(args.kf_max_dist)
  args.min_dist = 24

  print("Max workers:", args.workers)
  print("Passes:", args.passes)

  core = vs.core

  if args.use:
    source_filter = get_attr(core, args.use, True)
  else:
    args.use, source_filter = get_source_filter(core)
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

  video = source_filter(args.input)

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
    if args.fps:
      print("Can't have --fps and --copy-timestamps")
      exit(1)
    if args.timestamps:
      print("Can't have --timestamps and --copy-timestamps")
      exit(1)
    args.mkvextract = require_exec(args.mkvextract)

  if args.fps:
    if args.timestamps:
      print("Can't have --fps and --timestamps")
      exit(1)

  if args.timestamps and not os.path.isfile(args.timestamps):
    print("Timestamps file not found:", args.timestamps)
    exit(1)

  if args.darkboost:
    if args.darkboost_file:
      darkboost_path = args.darkboost_file
    else:
      filename = os.path.basename(args.input) + "_darkboost.json"
      darkboost_path = os.path.join(os.path.dirname(args.input), filename)
    darkboost = DarkBoost(clip, darkboost_path)

  print("aomenc:", args.aomenc)
  print("vspipe:", args.vspipe)
  print("mkvmerge:", args.mkvmerge)
  print("onepass_keyframes:", onepass_keyframes)

  extras = []
  if args.fps:
    extras.extend([f"Output framerate: {args.fps}"])

  if args.darkboost:
    extras.extend(["Darkboost", f"Profile: {args.darkboost_profile}"])

  print(" | ".join(extras))

  print("Encoder arguments:", " ".join(aom_args))

  if "--enable-keyframe-filtering=0" not in aom_args:
    print("WARNING: --enable-keyframe-filtering=0 is not set")

  if args.priority and CREATE_NO_WINDOW:
    args.priority = priorities[str(args.priority)]

  progress_bar = Progress(args.num_frames)

  script = """import vapoursynth as vs
vs.core.{}(r\"{}\").resize.Point(format=vs.YUV420P8).set_output()"""
  script = script.format(args.use, args.input)

  script_name = os.path.join(args._working_dir, "video.vpy")

  with open(script_name, "w+") as script_f:
    script_f.write(script)

  script_gop = """import vapoursynth as vs
v = vs.core.{}(r\"{}\")
r = v.height / v.width
w = min(1280, round(v.width / 1.5 / 2) * 2)
h = min(round(r * 1280), round(v.height / 1.5 / 2) * 2)
v.resize.Point(width=w, height=h, format=vs.YUV420P8).set_output()"""
  script_gop = script_gop.format(args.use, args.input)

  script_name_gop = os.path.join(args._working_dir, "gop.vpy")

  with open(script_name_gop, "w+") as script_f:
    script_f.write(script_gop)

  if args.workers <= 0:
    print("Number of workers set to 0, only getting keyframes")

  queue = Queue(args.start)
  workers = []
  frame = 0

  def update(n=None):
    active_workers = [worker for worker in workers if worker.segment]
    s = f"queue: {len(queue.queue)} workers: {len(active_workers)}"
    if frame < args.num_frames - 1:
      s = f"fp: {frame} {s}"
    progress_bar.update(n=n, description=s)

  update()

  queue.update = update

  n = [0]
  start = 0
  offset = 0
  output_log = []

  def add_job(start_frame, end_frame):
    n[0] += 1
    segment_output = os.path.join(args._working_dir,
                                  f"segment_{n[0]}.{args.segment_ext}")
    if os.path.isfile(segment_output):
      progress_bar._n -= end_frame - start_frame
      progress_bar.update(end_frame - start_frame)
    else:
      segment_args = []
      if args.darkboost:
        mid = (start_frame + end_frame) // 2

        if args.darkboost_profile == "conservative":
          if darkboost.count(mid, 32) > 0.25:
            segment_args.append(("cq", -2))
          elif darkboost.count(mid, 64) > 0.25:
            segment_args.append(("cq", -1))
        elif args.darkboost_profile == "light":
          if darkboost.count(mid, 32) > 0.25:
            segment_args.append(("cq", -2))
          elif darkboost.count(mid, 64) > 0.25:
            segment_args.append(("cq", -1))
          elif darkboost.count(mid, 160) < 0.33:
            segment_args.append(("cq", 1))
        elif args.darkboost_profile == "medium":
          if darkboost.count(mid, 32) > 0.25:
            segment_args.append(("cq", -3))
          elif darkboost.count(mid, 48) > 0.25:
            segment_args.append(("cq", -2))
          elif darkboost.count(mid, 64) > 0.25:
            segment_args.append(("cq", -1))
          elif darkboost.count(mid, 224) < 0.33:
            segment_args.append(("cq", 2))
          elif darkboost.count(mid, 160) < 0.33:
            segment_args.append(("cq", 1))

      queue.submit(start_frame, end_frame - 1, n[0], segment_args)

  def parse_keyframe(line, frame, start):
    match = re.match(re_keyframe, line.strip())
    if not match: return False, frame, start, 0

    frame = int(match.group(1))
    if frame >= args.num_frames:
      return False, frame, start, 0

    frame += offset
    frame_type = int(match.group(2))

    while frame - start > args.kf_max_dist * 2:
      add_job(start, start + args.kf_max_dist)
      start += args.kf_max_dist

    length = frame - start
    if frame - offset > 0 and frame_type == 1:
      if length > args.kf_max_dist:
        add_job(start, start + int(length / 2))
        add_job(start + int(length / 2), frame)
        start = frame
      elif length > args.min_dist:
        add_job(start, frame)
        start = frame

    return True, frame, start, frame_type

  if args._keyframes:
    if os.path.isfile(args._keyframes):
      with open(args._keyframes, "r") as f:
        for line in f.readlines():
          _kf, frame, start, _ft = parse_keyframe(line, frame, start)

      update()

  for _ in range(args.workers):
    workers.append(
      Worker(args, queue, aom_args, ranges, args.passes, script_name, update))

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

          match, frame, start, frame_type = parse_keyframe(line, frame, start)
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

  segments = concat(args, n[0])

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
  parser.add_argument("--priority", default=0, help="Process priority")
  parser.add_argument("--copy-timestamps",
                      default=False,
                      action="store_true",
                      help="Copy timestamps from input file.\n" \
                      "Support for variable frame rate")
  parser.add_argument("--timestamps", default=None, help="Timestamps file")
  parser.add_argument("--fps",
                      default=None,
                      help="Output framerate (ex. 24000/1001)")
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

  parser.add_argument("--darkboost", default=False, action="store_true")
  parser.add_argument("--darkboost-file",
                      default=None,
                      help="Path to darkboost cache")
  parser.add_argument("--darkboost-profile",
                      default="conservative",
                      help="Available profiles: conservative, medium")

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
