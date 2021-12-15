import argparse, os, platform, re, shutil, subprocess, sys, tempfile, time, traceback
import io
import vapoursynth as vs
import threading, inspect, ctypes
from threading import Condition, Event, Lock, Thread
from typing import BinaryIO, cast

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


# http://tomerfiliba.com/recipes/Thread2/
def _async_raise(tid, exctype):
  """raises the exception, performs cleanup if needed"""
  if not inspect.isclass(exctype):
    raise TypeError("Only types can be raised (not instances)")
  res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid,
                                                   ctypes.py_object(exctype))
  if res == 0:
    raise ValueError("invalid thread id")
  elif res != 1:
    # """if it returns a number greater than one, you're in trouble,
    # and you should call it again with exc=NULL to revert the effect"""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
    raise SystemError("PyThreadState_SetAsyncExc failed")


class Thread2(Thread):
  def _get_my_tid(self):
    """determines this (self's) thread id"""

    # do we have it cached?
    if hasattr(self, "_thread_id"):
      return self._thread_id

    # no, look for it in the _active dict
    for tid, tobj in threading._active.items():
      if tobj is self:
        self._thread_id = tid
        return tid

    raise AssertionError("could not determine the thread's id")

  def raise_exc(self, exctype):
    """raises the given exception type in the context of this thread"""

    if not self.is_alive():
      return

    _async_raise(self._get_my_tid(), exctype)

  def terminate(self):
    """raises SystemExit in the context of the given thread, which should 
        cause the thread to exit silently (unless caught)"""
    self.raise_exc(SystemExit)


class Queue:
  def __init__(self, offset_start):
    self.queue = []
    self.lock = Lock()
    self.empty_lock = Lock()
    self.empty = Condition(self.empty_lock)
    self.update = None
    self.offset_start = offset_start or 0

  def acquire(self, worker):
    with self.empty_lock:
      while len(self.queue) == 0:
        self.empty.wait()

    with self.lock:
      if len(self.queue) == 0:
        return None
      else:
        if self.update:
          self.update()
        worker.working.clear()
        pop = self.queue.pop(0)
        if len(self.queue) == 0:
          with self.empty_lock:
            self.empty.notify()
        return pop

  def wait_empty(self):
    with self.empty_lock:
      while len(self.queue) > 0:
        self.empty.wait()

  def clear(self):
    with self.lock:
      self.queue.clear()
      self.update()

  def submit(self, start, end, i):
    segment = (self.offset_start + start, self.offset_start + end, i)
    with self.lock:
      self.queue.append(segment)
      self.queue.sort(key=lambda x: x[0] - x[1])
      if self.update:
        self.update()
      with self.empty_lock:
        self.empty.notify()


class Worker:
  def __init__(self, args, filename, source_filter, threads, queue, aom_args,
               ranges, passes, update):
    self.args = args
    self.filename = filename
    self.source_filter = source_filter
    self.source_filter_threads = threads
    self.queue = queue
    self.aom_args = aom_args
    self.ranges = ranges
    self.passes = passes
    self.pipe = None
    self.pipe_output = None
    self.update = update
    self.working = Event()
    self.working.set()
    self.stopped = False
    self.segment = None
    Thread(target=self.loop, daemon=True).start()

  def encode(self, segment):
    ranges = [r for r in self.ranges if r[0] <= segment[0]]
    if ranges:
      aom_args = ranges[-1][1]
    else:
      aom_args = self.aom_args

    segment_tmp = tempfile.mktemp(dir=self.args._working_dir, suffix=".ivf")

    aomenc_cmd = [
      self.args.aomenc,
      "-",
      "--ivf",
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

      frame = 0
      s_output = []
      try:
        self.pipe = subprocess.Popen(pass_cmd,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     creationflags=self.args.priority)

        self.pipe_output = Thread2(target=pipe_video,
                                   args=(self.filename, self.source_filter,
                                         self.source_filter_threads, segment,
                                         self.pipe.stdin),
                                   daemon=True)
        self.pipe_output.start()

        stdout = io.TextIOWrapper(self.pipe.stdout, newline="")
        while True:
          line = stdout.readline()

          if not line and self.pipe.poll() is not None: break

          s_output.append(line.strip())
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
          print(segment[0], segment[1], "|", " ".join(pass_cmd))
          print("\n" + "\n".join(s_output))
          return False

        self.pipe.kill()
        self.pipe_output.terminate()
        self.pipe = None
        self.pipe_output = None

    segment_output = os.path.join(self.args._working_dir,
                                  f"segment_{segment[2]}.ivf")
    shutil.move(segment_tmp, segment_output)
    return True

  def _encode(self, segment):
    for _ in range(3):
      if self.encode(segment): return True

    return False

  def loop(self):
    while not self.stopped:
      try:
        self.segment = self.queue.acquire(self)
        if self.segment:
          self.update()
          self.working.clear()
          if not self._encode(self.segment):
            pass  # exit
      finally:
        self.segment = None
        self.update()
      self.working.set()

  def kill(self):
    self.stopped = True

    if self.pipe:
      self.pipe.kill()
    if self.pipe_output:
      self.pipe_output.terminate()


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


def concat(args, n_segments):
  print("\nConcatenating")
  segments = [f"segment_{n + 1}.ivf" for n in range(n_segments)]
  segments = [os.path.join(args._working_dir, segment) for segment in segments]

  for segment in segments:
    if not os.path.exists(segment):
      raise Exception(f"Segment {segment} is missing")

  out = _concat(args.mkvmerge, args._working_dir, segments, args.output)

  if not args.copy_timestamps and not args.mux:
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

    merge += ["--timestamps", f"0:{path_timestamps}"]

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

  if "threads" in inspect.getfullargspec(source_filter).args:
    video = source_filter(args.input, threads=args.source_filter_threads)
  else:
    video = source_filter(args.input)

  num_frames = video.num_frames

  args._start = args.start
  args.start = int(args.start or 0)
  args.end = int(args.end or num_frames - 1)

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

  return core, video, source_filter


def pipe_video(filename, source_filter, threads, segment, pipe):
  try:
    if "threads" in inspect.getfullargspec(source_filter).args:
      video = source_filter(filename, threads=threads)
    else:
      video = source_filter(filename)

    video = video[segment[0]:segment[1] + 1]
    video.output(cast(BinaryIO, pipe), y4m=True)
    pipe.close()
  except BrokenPipeError:
    pass


def main():
  if sys.platform == "win32" or sys.platform == "cygwin":
    onepass_keyframes = "bin/win64/onepass_keyframes.exe"
  else:
    onepass_keyframes = "bin/linux_amd64/onepass_keyframes"

  from pkg_resources import resource_filename

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
  parser.add_argument("-s",
                      "--start",
                      default=None,
                      help="Input start frame (inclusive, starting from 0)")
  parser.add_argument("-e",
                      "--end",
                      default=None,
                      help="Input end frame (inclusive, starting from 0)")
  parser.add_argument("-y",
                      help="Skip warning / overwrite output",
                      action="store_true")
  parser.add_argument("--priority", default=0, help="Process priority")
  parser.add_argument("--copy-timestamps",
                      default=False,
                      action="store_true",
                      help="Copy timestamps from input file.\n" \
                      "Support for variable frame rate")
  parser.add_argument("--mux",
                      default=False,
                      action="store_true",
                      help="Mux with contents of input file")
  parser.add_argument("--keyframes",
                      default=None,
                      help="Path to keyframes file")
  parser.add_argument("--working-dir",
                      default=None,
                      help="Path to working directory.\n" \
                      "Allows resuming and does not remove files after completion")
  parser.add_argument("--keep",
                      default=False,
                      action="store_true",
                      help="Do not delete temporary working directory.")

  parser.add_argument("--aomenc", default="aomenc", help="Path to aomenc")
  parser.add_argument("--mkvmerge",
                      default="mkvmerge",
                      help="Path to mkvmerge")
  parser.add_argument("--mkvextract",
                      default="mkvextract",
                      help="Path to mkvmerge. Required for VFR")

  parser.add_argument("--ranges",
                      default=None,
                      help="frame_n:arguments;frame_n2:arguments")
  parser.add_argument("--source_filter_threads", default=1)

  args, aom_args = parser.parse_known_args()

  ranges = []
  if args.ranges:
    for part_s in args.ranges.split(";"):
      part = part_s.split(":")
      part_frame = int(part[0])
      part_aom_args = (":".join(part[1:])).split(" ")
      args2_s = [arg.split("=")[0] for arg in part_aom_args]
      aom_args2 = [
        arg for arg in aom_args
        if not any(arg.startswith(arg2) for arg2 in args2_s)
      ]
      aom_args2 += part_aom_args
      ranges.append((part_frame, aom_args2))
      print("range:", part_frame, aom_args2)

  args.aomenc = require_exec(args.aomenc)
  args.mkvmerge = require_exec(args.mkvmerge)
  onepass_keyframes = require_exec("onepass_keyframes",
                                   (resource_filename,
                                    ("aomenc_by_gop", onepass_keyframes)))

  if args.copy_timestamps:
    args.mkvextract = require_exec(args.mkvextract)

  core, video, source_filter = parse_args(args)
  print(str(video))

  print("aomenc:", args.aomenc)
  print("mkvmerge:", args.mkvmerge)
  print("onepass_keyframes:", onepass_keyframes)

  print("Encoder arguments:", " ".join(aom_args))

  if args.priority and CREATE_NO_WINDOW:
    args.priority = priorities[str(args.priority)]

  progress_bar = Progress(args.num_frames)

  v_r = video.height / video.width
  v_w = round(min(1280, video.width / 1.5) / 2) * 2
  v_h = round(min(v_r * 1280, video.height / 1.5) / 2) * 2
  video_gop = core.resize.Point(video,
                                width=v_w,
                                height=v_h,
                                format=vs.YUV420P8)

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
    segment_output = os.path.join(args._working_dir, f"segment_{n[0]}.ivf")
    if os.path.isfile(segment_output):
      progress_bar._n -= end_frame - start_frame
      progress_bar.update(end_frame - start_frame)
    else:
      queue.submit(start_frame, end_frame - 1, n[0])

  def parse_keyframe(line, frame, start):
    match = re.match(re_keyframe, line.strip())
    if not match: return False, frame, start, 0

    frame = int(match.group(1)) + offset
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
      Worker(args, args.input, source_filter, args.source_filter_threads,
             queue, aom_args, ranges, args.passes, update))

  if frame < args.num_frames - 1:
    offset = max(0, frame - 3)
    args.start += offset
  else:
    args.start = frame

  if args.start < args.end:
    if args.end < args.num_frames - 1:
      video_gop = video_gop[:args.end + 1]

    if args.start > 0:
      video_gop = video_gop[args.start:]

    gop_lines = []
    try:
      with open(args._keyframes, "a+") as keyframes_file:
        pipe = subprocess.Popen(onepass_keyframes,
                                stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                creationflags=CREATE_NO_WINDOW)

        def gop_pipe(video, pipe):
          try:
            video.output(cast(BinaryIO, pipe), y4m=True)
            pipe.close()
          except BrokenPipeError:
            pass

        pipe_output = Thread2(target=gop_pipe,
                              args=(video_gop, pipe.stdin),
                              daemon=True)
        pipe_output.start()

        while True:
          line = pipe.stderr.readline().decode()

          if not line and pipe.poll() is not None: break

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
    except:
      print(traceback.format_exc())
      exit(1)
    finally:
      pipe.kill()
      pipe_output.terminate()
      if pipe.returncode != 0:
        for worker in workers:
          worker.kill()
        queue.clear()
        if pipe.returncode != None:
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

  queue.wait_empty()
  for worker in workers:
    worker.working.wait()

  segments = concat(args, n[0])

  if not args.working_dir and not args.keep:
    print("Cleaning up")
    # remove temporary files used by recursive concat
    tmp_files = [
      os.path.join(args._working_dir, f"{args.output}.tmp0.mkv"),
      os.path.join(args._working_dir, f"{args.output}.tmp1.mkv")
    ]

    timestamps = os.path.join(args._working_dir, "timestamps.txt")
    if os.path.isfile(timestamps):
      os.remove(timestamps)

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


if __name__ == "__main__":
  main()
