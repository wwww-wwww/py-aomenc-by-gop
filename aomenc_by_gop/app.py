import subprocess, os, sys, vapoursynth, re, traceback, argparse, platform, shutil, tempfile, time
from threading import Thread, Lock, Condition, Event

re_keyframe = r"frame *([0-9]+) *([0|1])"
re_aom_frame = r"Pass *([0-9]+)/[0-9]+ *frame * [0-9]+/([0-9]+)"

if hasattr(subprocess, "CREATE_NO_WINDOW"):
  CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
  CREATE_NO_WINDOW = 0

priorities = {
  "-2": 0x00000040,
  "-1": 0x00004000,
  "1": 0x00008000,
  "2": 0x00000080,
}


class Counter:
  def __init__(self):
    self.lock = Lock()
    self.n = 0

  def add(self, n):
    self.n += n

  def inc(self):
    self.n += 1
    return self.n


class Queue:
  def __init__(self, offset_start):
    self.queue = []
    self.lock = Lock()
    self.empty_lock = Lock()
    self.empty = Condition(self.empty_lock)
    self.not_empty_lock = Lock()
    self.not_empty = Condition(self.not_empty_lock)
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
          with self.not_empty_lock:
            self.not_empty.notify_all()
        return pop

  def wait_empty(self):
    with self.not_empty_lock:
      while len(self.queue) > 0:
        self.not_empty.wait()

  def submit(self, start, end, i):
    segment = (self.offset_start + start, self.offset_start + end, i)
    with self.lock:
      self.queue.append(segment)
      self.queue.sort(key=lambda x: x[1] - x[0], reverse=True)
      if self.update:
        self.update()
      with self.empty_lock:
        self.empty.notify()


class Worker:
  def __init__(self, args, queue, aom_args, passes, script, update1, update2):
    self.queue = queue
    self.args = args
    self.aom_args = aom_args
    self.passes = passes
    self.script = script
    self.vspipe = None
    self.pipe = None
    self.update = update1
    self.update2 = update2
    self.working = Event()
    self.working.set()
    self.stopped = False
    Thread(target=self.loop, daemon=True).start()

  def encode(self, segment):
    vspipe_cmd = [
      self.args.vspipe, self.script, "-y", "-", "-s",
      str(segment[0]), "-e",
      str(segment[1])
    ]

    segment_output = os.path.join(self.args.working_dir,
                                  f"segment_{segment[2]}.ivf")

    aomenc_cmd = [
      self.args.aomenc,
      "-",
      "--ivf",
      "-o",
      segment_output,
      f"--passes={self.passes}",
    ] + self.aom_args

    for p in range(self.passes):
      pass_cmd = aomenc_cmd + [f"--pass={p + 1}"]
      if self.passes == 2:
        segment_fpf = os.path.join(self.args.working_dir,
                                   f"segment_{segment[2]}.log")
        pass_cmd.append(f"--fpf={segment_fpf}")

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

      self.update2()

      frame = 0

      s_output = []
      try:
        while True:
          line = self.pipe.stdout.readline()

          if len(line) == 0 and self.pipe.poll() is not None:
            break

          s_output.append(line.strip())
          match = re.match(re_aom_frame, line.strip())
          if match:
            current_pass = int(match.group(1))
            if current_pass == self.passes:
              new_frame = int(match.group(2))
              if new_frame > frame:
                self.update(new_frame - frame)
                frame = new_frame

        if self.pipe.returncode != 0:
          if self.stopped: return False
          self.update(-frame)
          print(vspipe_cmd, "|", pass_cmd)
          print("\n" + "\n".join(s_output))
          return False
      except:
        print(traceback.format_exc())
      finally:
        self.vspipe.kill()
        self.pipe.kill()
        self.vspipe = None
        self.pipe = None
        self.update2()

    return True

  def loop(self):
    while True:
      if self.stopped: return
      segment = self.queue.acquire(self)
      if segment:
        for i in range(3):
          if self.encode(segment): break

      self.working.set()

  def kill(self):
    self.stopped = True

    if self.vspipe != None:
      self.vspipe.kill()
    if self.pipe != None:
      self.pipe.kill()


class Progress:
  def __init__(self, total):
    self.total = total
    self.description = ""
    self.lock = Lock()
    self.n = 0
    self.started = time.time()
    self.last_line = ""

  def update(self, n=None, description=None):
    if description:
      self.description = description

    if n:
      with self.lock:
        self.n += n

    self.print()

  def print(self):
    elapsed = round(time.time() - self.started)
    pct = int(self.n / self.total * 100)
    fps = self.n / (elapsed or 1)
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
  print("\nconcatenating")
  segments = [f"segment_{n + 1}.ivf" for n in range(n_segments)]
  segments = [os.path.join(args.working_dir, segment) for segment in segments]

  for segment in segments:
    if not os.path.exists(segment):
      raise Exception(f"Segment {segment} is missing")

  if platform.system() == "Linux":
    import resource
    file_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    cmd_limit = os.sysconf(os.sysconf_names["SC_ARG_MAX"])
  else:
    file_limit = -1
    cmd_limit = 32767

  out = _concat(args.mkvmerge, args.working_dir, segments, args.output,
                file_limit, cmd_limit)

  if not args.copy_timestamps and not args.mux:
    os.replace(out, args.output)
    return segments

  extract = []
  if args.copy_timestamps:
    path_timestamps = os.path.join(args.working_dir, "timestamps.txt")
    extract = ["--timestamps", f"0:{path_timestamps}"]
    subprocess.run([args.mkvextract, args.input, "timestamps_v2", extract[1]])
    if args.start != 0:
      with open(path_timestamps, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        ts = lines[1:]
        ts = [float(line) for line in ts]
        ts = ts[args.start:]

      with open(path_timestamps, "w+") as f:
        f.write(lines[0] + "\n")
        f.writelines([str(line) + "\n" for line in ts])

  merge = [args.mkvmerge]

  if args.mux:
    merge += ["-D", args.input]

  merge += extract + [out, "-o", args.output]

  subprocess.run(merge)

  return segments


def _concat(mkvmerge,
            working_dir,
            files,
            output,
            file_limit,
            cmd_limit,
            flip=False):
  tmp_out = os.path.join(working_dir, f"{output}.tmp{int(flip)}.mkv")
  cmd = [mkvmerge, "-o", tmp_out, files[0]]

  remaining = []
  for i, file in enumerate(files[1:]):
    new_cmd = cmd + [f"+{file}"]
    if sum(len(s) for s in new_cmd) < cmd_limit \
        and (file_limit == -1 or i < max(1, file_limit - 10)):
      cmd = new_cmd
    else:
      remaining = files[i + 1:]
      break

  concat = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
  message, _ = concat.communicate()
  concat.wait()

  if concat.returncode != 0:
    print(message)
    raise Exception

  if len(remaining) > 0:
    return _concat(mkvmerge, working_dir, [tmp_out] + remaining, output,
                   file_limit, cmd_limit, not flip)
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
      path = require_exec(default)
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

  core = vapoursynth.get_core()

  args.vpy = os.path.splitext(args.input)[1] == ".vpy"
  if args.vpy:
    script = open(args.input, "r").read()
    exec(script)
    video = vapoursynth.get_output()
  else:
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

  args.start = int(args.start or 0)
  args.end = int(args.end or num_frames - 1)

  if args.end:
    if args.end >= num_frames or (args.start and args.end < args.start):
      raise Exception("End frame out of bounds")

  args.num_frames = args.end - args.start + 1

  args.working_dir = tempfile.mkdtemp(dir=os.getcwd())
  print("Working directory:", args.working_dir)

  print(str(video))


def main():
  import sys

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
  parser.add_argument("-s", "--start", default=None, help="Input start frame")
  parser.add_argument("-e", "--end", default=None, help="Input end frame")
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

  parser.add_argument("--aomenc", default="aomenc", help="Path to aomenc")
  parser.add_argument("--vspipe", default="vspipe", help="Path to vspipe")
  parser.add_argument("--mkvmerge",
                      default="mkvmerge",
                      help="Path to mkvmerge")
  parser.add_argument("--mkvextract",
                      default="mkvextract",
                      help="Path to mkvmerge. Required for VFR")

  args, aom_args = parser.parse_known_args()

  args.aomenc = require_exec(args.aomenc)
  args.vspipe = require_exec(args.vspipe)
  args.mkvmerge = require_exec(args.mkvmerge)
  onepass_keyframes = require_exec(
    "onepass_keyframes", resource_filename("aomenc_by_gop", onepass_keyframes))

  if args.copy_timestamps:
    args.mkvextract = require_exec(args.mkvextract)

  parse_args(args)

  print("aomenc:", args.aomenc)
  print("vspipe:", args.vspipe)
  print("mkvmerge:", args.mkvmerge)
  print("onepass_keyframes:", onepass_keyframes)

  if args.priority and CREATE_NO_WINDOW:
    args.priority = priorities[str(args.priority)]

  progress_bar = Progress(args.num_frames)

  if args.vpy:
    script_name = args.input

    script_gop = """import vapoursynth as vs
exec(open(r"{}", "r").read())
v = vs.get_output()
vs.core.resize.Point(v, format=vs.YUV420P8).set_output()"""
    script_gop = script_gop.format(args.input)
  else:
    script = "from vapoursynth import core\n" \
            "core.{}(r\"{}\").set_output()".format(args.use, args.input)

    script_name = os.path.join(args.working_dir, "video.vpy")

    with open(script_name, "w+") as script_f:
      script_f.write(script)

    script_gop = """import vapoursynth as vs
v = vs.core.{}(r\"{}\")
w = round(v.width / 1.5 / 2) * 2
h = round(v.height / 1.5 / 2) * 2
resized = vs.core.resize.Point(v, width=w, height=h, format=vs.YUV420P8)
resized.set_output()"""
    script_gop = script_gop.format(args.use, args.input)

  script_name_gop = os.path.join(args.working_dir, "gop.vpy")

  with open(script_name_gop, "w+") as script_f:
    script_f.write(script_gop)

  queue = Queue(args.start)
  workers = []
  frame = 0

  def update():
    active_workers = [worker for worker in workers if worker.pipe != None]
    s = f"queue: {len(queue.queue)} workers: {len(active_workers)}"
    if frame < args.num_frames:
      s = f"fp: {frame} {s}"
    progress_bar.update(description=s)

  update()

  queue.update = update

  for i in range(args.workers):
    workers.append(
      Worker(args, queue, aom_args, args.passes, script_name,
             progress_bar.update, update))

  get_gop = [args.vspipe, script_name_gop, "-", "-y"]

  if args.start:
    get_gop.extend(["-s", str(args.start)])

  if args.end:
    get_gop.extend(["-e", str(args.end)])

  vspipe_pipe = subprocess.Popen(get_gop,
                                 stdout=subprocess.PIPE,
                                 creationflags=CREATE_NO_WINDOW)

  pipe = subprocess.Popen(onepass_keyframes,
                          stdin=vspipe_pipe.stdout,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          creationflags=CREATE_NO_WINDOW)

  counter = Counter()

  start = 0
  output_log = []

  try:
    while True:
      line = pipe.stderr.readline().strip().decode("utf-8")

      if len(line) == 0 and pipe.poll() is not None:
        break

      if len(line) > 0:
        output_log.append(line)
        match = re.match(re_keyframe, line)
        if match:
          frame = int(match.group(1))
          frame_type = int(match.group(2))
          length = frame - start
          if length - args.kf_max_dist > args.kf_max_dist:
            queue.submit(start, start + args.kf_max_dist - 1, counter.inc())
            start += args.kf_max_dist
          elif frame_type == 1:
            if length > args.kf_max_dist:
              queue.submit(start, start + int(length / 2) - 1, counter.inc())
              start += int(length / 2)
              queue.submit(start, frame - 1, counter.inc())
              start = frame
            elif length > args.min_dist:
              queue.submit(start, frame - 1, counter.inc())
              start = frame
          else:
            update()

    if pipe.returncode == 0:
      if args.num_frames > start:
        queue.submit(start, args.num_frames - 1, counter.inc())

      frame = args.num_frames
      update()

      queue.wait_empty()
      for worker in workers:
        worker.working.wait()

      segments = concat(args, counter.n)

      # cleanup

      # remove temporary files used by recursive concat
      tmp_files = [
        os.path.join(args.working_dir, f"{args.output}.tmp0.mkv"),
        os.path.join(args.working_dir, f"{args.output}.tmp1.mkv")
      ]

      timestamps = os.path.join(args.working_dir, "timestamps.txt")
      if os.path.isfile(timestamps):
        os.remove(timestamps)

      for file in tmp_files:
        if os.path.exists(file):
          os.remove(file)

      for segment in segments:
        os.remove(segment)
        if args.passes == 2:
          fpf = os.path.splitext(segment)[0] + ".log"
          os.remove(fpf)

      if not args.vpy:
        os.remove(script_name)
        os.remove(script_name_gop)

      os.rmdir(args.working_dir)

      print("completed")

    else:
      print()
      print("".join(output_log))

  except KeyboardInterrupt:
    print("\ncancelled")
  except:
    print(traceback.format_exc())
    exit(1)
  finally:
    queue.queue.clear()
    vspipe_pipe.kill()
    pipe.kill()
    for worker in workers:
      worker.kill()

    exit(0)


if __name__ == "__main__":
  main()
