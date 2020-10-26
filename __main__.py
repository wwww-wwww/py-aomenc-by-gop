#!/usr/bin/env python3


def require_exec(file):
  if not shutil.which(file):
    print(file, "not found. Exiting.")
    exit(1)


if __name__ == "__main__":
  import shutil

  require_exec("aomenc")
  require_exec("vspipe")
  require_exec("mkvmerge")
  require_exec("onepass_keyframes")

  from . import aomenc_by_gop
  aomenc_by_gop.main()
