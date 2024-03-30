import os
import json
import matplotlib.pyplot as plt

def graph(path):
  if os.path.isdir(path):
    statsfile = os.path.join(path, "stats.json")
    if os.path.exists(statsfile):
      return graph(statsfile)
    else:
      print("Can't find stats file")
      return False
  elif os.path.splitext(path)[1].lower() != ".json":
    print("Invalid stats file")
    return False

  j = json.load(open(path))
  j = sorted([j[s] for s in j], key=lambda x: x[0])
  x = [s[0] for s in j]
  y = [s[2] for s in j]
  print(sum(y) / len(y))
  plt.plot(x, y, "o-")
  plt.show()