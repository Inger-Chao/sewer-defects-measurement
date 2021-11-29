
from pipe_calibration import ShowDatasets
from config import conf

match, acc = ShowDatasets(conf.get("datasets"))
# print(match)
# print(acc)