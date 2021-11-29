
from pipe_calibration import getAllLevelAP
from config import conf

match, acc = getAllLevelAP(conf.get("datasets"))
print(match)
print(acc)