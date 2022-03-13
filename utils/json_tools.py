import json
from env.user import User
from env.basecenter import BaseCenter
import numpy as np
class JSONable(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if callable(obj):
            return str(obj).split(" ")[1]
        if isinstance(obj, User) or isinstance(obj,BaseCenter):
            return obj.id
        return super(JSONable, self).default(obj)