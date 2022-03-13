from matplotlib.legend_handler import HandlerBase
import matplotlib.pyplot as plt

list_color  = ["k", "k"]
list_mak    = ["o","x"]
list_lab    = ['Users','SBS']

class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup,xdescent, ydescent,
                        width, height, fontsize,trans):
        return [plt.Line2D([width/2], [height/2.],ls="",
                       marker=tup[1],color=tup[0], transform=trans)]
