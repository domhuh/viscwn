# Important information about cppvis.

To run the visualization, see main.py for example on how to run the visualization for the Boolean Network. 
To run the visualizaiton for SINR network using the generative network, you must train the network using vis/train.py.
For further documentation on the parameters of the Boolean Network, refer to env/boolean_network.py.
Other functional network is the SINR network, and refer to env/sinr_network.py for its parameters.
Run requirement.txt to install all dependencies.

The visualization is run on matplotlib interactive graphs. Refer to https://matplotlib.org/3.2.2/users/navigation_toolbar.html.
To see how this was developed, refer to the view function under the Boolean and SINR network classes.
To see how the generative model was trained, refer to vis/train.py, vis/data_collection.py, and vis/model.py for more information.
To see how the clustering method was developed, refer to vis/clustering.py.