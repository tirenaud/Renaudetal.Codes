import os
from typing import Optional

import matplotlib.colors as cm
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


colour_repo = '/Users/titouanrenaud/Documents/ENS/ARPE/plotly_cmaps'

def get_plotly_maps(scale_type: Optional[str] = None) -> dict[str, list[str]]:
	
    valid_types = [
	    'cyclical',
        'diverging',
        'qualitative',
        'sequential',
    ]

    if scale_type is not None:
        if scale_type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}; received {scale_type}")
        
        files = os.listdir(os.path.join(colour_repo, scale_type))
        files = [f.replace(".json","") for f in files]

        maps = {scale_type: files}

    else:
        maps = dict()
        for t in valid_types:
            files = os.listdir(os.path.join(colour_repo, t))
            files = [f.replace(".json","") for f in files]
            maps[t] = files
    
    return maps
             
        
def plotly_to_cmap(name: str) -> cm.ListedColormap:
    
    plotly_maps = get_plotly_maps()

    scale_type = [key for key in plotly_maps.keys() if name in plotly_maps[key]]

    if not scale_type:
        raise ValueError(f"{name} not a known plotly colour scale")
    
    plotly_cmap = pd.read_json(os.path.join(colour_repo,scale_type[0],f"{name}.json"))

    plotly_cmap.rename(columns={0:'value',1:'colour'}, inplace=True)

    plotly_cmap['colour'] = plotly_cmap['colour'].apply(lambda x: x.replace('rgb(','').replace(')','').split(','))

    plotly_cmap['r'] = plotly_cmap['colour'].apply(lambda x: int(x[0])/255.)
    plotly_cmap['g'] = plotly_cmap['colour'].apply(lambda x: int(x[1])/255.)
    plotly_cmap['b'] = plotly_cmap['colour'].apply(lambda x: int(x[2])/255.)

    cmap_vals = np.empty((len(plotly_cmap),3))
    cmap_vals[:,0] = plotly_cmap['r'].values
    cmap_vals[:,1] = plotly_cmap['g'].values
    cmap_vals[:,2] = plotly_cmap['b'].values

    cmap = cm.ListedColormap(cmap_vals)
    cmap = LinearSegmentedColormap.from_list('Sunsetdark', cmap_vals, N=100)

    return cmap