# Codes for 1D models figures in 'Global eruption reconstructions suggest weak atmospheric impacts on volcanic plume rise'

This Git gives access to all the codes used relative to the 1D models. Drawinf of the input parameters space, Monte-Carlo design, statisticall treatment and figures. 

A more detailed description is availbale in the Method section of the paper.

## Drawing input parameter space and running models

The runs described in Method section 'Eruption source parameters inputted in 1D model' are available in the `MONTE-CARLO_RUNS` folder. `inputs_MC.py` describes how we draw the input parameter space. These inputs are then used in `run_MC.py` to run the batch of inputs for each eruptive event with each model. 

## Codes for Figures

All the codes used to draw figures (main + supplementary) relative to 1D models are available in the `FIGURES` folder. These codes describe the different statistical treatments used. 

## Training - testing design

The last folder, `TRAINING-TESTING` describes the design of the 100 training-testing loops giving `Table S1`. 

