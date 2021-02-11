# PerSim: Data-Efficient Offline Reinforcement Learning  with Heterogeneous Agents via Personalized Simulators

This is the code accompaying the paper submission **PerSim: Data-Efficient Offline Reinforcement Learning  with Heterogeneous Agents via Personalized Simulators"** 


## Requirements

`python >3.6`
`Mujoco-py ` and its [prerequisites](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key). 
python packages in `requirements.txt`

## Datasets

We provide the offline datasets we performed the experiments on. The datasets can be downloaded via running `data.sh` through:
	`bash data.sh`


## Running PerSim

To run PerSim, run the following script:

	`python3 runner.py --env {env} --dataname {dataname}`

Choose env from {`mountainCar`, `cartPole`, `halfCheetah`}, and dataname from the available datasets in the `datasets` directory. e.g., `cartPole_pure_0.0_0`
