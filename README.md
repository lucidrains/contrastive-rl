<img src="./crtr.png" width="450px"></img>

## contrastive-rl

For following a [new line of research](https://arxiv.org/abs/2206.07568) that started in 2022 from [Eysenbach](https://ben-eysenbach.github.io/) et al.

This is important not because of contrastive learning, but because it happens to be a special case where the RL and SSL algorithm is one. It reveals how "traditional" RL is unable to build up representations alone.

## install

```shell
$ pip install contrastive-rl-pytorch
```

## usage

```python
import torch
from contrastive_rl_pytorch import ContrastiveRLTrainer

from x_mlps_pytorch import ResidualNormedMLP # https://arxiv.org/abs/2503.14858

encoder = ResidualNormedMLP(dim = 256, dim_in = 16, dim_out = 128)

trainer = ContrastiveRLTrainer(encoder)

trajectories = torch.randn(256, 512, 16)

trainer(trajectories, 100)

# train for 100 steps and save

torch.save(encoder.state_dict(), './trained.pt')
```

## quick test

make sure `uv` is installed `pip install uv`

then

```shell
$ uv run train_lunar.py
```

## citations

```bibtex
@misc{eysenbach2023contrastivelearninggoalconditionedreinforcement,
    title   = {Contrastive Learning as Goal-Conditioned Reinforcement Learning}, 
    author  = {Benjamin Eysenbach and Tianjun Zhang and Ruslan Salakhutdinov and Sergey Levine},
    year    = {2023},
    eprint  = {2206.07568},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2206.07568}, 
}
```

```bibtex
@misc{ziarko2025contrastiverepresentationstemporalreasoning,
    title   = {Contrastive Representations for Temporal Reasoning}, 
    author  = {Alicja Ziarko and Michal Bortkiewicz and Michal Zawalski and Benjamin Eysenbach and Piotr Milos},
    year    = {2025},
    eprint  = {2508.13113},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2508.13113}, 
}
```

```bibtex
@inproceedings{anonymous2025hierarchical,
    title   = {Hierarchical Contrastive Reinforcement Learning: learn representation more suitable for {RL} environments},
    author  = {Anonymous},
    booktitle = {Submitted to The Fourteenth International Conference on Learning Representations},
    year    = {2025},
    url     = {https://openreview.net/forum?id=rTCSFOzVcK},
    note    = {under review}
}
```

```bibtex
@misc{liu2024singlegoalneedskills,
    title   = {A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals}, 
    author  = {Grace Liu and Michael Tang and Benjamin Eysenbach},
    year    = {2024},
    eprint  = {2408.05804},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2408.05804}, 
}
```

```bibtex
@inproceedings{anonymous2025demystifying,
    title   = {Demystifying Emergent Exploration in Goal-Conditioned {RL}},
    author  = {Anonymous},
    booktitle = {Submitted to The Fourteenth International Conference on Learning Representations},
    year    = {2025},
    url     = {https://openreview.net/forum?id=mwgYORsqtv},
    note    = {under review}
}
```

```bibtex
@inproceedings{wang2025,
    title   = {1000 Layer Networks for Self-Supervised {RL}: Scaling Depth Can Enable New Goal-Reaching Capabilities},
    author  = {Kevin Wang and Ishaan Javali and Micha{\l} Bortkiewicz and Tomasz Trzcinski and Benjamin Eysenbach},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year    = {2025},
    url     = {https://openreview.net/forum?id=s0JVsx3bx1}
}
```

```bibtex
@misc{nimonkar2025selfsupervisedgoalreachingresultsmultiagent,
    title   = {Self-Supervised Goal-Reaching Results in Multi-Agent Cooperation and Exploration}, 
    author  = {Chirayu Nimonkar and Shlok Shah and Catherine Ji and Benjamin Eysenbach},
    year    = {2025},
    eprint  = {2509.10656},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2509.10656}, 
}
```
