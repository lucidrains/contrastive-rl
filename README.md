<img src="./crtr.png" width="450px"></img>

## contrastive-rl (wip)

For following a [new line of research](https://arxiv.org/abs/2206.07568) that started in 2022 from [Eysenbach](https://ben-eysenbach.github.io/) et al.

## install

```shell
$ pip install contrastive-rl-pytorch
```

## usage

```python
from contrastive_rl.contrastive_rl import ContrastiveRLTrainer
from x_mlps_pytorch import MLP

encoder = MLP(16, 256, 128)

trainer = ContrastiveRLTrainer(encoder)

trajectories = torch.randn(256, 512, 16)

trainer(trajectories, 100)

# train for 100 steps and save

torch.save(encoder.state_dict(), './trained.pt')
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
