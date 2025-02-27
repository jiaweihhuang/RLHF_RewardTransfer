# Instructions for Running the Code

```bash
# setup conda environments
conda env create -f vllm.yaml
conda env create -f rlhf.yaml

# standard online learning without transfer learning
bash ./run/run_T5Small_NoTransfer_K8_TS10K_Epoch3.sh

# our main transfer learning algorithm
bash ./run/run_T5Small_Transfer_K8_N32_Rp4_TS10K_Epoch3.sh

# purely exploit ROUGE-LSum reward model (the one with the lowest-quality)
bash ./run/run_T5Small_Transfer_K8_N32_Rp4_TS10K_Epoch3_PE_ROUGE.sh

# purely exploit T5-Large reward model (the one with the highest-quality)
bash ./run/run_T5Small_Transfer_K8_N32_Rp4_TS10K_Epoch3_PE_T5Large.sh
```

# Citation
If you find the content of this repo useful, please consider citing:
```bibtex
@misc{huang2025rlhfefficientimperfectreward,
      title={Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective}, 
      author={Jiawei Huang and Bingcong Li and Christoph Dann and Niao He},
      year={2025},
      eprint={2502.19255},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.19255}, 
}
```


# Acknowledgements
The code is based on RLHFlow/Online-RLHF repo. Their original code can be found in [https://github.com/RLHFlow/Online-RLHF](https://github.com/RLHFlow/Online-RLHF).
