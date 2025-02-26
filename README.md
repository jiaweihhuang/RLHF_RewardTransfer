# Instructions for Running the Code

```bash
# setup conda environments
conda env create -f vllm.yaml
conda env create -f rlhf.yaml

# standard online learning without transfer learning
bash run_T5Small_NoTransfer_K8_TS10K_Epoch3.sh

# our main transfer learning algorithm
bash run_T5Small_Transfer_K8_N32_Rp4_TS10K_Epoch3.sh

# purely exploit ROUGE-LSum reward model (the one with the lowest-quality)
bash run_T5Small_Transfer_K8_N32_Rp4_TS10K_Epoch3_PE_ROUGE.sh

# purely exploit T5-Large reward model (the one with the highest-quality)
bash run_T5Small_Transfer_K8_N32_Rp4_TS10K_Epoch3_PE_T5Large.sh
```


# Acknowledgements
The code is based on RLHFlow/Online-RLHF repo. Their original code can be found in [https://github.com/RLHFlow/Online-RLHF](https://github.com/RLHFlow/Online-RLHF).