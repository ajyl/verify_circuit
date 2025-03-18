
# Arbor: Verification Mechanisms.

Some preliminary code for studying verification mechanisms in transformers.
Some of the code is rather messy (sorry!) - please feel free to ping me on Discord if anything is unclear.


## Repo structure:

```
checkpoints --> model checkpoints.
data --> data files.
probe_checkpoints --> checkpoints for probes
```

Checkpoints and data are available [here](https://drive.google.com/drive/folders/1UYXQZjKzxkNkUY993IkiwNMeIF8Ppe-u?usp=sharing).
Download them and place them in the appropriate directories.


## Important files:

`src/HookedQwen.py`: Code to "hook" Qwen model. This allows us to add additional hooking points from the model, in order to extract their hidden states. If you want to hook a different module, feel free to add to this code.
`src/record_utils.py`: Additional utility code to extract hidden states.
`src/probe.py`: Contains the code for training and evaluating probes.
