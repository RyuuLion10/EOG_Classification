# Trianswer EOG -> M55M1 UART Inference

## Wiring

- `Trianswer TX` -> `M55M1 PB2 (UART1_RX)`
- `Trianswer RX` -> `M55M1 PB3 (UART1_TX)` optional
- `Trianswer GND` -> `M55M1 GND`
- Signal level must be `3.3V TTL`

## M55M1 UART Usage

- Debug log: original USB serial (`COM15`)
- Sensor input: `UART1`, `115200 8N1`

## Stream Format

Send one EOG frame per line:

```text
ch0,ch1
```

or

```text
ch0 ch1
```

Example:

```text
0.152,-0.041
0.149,-0.039
0.147,-0.035
```

Each line is one time step with 2 channels.

## Inference Rule

- First `256` frames fill the input window
- After that, M55M1 runs inference every `32` new frames
- Output is printed on the debug UART

## Board Commands

Commands are sent to `UART1` as a line:

- `help`
- `reset`
- `infer`

## Trianswer Side Requirement

Trianswer should send preprocessed EOG samples in the same numeric domain used during model training.
If Trianswer outputs raw ADC counts, add the same preprocessing on the Trianswer side before UART transmission.
