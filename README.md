<p align="center">
  <img src="cupcakke.jpg" width="220">
</p>

<h1 align="center">ChudSampler</h1>

<p align="center">
  <b>ChopSampler branch that doesn't use TD-PSOLA to synthesize harmonics</b>
</p>

<p align="center">
  From Chop to Chud
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-🦀-orange?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

---

## Why the name ChudSampler?

Because this resampler is a chopped chud just like MLo7

---

## ✨ Flag List | count: 12

NOTE: These flags are the same so um yea

| Flag | Range    | Default | Description |
|------|----------|---------|-------------|
| `g`  | `-100`-`100`  | `0` | Formant shift. Gender effect |
| `t`  | `-100`-`100`  | `0` | Off cent flag |
| `V`  | `0`-`100`     | `100` | Harmonic (voiced) level |
| `B`  | `-100`-`100`  | `0`   | Breathiness level |
| `U`  | `-100`-`100`  | `0`   | Unvoiced (fricative) level |
| `P`  | `0`-`100`      | `0`  | Normalization |
| `dg` | `0`-`100`     | `0`   | Distortion Growl effect|
| `dgs` | `0`-`100`     | `75`   | Rate of `dg` modulation|
| `fg` | `0`-`100`     | `0`   | Fry Growl effect|
| `gg` | `0`-`100`     | `0`   | Guttural Growl effect|
| `fv` | `0`-`1`     | `0`   | Force every frames to be voiced frame (affected by pitch)|
| `tn` | `-100`-`100`     | `0`   | Tension |
