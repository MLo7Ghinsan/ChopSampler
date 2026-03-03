<p align="center">
  <img src="cupcakke.jpg" width="220">
</p>

<h1 align="center">ChopSampler</h1>

<p align="center">
  <b>Resampler for Utau / OpenUtau that uses custom TD-PSOLA implementation</b>
</p>

<p align="center">
  TDPSOLA-based pitching/stretching | Period morphing for lesser artifacts
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-🦀-orange?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

---

## Why the name ChopSampler?

Because of how it works. This resampler chops your audio into small little grains and move them to perform pitching... And also this resampler is a chopped chud

---

## ✨ Flags (I'll make this nicer later)

Available parameters:
- gender
- velocity
- normalization

- distortion growl (gw). range 0-100. default 0
- distortion speed (gws). range 0-100. default 75
- frygrowl attempt (fg). range 0-100. default 0
- force voicing (fv). range 0-1. default 0
