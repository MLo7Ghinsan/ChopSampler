use hound::{WavReader, WavWriter, WavSpec};
use rayon::prelude::*;
use regex::Regex;
use rand_distr::{Normal, Distribution};
use rand::SeedableRng;
use rand::rngs::StdRng;
use walkdir::WalkDir;
use serde::{Serialize, Deserialize};
use std::env;
use std::fs::File;
use std::path::Path;
use std::f32::consts::PI;
use std::thread::available_parallelism;
use praatfan_core::Sound; 
use rustfft::{FftPlanner, num_complex::Complex};

fn seed_from_args(args: &[String]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for a in args {
        for byte in a.as_bytes() {
            h ^= *byte as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn searchsorted(a: &[f32], v: f32) -> usize {
    let mut low = 0;
    let mut high = a.len();
    while low < high {
        let mid = low + (high - low) / 2;
        if a[mid] < v { low = mid + 1; } else { high = mid; }
    }
    low
}

fn np_interp(x: &[f32], xp: &[f32], fp: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    if xp.is_empty() || fp.is_empty() { return vec![0.0; x.len()]; }
    for &xi in x {
        if xi <= xp[0] { out.push(fp[0]); continue; }
        if xi >= *xp.last().unwrap() { out.push(*fp.last().unwrap()); continue; }
        let idx = searchsorted(xp, xi).saturating_sub(1);
        let x0 = xp[idx]; let x1 = xp[idx + 1];
        let f0 = fp[idx]; let f1 = fp[idx + 1];
        if x1 == x0 { out.push(f0); } else { out.push(f0 + (xi - x0) * (f1 - f0) / (x1 - x0)); }
    }
    out
}

fn np_linspace(start: f32, stop: f32, num: usize) -> Vec<f32> {
    if num == 0 { return vec![]; }
    if num == 1 { return vec![start]; }
    let step = (stop - start) / (num as f32 - 1.0);
    (0..num).map(|i| start + (i as f32) * step).collect()
}

fn np_hanning(m: usize) -> Vec<f32> {
    if m == 0 { return vec![]; }
    if m == 1 { return vec![1.0]; }
    (0..m).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / (m as f32 - 1.0)).cos()).collect()
}

fn np_hanning_sqrt(m: usize) -> Vec<f32> {
    if m == 0 { return vec![]; }
    if m == 1 { return vec![1.0]; }
    (0..m).map(|n| (0.5 - 0.5 * (2.0 * PI * n as f32 / (m as f32 - 1.0)).cos()).max(0.0).sqrt()).collect()
}

fn build_time_map(
    seg_len: usize,
    cons_n: usize,
    cons_out: usize,
    out_n: usize,
    sr: u32,
) -> Vec<f32> {
    let mut time_map = vec![0.0_f32; out_n];
    if out_n == 0 || seg_len == 0 { return time_map; }

    let last_source = seg_len.saturating_sub(1) as f32;
    let consonant_source = cons_n.min(seg_len.saturating_sub(1)) as f32;
    let head_out = cons_out.min(out_n);

    // vel stuff
    if head_out > 0 {
        let denominator = head_out as f32;
        for i in 0..head_out {
            time_map[i] = consonant_source * i as f32 / denominator;
        }
    }

    let tail_out = out_n.saturating_sub(head_out);
    let sustain_span = (last_source - consonant_source).max(0.0);
    if tail_out > 0 {
        for j in 0..tail_out {
            time_map[head_out + j] = if sustain_span > 0.0 {
                let cycle = sustain_span * 2.0;
                let phase = j as f32 % cycle;
                consonant_source + if phase <= sustain_span { phase } else { cycle - phase }
            } else {
                consonant_source
            };
        }
    }

    if head_out > 1 && tail_out > 1 && sustain_span > 0.0 {
        let radius = ((sr as f32 * 0.02).round() as usize)
            .min(head_out / 2)
            .min(tail_out - 1)
            .min((sustain_span as usize) / 2);

        if radius > 1 {
            let left = head_out - radius;
            let right = head_out + radius;
            let y0 = time_map[left];
            let y1 = time_map[right];
            let interval = (right - left) as f32;
            let secant = ((y1 - y0) / interval).max(0.0);
            let head_rate = (consonant_source / head_out as f32).min(3.0 * secant);
            let tail_rate = 1.0_f32.min(3.0 * secant);

            for i in left..=right {
                let u = (i - left) as f32 / interval;
                let u2 = u * u;
                let u3 = u2 * u;
                let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
                let h10 = u3 - 2.0 * u2 + u;
                let h01 = -2.0 * u3 + 3.0 * u2;
                let h11 = u3 - u2;
                time_map[i] = (h00 * y0
                    + h10 * interval * head_rate
                    + h01 * y1
                    + h11 * interval * tail_rate)
                    .clamp(0.0, last_source);
            }
        }
    }

    time_map
}

#[derive(Serialize, Deserialize)]
struct PitchData {
    epochs: Vec<f32>,
    is_voiced: Vec<f32>,
    t0_array: Vec<f32>,
    sr: u32,
}

fn extract_pitch_features(audio: &[f32], sr: u32) -> PitchData {
    let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();
    let praat_sound = Sound::from_samples_owned(audio_f64, sr as f64);
    let pitch = praat_sound.to_pitch(0.0, 50.0, 1200.0);

    let mut epochs = Vec::new();
    let mut is_voiced = Vec::new();
    let mut t0_array = Vec::new();
    let hop_unvoiced = (0.01 * sr as f32) as usize;

    let mut t = 0;
    let mut in_voiced_region = false;
    let mut prev_epoch = 0;
    let mut prev_t0_samples = 0;

    while t < audio.len() {
        let time_sec = t as f64 / sr as f64;
        let frame_idx = pitch.get_frame_from_time(time_sec);
        let f0 = pitch.get_value_at_frame(frame_idx).unwrap_or(0.0) as f32;

        if f0 > 0.0 {
            let t0_samples = (sr as f32 / f0).round() as usize;
            let mut peak_idx = t;
            let mut sub_offset = 0.0_f32;

            if !in_voiced_region {
                let scan_end = (t + t0_samples).min(audio.len());
                let mut max_amp = -1.0;
                for i in t..scan_end {
                    if audio[i].abs() > max_amp { max_amp = audio[i].abs(); peak_idx = i; }
                }
                in_voiced_region = true;
            } else {
                let expected_epoch = t;
                let window_size = t0_samples.min(prev_t0_samples);
                let half_win = window_size / 2;
                let search_radius = (t0_samples as f32 * 0.20) as isize; 
                
                let mut best_tau = 0;
                let mut max_score = -1.0_f32;

                let n_tau = (2 * search_radius + 1).max(1) as usize;
                let mut scores = vec![f32::NEG_INFINITY; n_tau];

                let half_win_i = half_win as isize;
                let audio_len_i = audio.len() as isize;
                let prev_epoch_i = prev_epoch as isize;

                for tau in -search_radius..=search_radius {
                    let target_center = expected_epoch as isize + tau;
                    if target_center - half_win_i < 0 || target_center + half_win_i >= audio_len_i { continue; }
                    if prev_epoch_i - half_win_i < 0 || prev_epoch + half_win >= audio.len() { continue; }

                    let mut cross_corr = 0.0;
                    let mut energy = 0.0;
                    for i in -half_win_i..=half_win_i {
                        let src_val = audio[(prev_epoch_i + i) as usize];
                        let tgt_val = audio[(target_center + i) as usize];
                        cross_corr += src_val * tgt_val;
                        energy += tgt_val * tgt_val;
                    }

                    let score = if energy > 0.0 { cross_corr / energy.sqrt() } else { 0.0 };
                    scores[(tau + search_radius) as usize] = score;
                    if score > max_score { max_score = score; best_tau = tau; }
                }

                if max_score > 0.0 {
                    peak_idx = (expected_epoch as isize + best_tau) as usize;
                    
                    let bi = (best_tau + search_radius) as usize;
                    if bi > 0 && bi + 1 < n_tau {
                        let (sm, s0, sp) = (scores[bi - 1], scores[bi], scores[bi + 1]);
                        if sm.is_finite() && sp.is_finite() {
                            let denom = sm - 2.0 * s0 + sp;
                            if denom.abs() > 1e-12 {
                                let delta = 0.5 * (sm - sp) / denom;
                                if delta.abs() <= 1.0 { sub_offset = delta; }
                            }
                        }
                    }
                } else {
                    let search_start = t.saturating_sub(search_radius as usize);
                    let search_end = (t + search_radius as usize).min(audio.len());
                    let mut max_amp = -1.0;
                    for i in search_start..search_end {
                        if audio[i].abs() > max_amp { max_amp = audio[i].abs(); peak_idx = i; }
                    }
                }
            }

            epochs.push(peak_idx as f32 + sub_offset);
            is_voiced.push(1.0);
            t0_array.push(sr as f32 / f0);
            
            prev_epoch = peak_idx;
            prev_t0_samples = t0_samples;
            t = peak_idx + t0_samples; 
        } else {
            in_voiced_region = false;
            epochs.push(t as f32);
            is_voiced.push(0.0);
            t0_array.push(hop_unvoiced as f32);
            t += hop_unvoiced;
        }
    }

    PitchData { epochs, is_voiced, t0_array, sr }
}

fn reflect_index(i: isize, len: usize) -> usize {
    if len <= 1 { return 0; }
    let period = 2 * (len as isize - 1);
    let mut idx = i % period;
    if idx < 0 { idx += period; }
    if idx >= len as isize { idx = period - idx; }
    idx as usize
}

fn stft(x: &[f32], n_fft: usize, hop_length: usize, window: &[f32]) -> Vec<Vec<Complex<f32>>> {
    let pad = n_fft / 2;
    let mut x_padded = vec![0.0; x.len() + 2 * pad];
    if !x.is_empty() {
        for i in 0..x_padded.len() {
            let src = i as isize - pad as isize;
            x_padded[i] = x[reflect_index(src, x.len())];
        }
    }
    
    let num_frames = 1.max(1 + (x_padded.len().saturating_sub(n_fft)) / hop_length);
    let mut frames = Vec::with_capacity(num_frames);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    
    for i in 0..num_frames {
        let start = i * hop_length;
        let mut frame = vec![Complex::new(0.0, 0.0); n_fft];
        for j in 0..n_fft {
            if start + j < x_padded.len() {
                frame[j] = Complex::new(x_padded[start + j] * window[j], 0.0);
            }
        }
        fft.process(&mut frame);
        frames.push(frame[0..=n_fft / 2].to_vec()); 
    }
    frames
}

fn istft(spectra: &[Vec<Complex<f32>>], n_fft: usize, hop_length: usize, expected_len: usize, window: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    let pad = n_fft / 2;
    let num_frames = spectra.len();
    let total_len = n_fft + hop_length * num_frames.saturating_sub(1);
    let mut y = vec![0.0; total_len];
    let mut window_sum = vec![0.0_f32; total_len];

    let ifft_norm = n_fft as f32;
    
    for (i, frame) in spectra.iter().enumerate() {
        let mut full_frame = vec![Complex::new(0.0, 0.0); n_fft];
        for j in 0..=n_fft / 2 {
            full_frame[j] = frame[j];
            if j > 0 && j < n_fft / 2 { full_frame[n_fft - j] = frame[j].conj(); }
        }
        ifft.process(&mut full_frame);
        
        let start = i * hop_length;
        for j in 0..n_fft {
            y[start + j] += (full_frame[j].re / ifft_norm) * window[j];
            window_sum[start + j] += window[j] * window[j];
        }
    }
    
    for i in 0..total_len {
        if window_sum[i] > 1e-8 { y[i] /= window_sum[i]; }
    }
    
    let end = (pad + expected_len).min(y.len());
    let mut final_y = y[pad..end].to_vec();
    final_y.resize(expected_len, 0.0);
    final_y
}

fn separate_components(audio: &[f32], sr: u32, force_voicing: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_fft = 1024;
    let hop_length = 256;
    let window = np_hanning_sqrt(n_fft);

    // from epoch tracking peak-centred
    let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();
    let praat_sound = Sound::from_samples_owned(audio_f64, sr as f64);
    let pitch = praat_sound.to_pitch(0.0, 50.0, 1200.0);

    let zxx = stft(audio, n_fft, hop_length, &window);
    let num_frames = zxx.len();
    let freq_res = sr as f32 / n_fft as f32;

    let mut raw_f0 = vec![0.0_f32; num_frames];
    let mut voiced_indices = Vec::new();
    let mut voiced_f0_vals = Vec::new();
    for f in 0..num_frames {
        let t_sec = f as f64 * hop_length as f64 / sr as f64;
        let frame_idx = pitch.get_frame_from_time(t_sec);
        let detected_f0 = pitch.get_value_at_frame(frame_idx).unwrap_or(0.0) as f32;
        let f0 = if detected_f0 > 0.0 {
            detected_f0
        } else if force_voicing {
            100.0
        } else {
            0.0
        };
        raw_f0[f] = f0;
        if f0 > 0.0 {
            voiced_indices.push(f as f32);
            voiced_f0_vals.push(f0);
        }
    }

    let smooth_f0: Vec<f32> = if !voiced_indices.is_empty() {
        let all_indices: Vec<f32> = (0..num_frames).map(|i| i as f32).collect();
        np_interp(&all_indices, &voiced_indices, &voiced_f0_vals)
    } else {
        vec![100.0; num_frames]
    };

    let voiced_set: std::collections::HashSet<usize> =
        (0..num_frames).filter(|&f| raw_f0[f] > 0.0).collect();

    let mut harm_spectra = vec![vec![Complex::new(0.0, 0.0); n_fft / 2 + 1]; num_frames];
    let mut breath_spectra = vec![vec![Complex::new(0.0, 0.0); n_fft / 2 + 1]; num_frames];
    let mut unvoiced_spectra = vec![vec![Complex::new(0.0, 0.0); n_fft / 2 + 1]; num_frames];

    let max_voiced_freq = 13000.0_f32.min(sr as f32 / 2.0);

    for f in 0..num_frames {
        let voiced = voiced_set.contains(&f);
        let current_f0 = smooth_f0[f].clamp(50.0, 1200.0);

        let mut mag_db = vec![0.0_f32; n_fft / 2 + 1];
        for b in 0..=n_fft / 2 {
            mag_db[b] = 20.0 * (zxx[f][b].norm() + 1e-12).log10();
        }

        let mut mask = vec![0.0_f32; n_fft / 2 + 1];

        if voiced {
            let max_harmonics = (max_voiced_freq / current_f0) as usize;
            for h in 1..=max_harmonics {
                let target_hz = h as f32 * current_f0;
                if target_hz > max_voiced_freq { break; }

                let center_bin = (target_hz / freq_res).round() as usize;
                let search_radius = 1.max(((current_f0 * 0.3) / freq_res).round() as usize);
                let start_bin = 1.max(center_bin.saturating_sub(search_radius));
                let end_bin = (n_fft / 2).min(center_bin + search_radius + 1);

                if start_bin < end_bin {
                    let mut max_val = -999.0_f32;
                    let mut actual_bin = start_bin;
                    for b in start_bin..end_bin {
                        if mag_db[b] > max_val {
                            max_val = mag_db[b];
                            actual_bin = b;
                        }
                    }
                    let mask_start = actual_bin.saturating_sub(1);
                    let mask_end = (actual_bin + 2).min(n_fft / 2 + 1);
                    for value in &mut mask[mask_start..mask_end] {
                        *value = 1.0;
                    }
                }
            }
        }

        for b in 0..=n_fft / 2 {
            let harm_val = zxx[f][b] * mask[b];
            let noise_val = zxx[f][b] * (1.0 - mask[b]);
            harm_spectra[f][b] = harm_val;
            if voiced { breath_spectra[f][b] = noise_val; }
            else { unvoiced_spectra[f][b] = noise_val; }
        }
    }

    let expected_len = audio.len();
    let harm_audio = istft(&harm_spectra, n_fft, hop_length, expected_len, &window);
    let breath_audio = istft(&breath_spectra, n_fft, hop_length, expected_len, &window);
    let unvoiced_audio = istft(&unvoiced_spectra, n_fft, hop_length, expected_len, &window);

    (harm_audio, breath_audio, unvoiced_audio)
}

fn get_grain_directed(audio: &[f32], center: f32, size: usize, reverse: bool) -> Vec<f32> {
    let base = center.round();
    let d = center - base;
    let half = size as isize / 2;
    let mut grain = vec![0.0; size];

    if reverse {
        let start = base as isize + half;
        for i in 0..size as isize {
            let idx = start - i;
            if idx >= 0 && (idx as usize) < audio.len() {
                grain[i as usize] = audio[idx as usize];
            }
        }
        // flips
        if d.abs() < 1e-6 { grain } else { frac_delay(&grain, d) }
    } else {
        let start = base as isize - half;
        for i in 0..size as isize {
            let idx = start + i;
            if idx >= 0 && (idx as usize) < audio.len() {
                grain[i as usize] = audio[idx as usize];
            }
        }
        if d.abs() < 1e-6 { grain } else { frac_delay(&grain, -d) }
    }
}

fn map_runs_backward(time_map_abs: &[f32], output_pos: usize, span: usize) -> bool {
    if time_map_abs.len() < 2 { return false; }
    let last = time_map_abs.len() - 1;
    let half = (span / 2).max(1);
    let lo = output_pos.saturating_sub(half).min(last);
    let hi = output_pos.saturating_add(half).min(last);
    hi > lo && time_map_abs[hi] < time_map_abs[lo]
}

fn get_aligned_grain(audio: &[f32], center: f32, size: usize) -> Vec<f32> {
    get_grain_directed(audio, center, size, false)
}

fn frac_delay(x: &[f32], d: f32) -> Vec<f32> {
    if d.abs() < 1e-6 { return x.to_vec(); }
    let n = x.len();
    let mut out = vec![0.0_f32; n];
    let at = |i: isize| -> f32 { if i >= 0 && (i as usize) < n { x[i as usize] } else { 0.0 } };

    for i in 0..n {
        let pos = i as f32 - d;
        let i1 = pos.floor() as isize;
        let t = pos - i1 as f32;
        let (m1, p0, p1, p2) = (at(i1 - 1), at(i1), at(i1 + 1), at(i1 + 2));
        let c0 = p0;
        let c1 = 0.5 * (p1 - m1);
        let c2 = m1 - 2.5 * p0 + 2.0 * p1 - 0.5 * p2;
        let c3 = 0.5 * (p2 - m1) + 1.5 * (p0 - p1);
        out[i] = ((c3 * t + c2) * t + c1) * t + c0;
    }
    out
}

fn overlap_add_grain(
    output: &mut [f32],
    weight_sum: &mut [f32],
    grain: &[f32],
    window: &[f32],
    target_pos: f32,
    gain: f32,
) {
    if grain.is_empty() || grain.len() != window.len() { return; }

    let windowed: Vec<f32> = grain.iter().zip(window).map(|(&x, &w)| x * w).collect();
    let target_base = target_pos.floor() as isize;
    let target_fraction = target_pos - target_base as f32;
    let placed = frac_delay(&windowed, target_fraction);
    let placed_window = frac_delay(window, target_fraction);
    let start = target_base - placed.len() as isize / 2;

    for i in 0..placed.len() {
        let out_idx = start + i as isize;
        if out_idx >= 0 && (out_idx as usize) < output.len() {
            let out_idx = out_idx as usize;
            output[out_idx] += placed[i] * gain;
            weight_sum[out_idx] += placed_window[i].max(0.0);
        }
    }
}

fn wola_aperiodic_from_map(
    src: &[f32],
    time_map_abs: &[f32],
    out_len: usize,
    hop: usize,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; out_len];
    if src.is_empty() || time_map_abs.is_empty() || out_len == 0 || hop == 0 {
        return output;
    }

    let mut grain_size = hop.saturating_mul(4).max(8);
    grain_size += grain_size % 2;
    let window = np_hanning(grain_size);
    let mut weight_sum = vec![0.0_f32; out_len];

    let mut output_center = 0_usize;
    while output_center < out_len {
        let map_index = output_center.min(time_map_abs.len() - 1);
        let source_center = time_map_abs[map_index];
        let reverse = map_runs_backward(time_map_abs, output_center, hop);
        let grain = get_grain_directed(src, source_center, grain_size, reverse);
        overlap_add_grain(
            &mut output,
            &mut weight_sum,
            &grain,
            &window,
            output_center as f32,
            1.0,
        );
        output_center = output_center.saturating_add(hop);
    }

    for i in 0..out_len {
        if weight_sum[i] > 1e-4 {
            output[i] /= weight_sum[i];
        }
    }

    output
}

fn dynamic_onepole_filter(x: &mut [f32], f0_hz: &[f32], sr: u32, cutoff_factor: f32, order: usize, highpass: bool) {
    let n = x.len();
    if n == 0 { return; }
    let two_pi = 2.0 * PI;
    let sr_f = sr as f32;
    let ceil_fc = 0.45 * sr_f;

    for _ in 0..order {
        let mut yp = 0.0_f32;
        let mut prev_x = x[0];
        for i in 0..n {
            let fi = if i < f0_hz.len() { f0_hz[i] } else { *f0_hz.last().unwrap_or(&100.0) };
            let fc = (fi.max(20.0) * cutoff_factor).clamp(20.0, ceil_fc);
            if highpass {
                let alpha = sr_f / (two_pi * fc + sr_f);
                let xi = x[i];
                yp = alpha * (yp + xi - prev_x);
                x[i] = yp;
                prev_x = xi;
            } else {
                let alpha = (two_pi * fc) / (two_pi * fc + sr_f);
                yp = yp + alpha * (x[i] - yp);
                x[i] = yp;
            }
        }
    }
}

fn td_psola_utau(
    harm_audio: &[f32],
    breath_audio: &[f32],
    unvoiced_audio: &[f32],
    sr: u32,
    target_f0_hz: &[f32],
    time_map: &[f32],
    seg_start: usize,
    epochs: &[f32],
    is_voiced: &[f32],
    t0_array: &[f32],
    formant_semitones: f32,
    voice_drive: f32,
    drive_speed: f32,
    fry_intensity: f32,
    v_gain: f32,
    u_gain: f32,
    b_gain: f32,
    gg_intensity: f32,
    tension: f32,
    seed: u64,
) -> Vec<f32> {
    let formant_factor = 2.0_f32.powf(formant_semitones / 12.0);
    let hop_unvoiced = (0.01 * sr as f32) as usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal_dist = Normal::new(0.0, 1.0).unwrap();

    let mut abs_time_map = Vec::with_capacity(time_map.len());
    for &t in time_map { abs_time_map.push(t + seg_start as f32); }

    let out_len = abs_time_map.len();
    let buf_len = out_len + (sr * 2) as usize;
    let mut output_harm = vec![0.0_f32; buf_len];
    let rendered_breath = wola_aperiodic_from_map(
        breath_audio,
        &abs_time_map,
        out_len,
        hop_unvoiced,
    );
    let mut output_breath = vec![0.0_f32; buf_len];
    output_breath[..out_len].copy_from_slice(&rendered_breath);
    let mut output_unvoiced = vec![0.0_f32; buf_len];
    let mut unvoiced_weight = vec![0.0_f32; buf_len];

    if epochs.len() < 2 || is_voiced.len() < epochs.len() || t0_array.len() < epochs.len() {
        return vec![0.0; out_len];
    }

    let source_len = harm_audio.len().min(breath_audio.len()).min(unvoiced_audio.len());

    let mut t_s = 0.0_f32;
    let mut drive_phase = 0.0_f32;
    let mut prev_t_s = hop_unvoiced as f32;

    while (t_s as usize) < out_len {
        if source_len < 2 { break; }

        let t_a = abs_time_map[t_s as usize].clamp(0.0, (source_len - 1) as f32);

        let mut idx1 = searchsorted(epochs, t_a).saturating_sub(1);
        idx1 = idx1.clamp(0, epochs.len() - 2);
        let idx2 = idx1 + 1;
        let diff = epochs[idx2] - epochs[idx1];
        let weight = if diff > 0.0 {
            ((t_a - epochs[idx1]) / diff).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let voicing_mix = (1.0 - weight) * is_voiced[idx1] + weight * is_voiced[idx2.min(is_voiced.len().saturating_sub(1))];
        let t0_interp = (1.0 - weight) * t0_array[idx1] + weight * t0_array[idx2.min(t0_array.len().saturating_sub(1))];
        let current_target_hz = target_f0_hz[t_s as usize];

        let mut t_s_target = hop_unvoiced as f32;
        if current_target_hz > 0.0 {
            t_s_target = voicing_mix * (sr as f32 / current_target_hz) + (1.0 - voicing_mix) * hop_unvoiced as f32;
        }

        let t_s_step = t_s_target.max(1.0);

        let mut fry_offset = 0.0_f32;
        let mut fry_amp = 1.0_f32;

        // --- PATH A: VOICED ---
        if voicing_mix > 0.0 {
            let mut extract_win_size_v = (2.0 * t0_interp).round().max(4.0) as usize;
            extract_win_size_v += extract_win_size_v % 2;

            let g1_h = get_aligned_grain(harm_audio, epochs[idx1], extract_win_size_v);
            let g2_h = get_aligned_grain(harm_audio, epochs[idx2.min(epochs.len().saturating_sub(1))], extract_win_size_v);

            let mut morphed_harm = vec![0.0; extract_win_size_v];
            for i in 0..extract_win_size_v { morphed_harm[i] = (1.0 - weight) * g1_h[i] + weight * g2_h[i]; }

            let source_rms = (morphed_harm.iter().map(|x| x * x).sum::<f32>() / extract_win_size_v as f32).sqrt() + 1e-12;
            let mut shifted_harm = morphed_harm;
            if (formant_factor - 1.0).abs() > 0.001 {
                let orig_idx = np_linspace(0.0, 1.0, shifted_harm.len());
                let mut new_len = (shifted_harm.len() as f32 / formant_factor).round() as usize;
                new_len = new_len.max(4);
                new_len += new_len % 2;
                let new_idx = np_linspace(0.0, 1.0, new_len);
                shifted_harm = np_interp(&new_idx, &orig_idx, &shifted_harm);
            }

            let hanning = np_hanning(shifted_harm.len());
            for i in 0..shifted_harm.len() { shifted_harm[i] *= hanning[i]; }

            let current_rms = (shifted_harm.iter().map(|x| x * x).sum::<f32>() / shifted_harm.len() as f32).sqrt() + 1e-12;
            let density_comp = (t_s_step.max(1.0) / t0_interp.max(1.0)).sqrt();
            let mut gain = ((source_rms / current_rms) * density_comp).clamp(0.0, 5.0);

            if voice_drive > 0.0 {
                drive_phase += 2.0 * PI * drive_speed * (t_s_step / sr as f32);
                gain *= 1.0 + (drive_phase.sin() * voice_drive);
            }

            if fry_intensity > 0.0 {
                fry_offset = normal_dist.sample(&mut rng) * t0_interp * 0.12 * fry_intensity;
                if (t_s / t_s_step) as i32 % 2 == 0 { fry_amp = 1.0 - (0.5 * fry_intensity); }
                gain *= fry_amp;
            }

            if gg_intensity > 0.0 {
                let period_idx = (t_s / t0_interp.max(1.0)).round() as i32;
                let sub_factor = if period_idx % 2 == 0 { 1.0 - 0.95 * gg_intensity } else { 1.0 + 0.25 * gg_intensity };
                let drive = 1.0 + gg_intensity * 12.0;
                let pre_rms = (shifted_harm.iter().map(|x| x * x).sum::<f32>() / shifted_harm.len() as f32).sqrt() + 1e-12;

                for s in shifted_harm.iter_mut() { *s = (*s * drive).tanh() * sub_factor; }

                let post_rms = (shifted_harm.iter().map(|x| x * x).sum::<f32>() / shifted_harm.len() as f32).sqrt() + 1e-12;
                let rms_comp = pre_rms / post_rms;
                for s in shifted_harm.iter_mut() { *s *= rms_comp; }

                fry_offset += normal_dist.sample(&mut rng) as f32 * t0_interp * 0.1 * gg_intensity;
            }

            let target_pos = t_s + fry_offset;
            let ts_pos = target_pos.floor() as isize;
            let placed = frac_delay(&shifted_harm, target_pos - ts_pos as f32);

            let start_s_h = ts_pos - (placed.len() as isize / 2);
            for i in 0..placed.len() {
                let out_idx = start_s_h + i as isize;
                if out_idx >= 0 && (out_idx as usize) < buf_len {
                    output_harm[out_idx as usize] += placed[i] * gain * voicing_mix;
                }
            }
        }

        // --- PATH B: UNVOICED ---
        if voicing_mix < 1.0 {
            let required_final_size = t_s_step.max(prev_t_s) * 2.0;
            let mut extract_win_size_u = (hop_unvoiced as f32 * 2.0).max(required_final_size).round() as usize;
            extract_win_size_u += extract_win_size_u % 2;

            let reverse_u = map_runs_backward(&abs_time_map, t_s as usize, hop_unvoiced);
            let unvoiced_grain = get_grain_directed(unvoiced_audio, t_a, extract_win_size_u, reverse_u);
            let unvoiced_window = np_hanning(extract_win_size_u);
            overlap_add_grain(
                &mut output_unvoiced,
                &mut unvoiced_weight,
                &unvoiced_grain,
                &unvoiced_window,
                t_s,
                1.0 - voicing_mix,
            );
        }

        t_s += t_s_step;
        prev_t_s = t_s_step;
    }

    for i in 0..buf_len {
        if unvoiced_weight[i] > 1e-4 { output_unvoiced[i] /= unvoiced_weight[i]; }
    }

    let mut actual_len = buf_len;
    while actual_len > out_len && output_harm[actual_len - 1].abs() + output_breath[actual_len - 1].abs() + output_unvoiced[actual_len - 1].abs() < 1e-6 {
        actual_len -= 1;
    }
    output_harm.truncate(actual_len);
    output_breath.truncate(actual_len);
    output_unvoiced.truncate(actual_len);

    // tension
    if tension.abs() > 0.001 {
        let mut out_f0 = vec![100.0_f32; actual_len];
        for i in 0..actual_len.min(out_len) {
            out_f0[i] = target_f0_hz[i].max(80.0);
        }

        let abs_ten = tension.abs();
        
        if tension < 0.0 {
            // low tens
            let mut peak_before = 0.0_f32;
            for i in 0..actual_len { if output_harm[i].abs() > peak_before { peak_before = output_harm[i].abs(); } }

            let order = (1.0 + abs_ten * 4.0).round() as usize;
            let order = order.clamp(1, 6);
            let lp_factor = 2.0 - abs_ten * 0.75;
            dynamic_onepole_filter(&mut output_harm, &out_f0, sr, lp_factor, order, false);
            
            // to prevent bass
            let mut peak_after = 0.0_f32;
            for i in 0..actual_len { if output_harm[i].abs() > peak_after { peak_after = output_harm[i].abs(); } }
            let gain = peak_before / (peak_after + 1e-12); 
            for i in 0..actual_len { output_harm[i] *= gain; }

            let mut highpassed_breath = output_breath.clone();
            dynamic_onepole_filter(&mut highpassed_breath, &out_f0, sr, 3.0, 2, true);
            let breath_boost = abs_ten * 12.0; 
            for i in 0..actual_len {
                output_breath[i] += highpassed_breath[i] * breath_boost;
            }
        } else {
            // high tens
            let rms_before = {
                let mut s = 0.0_f32;
                for i in 0..actual_len { s += output_harm[i] * output_harm[i]; }
                (s / actual_len.max(1) as f32).sqrt() + 1e-12
            };

            let mut highpassed = output_harm.clone();
            dynamic_onepole_filter(&mut highpassed, &out_f0, sr, abs_ten * 4.0, 4, true);
            let boost = 1.0 + abs_ten * 20.0;
            for i in 0..actual_len { output_harm[i] += highpassed[i] * boost; }

            let mut highpassed_breath = output_breath.clone();
            dynamic_onepole_filter(&mut highpassed_breath, &out_f0, sr, 4.0, 2, true);
            let breath_boost = abs_ten * 8.0; 
            for i in 0..actual_len {
                output_breath[i] += highpassed_breath[i] * breath_boost;
            }

            let rms_after = {
                let mut s = 0.0_f32;
                for i in 0..actual_len { s += output_harm[i] * output_harm[i]; }
                (s / actual_len.max(1) as f32).sqrt() + 1e-12
            };
            let gain = rms_before / rms_after;
            for i in 0..actual_len { output_harm[i] *= gain; }
        }
    }

    let mut output = Vec::with_capacity(actual_len);
    for i in 0..actual_len {
        let h = output_harm[i] * v_gain;
        let b = output_breath[i] * b_gain;
        let u = output_unvoiced[i] * u_gain;
        output.push(h + b + u);
    }

    output
}

fn process_single_file(wav_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(wav_path)?;
    let sr = reader.spec().sample_rate;
    let audio: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Int => {
            let max = 2f32.powi(reader.spec().bits_per_sample as i32 - 1);
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        },
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect()
    };
    let pitch_data = extract_pitch_features(&audio, sr);
    let out_path = wav_path.with_extension("chopped");
    let file = File::create(out_path)?;
    bincode::serialize_into(file, &pitch_data)?;
    Ok(())
}

fn preprocess_folder(folder_path: &str) {
    let threads = (available_parallelism().map(|n| n.get()).unwrap_or(4) / 4).max(1);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap_or(());

    let wav_files: Vec<_> = WalkDir::new(folder_path).into_iter().filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "wav"))
        .map(|e| e.path().to_owned()).collect();

    if wav_files.is_empty() { return; }
    println!("Found {} WAV files. Extracting features", wav_files.len());

    let success: usize = wav_files.par_iter().map(|path| match process_single_file(path) {
        Ok(_) => { println!("[OK] Preprocessed: {:?}", path.file_name().unwrap()); 1 }
        Err(e) => { println!("[ERROR] Failed {:?}: {}", path.file_name().unwrap(), e); 0 }
    }).sum();
    println!("\nFinished! Successfully processed {}/{} files.", success, wav_files.len());
}

fn parse_utau_pitch(pitch_string: &str) -> Vec<f32> {
    if pitch_string.is_empty() || pitch_string == "AA" { return vec![0.0]; }
    let to_uint6 = |c: char| -> u32 {
        let o = c as u32;
        if o >= 97 { o - 71 } else if o >= 65 { o - 65 } else if o >= 48 { o + 4 } else if o == 43 { 62 } else if o == 47 { 63 } else { 0 }
    };
    let to_int12 = |s: &str| -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let u = (to_uint6(chars[0]) << 6) + to_uint6(chars[1]);
        let v = u & 0xFFF;
        if (v & 0x800) != 0 { (v as i32) - 4096 } else { v as i32 }
    };
    let mut out = Vec::new();
    let parts: Vec<&str> = pitch_string.split('#').collect();
    for chunk in parts.chunks(2) {
        if chunk.len() == 2 {
            let ps = chunk[0];
            let run: usize = chunk[1].parse().unwrap_or(0);
            for i in (0..ps.len()).step_by(2) { out.push(to_int12(&ps[i..i+2]) as f32); }
            if let Some(&last) = out.last() { out.extend(vec![last; run]); }
        } else {
            let ps = chunk[0];
            for i in (0..ps.len()).step_by(2) { out.push(to_int12(&ps[i..i+2]) as f32); }
        }
    }
    if out.is_empty() { vec![0.0] } else { out }
}

fn note_to_midi(note: &str) -> f32 {
    if let Some(caps) = Regex::new(r"([A-G]#?)(-?\d+)").unwrap().captures(note) {
        let nm = match &caps[1] { "C"=>0, "C#"=>1, "D"=>2, "D#"=>3, "E"=>4, "F"=>5, "F#"=>6, "G"=>7, "G#"=>8, "A"=>9, "A#"=>10, "B"=>11, _=>0 };
        let octv: i32 = caps[2].parse().unwrap_or(4);
        ((octv + 1) * 12 + nm) as f32
    } else { 60.0 }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && Path::new(&args[1]).is_dir() {
        preprocess_folder(&args[1]);
        std::process::exit(0);
    }

    if args.len() < 14 {
        eprintln!("Expected 13 UTAU args. Got {}", args.len().saturating_sub(1));
        std::process::exit(1);
    }

    let in_file = &args[1]; let out_file = &args[2]; let pitch = &args[3];
    let velocity: f32 = args[4].parse().unwrap_or(100.0);
    let flags = &args[5];
    let offset_s: f32 = args[6].parse::<f32>().unwrap_or(0.0) / 1000.0;
    let length_s: f32 = args[7].parse::<f32>().unwrap_or(1000.0).max(0.001) / 1000.0;
    let consonant_s: f32 = args[8].parse::<f32>().unwrap_or(0.0) / 1000.0;
    let cutoff_s: f32 = args[9].parse::<f32>().unwrap_or(0.0) / 1000.0;
    let volume: f32 = args[10].parse::<f32>().unwrap_or(100.0) / 100.0;
    let tempo: f32 = args[12].replace("!", "").parse().unwrap_or(120.0);
    let pitch_string = &args[13];

    let pitch_m = note_to_midi(pitch);
    let bend_cents = parse_utau_pitch(pitch_string);

    let mut fv = 0.0; let mut dg = 0.0; let mut dgs = 75.0;
    let mut fg = 0.0; let mut g_gender = 0.0;
    
    let mut v_gain = 1.0;
    let mut u_gain = 1.0;
    let mut b_gain = 1.0;
    let mut b_duck = 1.0;
    let mut gg_intensity = 0.0;
    let mut p_norm = 0.86;
    let mut t_off_cent: f32 = 0.0;
    let mut tension: f32 = 0.0;

    // general gain curve
    let scale_gain_u = |val: f32| -> f32 {
        if val <= -100.0 { 0.0 }
        else if val < 0.0 { 1.0 + val / 100.0 }
        else { 1.0 + (val / 20.0) }
    };

    // V flag (Harmonic strength) linear
    const B_MAX: f32 = 4.0;
    let scale_gain_b = |val: f32| -> f32 {
        if val <= -100.0 { 0.0 }
        else if val < 0.0 { 1.0 + val / 100.0 }
        else { 1.0 + (val / 100.0) * (B_MAX - 1.0) }
    };

    for cap in Regex::new(r"([a-zA-Z]+)([-+]?\d*)").unwrap().captures_iter(flags) {
        let val_str = &cap[2];
        let val = if val_str.is_empty() { 0.0 } else { val_str.parse::<f32>().unwrap_or(0.0) };

        match &cap[1] {
            "fv" => fv = if val_str.is_empty() { 1.0 } else { val.clamp(0.0, 1.0) },
            "dg" => dg = val.clamp(0.0, 100.0),
            "dgs" => dgs = val.clamp(0.0, 100.0), 
            "fg" => fg = val.clamp(0.0, 100.0),
            "g" => g_gender = val,
            "V" => v_gain = val.clamp(0.0, 100.0) / 100.0,
            "U" => u_gain = scale_gain_u(val),
            "B" => { b_gain = scale_gain_b(val); b_duck = 1.0 - val.clamp(0.0, 100.0) / 100.0; },
            "gg" => gg_intensity = val.clamp(0.0, 100.0) / 100.0,
            "P" => p_norm = if val_str.is_empty() { 1.0 } else { val.clamp(0.0, 100.0) / 100.0 },
            "t" => t_off_cent = val,
            "tn" => tension = val.clamp(-100.0, 100.0) / 100.0,
            _ => {}
        }
    }

    let seed_src: Vec<String> = std::iter::once(in_file.clone())
        .chain(args[3..].iter().cloned())
        .collect();
    let seed = seed_from_args(&seed_src);

    let mut reader = WavReader::open(in_file).unwrap();
    let sr = reader.spec().sample_rate;
    
    let audio: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Int => {
            let max_val = 2f32.powi(reader.spec().bits_per_sample as i32 - 1);
            let raw: Vec<f32> = reader.samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect();
            if reader.spec().channels > 1 {
                let channels = reader.spec().channels as usize;
                raw.chunks_exact(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
            } else { raw }
        },
        hound::SampleFormat::Float => {
            let raw: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();
            if reader.spec().channels > 1 {
                let channels = reader.spec().channels as usize;
                raw.chunks_exact(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
            } else { raw }
        }
    };

    let chopped_path = Path::new(in_file).with_extension("chopped");
    let mut pitch_data = if chopped_path.exists() {
        if let Ok(file) = File::open(&chopped_path) {
            if let Ok(data) = bincode::deserialize_from::<_, PitchData>(file) {
                if data.sr == sr { data } else { extract_pitch_features(&audio, sr) }
            } else { extract_pitch_features(&audio, sr) }
        } else { extract_pitch_features(&audio, sr) }
    } else { extract_pitch_features(&audio, sr) };

    if fv == 1.0 {
        for i in 0..pitch_data.is_voiced.len() {
            if pitch_data.is_voiced[i] == 0.0 {
                pitch_data.is_voiced[i] = 1.0;
                pitch_data.t0_array[i] = sr as f32 / 100.0;
            }
        }
    }

    let (harm_audio, breath_audio, unvoiced_audio) = separate_components(&audio, sr, fv == 1.0);

    let a = (offset_s.max(0.0) * sr as f32) as usize;
    let mut b = if cutoff_s < 0.0 { a + (-cutoff_s * sr as f32) as usize } 
                else { audio.len().saturating_sub((cutoff_s * sr as f32) as usize) };
    if b <= a { b = a + (length_s * sr as f32) as usize; }
    b = b.clamp(a + 1, audio.len());
    let seg_len = b - a;

    // tryna match sillysampler's calculation of total = consonant*vel_factor + length
    let cons_n = (consonant_s.max(0.0) * sr as f32) as usize;
    let vel_factor = 2.0_f32.powf(1.0 - (velocity / 100.0));
    let cons_out = (cons_n as f32 * vel_factor).round() as usize;
    let tail_n = ((length_s * sr as f32) as usize).max(256);
    let out_n = (cons_out + tail_n).max(256);

    let time_map = build_time_map(seg_len, cons_n, cons_out, out_n, sr);

    let tick_dt = 60.0 / (tempo * 96.0);
    let t_pitch_sec = np_linspace(0.0, (bend_cents.len() as f32 - 1.0) * tick_dt, bend_cents.len());
    let t_audio_sec = np_linspace(0.0, (out_n as f32 - 1.0) / sr as f32, out_n);
    
    let pitch_at = if bend_cents.len() == 1 { vec![bend_cents[0]; out_n] } 
                   else { np_interp(&t_audio_sec, &t_pitch_sec, &bend_cents) };
                   
    let mut target_f0_hz = vec![0.0; out_n];
    for i in 0..out_n { target_f0_hz[i] = 440.0 * 2.0_f32.powf(((pitch_at[i] / 100.0 + t_off_cent / 100.0 + pitch_m) - 69.0) / 12.0); }

    let mut audio_out = td_psola_utau(
        &harm_audio, &breath_audio, &unvoiced_audio, sr, &target_f0_hz, &time_map, a,
        &pitch_data.epochs, &pitch_data.is_voiced, &pitch_data.t0_array,
        -g_gender / 10.0, (dg / 100.0) * 2.0, dgs, fg / 100.0,
        v_gain * b_duck, u_gain, b_gain, gg_intensity,
        tension, seed
    );

    let norm_target = 0.5_f32;
    let mut peak = 0.0_f32;
    for sample in &audio_out { if sample.abs() > peak { peak = sample.abs(); } }
    peak += 1e-9;
    let norm_gain = 1.0 + ((norm_target / peak) - 1.0) * p_norm;

    let mut mx = 1e-9_f32;
    for sample in &mut audio_out { *sample *= norm_gain * volume; if sample.abs() > mx { mx = sample.abs(); } }
    if mx > 1.0 { for sample in &mut audio_out { *sample /= mx; } }

    let spec = WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = WavWriter::create(out_file, spec).unwrap();
    for sample in audio_out { writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_overlap(mut signal: Vec<f32>, weights: &[f32]) -> Vec<f32> {
        for (sample, &weight) in signal.iter_mut().zip(weights) {
            if weight > 1e-4 { *sample /= weight; }
        }
        signal
    }

    #[test]
    fn shared_overlap_add_preserves_component_linearity() {
        let grain_len = 64;
        let harmonic: Vec<f32> = (0..grain_len)
            .map(|i| (2.0 * PI * i as f32 / 16.0).sin())
            .collect();
        let noise: Vec<f32> = (0..grain_len)
            .map(|i| 0.2 * (2.0 * PI * i as f32 / 7.0).cos())
            .collect();
        let combined: Vec<f32> = harmonic.iter().zip(&noise).map(|(&h, &n)| h + n).collect();
        let window = np_hanning(grain_len);

        let mut out_h = vec![0.0_f32; 160];
        let mut out_n = vec![0.0_f32; 160];
        let mut out_combined = vec![0.0_f32; 160];
        let mut weight_h = vec![0.0_f32; 160];
        let mut weight_n = vec![0.0_f32; 160];
        let mut weight_combined = vec![0.0_f32; 160];

        for &position in &[40.25_f32, 72.5, 104.75] {
            overlap_add_grain(&mut out_h, &mut weight_h, &harmonic, &window, position, 1.0);
            overlap_add_grain(&mut out_n, &mut weight_n, &noise, &window, position, 1.0);
            overlap_add_grain(
                &mut out_combined,
                &mut weight_combined,
                &combined,
                &window,
                position,
                1.0,
            );
        }

        let out_h = normalize_overlap(out_h, &weight_h);
        let out_n = normalize_overlap(out_n, &weight_n);
        let out_combined = normalize_overlap(out_combined, &weight_combined);
        for i in 0..out_combined.len() {
            assert!((out_h[i] + out_n[i] - out_combined[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn aperiodic_wola_reconstructs_an_identity_map() {
        let len = 4096;
        let source: Vec<f32> = (0..len)
            .map(|i| {
                0.4 * (2.0 * PI * i as f32 / 37.0).sin()
                    + 0.2 * (2.0 * PI * i as f32 / 13.0).cos()
            })
            .collect();
        let time_map: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let rendered = wola_aperiodic_from_map(&source, &time_map, len, 64);

        for i in 256..(len - 256) {
            assert!((rendered[i] - source[i]).abs() < 2e-5);
        }
    }

    #[test]
    fn aperiodic_wola_has_no_geometry_level_dip() {
        let source = vec![1.0_f32; 4096];
        let time_map: Vec<f32> = (0..2048)
            .map(|i| 800.0 + i as f32 * 0.23)
            .collect();
        let rendered = wola_aperiodic_from_map(&source, &time_map, time_map.len(), 64);

        for sample in &rendered[128..(rendered.len() - 128)] {
            assert!((*sample - 1.0).abs() < 2e-5);
        }
    }

    #[test]
    fn residual_render_is_independent_of_target_pitch() {
        let sr = 16_000_u32;
        let source_len = 4096;
        let harm = vec![0.0_f32; source_len];
        let unvoiced = vec![0.0_f32; source_len];
        let breath: Vec<f32> = (0..source_len)
            .map(|i| {
                0.25 * (2.0 * PI * i as f32 / 23.0).sin()
                    + 0.1 * (2.0 * PI * i as f32 / 7.0).cos()
            })
            .collect();
        let time_map: Vec<f32> = (0..2048).map(|i| i as f32 * 0.75).collect();
        let epochs: Vec<f32> = (0..26).map(|i| (i * 160) as f32).collect();
        let voiced = vec![1.0_f32; epochs.len()];
        let periods = vec![160.0_f32; epochs.len()];
        let low_pitch = vec![100.0_f32; time_map.len()];
        let high_pitch = vec![320.0_f32; time_map.len()];

        let render = |pitch: &[f32]| {
            td_psola_utau(
                &harm,
                &breath,
                &unvoiced,
                sr,
                pitch,
                &time_map,
                800,
                &epochs,
                &voiced,
                &periods,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1234,
            )
        };

        let low = render(&low_pitch);
        let high = render(&high_pitch);
        assert_eq!(low.len(), high.len());
        for (a, b) in low.iter().zip(&high) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn velocity_scaled_consonant_joins_loop_continuously() {
        let map = build_time_map(1000, 200, 400, 1200, 100);
        assert_eq!(map.len(), 1200);
        assert!((map[0] - 0.0).abs() < 1e-6);
        assert!((map[1199] - 999.0).abs() < 1e-4);
        assert!(map.windows(2).all(|pair| pair[1] >= pair[0]));
        assert!(map[399] < map[400]);
    }

    #[test]
    fn loop_map_ping_pongs_only_the_sustain() {
        let map = build_time_map(1000, 200, 200, 2001, 100);
        assert!((map[200] - 200.0).abs() < 1e-4);
        assert!((map[999] - 999.0).abs() < 1e-4);
        assert!((map[1798] - 200.0).abs() < 1e-4);
        assert!(map.iter().all(|&position| position >= 0.0 && position <= 999.0));
    }
}
