use hound::{WavReader, WavWriter, WavSpec};
use rayon::prelude::*;
use regex::Regex;
use rand_distr::{Normal, Distribution};
use walkdir::WalkDir;
use serde::{Serialize, Deserialize};
use std::env;
use std::fs::File;
use std::path::Path;
use std::f32::consts::PI;
use std::thread::available_parallelism;
use praatfan_core::Sound; 
use rustfft::{FftPlanner, num_complex::Complex};

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
                    if score > max_score { max_score = score; best_tau = tau; }
                }

                if max_score > 0.0 {
                    peak_idx = (expected_epoch as isize + best_tau) as usize;
                } else {
                    let search_start = t.saturating_sub(search_radius as usize);
                    let search_end = (t + search_radius as usize).min(audio.len());
                    let mut max_amp = -1.0;
                    for i in search_start..search_end {
                        if audio[i].abs() > max_amp { max_amp = audio[i].abs(); peak_idx = i; }
                    }
                }
            }

            epochs.push(peak_idx as f32);
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

fn separate_components(audio: &[f32], sr: u32, _pitch_data: &PitchData) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_fft = 1024;
    let hop_length = 256;
    let window = np_hanning_sqrt(n_fft);

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
        let f0 = pitch.get_value_at_frame(frame_idx).unwrap_or(0.0) as f32;
        raw_f0[f] = if f0 > 0.0 { f0 } else { 0.0 };
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

    let voiced_set: std::collections::HashSet<usize> = (0..num_frames).filter(|&f| raw_f0[f] > 0.0).collect();

    let mut harm_spectra = vec![vec![Complex::new(0.0, 0.0); n_fft / 2 + 1]; num_frames];
    let mut breath_spectra = vec![vec![Complex::new(0.0, 0.0); n_fft / 2 + 1]; num_frames];
    let mut unvoiced_spectra = vec![vec![Complex::new(0.0, 0.0); n_fft / 2 + 1]; num_frames];

    let max_voiced_freq = 13000.0_f32.min(sr as f32 / 2.0);

    for f in 0..num_frames {
        let voiced = voiced_set.contains(&f);
        let current_f0 = smooth_f0[f].clamp(50.0, 1200.0);

        let mut mag_db = vec![0.0; n_fft / 2 + 1];
        for b in 0..=n_fft / 2 { mag_db[b] = 20.0 * (zxx[f][b].norm() + 1e-12).log10(); }

        let mut mask = vec![0.0; n_fft / 2 + 1];

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
                    let mut max_val = -999.0;
                    let mut actual_bin = start_bin;
                    for b in start_bin..end_bin {
                        if mag_db[b] > max_val { max_val = mag_db[b]; actual_bin = b; }
                    }
                    let m_start = actual_bin.saturating_sub(1);
                    let m_end = (actual_bin + 2).min(n_fft / 2 + 1);
                    for b in m_start..m_end { mask[b] = 1.0; }
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

fn get_aligned_grain(audio: &[f32], center: f32, size: usize) -> Vec<f32> {
    let mut grain = vec![0.0; size];
    let start = center.round() as isize - (size as isize / 2);
    for i in 0..size as isize {
        let idx = start + i;
        if idx >= 0 && (idx as usize) < audio.len() {
            grain[i as usize] = audio[idx as usize];
        }
    }
    grain
}

fn td_psola_utau(
    harm_audio: &[f32],
    breath_audio: &[f32],
    unvoiced_audio: &[f32],
    orig_audio: &[f32],
    sr: u32,
    target_f0_hz: &[f32],
    time_map: &[f32],
    seg_start: usize,
    _seg_end: usize,
    epochs: &[f32],
    mut is_voiced: Vec<f32>,
    mut t0_array: Vec<f32>,
    formant_semitones: f32,
    force_voicing: bool,
    voice_drive: f32,
    drive_speed: f32,
    fry_intensity: f32,
    v_gain: f32,
    u_gain: f32,
    b_gain: f32,
    gg_intensity: f32,
    p_norm: f32,
) -> Vec<f32> {
    let formant_factor = 2.0_f32.powf(formant_semitones / 12.0);
    let hop_unvoiced = (0.01 * sr as f32) as usize;
    let mut rng = rand::thread_rng();
    let normal_dist = Normal::new(0.0, 1.0).unwrap();

    if force_voicing {
        for i in 0..is_voiced.len() {
            if is_voiced[i] == 0.0 {
                is_voiced[i] = 1.0;
                t0_array[i] = sr as f32 / 100.0;
            }
        }
    }

    let mut abs_time_map = Vec::with_capacity(time_map.len());
    for &t in time_map { abs_time_map.push(t + seg_start as f32); }

    let out_len = abs_time_map.len();
    let buf_len = out_len + (sr * 2) as usize;
    let mut output_harm = vec![0.0_f32; buf_len];
    let mut output_breath = vec![0.0_f32; buf_len];
    let mut output_unvoiced = vec![0.0_f32; buf_len];

    let mut t_s = 0.0_f32;
    let mut drive_phase = 0.0_f32;
    let mut prev_t_s = hop_unvoiced as f32;

    while (t_s as usize) < out_len {
        let t_a = abs_time_map[t_s as usize];
        if t_a >= (orig_audio.len() - 1) as f32 { break; }

        let mut idx1 = searchsorted(epochs, t_a).saturating_sub(1);
        idx1 = idx1.clamp(0, epochs.len().saturating_sub(2));
        let idx2 = idx1 + 1;

        let diff = epochs[idx2] - epochs[idx1];
        let weight = if diff > 0.0 { (t_a - epochs[idx1]) / diff } else { 0.0 };

        let voicing_mix = (1.0 - weight) * is_voiced[idx1] + weight * is_voiced[idx2];
        let t0_interp = (1.0 - weight) * t0_array[idx1] + weight * t0_array[idx2];
        let current_target_hz = target_f0_hz[t_s as usize];

        let mut t_s_target = hop_unvoiced as f32;
        if current_target_hz > 0.0 {
            t_s_target = voicing_mix * (sr as f32 / current_target_hz) + (1.0 - voicing_mix) * hop_unvoiced as f32;
        }

        let max_delta = hop_unvoiced as f32 * 0.5;
        let mut t_s_step = t_s_target;
        if t_s_target > prev_t_s + max_delta { t_s_step = prev_t_s + max_delta; }
        else if t_s_target < prev_t_s - max_delta { t_s_step = prev_t_s - max_delta; }

        let mut fry_offset = 0.0_f32;
        let mut fry_amp = 1.0_f32;

        // --- PATH A: VOICED ---
        if voicing_mix > 0.0 {
            let mut extract_win_size_v = (2.0 * t0_interp).round() as usize;
            extract_win_size_v += extract_win_size_v % 2;

            let g1_h = get_aligned_grain(harm_audio, epochs[idx1], extract_win_size_v);
            let g2_h = get_aligned_grain(harm_audio, epochs[idx2], extract_win_size_v);
            
            let mut morphed_harm = vec![0.0; extract_win_size_v];
            for i in 0..extract_win_size_v { morphed_harm[i] = (1.0 - weight) * g1_h[i] + weight * g2_h[i]; }

            let source_rms = (morphed_harm.iter().map(|x| x * x).sum::<f32>() / extract_win_size_v as f32).sqrt() + 1e-12;

            let mut shifted_harm = morphed_harm;
            if (formant_factor - 1.0).abs() > 0.001 {
                let orig_idx = np_linspace(0.0, 1.0, shifted_harm.len());
                let mut new_len = (shifted_harm.len() as f32 / formant_factor).round() as usize;
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

            let ts_pos = (t_s + fry_offset).round() as isize;
            let start_s_h = ts_pos - (shifted_harm.len() as isize / 2);
            for i in 0..shifted_harm.len() {
                let out_idx = start_s_h + i as isize;
                if out_idx >= 0 && (out_idx as usize) < buf_len {
                    output_harm[out_idx as usize] += shifted_harm[i] * gain * voicing_mix;
                }
            }
        }

        // --- PATH B: UNVOICED ---
        if voicing_mix < 1.0 {
            let required_final_size = t_s_step.max(prev_t_s) * 2.0;
            let mut extract_win_size_u = (hop_unvoiced as f32 * 2.0).max(required_final_size).round() as usize;
            extract_win_size_u += extract_win_size_u % 2;

            let mut stretch_comp = 1.0;
            if extract_win_size_u as f32 > required_final_size {
                stretch_comp = required_final_size / extract_win_size_u as f32;
            }

            let start = t_a as isize - (extract_win_size_u as isize / 2);
            let mut morphed_pulse_u = vec![0.0; extract_win_size_u];
            for i in 0..extract_win_size_u as isize {
                let idx = start + i;
                if idx >= 0 && idx < unvoiced_audio.len() as isize {
                    morphed_pulse_u[i as usize] = unvoiced_audio[idx as usize];
                }
            }

            let hanning = np_hanning(extract_win_size_u);
            let ts_pos = t_s.round() as isize;
            let start_s = ts_pos - (extract_win_size_u as isize / 2);

            for i in 0..extract_win_size_u {
                let out_idx = start_s + i as isize;
                if out_idx >= 0 && (out_idx as usize) < buf_len {
                    output_unvoiced[out_idx as usize] += morphed_pulse_u[i] * hanning[i]
                        * stretch_comp * (1.0 - voicing_mix);
                }
            }
        }

        t_s += t_s_step;
        prev_t_s = t_s_step;
    }

    let breath_hop = hop_unvoiced as f32;
    let breath_win = hop_unvoiced * 2;
    let breath_hanning = np_hanning(breath_win);
    let mut t_s_b = 0.0_f32;

    while (t_s_b as usize) < out_len {
        let t_a = abs_time_map[t_s_b as usize];
        if t_a >= (orig_audio.len() - 1) as f32 { break; }

        let mut idx1 = searchsorted(epochs, t_a).saturating_sub(1);
        idx1 = idx1.clamp(0, epochs.len().saturating_sub(2));
        let idx2 = idx1 + 1;
        
        let diff = epochs[idx2] - epochs[idx1];
        let weight = if diff > 0.0 { (t_a - epochs[idx1]) / diff } else { 0.0 };
        let voicing_mix = (1.0 - weight) * is_voiced[idx1] + weight * is_voiced[idx2];

        if voicing_mix > 0.0 {
            let g_b = get_aligned_grain(breath_audio, t_a, breath_win);
            let ts_pos = t_s_b.round() as isize;
            let start_s = ts_pos - (breath_win as isize / 2);

            for i in 0..breath_win {
                let out_idx = start_s + i as isize;
                if out_idx >= 0 && (out_idx as usize) < buf_len {
                    output_breath[out_idx as usize] += g_b[i] * breath_hanning[i] * voicing_mix;
                }
            }
        }
        t_s_b += breath_hop;
    }

    let mut actual_len = buf_len;
    while actual_len > out_len && output_harm[actual_len - 1].abs() + output_breath[actual_len - 1].abs() + output_unvoiced[actual_len - 1].abs() < 1e-6 {
        actual_len -= 1;
    }
    output_harm.truncate(actual_len);
    output_breath.truncate(actual_len);
    output_unvoiced.truncate(actual_len);

    let mut norm_mult = 1.0_f32;
    if p_norm > 0.0 {
        let mut raw_peak = 0.0_f32;
        for i in 0..actual_len {
            let raw = (output_harm[i] + output_breath[i] + output_unvoiced[i]).abs();
            if raw > raw_peak { raw_peak = raw; }
        }
        raw_peak += 1e-9;
        norm_mult = p_norm / raw_peak;
    }

    let mut output = Vec::with_capacity(actual_len);
    for i in 0..actual_len {
        let h = output_harm[i] * norm_mult * v_gain;
        let b = output_breath[i] * norm_mult * b_gain;
        let u = output_unvoiced[i] * norm_mult * u_gain;
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
    let mut gg_intensity = 0.0;
    let mut p_norm = 0.0;

    // general gain curve
    let scale_gain_u = |val: f32| -> f32 {
        if val <= -100.0 { 0.0 }
        else if val < 0.0 { 1.0 + val / 100.0 }
        else { 1.0 + (val / 20.0) }
    };

    // V flag (Harmonic strength) linear

    // aggressive exponential curve
    let scale_gain_b = |val: f32| -> f32 {
        if val <= -100.0 { 0.0 }
        else { 10.0_f32.powf(val / 50.0) }
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
            "B" => b_gain = scale_gain_b(val),
            "gg" => gg_intensity = val.clamp(0.0, 100.0) / 100.0,
            "P" => p_norm = if val_str.is_empty() { 1.0 } else { val.clamp(0.0, 100.0) / 100.0 },
            _ => {}
        }
    }

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
    let pitch_data = if chopped_path.exists() {
        if let Ok(file) = File::open(&chopped_path) {
            if let Ok(data) = bincode::deserialize_from::<_, PitchData>(file) {
                if data.sr == sr { data } else { extract_pitch_features(&audio, sr) }
            } else { extract_pitch_features(&audio, sr) }
        } else { extract_pitch_features(&audio, sr) }
    } else { extract_pitch_features(&audio, sr) };

    let (harm_audio, breath_audio, unvoiced_audio) = separate_components(&audio, sr, &pitch_data);

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

    let mut time_map = vec![0.0; out_n];

    if cons_out > 0 && cons_n > 0 {
        let lin = np_linspace(0.0, cons_n as f32, cons_out);
        for i in 0..cons_out { time_map[i] = lin[i]; }
    }

    if out_n > cons_out {
        let tail_src_len = seg_len.saturating_sub(cons_n).max(1);
        let lin = np_linspace(cons_n as f32, (cons_n + tail_src_len - 1) as f32, out_n - cons_out);
        for i in 0..(out_n - cons_out) { time_map[cons_out + i] = lin[i]; }
    }

    let tick_dt = 60.0 / (tempo * 96.0);
    let t_pitch_sec = np_linspace(0.0, (bend_cents.len() as f32 - 1.0) * tick_dt, bend_cents.len());
    let t_audio_sec = np_linspace(0.0, (out_n as f32 - 1.0) / sr as f32, out_n);
    
    let pitch_at = if bend_cents.len() == 1 { vec![bend_cents[0]; out_n] } 
                   else { np_interp(&t_audio_sec, &t_pitch_sec, &bend_cents) };
                   
    let mut target_f0_hz = vec![0.0; out_n];
    for i in 0..out_n { target_f0_hz[i] = 440.0 * 2.0_f32.powf(((pitch_at[i] / 100.0 + pitch_m) - 69.0) / 12.0); }

    let mut audio_out = td_psola_utau(
        &harm_audio, &breath_audio, &unvoiced_audio, &audio, sr, &target_f0_hz, &time_map, a, b,
        &pitch_data.epochs, pitch_data.is_voiced, pitch_data.t0_array,
        -g_gender / 10.0, fv == 1.0, (dg / 100.0) * 2.0, dgs, fg / 100.0,
        v_gain, u_gain, b_gain, gg_intensity, p_norm
    );

    let mut mx = 1e-9_f32;
    for sample in &mut audio_out { *sample *= volume; if sample.abs() > mx { mx = sample.abs(); } }
    if mx > 1.0 { for sample in &mut audio_out { *sample /= mx; } }

    let spec = WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = WavWriter::create(out_file, spec).unwrap();
    for sample in audio_out { writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap(); }
}
