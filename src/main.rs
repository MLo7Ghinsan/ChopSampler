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
        let x0 = xp[idx];
        let x1 = xp[idx + 1];
        let f0 = fp[idx];
        let f1 = fp[idx + 1];
        
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
                    
                    // Safe Bounds checking
                    if target_center - half_win_i < 0 || target_center + half_win_i >= audio_len_i {
                        continue;
                    }
                    if prev_epoch_i - half_win_i < 0 || prev_epoch + half_win >= audio.len() {
                        continue;
                    }

                    let mut cross_corr = 0.0;
                    let mut energy = 0.0;

                    for i in -half_win_i..=half_win_i {
                        let src_idx = (prev_epoch_i + i) as usize;
                        let tgt_idx = (target_center + i) as usize;
                        
                        let src_val = audio[src_idx];
                        let tgt_val = audio[tgt_idx];
                        
                        cross_corr += src_val * tgt_val;
                        energy += tgt_val * tgt_val;
                    }

                    let score = if energy > 0.0 { cross_corr / energy.sqrt() } else { 0.0 };
                    
                    if score > max_score {
                        max_score = score;
                        best_tau = tau;
                    }
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
    full_audio: &[f32],
    sr: u32,
    target_f0_hz: &[f32],
    time_map: &[f32],
    seg_start: usize,
    seg_end: usize,
    epochs: &[f32],
    mut is_voiced: Vec<f32>,
    mut t0_array: Vec<f32>,
    formant_semitones: f32,
    force_voicing: bool,
    voice_drive: f32,
    drive_speed: f32,
    fry_intensity: f32,
    norm_val: f32,
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
    let mut output = vec![0.0; out_len + (sr * 2) as usize];

    let mut t_s = 0.0_f32;
    let mut drive_phase = 0.0_f32;
    let mut prev_t_s = hop_unvoiced as f32;

    while (t_s as usize) < out_len {
        let t_a = abs_time_map[t_s as usize];
        if t_a >= (full_audio.len() - 1) as f32 { break; }

        let mut idx1 = searchsorted(epochs, t_a).saturating_sub(1);
        idx1 = idx1.clamp(0, epochs.len().saturating_sub(2));
        let idx2 = idx1 + 1;

        let diff = epochs[idx2] - epochs[idx1];
        let weight = if diff > 0.0 { (t_a - epochs[idx1]) / diff } else { 0.0 };

        let v1 = is_voiced[idx1];
        let v2 = is_voiced[idx2];
        let voicing_mix = (1.0 - weight) * v1 + weight * v2;
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

            let g1 = get_aligned_grain(full_audio, epochs[idx1], extract_win_size_v);
            let g2 = get_aligned_grain(full_audio, epochs[idx2], extract_win_size_v);
            
            let mut morphed_pulse = vec![0.0; extract_win_size_v];
            for i in 0..extract_win_size_v {
                morphed_pulse[i] = (1.0 - weight) * g1[i] + weight * g2[i];
            }

            let source_rms = (morphed_pulse.iter().map(|x| x * x).sum::<f32>() / extract_win_size_v as f32).sqrt() + 1e-12;

            let mut shifted_pulse = morphed_pulse;
            if (formant_factor - 1.0).abs() > 0.001 {
                let orig_idx = np_linspace(0.0, 1.0, shifted_pulse.len());
                let mut new_len = (shifted_pulse.len() as f32 / formant_factor).round() as usize;
                new_len += new_len % 2;
                let new_idx = np_linspace(0.0, 1.0, new_len);
                shifted_pulse = np_interp(&new_idx, &orig_idx, &shifted_pulse);
            }

            let hanning = np_hanning(shifted_pulse.len());
            for i in 0..shifted_pulse.len() { shifted_pulse[i] *= hanning[i]; }

            let current_rms = (shifted_pulse.iter().map(|x| x * x).sum::<f32>() / shifted_pulse.len() as f32).sqrt() + 1e-12;
            let density_comp = (t_s_step.max(1.0) / t0_interp.max(1.0)).sqrt();
            let gain = ((source_rms / current_rms) * density_comp).clamp(0.0, 5.0);
            
            for sample in &mut shifted_pulse { *sample *= gain; }

            if voice_drive > 0.0 {
                drive_phase += 2.0 * PI * drive_speed * (t_s_step / sr as f32);
                let drive_mod = 1.0 + (drive_phase.sin() * voice_drive);
                for sample in &mut shifted_pulse { *sample *= drive_mod; }
            }

            if fry_intensity > 0.0 {
                let rand_val = normal_dist.sample(&mut rng);
                fry_offset = rand_val * t0_interp * 0.12 * fry_intensity;
                if (t_s / t_s_step) as i32 % 2 == 0 { fry_amp = 1.0 - (0.5 * fry_intensity); }
                for sample in &mut shifted_pulse { *sample *= fry_amp; }
            }

            for sample in &mut shifted_pulse { *sample *= voicing_mix; }

            let ts_pos = (t_s + fry_offset).round() as isize;
            let start_s = ts_pos - (shifted_pulse.len() as isize / 2);

            for (i, &sample) in shifted_pulse.iter().enumerate() {
                let out_idx = start_s + i as isize;
                if out_idx >= 0 && (out_idx as usize) < output.len() {
                    output[out_idx as usize] += sample;
                }
            }
        }

        // --- PATH B: UNVOICED ---
        if voicing_mix < 1.0 {
            let required_final_size = t_s_step.max(prev_t_s) * 2.0;
            let mut extract_win_size_u = (hop_unvoiced as f32 * 2.0).max(required_final_size).round() as usize;
            extract_win_size_u += extract_win_size_u % 2;

            let mut unvoiced_gain = 1.0 - voicing_mix;
            if extract_win_size_u as f32 > required_final_size {
                unvoiced_gain *= required_final_size / extract_win_size_u as f32;
            }

            let read_pos = t_a as isize;
            let start = read_pos - (extract_win_size_u as isize / 2);

            let mut morphed_pulse_u = vec![0.0; extract_win_size_u];
            for i in 0..extract_win_size_u as isize {
                let idx = start + i;
                if idx >= 0 && idx < full_audio.len() as isize {
                    morphed_pulse_u[i as usize] = full_audio[idx as usize];
                }
            }

            let hanning = np_hanning(extract_win_size_u);
            for i in 0..extract_win_size_u {
                morphed_pulse_u[i] *= hanning[i] * unvoiced_gain;
            }

            let ts_pos = t_s.round() as isize;
            let start_s = ts_pos - (extract_win_size_u as isize / 2);

            for (i, &sample) in morphed_pulse_u.iter().enumerate() {
                let out_idx = start_s + i as isize;
                if out_idx >= 0 && (out_idx as usize) < output.len() {
                    output[out_idx as usize] += sample;
                }
            }
        }

        t_s += t_s_step;
        prev_t_s = t_s_step;
    }

    let mut actual_len = output.len();
    while actual_len > out_len && output[actual_len - 1].abs() < 1e-6 {
        actual_len -= 1;
    }
    output.truncate(actual_len);

    let out_max = output.iter().fold(0.0_f32, |m, x| m.max(x.abs())) + 1e-9;
    let orig_max = full_audio[seg_start..seg_end.min(full_audio.len())].iter().fold(0.0_f32, |m, x| m.max(x.abs())) + 1e-9;
    
    for sample in &mut output {
        let base_output = (*sample / out_max) * orig_max;
        let peak_output = *sample / out_max;
        *sample = base_output * (1.0 - norm_val) + peak_output * norm_val;
    }
    output
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
    let mut fg = 0.0; let mut p_norm = 0.0; let mut g_gender = 0.0;

    for cap in Regex::new(r"([a-zA-Z]+)([-+]?\d*)").unwrap().captures_iter(flags) {
        let val = if &cap[2] == "" { 1.0 } else { cap[2].parse::<f32>().unwrap_or(0.0) };
        match &cap[1] {
            "fv" => fv = val.clamp(0.0, 1.0), "dg" => dg = val.clamp(0.0, 100.0),
            "dgs" => dgs = val.clamp(0.0, 100.0), "fg" => fg = val.clamp(0.0, 100.0),
            "P" => p_norm = val.clamp(0.0, 100.0), "g" => g_gender = val, _ => {}
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

    let a = (offset_s.max(0.0) * sr as f32) as usize;
    let mut b = if cutoff_s < 0.0 { a + (-cutoff_s * sr as f32) as usize } 
                else { audio.len().saturating_sub((cutoff_s * sr as f32) as usize) };
    if b <= a { b = a + (length_s * sr as f32) as usize; }
    b = b.clamp(a + 1, audio.len());
    let seg_len = b - a;

    let out_n = ((length_s * sr as f32) as usize).max(256);
    let cons_n = (consonant_s.max(0.0) * sr as f32) as usize;
    let vel_factor = 2.0_f32.powf(1.0 - (velocity / 100.0));
    let cons_out = ((cons_n as f32 * vel_factor).round() as usize).clamp(0, out_n);

    let mut time_map = vec![0.0; out_n];
    if cons_out > 0 {
        let lin = np_linspace(0.0, cons_n as f32, cons_out);
        for i in 0..cons_out { time_map[i] = lin[i]; }
    }
    if out_n > cons_out {
        let lin = np_linspace(cons_n as f32, (seg_len - 1) as f32, out_n - cons_out);
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
        &audio, sr, &target_f0_hz, &time_map, a, b,
        &pitch_data.epochs, pitch_data.is_voiced, pitch_data.t0_array,
        -g_gender / 10.0, fv == 1.0, (dg / 100.0) * 2.0, dgs, fg / 100.0, p_norm / 100.0
    );

    let mut mx = 1e-9_f32;
    for sample in &mut audio_out { *sample *= volume; if sample.abs() > mx { mx = sample.abs(); } }
    if mx > 1.0 { for sample in &mut audio_out { *sample /= mx; } }

    let spec = WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = WavWriter::create(out_file, spec).unwrap();
    for sample in audio_out { writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap(); }
}