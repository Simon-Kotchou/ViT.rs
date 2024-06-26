use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem;
use std::os::raw::c_int;

const NUM_PARAMETER_TENSORS: usize = 16;
const NUM_ACTIVATION_TENSORS: usize = 23;

#[repr(C)]
struct ParameterTensors {
    wte: *mut f32,
    wpe: *mut f32,
    ln1w: *mut f32,
    ln1b: *mut f32,
    qkvw: *mut f32,
    qkvb: *mut f32,
    attprojw: *mut f32,
    attprojb: *mut f32,
    ln2w: *mut f32,
    ln2b: *mut f32,
    fcw: *mut f32,
    fcb: *mut f32,
    fcprojw: *mut f32,
    fcprojb: *mut f32,
    lnfw: *mut f32,
    lnfb: *mut f32,
}

#[repr(C)]
struct ActivationTensors {
    encoded: *mut f32,
    ln1: *mut f32,
    ln1_mean: *mut f32,
    ln1_rstd: *mut f32,
    qkv: *mut f32,
    atty: *mut f32,
    preatt: *mut f32,
    att: *mut f32,
    attproj: *mut f32,
    residual2: *mut f32,
    ln2: *mut f32,
    ln2_mean: *mut f32,
    ln2_rstd: *mut f32,
    fch: *mut f32,
    fch_gelu: *mut f32,
    fcproj: *mut f32,
    residual3: *mut f32,
    lnf: *mut f32,
    lnf_mean: *mut f32,
    lnf_rstd: *mut f32,
    logits: *mut f32,
    probs: *mut f32,
    losses: *mut f32,
}

#[derive(Debug, Clone, Copy)]
struct ViTConfig {
    max_seq_len: c_int,
    vocab_size: c_int,
    num_layers: c_int,
    num_heads: c_int,
    channels: c_int,
}

struct ViT {
    config: ViTConfig,
    params: ParameterTensors,
    param_sizes: [usize; NUM_PARAMETER_TENSORS],
    params_memory: *mut f32,
    num_parameters: usize,
    grads: ParameterTensors,
    grads_memory: *mut f32,
    m_memory: *mut f32,
    v_memory: *mut f32,
    acts: ActivationTensors,
    act_sizes: [usize; NUM_ACTIVATION_TENSORS],
    acts_memory: *mut f32,
    num_activations: usize,
    grads_acts: ActivationTensors,
    grads_acts_memory: *mut f32,
    batch_size: c_int,
    seq_len: c_int,
    inputs: *mut c_int,
    targets: *mut c_int,
    mean_loss: f32,
}

impl ViT {
    fn build_from_checkpoint(checkpoint_path: &str) -> Self {
        let mut model_file = File::open(checkpoint_path).expect("Error opening model file");
        let mut model_header = [0; 256];
        model_file.read_exact(&mut model_header).expect("Error reading model header");
        
        let max_seq_len = model_header[2] as c_int;
        let vocab_size = model_header[3] as c_int;
        let num_layers = model_header[4] as c_int;
        let num_heads = model_header[5] as c_int;
        let channels = model_header[6] as c_int;
        
        println!("[ViT]");
        println!("max_seq_len: {}", max_seq_len);
        println!("vocab_size: {}", vocab_size);
        println!("num_layers: {}", num_layers);
        println!("num_heads: {}", num_heads);
        println!("channels: {}", channels);
        
        let config = ViTConfig {
            max_seq_len,
            vocab_size,
            num_layers,
            num_heads,
            channels,
        };
        
        let mut param_sizes = [0; NUM_PARAMETER_TENSORS];
        param_sizes[0] = (vocab_size * channels) as usize;
        param_sizes[1] = (max_seq_len * channels) as usize;
        param_sizes[2] = (num_layers * channels) as usize;
        param_sizes[3] = (num_layers * channels) as usize;
        param_sizes[4] = (num_layers * (3 * channels) * channels) as usize;
        param_sizes[5] = (num_layers * (3 * channels)) as usize;
        param_sizes[6] = (num_layers * channels * channels) as usize;
        param_sizes[7] = (num_layers * channels) as usize;
        param_sizes[8] = (num_layers * channels) as usize;
        param_sizes[9] = (num_layers * channels) as usize;
        param_sizes[10] = (num_layers * (4 * channels) * channels) as usize;
        param_sizes[11] = (num_layers * (4 * channels)) as usize;
        param_sizes[12] = (num_layers * channels * (4 * channels)) as usize;
        param_sizes[13] = (num_layers * channels) as usize;
        param_sizes[14] = channels as usize;
        param_sizes[15] = channels as usize;
        
        let num_parameters: usize = param_sizes.iter().sum();
        println!("num_parameters: {}", num_parameters);
        
        let params_memory = unsafe {
            let layout = std::alloc::Layout::array::<f32>(num_parameters).unwrap();
            std::alloc::alloc(layout) as *mut f32
        };
        
        model_file.seek(SeekFrom::Start(1024)).expect("Error seeking in model file");
        let mut params_memory_slice = unsafe { std::slice::from_raw_parts_mut(params_memory, num_parameters) };
        model_file.read_exact(bytemuck::cast_slice_mut(params_memory_slice)).expect("Error reading model parameters");
        
        let params = ParameterTensors {
            wte: params_memory,
            wpe: unsafe { params_memory.offset(param_sizes[0] as isize) },
            ln1w: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1]) as isize) },
            ln1b: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2]) as isize) },
            qkvw: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3]) as isize) },
            qkvb: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4]) as isize) },
            attprojw: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5]) as isize) },
            attprojb: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6]) as isize) },
            ln2w: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7]) as isize) },
            ln2b: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8]) as isize) },
            fcw: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9]) as isize) },
            fcb: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10]) as isize) },
            fcprojw: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11]) as isize) },
            fcprojb: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12]) as isize) },
            lnfw: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13]) as isize) },
            lnfb: unsafe { params_memory.offset((param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13] + param_sizes[14]) as isize) },
        };
        
        Self {
            config,
            params,
            param_sizes,
            params_memory,
            num_parameters,
            grads: unsafe { mem::zeroed() },
            grads_memory: std::ptr::null_mut(),
            m_memory: std::ptr::null_mut(),
            v_memory: std::ptr::null_mut(),
            acts: unsafe { mem::zeroed() },
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: std::ptr::null_mut(),
            num_activations: 0,
            grads_acts: unsafe { mem::zeroed() },
            grads_acts_memory: std::ptr::null_mut(),
            batch_size: 0,
            seq_len: 0,
            inputs: std::ptr::null_mut(),
            targets: std::ptr::null_mut(),
            mean_loss: -1.0,
        }
    }
    
    fn forward(&mut self, inputs: *const c_int, targets: *const c_int, b: c_int, t: c_int) {
        // ... (previous code for memory allocation and input/target copying)

        let params = &self.params;
        let acts = &mut self.acts;

        // Forward pass
        unsafe {
            encoder_forward(acts.encoded, inputs, params.wte, params.wpe, b, t, c);

            let mut residual: *mut f32 = std::ptr::null_mut();
            for l in 0..l {
                residual = if l == 0 {
                    acts.encoded
                } else {
                    acts.residual3.offset((l - 1) * b * t * c)
                };

                let l_ln1w = params.ln1w.offset(l * c);
                let l_ln1b = params.ln1b.offset(l * c);
                let l_qkvw = params.qkvw.offset(l * 3 * c * c);
                let l_qkvb = params.qkvb.offset(l * 3 * c);
                let l_attprojw = params.attprojw.offset(l * c * c);
                let l_attprojb = params.attprojb.offset(l * c);
                let l_ln2w = params.ln2w.offset(l * c);
                let l_ln2b = params.ln2b.offset(l * c);
                let l_fcw = params.fcw.offset(l * 4 * c * c);
                let l_fcb = params.fcb.offset(l * 4 * c);
                let l_fcprojw = params.fcprojw.offset(l * c * 4 * c);
                let l_fcprojb = params.fcprojb.offset(l * c);

                let l_ln1 = acts.ln1.offset(l * b * t * c);
                let l_ln1_mean = acts.ln1_mean.offset(l * b * t);
                let l_ln1_rstd = acts.ln1_rstd.offset(l * b * t);
                let l_qkv = acts.qkv.offset(l * b * t * 3 * c);
                let l_atty = acts.atty.offset(l * b * t * c);
                let l_preatt = acts.preatt.offset(l * b * nh * t * t);
                let l_att = acts.att.offset(l * b * nh * t * t);
                let l_attproj = acts.attproj.offset(l * b * t * c);
                let l_residual2 = acts.residual2.offset(l * b * t * c);
                let l_ln2 = acts.ln2.offset(l * b * t * c);
                let l_ln2_mean = acts.ln2_mean.offset(l * b * t);
                let l_ln2_rstd = acts.ln2_rstd.offset(l * b * t);
                let l_fch = acts.fch.offset(l * b * t * 4 * c);
                let l_fch_gelu = acts.fch_gelu.offset(l * b * t * 4 * c);
                let l_fcproj = acts.fcproj.offset(l * b * t * c);
                let l_residual3 = acts.residual3.offset(l * b * t * c);

                layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, b, t, c);
                matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, b, t, c, 3 * c);
                attention_forward(l_atty, l_preatt, l_att, l_qkv, b, t, c, nh);
                matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, b, t, c, c);
                residual_forward(l_residual2, residual, l_attproj, b * t * c);
                layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, b, t, c);
                matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, b, t, c, 4 * c);
                gelu_forward(l_fch_gelu, l_fch, b * t * 4 * c);
                matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, b, t, 4 * c, c);
                residual_forward(l_residual3, l_residual2, l_fcproj, b * t * c);
            }

            residual = acts.residual3.offset((l - 1) * b * t * c);
            layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, b, t, c);
            matmul_forward(acts.logits, acts.lnf, params.wte, std::ptr::null_mut(), b, t, c, v);
            softmax_forward(acts.probs, acts.logits, b, t, v);
        }

        if !targets.is_null() {
            unsafe {
                crossentropy_forward(acts.losses, acts.probs, targets, b, t, v);

                let mut mean_loss = 0.0;
                for i in 0..(b * t) {
                    mean_loss += *acts.losses.offset(i as isize);
                }
                mean_loss /= (b * t) as f32;
                self.mean_loss = mean_loss;
            }
        } else {
            self.mean_loss = -1.0;
        }
    }

    
    fn backward(&mut self) {
        // ... (previous code for memory allocation and gradient zeroing)

        let b = self.batch_size;
        let t = self.seq_len;
        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let c = self.config.channels;

        let params = &self.params;
        let grads = &mut self.grads;
        let acts = &self.acts;
        let grads_acts = &mut self.grads_acts;

        // Backward pass
        unsafe {
            let dloss_mean = 1.0 / (b * t) as f32;
            for i in 0..(b * t) {
                *grads_acts.losses.offset(i as isize) = dloss_mean;
            }

            crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, self.targets, b, t, v);
            matmul_backward(grads_acts.lnf, grads.wte, std::ptr::null_mut(), grads_acts.logits, acts.lnf, params.wte, b, t, c, v);

            let mut residual = acts.residual3.offset((l - 1) * b * t * c);
            let mut dresidual = grads_acts.residual3.offset((l - 1) * b * t * c);

            layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, b, t, c);

            for l in (0..l).rev() {
                residual = if l == 0 {
                    acts.encoded
                } else {
                    acts.residual3.offset((l - 1) * b * t * c)
                };
                dresidual = if l == 0 {
                    grads_acts.encoded
                } else {
                    grads_acts.residual3.offset((l - 1) * b * t * c)
                };

                let l_ln1w = params.ln1w.offset(l * c);
                let l_qkvw = params.qkvw.offset(l * 3 * c * c);
                let l_attprojw = params.attprojw.offset(l * c * c);
                let l_ln2w = params.ln2w.offset(l * c);
                let l_fcw = params.fcw.offset(l * 4 * c * c);
                let l_fcprojw = params.fcprojw.offset(l * c * 4 * c);

                let dl_ln1w = grads.ln1w.offset(l * c);
                let dl_ln1b = grads.ln1b.offset(l * c);
                let dl_qkvw = grads.qkvw.offset(l * 3 * c * c);
                let dl_qkvb = grads.qkvb.offset(l * 3 * c);
                let dl_attprojw = grads.attprojw.offset(l * c * c);
                let dl_attprojb = grads.attprojb.offset(l * c);
                let dl_ln2w = grads.ln2w.offset(l * c);
                let dl_ln2b = grads.ln2b.offset(l * c);
                let dl_fcw = grads.fcw.offset(l * 4 * c * c);
                let dl_fcb = grads.fcb.offset(l * 4 * c);
                let dl_fcprojw = grads.fcprojw.offset(l * c * 4 * c);
                let dl_fcprojb = grads.fcprojb.offset(l * c);

                let l_ln1 = acts.ln1.offset(l * b * t * c);
                let l_ln1_mean = acts.ln1_mean.offset(l * b * t);
                let l_ln1_rstd = acts.ln1_rstd.offset(l * b * t);
                let l_qkv = acts.qkv.offset(l * b * t * 3 * c);
                let l_atty = acts.atty.offset(l * b * t * c);
                let l_att = acts.att.offset(l * b * nh * t * t);
                let l_residual2 = acts.residual2.offset(l * b * t * c);
                let l_ln2 = acts.ln2.offset(l * b * t * c);
                let l_ln2_mean = acts.ln2_mean.offset(l * b * t);
                let l_ln2_rstd = acts.ln2_rstd.offset(l * b * t);
                let l_fch = acts.fch.offset(l * b * t * 4 * c);
                let l_fch_gelu = acts.fch_gelu.offset(l * b * t * 4 * c);

                let dl_ln1 = grads_acts.ln1.offset(l * b * t * c);
                let dl_qkv = grads_acts.qkv.offset(l * b * t * 3 * c);
                let dl_atty = grads_acts.atty.offset(l * b * t * c);
                let dl_preatt = grads_acts.preatt.offset(l * b * nh * t * t);
                let dl_att = grads_acts.att.offset(l * b * nh * t * t);
                let dl_attproj = grads_acts.attproj.offset(l * b * t * c);
                let dl_residual2 = grads_acts.residual2.offset(l * b * t * c);
                let dl_ln2 = grads_acts.ln2.offset(l * b * t * c);
                let dl_fch = grads_acts.fch.offset(l * b * t * 4 * c);
                let dl_fch_gelu = grads_acts.fch_gelu.offset(l * b * t * 4 * c);
                let dl_fcproj = grads_acts.fcproj.offset(l * b * t * c);
                let dl_residual3 = grads_acts.residual3.offset(l * b * t * c);

                residual_backward(dl_residual2, dl_fcproj, dl_residual3, b * t * c);
                matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, b, t, 4 * c, c);
                gelu_backward(dl_fch, l_fch, dl_fch_gelu, b * t * 4 * c);
                matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, b, t, c, 4 * c);
                layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, b, t, c);
                residual_backward(dresidual, dl_attproj, dl_residual2, b * t * c);
                matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, b, t, c, c);
                attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, b, t, c, nh);
                matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, b, t, c, 3 * c);
                layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, b, t, c);
            }

            encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, self.inputs, b, t, c);
        }
    }
}

fn residual_forward(out: *mut f32, inp1: *const f32, inp2: *const f32, n: c_int) {
    unsafe {
        for i in 0..n {
            *out.offset(i as isize) = *inp1.offset(i as isize) + *inp2.offset(i as isize);
        }
    }
}

fn matmul_forward(out: *mut f32, inp: *const f32, weight: *const f32, bias: *const f32, b: c_int, t: c_int, c: c_int, oc: c_int) {
    unsafe {
        for bt in 0..(b * t) {
            for o in 0..oc {
                let mut val = if !bias.is_null() { *bias.offset(o as isize) } else { 0.0 };
                let wrow = weight.offset((o * c) as isize);
                let inp_bt = inp.offset((bt * c) as isize);
                for i in 0..c {
                    val += *inp_bt.offset(i as isize) * *wrow.offset(i as isize);
                }
                *out.offset((bt * oc + o) as isize) = val;
            }
        }
    }
}

fn attention_forward(out: *mut f32, preatt: *mut f32, att: *mut f32, inp: *const f32, b: c_int, t: c_int, c: c_int, nh: c_int) {
    let c3 = c * 3;
    let hs = c / nh;
    let scale = 1.0 / (hs as f32).sqrt();

    unsafe {
        for bth in 0..(b * t * nh) {
            let (b, t, h) = (bth / (t * nh), (bth / nh) % t, bth % nh);
            let query_t = inp.offset(((b * t + t) * c3 + h * hs) as isize);
            let preatt_bth = preatt.offset((bth * t) as isize);
            let att_bth = att.offset((bth * t) as isize);

            let mut maxval = -10000.0;
            for t2 in 0..=t {
                let key_t2 = inp.offset(((b * t + t2) * c3 + h * hs + c) as isize);
                let mut val = 0.0;
                for i in 0..hs {
                    val += *query_t.offset(i as isize) * *key_t2.offset(i as isize);
                }
                val *= scale;
                if val > maxval {
                    maxval = val;
                }
                *preatt_bth.offset(t2 as isize) = val;
            }

            let mut expsum = 0.0;
            for t2 in 0..=t {
                let expv = (*preatt_bth.offset(t2 as isize) - maxval).exp();
                expsum += expv;
                *att_bth.offset(t2 as isize) = expv;
            }
            let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

            for t2 in 0..t {
                *att_bth.offset(t2 as isize) *= expsum_inv;
            }

            let out_bth = out.offset(((b * t + t) * c + h * hs) as isize);
            for i in 0..hs {
                *out_bth.offset(i as isize) = 0.0;
            }
            for t2 in 0..=t {
                let value_t2 = inp.offset(((b * t + t2) * c3 + h * hs + c * 2) as isize);
                let att_btht2 = *att_bth.offset(t2 as isize);
                for i in 0..hs {
                    *out_bth.offset(i as isize) += att_btht2 * *value_t2.offset(i as isize);
                }
            }
        }
    }
}

fn layernorm_forward(out: *mut f32, mean: *mut f32, rstd: *mut f32, inp: *const f32, weight: *const f32, bias: *const f32, b: c_int, t: c_int, c: c_int) {
    let eps = 1e-5;
    unsafe {
        for bt in 0..(b * t) {
            let x = inp.offset((bt * c) as isize);
            let mut m = 0.0;
            for i in 0..c {
                m += *x.offset(i as isize);
            }
            m /= c as f32;
            let mut v = 0.0;
            for i in 0..c {
                let xshift = *x.offset(i as isize) - m;
                v += xshift * xshift;
            }
            v /= c as f32;
            let s = 1.0 / (v + eps).sqrt();
            let out_bt = out.offset((bt * c) as isize);
            for i in 0..c {
                let n = s * (*x.offset(i as isize) - m);
                let o = n * *weight.offset(i as isize) + *bias.offset(i as isize);
                *out_bt.offset(i as isize) = o;
            }
            *mean.offset(bt as isize) = m;
            *rstd.offset(bt as isize) = s;
        }
    }
}

fn gelu_forward(out: *mut f32, inp: *const f32, n: c_int) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    unsafe {
        for i in 0..n {
            let x = *inp.offset(i as isize);
            let cube = 0.044715 * x * x * x;
            *out.offset(i as isize) = 0.5 * x * (1.0 + ((s * (x + cube)).tanh()));
        }
    }
}

fn softmax_forward(probs: *mut f32, logits: *const f32, b: c_int, t: c_int, v: c_int) {
    unsafe {
        for bt in 0..(b * t) {
            let logits_bt = logits.offset((bt * v) as isize);
            let probs_bt = probs.offset((bt * v) as isize);

            let mut maxval = -10000.0;
            for i in 0..v {
                if *logits_bt.offset(i as isize) > maxval {
                    maxval = *logits_bt.offset(i as isize);
                }
            }

            let mut sum = 0.0;
            for i in 0..v {
                *probs_bt.offset(i as isize) = (*logits_bt.offset(i as isize) - maxval).exp();
                sum += *probs_bt.offset(i as isize);
            }

            for i in 0..v {
                *probs_bt.offset(i as isize) /= sum;
            }
        }
    }
}

// Backward functions

fn residual_backward(dinp1: *mut f32, dinp2: *mut f32, dout: *const f32, n: c_int) {
    unsafe {
        for i in 0..n {
            *dinp1.offset(i as isize) += *dout.offset(i as isize);
            *dinp2.offset(i as isize) += *dout.offset(i as isize);
        }
    }
}

fn matmul_backward(dinp: *mut f32, dweight: *mut f32, dbias: *mut f32, dout: *const f32, inp: *const f32, weight: *const f32, b: c_int, t: c_int, c: c_int, oc: c_int) {
    unsafe {
        for bt in 0..(b * t) {
            for o in 0..oc {
                let dout_bt = dout.offset((bt * oc + o) as isize);
                let wrow = weight.offset((o * c) as isize);
                let dinp_bt = dinp.offset((bt * c) as isize);
                for i in 0..c {
                    *dinp_bt.offset(i as isize) += *wrow.offset(i as isize) * *dout_bt;
                }
            }
        }

        for o in 0..oc {
            for bt in 0..(b * t) {
                let dout_bt = dout.offset((bt * oc + o) as isize);
                let inp_bt = inp.offset((bt * c) as isize);
                let dwrow = dweight.offset((o * c) as isize);
                if !dbias.is_null() {
                    *dbias.offset(o as isize) += *dout_bt;
                }
                for i in 0..c {
                    *dwrow.offset(i as isize) += *inp_bt.offset(i as isize) * *dout_bt;
                }
            }
        }
    }
}

fn attention_backward(dinp: *mut f32, dpreatt: *mut f32, datt: *mut f32, dout: *const f32, inp: *const f32, att: *const f32, b: c_int, t: c_int, c: c_int, nh: c_int) {
    let c3 = c * 3;
    let hs = c / nh;
    let scale = 1.0 / (hs as f32).sqrt();

    unsafe {
        for bth in 0..(b * t * nh) {
            let (b, t, h) = (bth / (t * nh), (bth / nh) % t, bth % nh);
            let att_bth = att.offset((bth * t) as isize);
            let datt_bth = datt.offset((bth * t) as isize);
            let dpreatt_bth = dpreatt.offset((bth * t) as isize);
            let dquery_t = dinp.offset(((b * t + t) * c3 + h * hs) as isize);
            let query_t = inp.offset(((b * t + t) * c3 + h * hs) as isize);

            let dout_bth = dout.offset(((b * t + t) * c + h * hs) as isize);
            for t2 in 0..=t {
                let value_t2 = inp.offset(((b * t + t2) * c3 + h * hs + c * 2) as isize);
                let dvalue_t2 = dinp.offset(((b * t + t2) * c3 + h * hs + c * 2) as isize);
                for i in 0..hs {
                    *datt_bth.offset(t2 as isize) += *value_t2.offset(i as isize) * *dout_bth.offset(i as isize);
                    *dvalue_t2.offset(i as isize) += *att_bth.offset(t2 as isize) * *dout_bth.offset(i as isize);
                }
            }

            for t2 in 0..=t {
                for t3 in 0..=t {
                    let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                    let local_derivative = *att_bth.offset(t2 as isize) * (indicator - *att_bth.offset(t3 as isize));
                    *dpreatt_bth.offset(t3 as isize) += local_derivative * *datt_bth.offset(t2 as isize);
                }
            }

            for t2 in 0..=t {
                let key_t2 = inp.offset(((b * t + t2) * c3 + h * hs + c) as isize);
                let dkey_t2 = dinp.offset(((b * t + t2) * c3 + h * hs + c) as isize);
                for i in 0..hs {
                    *dquery_t.offset(i as isize) += *key_t2.offset(i as isize) * *dpreatt_bth.offset(t2 as isize) * scale;
                    *dkey_t2.offset(i as isize) += *query_t.offset(i as isize) * *dpreatt_bth.offset(t2 as isize) * scale;
                }
            }
        }
    }
}

fn layernorm_backward(dinp: *mut f32, dweight: *mut f32, dbias: *mut f32, dout: *const f32, inp: *const f32, weight: *const f32, mean: *const f32, rstd: *const f32, b: c_int, t: c_int, c: c_int) {
    unsafe {
        for bt in 0..(b * t) {
            let dout_bt = dout.offset((bt * c) as isize);
            let inp_bt = inp.offset((bt * c) as isize);
            let dinp_bt = dinp.offset((bt * c) as isize);
            let mean_bt = *mean.offset(bt as isize);
            let rstd_bt = *rstd.offset(bt as isize);

            let mut dnorm_mean = 0.0;
            let mut dnorm_norm_mean = 0.0;
            for i in 0..c {
                let norm_bti = (inp_bt.offset(i as isize) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.offset(i as isize) * *dout_bt.offset(i as isize);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= c as f32;
            dnorm_norm_mean /= c as f32;

            for i in 0..c {
                let norm_bti = (*inp_bt.offset(i as isize) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.offset(i as isize) * *dout_bt.offset(i as isize);
                *dbias.offset(i as isize) += *dout_bt.offset(i as isize);
                *dweight.offset(i as isize) += norm_bti * *dout_bt.offset(i as isize);
                let mut dval = 0.0;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                *dinp_bt.offset(i as isize) += dval;
            }
        }
    }
}

fn gelu_backward(dinp: *mut f32, inp: *const f32, dout: *const f32, n: c_int) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    unsafe {
        for i in 0..n {
            let x = *inp.offset(i as isize);
            let cube = 0.044715 * x * x * x;
            let tanh_arg = s * (x + cube);
            let tanh_out = tanh_arg.tanh();
            let coshf_out = (2.0 * tanh_arg).cosh();
            let sech_out = 1.0 / (coshf_out * coshf_out);
            let local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * s * (1.0 + 3.0 * 0.044715 * x * x);
            *dinp.offset(i as isize) += local_grad * *dout.offset(i as isize);
        }
    }
}

fn softmax_backward(dinp: *mut f32, dout: *const f32, probs: *const f32, b: c_int, t: c_int, v: c_int) {
    unsafe {
        for bt in 0..(b * t) {
            let dout_bt = dout.offset((bt * v) as isize);
            let probs_bt = probs.offset((bt * v) as isize);
            let dinp_bt = dinp.offset((bt * v) as isize);
            for i in 0..v {
                let p = *probs_bt.offset(i as isize);
                for j in 0..v {
                    let indicator = if i == j { 1.0 } else { 0.0 };
                    *dinp_bt.offset(i as isize) += (p - indicator) * *dout_bt.offset(j as isize);
                }
            }
        }
    }
}

// Utility functions

fn init_parameters(params: &mut ParameterTensors, config: &ViTConfig) {
    let v = config.vocab_size as usize;
    let c = config.channels as usize;
    let l = config.num_layers as usize;

    unsafe {
        for i in 0..(v * c) {
            *params.wte.offset(i as isize) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(config.max_seq_len as usize * c) {
            *params.wpe.offset(i as isize) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * c) {
            *params.ln1w.offset(i as isize) = 1.0;
            *params.ln2w.offset(i as isize) = 1.0;
        }

        for i in 0..(l * 3 * c * c) {
            *params.qkvw.offset(i as isize) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * c * c) {
            *params.attprojw.offset(i as isize) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * 4 * c * c) {
            *params.fcw.offset(i as isize) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * c * 4 * c) {
            *params.fcprojw.offset(i as isize) = rand::random::<f32>() * 0.02;
        }

        for i in 0..c {
            *params.lnfw.offset(i as isize) = 1.0;
        }
    }
}

fn save_checkpoint(params: &ParameterTensors, config: &ViTConfig, filepath: &str) {
    let mut file = File::create(filepath).unwrap();
    unsafe {
        file.write_all(std::slice::from_raw_parts(
            params.wte as *const u8,
            config.vocab_size as usize * config.channels as usize * std::mem::size_of::<f32>(),
        )).unwrap();
        // Save other parameters similarly
    }
}

fn load_checkpoint(params: &mut ParameterTensors, config: &ViTConfig, filepath: &str) {
    let mut file = File::open(filepath).unwrap();
    unsafe {
        file.read_exact(std::slice::from_raw_parts_mut(
            params.wte as *mut u8,
            config.vocab_size as usize * config.channels as usize * std::mem::size_of::<f32>(),
        )).unwrap();
        // Load other parameters similarly
    }
}

fn optimizer_step(model: &mut ViT, learning_rate: f32) {
    unsafe {
        for i in 0..model.num_parameters {
            *model.params_memory.offset(i as isize) -= learning_rate * *model.grads_memory.offset(i as isize);
        }
    }
}