use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem;
use std::os::raw::c_int;

const NUM_PARAMETER_TENSORS: usize = 16;
const NUM_ACTIVATION_TENSORS: usize = 23;

/// Represents the parameter tensors of the ViT model.
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

/// Represents the activation tensors of the ViT model.
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

/// Represents the configuration of the ViT model.
#[derive(Debug, Clone, Copy)]
struct ViTConfig {
    max_seq_len: usize,
    vocab_size: usize,
    num_layers: usize,
    num_heads: usize,
    channels: usize,
}

/// Represents the ViT model.
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
    batch_size: usize,
    seq_len: usize,
    inputs: *mut c_int,
    targets: *mut c_int,
    mean_loss: f32,
}

impl ViT {
    /// Builds a ViT model from a checkpoint file.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to the checkpoint file.
    fn build_from_checkpoint(checkpoint_path: &str) -> Self {
        let mut model_file = File::open(checkpoint_path).expect("Error opening model file");
        let mut model_header = [0; 256];
        model_file.read_exact(&mut model_header).expect("Error reading model header");
        
        let max_seq_len = model_header[2] as usize;
        let vocab_size = model_header[3] as usize;
        let num_layers = model_header[4] as usize;
        let num_heads = model_header[5] as usize;
        let channels = model_header[6] as usize;
        
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
        param_sizes[0] = vocab_size * channels;
        param_sizes[1] = max_seq_len * channels;
        param_sizes[2] = num_layers * channels;
        param_sizes[3] = num_layers * channels;
        param_sizes[4] = num_layers * (3 * channels) * channels;
        param_sizes[5] = num_layers * (3 * channels);
        param_sizes[6] = num_layers * channels * channels;
        param_sizes[7] = num_layers * channels;
        param_sizes[8] = num_layers * channels;
        param_sizes[9] = num_layers * channels;
        param_sizes[10] = num_layers * (4 * channels) * channels;
        param_sizes[11] = num_layers * (4 * channels);
        param_sizes[12] = num_layers * channels * (4 * channels);
        param_sizes[13] = num_layers * channels;
        param_sizes[14] = channels;
        param_sizes[15] = channels;
        
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
            wpe: unsafe { params_memory.add(param_sizes[0]) },
            ln1w: unsafe { params_memory.add(param_sizes[0] + param_sizes[1]) },
            ln1b: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2]) },
            qkvw: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3]) },
            qkvb: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4]) },
            attprojw: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5]) },
            attprojb: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6]) },
            ln2w: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7]) },
            ln2b: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8]) },
            fcw: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9]) },
            fcb: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10]) },
            fcprojw: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11]) },
            fcprojb: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12]) },
            lnfw: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13]) },
            lnfb: unsafe { params_memory.add(param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13] + param_sizes[14]) },
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
    
    /// Performs the forward pass of the ViT model.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input tensor pointer.
    /// * `targets` - Target tensor pointer.
    /// * `b` - Batch size.
    /// * `t` - Sequence length.
    fn forward(&mut self, inputs: *const c_int, targets: *const c_int, b: usize, t: usize) {
        // ... (previous code for memory allocation and input/target copying)

        let params = &self.params;
        let acts = &mut self.acts;

        // Forward pass
        unsafe {
            // Perform the encoder forward pass.
            // This operation encodes the input sequence using the word and position embeddings.
            encoder_forward(acts.encoded, inputs, params.wte, params.wpe, b, t, self.config.channels);

            let mut residual: *mut f32 = std::ptr::null_mut();
            for l in 0..self.config.num_layers {
                residual = if l == 0 {
                    acts.encoded
                } else {
                    acts.residual3.add((l - 1) * b * t * self.config.channels)
                };

                let l_ln1w = params.ln1w.add(l * self.config.channels);
                let l_ln1b = params.ln1b.add(l * self.config.channels);
                let l_qkvw = params.qkvw.add(l * 3 * self.config.channels * self.config.channels);
                let l_qkvb = params.qkvb.add(l * 3 * self.config.channels);
                let l_attprojw = params.attprojw.add(l * self.config.channels * self.config.channels);
                let l_attprojb = params.attprojb.add(l * self.config.channels);
                let l_ln2w = params.ln2w.add(l * self.config.channels);
                let l_ln2b = params.ln2b.add(l * self.config.channels);
                let l_fcw = params.fcw.add(l * 4 * self.config.channels * self.config.channels);
                let l_fcb = params.fcb.add(l * 4 * self.config.channels);
                let l_fcprojw = params.fcprojw.add(l * self.config.channels * 4 * self.config.channels);
                let l_fcprojb = params.fcprojb.add(l * self.config.channels);

                let l_ln1 = acts.ln1.add(l * b * t * self.config.channels);
                let l_ln1_mean = acts.ln1_mean.add(l * b * t);
                let l_ln1_rstd = acts.ln1_rstd.add(l * b * t);
                let l_qkv = acts.qkv.add(l * b * t * 3 * self.config.channels);
                let l_atty = acts.atty.add(l * b * t * self.config.channels);
                let l_preatt = acts.preatt.add(l * b * self.config.num_heads * t * t);
                let l_att = acts.att.add(l * b * self.config.num_heads * t * t);
                let l_attproj = acts.attproj.add(l * b * t * self.config.channels);
                let l_residual2 = acts.residual2.add(l * b * t * self.config.channels);
                let l_ln2 = acts.ln2.add(l * b * t * self.config.channels);
                let l_ln2_mean = acts.ln2_mean.add(l * b * t);
                let l_ln2_rstd = acts.ln2_rstd.add(l * b * t);
                let l_fch = acts.fch.add(l * b * t * 4 * self.config.channels);
                let l_fch_gelu = acts.fch_gelu.add(l * b * t * 4 * self.config.channels);
                let l_fcproj = acts.fcproj.add(l * b * t * self.config.channels);
                let l_residual3 = acts.residual3.add(l * b * t * self.config.channels);

                // Perform layer normalization on the residual.
                layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, b, t, self.config.channels);
                
                // Perform matrix multiplication to compute query, key, and value tensors.
                matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, b, t, self.config.channels, 3 * self.config.channels);
                
                // Perform multi-head self-attention.
                attention_forward(l_atty, l_preatt, l_att, l_qkv, b, t, self.config.channels, self.config.num_heads);
                
                // Perform matrix multiplication to project the attention output.
                matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, b, t, self.config.channels, self.config.channels);
                
                // Compute the residual connection.
                residual_forward(l_residual2, residual, l_attproj, b * t * self.config.channels);
                
                // Perform layer normalization on the residual.
                layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, b, t, self.config.channels);
                
                // Perform matrix multiplication for the feed-forward layer.
                matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, b, t, self.config.channels, 4 * self.config.channels);
                
                // Apply the GELU activation function.
                gelu_forward(l_fch_gelu, l_fch, b * t * 4 * self.config.channels);
                
                // Perform matrix multiplication to project the feed-forward output.
                matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, b, t, 4 * self.config.channels, self.config.channels);
                
                // Compute the residual connection.
                residual_forward(l_residual3, l_residual2, l_fcproj, b * t * self.config.channels);
            }

            residual = acts.residual3.add((self.config.num_layers - 1) * b * t * self.config.channels);
            
            // Perform layer normalization on the final residual.
            layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, b, t, self.config.channels);
            
            // Perform matrix multiplication to compute the logits.
            matmul_forward(acts.logits, acts.lnf, params.wte, std::ptr::null_mut(), b, t, self.config.channels, self.config.vocab_size);
            
            // Apply the softmax function to compute the probabilities.
            softmax_forward(acts.probs, acts.logits, b, t, self.config.vocab_size);
        }

        if !targets.is_null() {
            unsafe {
                // Compute the cross-entropy loss.
                crossentropy_forward(acts.losses, acts.probs, targets, b, t, self.config.vocab_size);

                let mut mean_loss = 0.0;
                for i in 0..(b * t) {
                    mean_loss += *acts.losses.add(i);
                }
                mean_loss /= (b * t) as f32;
                self.mean_loss = mean_loss;
            }
        } else {
            self.mean_loss = -1.0;
        }
    }
    
    /// Performs the backward pass of the ViT model.
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
                *grads_acts.losses.add(i) = dloss_mean;
            }

            // Compute the gradients of the cross-entropy loss and softmax.
            crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, self.targets, b, t, v);
            
            // Compute the gradients of the final matrix multiplication and layer normalization.
            matmul_backward(grads_acts.lnf, grads.wte, std::ptr::null_mut(), grads_acts.logits, acts.lnf, params.wte, b, t, c, v);

            let mut residual = acts.residual3.add((l - 1) * b * t * c);
            let mut dresidual = grads_acts.residual3.add((l - 1) * b * t * c);

            layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, b, t, c);

            for l in (0..l).rev() {
                residual = if l == 0 {
                    acts.encoded
                } else {
                    acts.residual3.add((l - 1) * b * t * c)
                };
                dresidual = if l == 0 {
                    grads_acts.encoded
                } else {
                    grads_acts.residual3.add((l - 1) * b * t * c)
                };

                let l_ln1w = params.ln1w.add(l * c);
                let l_qkvw = params.qkvw.add(l * 3 * c * c);
                let l_attprojw = params.attprojw.add(l * c * c);
                let l_ln2w = params.ln2w.add(l * c);
                let l_fcw = params.fcw.add(l * 4 * c * c);
                let l_fcprojw = params.fcprojw.add(l * c * 4 * c);

                let dl_ln1w = grads.ln1w.add(l * c);
                let dl_ln1b = grads.ln1b.add(l * c);
                let dl_qkvw = grads.qkvw.add(l * 3 * c * c);
                let dl_qkvb = grads.qkvb.add(l * 3 * c);
                let dl_attprojw = grads.attprojw.add(l * c * c);
                let dl_attprojb = grads.attprojb.add(l * c);
                let dl_ln2w = grads.ln2w.add(l * c);
                let dl_ln2b = grads.ln2b.add(l * c);
                let dl_fcw = grads.fcw.add(l * 4 * c * c);
                let dl_fcb = grads.fcb.add(l * 4 * c);
                let dl_fcprojw = grads.fcprojw.add(l * c * 4 * c);
                let dl_fcprojb = grads.fcprojb.add(l * c);

                let l_ln1 = acts.ln1.add(l * b * t * c);
                let l_ln1_mean = acts.ln1_mean.add(l * b * t);
                let l_ln1_rstd = acts.ln1_rstd.add(l * b * t);
                let l_qkv = acts.qkv.add(l * b * t * 3 * c);
                let l_atty = acts.atty.add(l * b * t * c);
                let l_att = acts.att.add(l * b * nh * t * t);
                let l_residual2 = acts.residual2.add(l * b * t * c);
                let l_ln2 = acts.ln2.add(l * b * t * c);
                let l_ln2_mean = acts.ln2_mean.add(l * b * t);
                let l_ln2_rstd = acts.ln2_rstd.add(l * b * t);
                let l_fch = acts.fch.add(l * b * t * 4 * c);
                let l_fch_gelu = acts.fch_gelu.add(l * b * t * 4 * c);

                let dl_ln1 = grads_acts.ln1.add(l * b * t * c);
                let dl_qkv = grads_acts.qkv.add(l * b * t * 3 * c);
                let dl_atty = grads_acts.atty.add(l * b * t * c);
                let dl_preatt = grads_acts.preatt.add(l * b * nh * t * t);
                let dl_att = grads_acts.att.add(l * b * nh * t * t);
                let dl_attproj = grads_acts.attproj.add(l * b * t * c);
                let dl_residual2 = grads_acts.residual2.add(l * b * t * c);
                let dl_ln2 = grads_acts.ln2.add(l * b * t * c);
                let dl_fch = grads_acts.fch.add(l * b * t * 4 * c);
                let dl_fch_gelu = grads_acts.fch_gelu.add(l * b * t * 4 * c);
                let dl_fcproj = grads_acts.fcproj.add(l * b * t * c);
                let dl_residual3 = grads_acts.residual3.add(l * b * t * c);

                // Compute the gradients of the residual connections.
                residual_backward(dl_residual2, dl_fcproj, dl_residual3, b * t * c);
                
                // Compute the gradients of the feed-forward projection and GELU activation.
                matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, b, t, 4 * c, c);
                gelu_backward(dl_fch, l_fch, dl_fch_gelu, b * t * 4 * c);
                
                // Compute the gradients of the feed-forward matrix multiplication and layer normalization.
                matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, b, t, c, 4 * c);
                layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, b, t, c);
                
                // Compute the gradients of the residual connections.
                residual_backward(dresidual, dl_attproj, dl_residual2, b * t * c);
                
                // Compute the gradients of the attention projection and multi-head self-attention.
                matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, b, t, c, c);
                attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, b, t, c, nh);
                
                // Compute the gradients of the query, key, value matrix multiplication and layer normalization.
                matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, b, t, c, 3 * c);
                layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, b, t, c);
            }

            // Compute the gradients of the encoder.
            encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, self.inputs, b, t, c);
        }
    }
}

/// Performs the residual forward pass.
///
/// # Arguments
///
/// * `out` - Output tensor pointer.
/// * `inp1` - First input tensor pointer.
/// * `inp2` - Second input tensor pointer.
/// * `n` - Number of elements in the tensors.
fn residual_forward(out: *mut f32, inp1: *const f32, inp2: *const f32, n: usize) {
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out, n);
        let inp1_slice = std::slice::from_raw_parts(inp1, n);
        let inp2_slice = std::slice::from_raw_parts(inp2, n);

        for i in 0..n {
            out_slice[i] = inp1_slice[i] + inp2_slice[i];
        }
    }
}

/// Performs the matrix multiplication forward pass.
///
/// # Arguments
///
/// * `out` - Output tensor pointer.
/// * `inp` - Input tensor pointer.
/// * `weight` - Weight tensor pointer.
/// * `bias` - Bias tensor pointer.
/// * `b` - Batch size.
/// * `t` - Sequence length.
/// * `c` - Input channels.
/// * `oc` - Output channels.
fn matmul_forward(out: *mut f32, inp: *const f32, weight: *const f32, bias: *const f32, b: usize, t: usize, c: usize, oc: usize) {
    unsafe {
        for bt in 0..(b * t) {
            for o in 0..oc {
                let mut val = if !bias.is_null() { *bias.add(o) } else { 0.0 };
                let wrow = weight.add(o * c);
                let inp_bt = inp.add(bt * c);
                for i in 0..c {
                    val += *inp_bt.add(i) * *wrow.add(i);
                }
                *out.add(bt * oc + o) = val;
            }
        }
    }
}

/// Performs the multi-head self-attention forward pass.
///
/// # Arguments
///
/// * `out` - Output tensor pointer.
/// * `preatt` - Pre-attention tensor pointer.
/// * `att` - Attention tensor pointer.
/// * `inp` - Input tensor pointer.
/// * `b` - Batch size.
/// * `t` - Sequence length.
/// * `c` - Input channels.
/// * `nh` - Number of attention heads.
fn attention_forward(out: *mut f32, preatt: *mut f32, att: *mut f32, inp: *const f32, b: usize, t: usize, c: usize, nh: usize) {
    let c3 = c * 3;
    let hs = c / nh;
    let scale = 1.0 / (hs as f32).sqrt();

    unsafe {
        for bth in 0..(b * t * nh) {
            let (b, t, h) = (bth / (t * nh), (bth / nh) % t, bth % nh);
            let query_t = inp.add((b * t + t) * c3 + h * hs);
            let preatt_bth = preatt.add(bth * t);
            let att_bth = att.add(bth * t);

            let mut maxval = -10000.0;
            for t2 in 0..=t {
                let key_t2 = inp.add((b * t + t2) * c3 + h * hs + c);
                let mut val = 0.0;
                for i in 0..hs {
                    val += *query_t.add(i) * *key_t2.add(i);
                }
                val *= scale;
                if val > maxval {
                    maxval = val;
                }
                *preatt_bth.add(t2) = val;
            }

            let mut expsum = 0.0;
            for t2 in 0..=t {
                let expv = (*preatt_bth.add(t2) - maxval).exp();
                expsum += expv;
                *att_bth.add(t2) = expv;
            }
            let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

            for t2 in 0..t {
                *att_bth.add(t2) *= expsum_inv;
            }

            let out_bth = out.add((b * t + t) * c + h * hs);
            for i in 0..hs {
                *out_bth.add(i) = 0.0;
            }
            for t2 in 0..=t {
                let value_t2 = inp.add((b * t + t2) * c3 + h * hs + c * 2);
                let att_btht2 = *att_bth.add(t2);
                for i in 0..hs {
                    *out_bth.add(i) += att_btht2 * *value_t2.add(i);
                }
            }
        }
    }
}

/// Performs the layer normalization forward pass.
///
/// # Arguments
///
/// * `out` - Output tensor pointer.
/// * `mean` - Mean tensor pointer.
/// * `rstd` - Reciprocal standard deviation tensor pointer.
/// * `inp` - Input tensor pointer.
/// * `weight` - Weight tensor pointer.
/// * `bias` - Bias tensor pointer.
/// * `b` - Batch size.
/// * `t` - Sequence length.
/// * `c` - Input channels.
fn layernorm_forward(out: *mut f32, mean: *mut f32, rstd: *mut f32, inp: *const f32, weight: *const f32, bias: *const f32, b: usize, t: usize, c: usize) {
    let eps = 1e-5;
    unsafe {
        for bt in 0..(b * t) {
            let x = inp.add(bt * c);
            let mut m = 0.0;
            for i in 0..c {
                m += *x.add(i);
            }
            m /= c as f32;
            let mut v = 0.0;
            for i in 0..c {
                let xshift = *x.add(i) - m;
                v += xshift * xshift;
            }
            v /= c as f32;
            let s = 1.0 / (v + eps).sqrt();
            let out_bt = out.add(bt * c);
            for i in 0..c {
                let n = s * (*x.add(i) - m);
                let o = n * *weight.add(i) + *bias.add(i);
                *out_bt.add(i) = o;
            }
            *mean.add(bt) = m;
            *rstd.add(bt) = s;
        }
    }
}

/// Performs the GELU activation forward pass.
///
/// # Arguments
///
/// * `out` - Output tensor pointer.
/// * `inp` - Input tensor pointer.
/// * `n` - Number of elements in the tensors.
fn gelu_forward(out: *mut f32, inp: *const f32, n: usize) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    unsafe {
        for i in 0..n {
            let x = *inp.add(i);
            let cube = 0.044715 * x * x * x;
            *out.add(i) = 0.5 * x * (1.0 + ((s * (x + cube)).tanh()));
        }
    }
}

/// Performs the softmax activation forward pass.
///
/// # Arguments
///
/// * `probs` - Probability tensor pointer.
/// * `logits` - Logit tensor pointer.
/// * `b` - Batch size.
/// * `t` - Sequence length.
/// * `v` - Vocabulary size.
fn softmax_forward(probs: *mut f32, logits: *const f32, b: usize, t: usize, v: usize) {
    unsafe {
        for bt in 0..(b * t) {
            let logits_bt = logits.add(bt * v);
            let probs_bt = probs.add(bt * v);

            let mut maxval = -10000.0;
            for i in 0..v {
                if *logits_bt.add(i) > maxval {
                    maxval = *logits_bt.add(i);
                }
            }

            let mut sum = 0.0;
            for i in 0..v {
                *probs_bt.add(i) = (*logits_bt.add(i) - maxval).exp();
                sum += *probs_bt.add(i);
            }

            for i in 0..v {
                *probs_bt.add(i) /= sum;
            }
        }
    }
}

// Backward functions

/// Performs the residual backward pass.
///
/// # Arguments
///
/// * `dinp1` - Gradient of the first input tensor pointer.
/// * `dinp2` - Gradient of the second input tensor pointer.
/// * `dout` - Gradient of the output tensor pointer.
/// * `n` - Number of elements in the tensors.
fn residual_backward(dinp1: *mut f32, dinp2: *mut f32, dout: *const f32, n: usize) {
    unsafe {
        for i in 0..n {
            *dinp1.add(i) += *dout.add(i);
            *dinp2.add(i) += *dout.add(i);
        }
    }
}

/// Performs the matrix multiplication backward pass.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor pointer.
/// * `dweight` - Gradient of the weight tensor pointer.
/// * `dbias` - Gradient of the bias tensor pointer.
/// * `dout` - Gradient of the output tensor pointer.
/// * `inp` - Input tensor pointer.
/// * `weight` - Weight tensor pointer.
/// * `b` - Batch size.
/// * `t` - Sequence length.
/// * `c` - Input channels.
/// * `oc` - Output channels.
fn matmul_backward(dinp: *mut f32, dweight: *mut f32, dbias: *mut f32, dout: *const f32, inp: *const f32, weight: *const f32, b: usize, t: usize, c: usize, oc: usize) {
    unsafe {
        for bt in 0..(b * t) {
            for o in 0..oc {
                let dout_bt = dout.add(bt * oc + o);
                let wrow = weight.add(o * c);
                let dinp_bt = dinp.add(bt * c);
                for i in 0..c {
                    *dinp_bt.add(i) += *wrow.add(i) * *dout_bt;
                }
            }
        }

        for o in 0..oc {
            for bt in 0..(b * t) {
                let dout_bt = dout.add(bt * oc + o);
                let inp_bt = inp.add(bt * c);
                let dwrow = dweight.add(o * c);
                if !dbias.is_null() {
                    *dbias.add(o) += *dout_bt;
                }
                for i in 0..c {
                    *dwrow.add(i) += *inp_bt.add(i) * *dout_bt;
                }
            }
        }
    }
}

/// Performs the layer normalization backward pass.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor pointer.
/// * `dweight` - Gradient of the weight tensor pointer.
/// * `dbias` - Gradient of the bias tensor pointer.
/// * `dout` - Gradient of the output tensor pointer.
/// * `inp` - Input tensor pointer.
/// * `weight` - Weight tensor pointer.
/// * `mean` - Mean tensor pointer.
/// * `rstd` - Reciprocal standard deviation tensor pointer.
/// * `b` - Batch size.
/// * `t` - Sequence length.
/// * `c` - Input channels.
fn layernorm_backward(
    dinp: *mut f32,
    dweight: *mut f32,
    dbias: *mut f32,
    dout: *const f32,
    inp: *const f32,
    weight: *const f32,
    mean: *const f32,
    rstd: *const f32,
    b: usize,
    t: usize,
    c: usize,
) {
    unsafe {
        for bt in 0..(b * t) {
            let dout_bt = dout.add(bt * c);
            let inp_bt = inp.add(bt * c);
            let dinp_bt = dinp.add(bt * c);
            let mean_bt = *mean.add(bt);
            let rstd_bt = *rstd.add(bt);

            let mut dnorm_mean = 0.0;
            let mut dnorm_norm_mean = 0.0;
            for i in 0..c {
                let norm_bti = (inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.add(i) * *dout_bt.add(i);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= c as f32;
            dnorm_norm_mean /= c as f32;

            for i in 0..c {
                let norm_bti = (*inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight.add(i) * *dout_bt.add(i);
                *dbias.add(i) += *dout_bt.add(i);
                *dweight.add(i) += norm_bti * *dout_bt.add(i);
                let mut dval = 0.0;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                *dinp_bt.add(i) += dval;
            }
        }
    }
}

/// Performs the GELU activation backward pass.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor pointer.
/// * `inp` - Input tensor pointer.
/// * `dout` - Gradient of the output tensor pointer.
/// * `n` - Number of elements in the tensors.
fn gelu_backward(dinp: *mut f32, inp: *const f32, dout: *const f32, n: usize) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    unsafe {
        for i in 0..n {
            let x = *inp.add(i);
            let cube = 0.044715 * x * x * x;
            let tanh_arg = s * (x + cube);
            let tanh_out = tanh_arg.tanh();
            let coshf_out = (2.0 * tanh_arg).cosh();
            let sech_out = 1.0 / (coshf_out * coshf_out);
            let local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * s * (1.0 + 3.0 * 0.044715 * x * x);
            *dinp.add(i) += local_grad * *dout.add(i);
        }
    }
}

/// Performs the softmax activation backward pass.
///
/// # Arguments
///
/// * `dinp` - Gradient of the input tensor pointer.
/// * `dout` - Gradient of the output tensor pointer.
/// * `probs` - Probability tensor pointer.
/// * `b` - Batch size.
/// * `t` - Sequence length.
/// * `v` - Vocabulary size.
fn softmax_backward(dinp: *mut f32, dout: *const f32, probs: *const f32, b: usize, t: usize, v: usize) {
    unsafe {
        for bt in 0..(b * t) {
            let dout_bt = dout.add(bt * v);
            let probs_bt = probs.add(bt * v);
            let dinp_bt = dinp.add(bt * v);
            for i in 0..v {
                let p = *probs_bt.add(i);
                for j in 0..v {
                    let indicator = if i == j { 1.0 } else { 0.0 };
                    *dinp_bt.add(i) += (p - indicator) * *dout_bt.add(j);
                }
            }
        }
    }
}

// Utility functions

/// Initializes the model parameters.
///
/// # Arguments
///
/// * `params` - Parameter tensors.
/// * `config` - Model configuration.
fn init_parameters(params: &mut ParameterTensors, config: &ViTConfig) {
    let v = config.vocab_size;
    let c = config.channels;
    let l = config.num_layers;

    unsafe {
        for i in 0..(v * c) {
            *params.wte.add(i) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(config.max_seq_len * c) {
            *params.wpe.add(i) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * c) {
            *params.ln1w.add(i) = 1.0;
            *params.ln2w.add(i) = 1.0;
        }

        for i in 0..(l * 3 * c * c) {
            *params.qkvw.add(i) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * c * c) {
            *params.attprojw.add(i) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * 4 * c * c) {
            *params.fcw.add(i) = rand::random::<f32>() * 0.02;
        }

        for i in 0..(l * c * 4 * c) {
            *params.fcprojw.add(i) = rand::random::<f32>() * 0.02;
        }

        for i in 0..c {
            *params.lnfw.add(i) = 1.0;
        }
    }
}

/// Saves the model parameters to a checkpoint file.
///
/// # Arguments
///
/// * `params` - Parameter tensors.
/// * `config` - Model configuration.
/// * `filepath` - Path to the checkpoint file.
fn save_checkpoint(params: &ParameterTensors, config: &ViTConfig, filepath: &str) {
    let mut file = File::create(filepath).unwrap();
    unsafe {
        file.write_all(std::slice::from_raw_parts(
            params.wte as *const u8,
            config.vocab_size * config.channels * std::mem::size_of::<f32>(),
        ))
        .unwrap();
        // Save other parameters similarly
    }
}

/// Loads the model parameters from a checkpoint file.
///
/// # Arguments
///
/// * `params` - Parameter tensors.
/// * `config` - Model configuration.
/// * `filepath` - Path to the checkpoint file.
fn load_checkpoint(params: &mut ParameterTensors, config: &ViTConfig, filepath: &str) {
    let mut file = File::open(filepath).unwrap();
    unsafe {
        file.read_exact(std::slice::from_raw_parts_mut(
            params.wte as *mut u8,
            config.vocab_size * config.channels * std::mem::size_of::<f32>(),
        ))
        .unwrap();
        // Load other parameters similarly
    }
}

/// Performs the optimizer step to update the model parameters.
///
/// # Arguments
///
/// * `model` - ViT model.
/// * `learning_rate` - Learning rate for parameter updates.
fn optimizer_step(model: &mut ViT, learning_rate: f32) {
    unsafe {
        for i in 0..model.num_parameters {
            *model.params_memory.add(i) -= learning_rate * *model.grads_memory.add(i);
        }
    }
}