#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_from_checkpoint() {
        let checkpoint_path = "path/to/checkpoint.bin";
        let vit = ViT::build_from_checkpoint(checkpoint_path);
        
        assert_eq!(vit.config.max_seq_len, 1024);
        assert_eq!(vit.config.vocab_size, 50257);
        assert_eq!(vit.config.num_layers, 12);
        assert_eq!(vit.config.num_heads, 12);
        assert_eq!(vit.config.channels, 768);
        assert_eq!(vit.num_parameters, 124439808);
    }

    #[test]
    fn test_forward_pass() {
        let mut vit = ViT::build_from_checkpoint("path/to/checkpoint.bin");
        let b = 4;
        let t = 64;
        let v = vit.config.vocab_size;
        let c = vit.config.channels;

        let inputs = vec![0; b * t];
        let targets = vec![0; b * t];
        vit.forward(inputs.as_ptr(), targets.as_ptr(), b as c_int, t as c_int);

        assert!(vit.mean_loss > 0.0);

        unsafe {
            assert_ne!(vit.acts.encoded, std::ptr::null_mut());
            assert_ne!(vit.acts.logits, std::ptr::null_mut());
            assert_ne!(vit.acts.probs, std::ptr::null_mut());
            assert_ne!(vit.acts.losses, std::ptr::null_mut());

            let encoded_slice = std::slice::from_raw_parts(vit.acts.encoded, (b * t * c) as usize);
            assert!(!encoded_slice.iter().all(|&x| x == 0.0));

            let logits_slice = std::slice::from_raw_parts(vit.acts.logits, (b * t * v) as usize);
            assert!(!logits_slice.iter().all(|&x| x == 0.0));

            let probs_slice = std::slice::from_raw_parts(vit.acts.probs, (b * t * v) as usize);
            assert!(!probs_slice.iter().all(|&x| x == 0.0));

            let losses_slice = std::slice::from_raw_parts(vit.acts.losses, (b * t) as usize);
            assert!(!losses_slice.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_backward_pass() {
        let mut vit = ViT::build_from_checkpoint("path/to/checkpoint.bin");
        let b = 4;
        let t = 64;
        let v = vit.config.vocab_size;
        let c = vit.config.channels;

        let inputs = vec![0; b * t];
        let targets = vec![0; b * t];
        vit.forward(inputs.as_ptr(), targets.as_ptr(), b as c_int, t as c_int);

        vit.backward();

        unsafe {
            assert_ne!(vit.grads.wte, std::ptr::null_mut());
            assert_ne!(vit.grads.wpe, std::ptr::null_mut());
            assert_ne!(vit.grads.ln1w, std::ptr::null_mut());
            assert_ne!(vit.grads.ln1b, std::ptr::null_mut());
            assert_ne!(vit.grads.qkvw, std::ptr::null_mut());
            assert_ne!(vit.grads.qkvb, std::ptr::null_mut());
            assert_ne!(vit.grads.attprojw, std::ptr::null_mut());
            assert_ne!(vit.grads.attprojb, std::ptr::null_mut());
            assert_ne!(vit.grads.ln2w, std::ptr::null_mut());
            assert_ne!(vit.grads.ln2b, std::ptr::null_mut());
            assert_ne!(vit.grads.fcw, std::ptr::null_mut());
            assert_ne!(vit.grads.fcb, std::ptr::null_mut());
            assert_ne!(vit.grads.fcprojw, std::ptr::null_mut());
            assert_ne!(vit.grads.fcprojb, std::ptr::null_mut());
            assert_ne!(vit.grads.lnfw, std::ptr::null_mut());
            assert_ne!(vit.grads.lnfb, std::ptr::null_mut());

            assert_ne!(vit.grads_acts.encoded, std::ptr::null_mut());
            assert_ne!(vit.grads_acts.logits, std::ptr::null_mut());
            assert_ne!(vit.grads_acts.probs, std::ptr::null_mut());
            assert_ne!(vit.grads_acts.losses, std::ptr::null_mut());
        }
    }

    #[test]
    fn test_residual_forward() {
        let n = 10;
        let inp1 = vec![1.0; n];
        let inp2 = vec![2.0; n];
        let mut out = vec![0.0; n];

        residual_forward(out.as_mut_ptr(), inp1.as_ptr(), inp2.as_ptr(), n as c_int);

        assert_eq!(out, vec![3.0; n]);
    }

    #[test]
    fn test_matmul_forward() {
        let b = 2;
        let t = 3;
        let c = 4;
        let oc = 5;

        let inp = vec![1.0; b * t * c];
        let weight = vec![2.0; oc * c];
        let bias = vec![3.0; oc];
        let mut out = vec![0.0; b * t * oc];

        matmul_forward(
            out.as_mut_ptr(),
            inp.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            b as c_int,
            t as c_int,
            c as c_int,
            oc as c_int,
        );

        let expected_out = vec![
            35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0,
            35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0,
            35.0, 35.0,
        ];
        assert_eq!(out, expected_out);
    }

    #[test]
    fn test_attention_forward() {
        let b = 2;
        let b = 2;
        let t = 3;
        let c = 4;
        let nh = 2;
        let inp = vec![1.0; b * t * 3 * c];
        let mut out = vec![0.0; b * t * c];
        let mut preatt = vec![0.0; b * nh * t * t];
        let mut att = vec![0.0; b * nh * t * t];
    
        attention_forward(
            out.as_mut_ptr(),
            preatt.as_mut_ptr(),
            att.as_mut_ptr(),
            inp.as_ptr(),
            b as c_int,
            t as c_int,
            c as c_int,
            nh as c_int,
        );
    
        assert_ne!(out, vec![0.0; b * t * c]);
        assert_ne!(preatt, vec![0.0; b * nh * t * t]);
        assert_ne!(att, vec![0.0; b * nh * t * t]);
    }

    #[test]
    fn test_layernorm_forward() {
        let b = 2;
        let t = 3;
        let c = 4;

        let inp = vec![1.0; b * t * c];
        let weight = vec![2.0; c];
        let bias = vec![3.0; c];
        let mut out = vec![0.0; b * t * c];
        let mut mean = vec![0.0; b * t];
        let mut rstd = vec![0.0; b * t];

        layernorm_forward(
            out.as_mut_ptr(),
            mean.as_mut_ptr(),
            rstd.as_mut_ptr(),
            inp.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            b as c_int,
            t as c_int,
            c as c_int,
        );

        assert_ne!(out, vec![0.0; b * t * c]);
        assert_ne!(mean, vec![0.0; b * t]);
        assert_ne!(rstd, vec![0.0; b * t]);
    }

    #[test]
    fn test_gelu_forward() {
        let n = 10;
        let inp = vec![1.0; n];
        let mut out = vec![0.0; n];

        gelu_forward(out.as_mut_ptr(), inp.as_ptr(), n as c_int);

        assert_ne!(out, vec![0.0; n]);
    }

    #[test]
    fn test_softmax_forward() {
        let b = 2;
        let t = 3;
        let v = 4;

        let logits = vec![1.0; b * t * v];
        let mut probs = vec![0.0; b * t * v];

        softmax_forward(
            probs.as_mut_ptr(),
            logits.as_ptr(),
            b as c_int,
            t as c_int,
            v as c_int,
        );

        assert_ne!(probs, vec![0.0; b * t * v]);

        // Check if the probabilities sum to 1 for each (b, t) position
        for b in 0..b {
            for t in 0..t {
                let sum: f32 = probs[(b * t * v)..(b * t * v + v)]
                    .iter()
                    .sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
    }