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