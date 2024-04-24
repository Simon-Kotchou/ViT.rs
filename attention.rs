fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
) {
    let c3 = c * 3;
    let hs = c / nh;
    let scale = (hs as f32).sqrt().recip();

    for bth in 0..(b * t * nh) {
        let (b, t, h) = (bth / (t * nh), (bth / nh) % t, bth % nh);
        let query_offset = (b * t + t) * c3 + h * hs;
        let query_t = &inp[query_offset..query_offset + hs];
        let preatt_offset = bth * t;
        let att_offset = bth * t;

        let mut maxval = f32::NEG_INFINITY;
        for t2 in 0..=t {
            let key_offset = (b * t + t2) * c3 + h * hs + c;
            let key_t2 = &inp[key_offset..key_offset + hs];
            let mut val = query_t.iter().zip(key_t2).map(|(&q, &k)| q * k).sum::<f32>();
            val *= scale;
            if val > maxval {
                maxval = val;
            }
            preatt[preatt_offset + t2] = val;
        }

        let mut expsum = 0.0;
        for t2 in 0..=t {
            let expv = (preatt[preatt_offset + t2] - maxval).exp();
            expsum += expv;
            att[att_offset + t2] = expv;
        }
        let expsum_inv = expsum.recip();

        for t2 in 0..t {
            att[att_offset + t2] *= expsum_inv;
        }

        let out_offset = (b * t + t) * c + h * hs;
        let out_bth = &mut out[out_offset..out_offset + hs];
        out_bth.fill(0.0);
        for t2 in 0..=t {
            let value_offset = (b * t + t2) * c3 + h * hs + c * 2;
            let value_t2 = &inp[value_offset..value_offset + hs];
            let att_btht2 = att[att_offset + t2];
            for (o, &v) in out_bth.iter_mut().zip(value_t2) {
                *o += att_btht2 * v;
            }
        }
    }
}