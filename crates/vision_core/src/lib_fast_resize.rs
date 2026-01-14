use anyhow::{anyhow, Result};
use fast_image_resize as fir;
use fast_image_resize::pixels::U8x3;
use image::RgbImage;
use std::num::NonZeroU32;

pub struct ResizeWorkspace {
    resizer: fir::Resizer,
}

impl ResizeWorkspace {
    pub fn new(algorithm: fir::ResizeAlg) -> Self {
        Self {
            resizer: fir::Resizer::new(algorithm),
        }
    }

    pub fn resize_rgb_into(
        &mut self,
        src: &RgbImage,
        dst_w: u32,
        dst_h: u32,
        dst_buf: &mut Vec<u8>,
    ) -> Result<()> {
        let src_w = NonZeroU32::new(src.width()).ok_or_else(|| anyhow!("src width=0"))?;
        let src_h = NonZeroU32::new(src.height()).ok_or_else(|| anyhow!("src height=0"))?;
        let dst_w = NonZeroU32::new(dst_w).ok_or_else(|| anyhow!("dst width=0"))?;
        let dst_h = NonZeroU32::new(dst_h).ok_or_else(|| anyhow!("dst height=0"))?;
        let dst_len = dst_w.get() as usize * dst_h.get() as usize * 3;
        if dst_buf.len() != dst_len {
            dst_buf.resize(dst_len, 0);
        }

        let src_view = fir::ImageView::<U8x3>::from_buffer(src_w, src_h, src.as_raw())?;
        let dst_view =
            fir::ImageViewMut::<U8x3>::from_buffer(dst_w, dst_h, dst_buf.as_mut_slice())?;
        let src_dyn = fir::DynamicImageView::from(src_view);
        let mut dst_dyn = fir::DynamicImageViewMut::from(dst_view);
        self.resizer.resize(&src_dyn, &mut dst_dyn)?;
        Ok(())
    }

    pub fn resize_with_pad_into(
        &mut self,
        src: &RgbImage,
        target_w: u32,
        target_h: u32,
        tmp_buf: &mut Vec<u8>,
        dst_buf: &mut Vec<u8>,
    ) -> Result<f32> {
        let img_w = src.width();
        let img_h = src.height();
        let im_ratio = img_h as f32 / img_w as f32;
        let model_ratio = target_h as f32 / target_w as f32;

        let (new_w, new_h) = if im_ratio > model_ratio {
            let new_h = target_h;
            let new_w = (new_h as f32 / im_ratio).round().max(1.0) as u32;
            (new_w, new_h)
        } else {
            let new_w = target_w;
            let new_h = (new_w as f32 * im_ratio).round().max(1.0) as u32;
            (new_w, new_h)
        };

        let det_scale = new_h as f32 / img_h as f32;
        if new_w == target_w && new_h == target_h {
            self.resize_rgb_into(src, target_w, target_h, dst_buf)?;
            return Ok(det_scale);
        }

        self.resize_rgb_into(src, new_w, new_h, tmp_buf)?;
        let dst_len = target_w as usize * target_h as usize * 3;
        if dst_buf.len() != dst_len {
            dst_buf.resize(dst_len, 0);
        }
        dst_buf.fill(0);

        let row_bytes = new_w as usize * 3;
        let dst_row_bytes = target_w as usize * 3;
        for y in 0..new_h as usize {
            let src_off = y * row_bytes;
            let dst_off = y * dst_row_bytes;
            dst_buf[dst_off..dst_off + row_bytes]
                .copy_from_slice(&tmp_buf[src_off..src_off + row_bytes]);
        }

        Ok(det_scale)
    }
}
