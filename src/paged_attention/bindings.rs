/* automatically generated by rust-bindgen 0.69.1 */
// Edited by Eric Buehler, added the Optional & Storage types from:
// https://github.com/pytorch/pytorch/blob/0f5e24bda9450a89ba56d2fdd471f56d97fe4546/c10/util/Optional.h#L209

use std::ffi::c_uchar;

#[repr(C)]
pub union Storage<T: Copy> {
    pub dummy_: c_uchar,
    pub value_: T,
}

#[repr(C)]
pub struct Optional<T: Copy> {
    pub init_: bool,
    pub storage_: Storage<T>,
}

extern "C" {
    #[link_name = "\u{1}_Z18paged_attention_v1RlS_S_S_S_fS_S_iiRK8optionalIlE"]
    pub fn paged_attention_v1(
        out: *mut torch_sys::C_tensor,
        query: *mut torch_sys::C_tensor,
        key_cache: *mut torch_sys::C_tensor,
        value_cache: *mut torch_sys::C_tensor,
        head_mapping: *mut torch_sys::C_tensor,
        scale: f32,
        block_tables: *mut torch_sys::C_tensor,
        context_lens: *mut torch_sys::C_tensor,
        block_size: ::std::ffi::c_int,
        max_context_len: ::std::ffi::c_int,
        alibi_slopes: *const Optional<torch_sys::C_tensor>,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z18paged_attention_v2RlS_S_S_S_S_S_S_fS_S_iiRK8optionalIlE"]
    pub fn paged_attention_v2(
        out: *mut torch_sys::C_tensor,
        exp_sums: *mut torch_sys::C_tensor,
        max_logits: *mut torch_sys::C_tensor,
        tmp_out: *mut torch_sys::C_tensor,
        query: *mut torch_sys::C_tensor,
        key_cache: *mut torch_sys::C_tensor,
        value_cache: *mut torch_sys::C_tensor,
        head_mapping: *mut torch_sys::C_tensor,
        scale: f32,
        block_tables: *mut torch_sys::C_tensor,
        context_lens: *mut torch_sys::C_tensor,
        block_size: ::std::ffi::c_int,
        max_context_len: ::std::ffi::c_int,
        alibi_slopes: *const Optional<torch_sys::C_tensor>,
    );
}

/* automatically generated by rust-bindgen 0.69.1 */
// Edited by Eric Buehler

extern "C" {
    #[link_name = "\u{1}_Z17reshape_and_cacheRlS_S_S_S_"]
    pub fn reshape_and_cache(
        key: *mut torch_sys::C_tensor,
        value: *mut torch_sys::C_tensor,
        key_cache: *mut torch_sys::C_tensor,
        value_cache: *mut torch_sys::C_tensor,
        slot_mapping: *mut torch_sys::C_tensor,
    );
}